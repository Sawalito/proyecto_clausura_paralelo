// =============================================================================
// bow_mpi.cpp
// -----------------------------------------------------------------------------
// Version PARALELA del proyecto Bag of Words usando MPI.
//
// Modelo de paralelizacion: SPMD (Single Program Multiple Data). El mismo
// binario corre en `q` procesos; cada uno se identifica por su `rank`
// (0..q-1) y procesa un subconjunto de los libros. Rank 0 actua como
// "raiz" (root) para las comunicaciones colectivas y para la escritura
// final del CSV.
//
// Pipeline:
//   1. Lectura + Bcast de URLs       -> rank 0 lee, todos reciben.
//   2. Asignacion local con LPT      -> rank 0 hace stat al cache, broadcast
//                                       de pesos, todos corren LPT identicamente.
//   3. Descarga local                -> cada rank descarga sus libros.
//   4. Tokenizacion local            -> cada rank tokeniza lo suyo.
//   5. Vocabulario global            -> Gatherv (palabras locales) -> union
//                                       en rank 0 -> Bcast a todos.
//   6. Construccion de filas locales -> cada rank arma sus filas de la matriz.
//   7. Gatherv de filas              -> rank 0 reordena por indice original.
//   8. Escritura del CSV             -> solo rank 0 (un unico CSV de salida).
//
// Medicion: cada fase mide tiempo local con MPI_Wtime, y al final hacemos
// MPI_Reduce con MPI_MAX. El proceso mas lento dicta el tiempo paralelo real.
//
// Compilacion (ver Makefile):
//   mpic++ -O3 -march=native -DOMPI_SKIP_MPICXX -std=c++17 -o bow_mpi
//          bow_mpi.cpp -lcurl
//
// Ejecucion:
//   mpirun --oversubscribe -np <q> ./bow_mpi <urls.txt> <out.csv> [cache_dir]
// =============================================================================

#include "bow_common.hpp"
#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>

int main(int argc, char** argv) {
    // MPI_Init: arranca el entorno MPI y procesa los flags de mpirun.
    // Debe ser la primera llamada MPI del programa.
    MPI_Init(&argc, &argv);

    int rank;  // id de este proceso (0..size-1)
    int size;  // numero total de procesos en el comunicador
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // BOW_VERBOSE=1 reactiva los prints por libro/rank. Por defecto los
    // silenciamos porque mpirun serializa stdout entre ranks, y la chatter
    // anadia ~10-30 ms de wall-clock a cada corrida (notable cuando el
    // computo total puede ser ~0.4 s con cache caliente).
    const bool verbose = std::getenv("BOW_VERBOSE") != nullptr;

    // Validacion: solo rank 0 imprime el error para no spamear.
    if (argc < 3) {
        if (rank == 0)
            std::cerr << "Uso: mpirun -np <q> " << argv[0]
                      << " <urls.txt> <output.csv> [cache_dir]\n";
        MPI_Finalize();
        return 1;
    }
    std::string cache_dir = (argc >= 4) ? argv[3] : "";

    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Barrier inicial para que todos los ranks empiecen a medir desde el
    // mismo punto (de lo contrario los rezagados arrastrarian un offset).
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // =========================================================================
    // FASE 1: Broadcast de URLs
    // -------------------------------------------------------------------------
    // Solo rank 0 conoce el archivo de entrada. Lo lee, serializa el
    // vector<string> a un buffer plano de chars, y lo distribuye:
    //
    //   1) Bcast del numero de URLs (k).
    //   2) Bcast del tamanio del buffer serializado.
    //   3) Bcast del buffer en si.
    //
    // El resto de ranks deserializa para recuperar el vector<string>.
    // =========================================================================
    std::vector<std::string> urls;
    int k = 0;              // numero total de libros (lo sabran todos)
    int urls_buf_size = 0;  // tamanio en bytes del buffer serializado
    std::vector<char> urls_buf;

    if (rank == 0) {
        urls          = read_urls(argv[1]);
        k             = static_cast<int>(urls.size());
        urls_buf      = serialize_strings(urls);
        urls_buf_size = static_cast<int>(urls_buf.size());
    }
    MPI_Bcast(&k,             1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&urls_buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Los ranks != 0 ahora saben cuanto reservar para el buffer.
    if (rank != 0) urls_buf.resize(urls_buf_size);
    MPI_Bcast(urls_buf.data(), urls_buf_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) urls = deserialize_strings(urls_buf);

    if (k == 0) {
        if (rank == 0) std::cerr << "No hay URLs.\n";
        MPI_Finalize();
        return 1;
    }

    // =========================================================================
    // FASE 2: Asignacion local con LPT (balanceo por tamanio de archivo)
    // -------------------------------------------------------------------------
    // Idea: en lugar de repartir libros contiguos por indice (lo que puede
    // caer en el caso patologico de "Shakespeare completo de 5MB cae todo
    // a un solo rank"), usamos LPT para asignar libros grandes primero al
    // rank menos cargado.
    //
    // Como conocemos los pesos sin descargar: leemos el tamanio del archivo
    // cacheado en disco (file_size). En la PRIMERA corrida (cache vacio)
    // todos los pesos son 0 y LPT cae al reparto contiguo automaticamente.
    //
    // Cada rank corre LPT con los MISMOS inputs (book_sizes broadcasteado),
    // asi llegan al MISMO vector `owners` sin tener que comunicarse mas.
    // =========================================================================
    std::vector<long long> book_sizes(k, 0);
    if (rank == 0) {
        for (int i = 0; i < k; ++i)
            book_sizes[i] = static_cast<long long>(cached_file_size(urls[i], cache_dir));
    }
    MPI_Bcast(book_sizes.data(), k, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Convertimos long long -> size_t porque es lo que espera lpt_assign.
    std::vector<size_t> sz_vec(k);
    for (int i = 0; i < k; ++i) sz_vec[i] = static_cast<size_t>(book_sizes[i]);
    std::vector<int> owners = lpt_assign(sz_vec, size);

    // Cada rank se queda con los indices que le toca procesar.
    std::vector<int> my_indices;
    my_indices.reserve(k / size + 1);
    for (int i = 0; i < k; ++i) if (owners[i] == rank) my_indices.push_back(i);
    int local_k = static_cast<int>(my_indices.size());

    // Resumen inicial (solo rank 0).
    if (rank == 0) {
        long long total_sz = 0;
        for (auto s : book_sizes) total_sz += s;
        std::cout << "[MPI] q=" << size << " procesos, k=" << k
                  << " libros, cache=" << (cache_dir.empty() ? "(off)" : cache_dir)
                  << ", balanceo=" << (total_sz > 0 ? "LPT" : "contiguo")
                  << "\n";
    }
    // Detalle por rank (opcional, util para auditar el balanceo).
    if (verbose) {
        std::string idx_str;
        for (size_t i = 0; i < my_indices.size(); ++i) {
            if (i) idx_str += ",";
            idx_str += std::to_string(my_indices[i]);
        }
        std::cout << "[Rank " << rank << "] libros {" << idx_str << "}\n";
    }

    // =========================================================================
    // FASE 3: Descarga local
    // -------------------------------------------------------------------------
    // Cada rank descarga (o lee del cache) solo los libros que le tocan.
    // Es la fase donde se gana mas si hay descarga real, porque las q
    // descargas suceden en paralelo (limitadas por ancho de banda).
    //
    // No ponemos MPI_Barrier alrededor porque reducimos cada t_*_local con
    // MPI_MAX al final: el wall-clock real no se pierde y ahorramos
    // sincronizaciones (cada barrier cuesta ms en clusters chicos).
    // =========================================================================
    double t_dl_start = MPI_Wtime();

    std::vector<std::string> raw_texts(local_k);
    for (int i = 0; i < local_k; ++i) {
        double td0 = MPI_Wtime();
        raw_texts[i] = download_url_cached(urls[my_indices[i]], cache_dir);
        double td1 = MPI_Wtime();
        if (verbose) {
            std::cout << "[Rank " << rank << "] book " << my_indices[i]
                      << " dl=" << (td1 - td0) << "s "
                      << urls[my_indices[i]] << "\n";
        }
    }
    double t_dl_end = MPI_Wtime();

    // =========================================================================
    // FASES 4-6: Computo local
    //   4. Tokenizar cada libro local.
    //   5. Reducir vocabularios locales -> vocabulario global.
    //   6. Construir las filas locales de la matriz.
    // =========================================================================
    double t_cp_start = MPI_Wtime();

    // ----- Fase 4: tokenizacion -----
    std::vector<std::unordered_map<std::string, int>> local_counts(local_k);
    std::unordered_set<std::string> local_vocab_set;
    local_vocab_set.reserve(20000);
    for (int i = 0; i < local_k; ++i) {
        double tt0 = MPI_Wtime();
        local_counts[i] = tokenize_and_count_fast(raw_texts[i]);
        double tt1 = MPI_Wtime();
        // Acumulamos el vocabulario local (union de palabras de mis libros).
        for (const auto& kv : local_counts[i])
            local_vocab_set.insert(kv.first);
        if (verbose) {
            std::cout << "[Rank " << rank << "] book " << my_indices[i]
                      << " tk=" << (tt1 - tt0)
                      << "s unique=" << local_counts[i].size() << "\n";
        }
    }
    double t_cp_local_end = MPI_Wtime();

    // ----- Fase 5: vocabulario global -----
    // Patron clasico: Gatherv -> union en root -> Bcast.
    //
    // Serializamos directo desde el set para evitar copiarlo a un vector
    // intermedio (era una copia de ~600 KB de strings sin necesidad).
    std::vector<char> lvb;  // local_vocab_buf: "palabra\0palabra\0..."
    {
        size_t total = 0;
        for (const auto& w : local_vocab_set) total += w.size() + 1;
        lvb.reserve(total);
        for (const auto& w : local_vocab_set) {
            lvb.insert(lvb.end(), w.begin(), w.end());
            lvb.push_back('\0');
        }
    }
    int lvbs = static_cast<int>(lvb.size());  // tamanio del buffer local

    // Paso A: cada rank reporta a root cuantos bytes va a enviar.
    // rc[i] = bytes que envia el rank i. dp[i] = offset donde se ubica.
    std::vector<int> rc(size), dp(size);
    MPI_Gather(&lvbs, 1, MPI_INT, rc.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Paso B: rank 0 calcula displacements y asigna espacio para recibir
    // la concatenacion de todos los buffers locales.
    std::vector<char> avb;  // all_vocab_buf: concat de todos los lvb
    if (rank == 0) {
        int t = 0;
        for (int i = 0; i < size; ++i) { dp[i] = t; t += rc[i]; }
        avb.resize(t);
    }
    // Paso C: Gatherv junta los buffers de tamanio variable en avb.
    MPI_Gatherv(lvb.data(), lvbs, MPI_CHAR,
                avb.data(), rc.data(), dp.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    // Paso D: rank 0 deduplica con un unordered_set y ordena lex para
    // que las columnas del CSV salgan en orden alfabetico (determinismo
    // bit-a-bit vs la version serial).
    std::vector<std::string> gv;  // global_vocab (solo rank 0 lo llena)
    std::vector<char> gvb;        // global_vocab_buf (serializado para Bcast)
    int gvbs = 0;
    if (rank == 0) {
        std::unordered_set<std::string> gs;
        gs.reserve(50000);
        for (auto& w : deserialize_strings(avb)) gs.insert(std::move(w));
        gv.assign(gs.begin(), gs.end());
        std::sort(gv.begin(), gv.end());
        gvb  = serialize_strings(gv);
        gvbs = static_cast<int>(gvb.size());
    }

    // Paso E: rank 0 distribuye el vocabulario global a todos.
    int V = (rank == 0) ? static_cast<int>(gv.size()) : 0;
    MPI_Bcast(&V,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gvbs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) gvb.resize(gvbs);
    MPI_Bcast(gvb.data(), gvbs, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) gv = deserialize_strings(gvb);

    double t_vocab_end = MPI_Wtime();

    // ----- Fase 6: construir filas locales -----
    // Indice palabra -> columna global (cada rank arma el suyo, son iguales).
    std::unordered_map<std::string, int> word_to_idx;
    word_to_idx.reserve(static_cast<size_t>(V) * 2);
    for (int j = 0; j < V; ++j) word_to_idx[gv[j]] = j;

    // Matriz parcial: local_k filas x V columnas, en orden de `my_indices`.
    // Cuidado: el orden NO es por indice original todavia; eso se reordena
    // en rank 0 mas adelante usando all_indices.
    std::vector<int> local_matrix(static_cast<size_t>(local_k) * V, 0);
    for (int i = 0; i < local_k; ++i) {
        for (const auto& kv : local_counts[i]) {
            auto it = word_to_idx.find(kv.first);
            if (it != word_to_idx.end())
                local_matrix[static_cast<size_t>(i) * V + it->second] = kv.second;
        }
    }
    double t_cp_end = MPI_Wtime();

    // =========================================================================
    // FASE 7: Gatherv de filas al rank 0
    // -------------------------------------------------------------------------
    // Problema: cada rank tiene local_k filas correspondientes a indices NO
    // contiguos (los que le asigno LPT). Necesitamos:
    //   a) Saber cuantas filas envia cada rank (Gather de local_k).
    //   b) Recibir todas las filas concatenadas (Gatherv de local_matrix).
    //   c) Saber a que indice original corresponde cada fila recibida
    //      (Gatherv de my_indices) para poder reordenar.
    // =========================================================================

    // (a) Cuantas filas envia cada rank.
    std::vector<int> all_local_k(size);
    MPI_Gather(&local_k, 1, MPI_INT, all_local_k.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // (c) Que indices originales envia cada rank.
    std::vector<int> idx_counts(size), idx_displs(size);
    std::vector<int> all_indices;
    if (rank == 0) {
        int t = 0;
        for (int i = 0; i < size; ++i) {
            idx_counts[i] = all_local_k[i];
            idx_displs[i] = t;
            t            += idx_counts[i];
        }
        all_indices.resize(t);
    }
    MPI_Gatherv(my_indices.data(), local_k, MPI_INT,
                all_indices.data(), idx_counts.data(), idx_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // (b) Las filas en si: local_k * V ints por rank.
    std::vector<int> mc(size), md(size);  // matrix counts / displs
    std::vector<int> gathered;
    if (rank == 0) {
        int t = 0;
        for (int i = 0; i < size; ++i) {
            mc[i] = all_local_k[i] * V;
            md[i] = t;
            t    += mc[i];
        }
        gathered.resize(t);
    }
    MPI_Gatherv(local_matrix.data(), local_k * V, MPI_INT,
                gathered.data(), mc.data(), md.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Reordenamiento: rank 0 mueve cada fila recibida al slot que le
    // corresponde por su indice original (el del archivo urls.txt). Esto
    // garantiza que out_mpi.csv tenga las filas EXACTAMENTE en el mismo
    // orden que out_serial.csv -> diff bit-a-bit.
    std::vector<int> global_matrix;
    if (rank == 0) {
        global_matrix.assign(static_cast<size_t>(k) * V, 0);
        int gathered_row = 0;
        for (int r = 0; r < size; ++r) {
            for (int b = 0; b < all_local_k[r]; ++b) {
                int orig = all_indices[idx_displs[r] + b];
                const int* src = gathered.data() + static_cast<size_t>(gathered_row) * V;
                int* dst       = global_matrix.data() + static_cast<size_t>(orig) * V;
                std::copy(src, src + V, dst);
                ++gathered_row;
            }
        }
    }

    // =========================================================================
    // FASE 8: Escritura del CSV (solo rank 0)
    // -------------------------------------------------------------------------
    // Mismo truco que en la version serial: armar el CSV completo en un
    // std::string y hacer un solo out.write() al final. Solo rank 0 ejecuta
    // esto porque genera un unico archivo de salida.
    // =========================================================================
    double t_io_start = MPI_Wtime();
    if (rank == 0) {
        std::string csv;
        csv.reserve(static_cast<size_t>(k) * V * 4 + 100000);

        // Header con vocabulario global.
        csv.append("book_id");
        for (const auto& w : gv) { csv.push_back(','); csv.append(w); }
        csv.push_back('\n');

        // Filas en orden de indice original.
        for (int i = 0; i < k; ++i) {
            append_int(csv, i);
            for (int j = 0; j < V; ++j) {
                csv.push_back(',');
                append_int(csv, global_matrix[static_cast<size_t>(i) * V + j]);
            }
            csv.push_back('\n');
        }
        std::ofstream out(argv[2], std::ios::binary);
        out.write(csv.data(), csv.size());
    }
    double t_io_end = MPI_Wtime();

    curl_global_cleanup();
    double t_end = MPI_Wtime();

    // =========================================================================
    // Reduccion de tiempos: max sobre ranks = tiempo paralelo real
    // -------------------------------------------------------------------------
    // Cada rank termina la fase X en un wall-clock distinto. El tiempo
    // paralelo "real" de esa fase es el del proceso mas lento (porque los
    // demas esperaran a este en la siguiente comunicacion colectiva).
    // De ahi el MPI_MAX. Si quisieramos reportar tiempo "medio" usariamos
    // MPI_SUM y dividiriamos por `size`, pero eso enmascara desbalances.
    // =========================================================================
    double dl_local    = t_dl_end       - t_dl_start;
    double tk_local    = t_cp_local_end - t_cp_start;       // tokenizacion
    double voc_local   = t_vocab_end    - t_cp_local_end;   // vocab global
    double mat_local   = t_cp_end       - t_vocab_end;      // matriz local
    double cp_local    = t_cp_end       - t_cp_start;       // computo total
    double io_local    = t_io_end       - t_io_start;
    double total_local = t_end          - t_start;

    double dl_max, tk_max, voc_max, mat_max, cp_max, io_max, total_max;
    MPI_Reduce(&dl_local,    &dl_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tk_local,    &tk_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&voc_local,   &voc_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mat_local,   &mat_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&cp_local,    &cp_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&io_local,    &io_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "----------------------------------------\n";
        std::cout << "[MPI] Tiempo descarga:    " << dl_max    << " s\n";
        std::cout << "[MPI] Tiempo tokenize:    " << tk_max    << " s\n";
        std::cout << "[MPI] Tiempo vocab(comm): " << voc_max   << " s\n";
        std::cout << "[MPI] Tiempo matriz:      " << mat_max   << " s\n";
        std::cout << "[MPI] Tiempo computo:     " << cp_max    << " s\n";
        std::cout << "[MPI] Tiempo io:          " << io_max    << " s\n";
        std::cout << "[MPI] Tiempo total:       " << total_max << " s\n";
        std::cout << "[MPI] Vocabulario:        " << V << " palabras\n";
        std::cout << "[MPI] Matriz:             " << k << " x " << V << "\n";
        std::cout << "[MPI] CSV:                " << argv[2] << "\n";
    }

    // MPI_Finalize: cierra el entorno MPI. Despues de esto no se puede
    // llamar a ninguna funcion MPI_*.
    MPI_Finalize();
    return 0;
}
