// =============================================================================
// bow_mpi.cpp  (versión optimizada)
// -----------------------------------------------------------------------------
// Versión PARALELA del proyecto Bag of Words.
//
// Pipeline (igual que la versión anterior, ahora con timing detallado):
//   1. Lectura + Bcast de URLs               (rank 0 -> todos)
//   2. Asignación local de libros            (cálculo determinista)
//   3. Descarga local                        --> medida como T_download
//   4. Tokenización local                     |
//   5. Reducción del vocabulario global       |- medida como T_compute + T_comm
//   6. Construcción de filas locales          |
//   7. Gatherv de la matriz al rank 0         |
//   8. Escritura del CSV                      --> medida como T_io
//
// Para cada fase reportamos max sobre ranks (Reduce con MPI_MAX), porque el
// proceso más lento es el que dicta el tiempo paralelo real.
//
// Compilación:
//   mpic++ -O3 -march=native -DOMPI_SKIP_MPICXX -std=c++17 -o bow_mpi
//          bow_mpi.cpp -lcurl
// =============================================================================

#include "bow_common.hpp"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            std::cerr << "Uso: mpirun -np <q> " << argv[0]
                      << " <urls.txt> <output.csv> [cache_dir]\n";
        MPI_Finalize();
        return 1;
    }
    std::string cache_dir = (argc >= 4) ? argv[3] : "";

    curl_global_init(CURL_GLOBAL_DEFAULT);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // =========================================================================
    // FASE 1: Bcast de URLs
    // =========================================================================
    std::vector<std::string> urls;
    int k = 0, urls_buf_size = 0;
    std::vector<char> urls_buf;

    if (rank == 0) {
        urls          = read_urls(argv[1]);
        k             = static_cast<int>(urls.size());
        urls_buf      = serialize_strings(urls);
        urls_buf_size = static_cast<int>(urls_buf.size());
    }
    MPI_Bcast(&k,             1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&urls_buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) urls_buf.resize(urls_buf_size);
    MPI_Bcast(urls_buf.data(), urls_buf_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) urls = deserialize_strings(urls_buf);

    if (k == 0) {
        if (rank == 0) std::cerr << "No hay URLs.\n";
        MPI_Finalize();
        return 1;
    }

    // =========================================================================
    // FASE 2: Asignación local
    // =========================================================================
    int per_proc  = k / size;
    int remainder = k % size;
    int local_k   = per_proc + (rank < remainder ? 1 : 0);
    int start     = rank * per_proc + std::min(rank, remainder);

    if (rank == 0)
        std::cout << "[MPI] q=" << size << " procesos, k=" << k
                  << " libros, cache=" << (cache_dir.empty() ? "(off)" : cache_dir)
                  << "\n";
    std::cout << "[Rank " << rank << "] libros [" << start
              << ", " << (start + local_k) << ")\n";

    // =========================================================================
    // FASE 3: DESCARGA LOCAL (medida aparte)
    // =========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double t_dl_start = MPI_Wtime();

    std::vector<std::string> raw_texts(local_k);
    for (int i = 0; i < local_k; ++i) {
        raw_texts[i] = download_url_cached(urls[start + i], cache_dir);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_dl_end = MPI_Wtime();

    // =========================================================================
    // FASE 4-6: CÓMPUTO LOCAL
    //   4. Tokenización
    //   5. Vocabulario global (Gatherv -> unión -> Bcast)
    //   6. Construcción de filas locales
    // =========================================================================
    double t_cp_start = MPI_Wtime();

    // Tokenización
    std::vector<std::unordered_map<std::string, int>> local_counts(local_k);
    std::unordered_set<std::string> local_vocab_set;
    local_vocab_set.reserve(20000);
    for (int i = 0; i < local_k; ++i) {
        local_counts[i] = tokenize_and_count_fast(raw_texts[i]);
        for (const auto& kv : local_counts[i])
            local_vocab_set.insert(kv.first);
    }

    double t_cp_local_end = MPI_Wtime();

    // Vocabulario global: Gatherv -> unión en rank 0 -> Bcast
    std::vector<std::string> lvv(local_vocab_set.begin(),
                                 local_vocab_set.end());
    std::vector<char> lvb = serialize_strings(lvv);
    int lvbs = static_cast<int>(lvb.size());

    std::vector<int> rc(size), dp(size);
    MPI_Gather(&lvbs, 1, MPI_INT, rc.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> avb;
    if (rank == 0) {
        int t = 0;
        for (int i = 0; i < size; ++i) { dp[i] = t; t += rc[i]; }
        avb.resize(t);
    }
    MPI_Gatherv(lvb.data(), lvbs, MPI_CHAR,
                avb.data(), rc.data(), dp.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    std::vector<std::string> gv;
    std::vector<char> gvb;
    int gvbs = 0;
    if (rank == 0) {
        std::unordered_set<std::string> gs;
        gs.reserve(50000);
        for (auto& w : deserialize_strings(avb)) gs.insert(std::move(w));
        gv.assign(gs.begin(), gs.end());
        std::sort(gv.begin(), gv.end());  // ordenado lex para CSV determinístico
        gvb  = serialize_strings(gv);
        gvbs = static_cast<int>(gvb.size());
    }
    int V = (rank == 0) ? static_cast<int>(gv.size()) : 0;
    MPI_Bcast(&V,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gvbs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) gvb.resize(gvbs);
    MPI_Bcast(gvb.data(), gvbs, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) gv = deserialize_strings(gvb);

    double t_vocab_end = MPI_Wtime();

    // Construcción de filas locales (con índice O(1) word -> col)
    std::unordered_map<std::string, int> word_to_idx;
    word_to_idx.reserve(static_cast<size_t>(V) * 2);
    for (int j = 0; j < V; ++j) word_to_idx[gv[j]] = j;

    std::vector<int> local_matrix(static_cast<size_t>(local_k) * V, 0);
    for (int i = 0; i < local_k; ++i) {
        for (const auto& kv : local_counts[i]) {
            auto it = word_to_idx.find(kv.first);
            if (it != word_to_idx.end())
                local_matrix[static_cast<size_t>(i) * V + it->second] = kv.second;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_cp_end = MPI_Wtime();

    // =========================================================================
    // FASE 7: Gatherv de filas al rank 0
    // =========================================================================
    std::vector<int> all_local_k(size);
    MPI_Gather(&local_k, 1, MPI_INT, all_local_k.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> mc(size), md(size);
    std::vector<int> global_matrix;
    if (rank == 0) {
        int t = 0;
        for (int i = 0; i < size; ++i) {
            mc[i] = all_local_k[i] * V;
            md[i] = t;
            t    += mc[i];
        }
        global_matrix.resize(t);
    }
    MPI_Gatherv(local_matrix.data(), local_k * V, MPI_INT,
                global_matrix.data(), mc.data(), md.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // =========================================================================
    // FASE 8: Escritura del CSV (buffer en memoria, escrito en un solo write)
    // =========================================================================
    double t_io_start = MPI_Wtime();
    if (rank == 0) {
        std::string csv;
        csv.reserve(static_cast<size_t>(k) * V * 4 + 100000);
        csv.append("book_id");
        for (const auto& w : gv) { csv.push_back(','); csv.append(w); }
        csv.push_back('\n');
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
    MPI_Barrier(MPI_COMM_WORLD);
    double t_io_end = MPI_Wtime();

    curl_global_cleanup();
    double t_end = MPI_Wtime();

    // =========================================================================
    // Reducción de tiempos (max sobre ranks = el tiempo paralelo real)
    // =========================================================================
    double dl_local    = t_dl_end       - t_dl_start;
    double tk_local    = t_cp_local_end - t_cp_start;       // tokenización
    double voc_local   = t_vocab_end    - t_cp_local_end;   // vocab global
    double mat_local   = t_cp_end       - t_vocab_end;      // matriz local
    double cp_local    = t_cp_end       - t_cp_start;       // cómputo total
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

    MPI_Finalize();
    return 0;
}