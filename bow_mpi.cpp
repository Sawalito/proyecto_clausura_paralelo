// =============================================================================
// bow_mpi.cpp
// -----------------------------------------------------------------------------
// Version paralela MPI del proyecto Bag of Words.
//
// Pipeline:
//   1. rank 0 lee URLs y las distribuye con MPI_Bcast.
//   2. cada rank procesa un bloque deterministico de libros.
//   3. cada rank descarga/tokeniza sus libros y construye conteos locales.
//   4. los vocabularios locales se reunen con MPI_Gather/MPI_Gatherv.
//   5. rank 0 une y ordena el vocabulario global, luego lo difunde.
//   6. cada rank construye sus filas de matriz y rank 0 las recolecta.
//   7. rank 0 escribe el CSV final.
// =============================================================================

#include "bow_common.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T> T *data_or_null(std::vector<T> &v) {
    return v.empty() ? nullptr : v.data();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // BOW_VERBOSE=1 reactiva los prints por libro/rank. Por defecto los
    // silenciamos porque mpirun serializa stdout entre ranks, y la chatter
    // anadia ~10-30 ms de wall-clock a cada corrida (notable cuando el
    // computo total puede ser ~0.4 s con cache caliente).
    const bool verbose = std::getenv("BOW_VERBOSE") != nullptr;

    // Validacion: solo rank 0 imprime el error para no spamear.
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Uso: mpirun -np <q> " << argv[0]
                      << " <urls.txt> <output.csv> [cache_dir]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::string urls_file = argv[1];
    const std::string output_file = argv[2];
    const std::string cache_dir = (argc >= 4) ? argv[3] : "";
    const bool verbose = std::getenv("BOW_VERBOSE") != nullptr;

    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Barrier inicial para que todos los ranks empiecen a medir desde el
    // mismo punto (de lo contrario los rezagados arrastrarian un offset).
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Fase 1: rank 0 lee las URLs. MPI_Bcast copia k y el buffer serializado a
    // todos los procesos para que cada rank pueda calcular su propio rango.
    std::vector<std::string> urls;
    int k = 0;
    int urls_buf_size = 0;
    std::vector<char> urls_buf;

    if (rank == 0) {
        urls = read_urls(urls_file);
        k = static_cast<int>(urls.size());
        urls_buf = serialize_strings(urls);
        urls_buf_size = static_cast<int>(urls_buf.size());
    }

    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&urls_buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        urls_buf.resize(urls_buf_size);
    MPI_Bcast(data_or_null(urls_buf), urls_buf_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
        urls = deserialize_strings(urls_buf);

    if (k == 0) {
        if (rank == 0) {
            std::cerr << "[MPI] No hay URLs validas en " << urls_file << "\n";
        }
        curl_global_cleanup();
        MPI_Finalize();
        return 1;
    }

    // Fase 2: reparto por bloques casi iguales. Si hay mas procesos que libros,
    // algunos ranks reciben local_k=0 y participan solo en los colectivos.
    const int per_proc = k / size;
    const int remainder = k % size;
    const int local_k = per_proc + (rank < remainder ? 1 : 0);
    const int start = rank * per_proc + std::min(rank, remainder);

    if (rank == 0) {
        std::cout << "[MPI] q=" << size << " procesos, k=" << k
                  << " libros, cache=" << (cache_dir.empty() ? "(off)" : cache_dir)
                  << ", balanceo=" << (total_sz > 0 ? "LPT" : "contiguo")
                  << "\n";
    }

    if (verbose) {
        for (int r = 0; r < size; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == r) {
                std::cout << "[Rank " << rank << "] libros [" << start << ", "
                          << (start + local_k) << ")\n";
                std::cout.flush();
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Fase 3: descarga local.
    MPI_Barrier(MPI_COMM_WORLD);
    double t_dl_start = MPI_Wtime();

    std::vector<std::string> raw_texts(local_k);
    int local_empty_downloads = 0;
    for (int i = 0; i < local_k; ++i) {
        const int global_i = start + i;
        raw_texts[i] = download_url_cached(urls[global_i], cache_dir);
        if (raw_texts[i].empty()) {
            ++local_empty_downloads;
            std::cerr << "[MPI rank " << rank << "] Descarga vacia para URL "
                      << (global_i + 1) << ": " << urls[global_i] << "\n";
        }
    }
    double t_dl_end = MPI_Wtime();

    int total_empty_downloads = 0;
    MPI_Allreduce(&local_empty_downloads, &total_empty_downloads, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (total_empty_downloads > 0) {
        if (rank == 0) {
            std::cerr << "[MPI] Fallaron " << total_empty_downloads
                      << " descargas; se detiene la corrida.\n";
        }
        curl_global_cleanup();
        MPI_Finalize();
        return 1;
    }

    // Fase 4: tokenizacion local.
    MPI_Barrier(MPI_COMM_WORLD);
    double t_cp_start = MPI_Wtime();

    std::vector<std::unordered_map<std::string, int>> local_counts(local_k);
    std::unordered_set<std::string> local_vocab_set;
    local_vocab_set.reserve(20000);
    for (int i = 0; i < local_k; ++i) {
        double tt0 = MPI_Wtime();
        local_counts[i] = tokenize_and_count_fast(raw_texts[i]);
        for (const auto &kv : local_counts[i])
            local_vocab_set.insert(kv.first);
        if (verbose) {
            std::cout << "[Rank " << rank << "] book " << my_indices[i]
                      << " tk=" << (tt1 - tt0)
                      << "s unique=" << local_counts[i].size() << "\n";
        }
    }

    double t_tokenize_end = MPI_Wtime();

    // Fase 5: vocabulario global.
    //
    // MPI_Gather primero junta tamanos de buffers. MPI_Gatherv permite que cada
    // rank envie una cantidad distinta de bytes, necesaria porque cada vocabulario
    // local tiene tamano diferente.
    std::vector<std::string> local_vocab(local_vocab_set.begin(),
                                         local_vocab_set.end());
    std::vector<char> local_vocab_buf = serialize_strings(local_vocab);
    int local_vocab_bytes = static_cast<int>(local_vocab_buf.size());

    std::vector<int> recv_counts(rank == 0 ? size : 0);
    std::vector<int> displs(rank == 0 ? size : 0);
    MPI_Gather(&local_vocab_bytes, 1, MPI_INT, rank == 0 ? recv_counts.data() : nullptr,
               1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> all_vocab_buf;
    if (rank == 0) {
        int total_bytes = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total_bytes;
            total_bytes += recv_counts[i];
        }
        all_vocab_buf.resize(total_bytes);
    }

    MPI_Gatherv(data_or_null(local_vocab_buf), local_vocab_bytes, MPI_CHAR,
                data_or_null(all_vocab_buf), rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<std::string> global_vocab;
    std::vector<char> global_vocab_buf;
    int global_vocab_bytes = 0;
    if (rank == 0) {
        std::unordered_set<std::string> global_vocab_set;
        global_vocab_set.reserve(50000);
        for (auto &word : deserialize_strings(all_vocab_buf)) {
            global_vocab_set.insert(std::move(word));
        }
        global_vocab.assign(global_vocab_set.begin(), global_vocab_set.end());
        std::sort(global_vocab.begin(), global_vocab.end());
        global_vocab_buf = serialize_strings(global_vocab);
        global_vocab_bytes = static_cast<int>(global_vocab_buf.size());
    }

    int V = (rank == 0) ? static_cast<int>(global_vocab.size()) : 0;

    // MPI_Bcast distribuye el vocabulario final a todos los ranks. Asi cada
    // proceso puede construir filas con las mismas columnas y el CSV queda
    // equivalente al serial.
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_vocab_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        global_vocab_buf.resize(global_vocab_bytes);
    MPI_Bcast(data_or_null(global_vocab_buf), global_vocab_bytes, MPI_CHAR, 0,
              MPI_COMM_WORLD);
    if (rank != 0)
        global_vocab = deserialize_strings(global_vocab_buf);

    double t_vocab_end = MPI_Wtime();

    // Fase 6: matriz local y recoleccion en rank 0.
    std::unordered_map<std::string, int> word_to_idx;
    word_to_idx.reserve(static_cast<size_t>(V) * 2);
    for (int j = 0; j < V; ++j)
        word_to_idx[global_vocab[j]] = j;

    // Matriz parcial: local_k filas x V columnas, en orden de `my_indices`.
    // Cuidado: el orden NO es por indice original todavia; eso se reordena
    // en rank 0 mas adelante usando all_indices.
    std::vector<int> local_matrix(static_cast<size_t>(local_k) * V, 0);
    for (int i = 0; i < local_k; ++i) {
        for (const auto &kv : local_counts[i]) {
            auto it = word_to_idx.find(kv.first);
            if (it != word_to_idx.end()) {
                local_matrix[static_cast<size_t>(i) * V + it->second] = kv.second;
            }
        }
    }

    // MPI_Gather recolecta cuantas filas genero cada rank. Despues MPI_Gatherv
    // reune bloques de matriz de tamano distinto en rank 0.
    std::vector<int> all_local_k(rank == 0 ? size : 0);
    MPI_Gather(&local_k, 1, MPI_INT, rank == 0 ? all_local_k.data() : nullptr, 1,
               MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> matrix_counts(rank == 0 ? size : 0);
    std::vector<int> matrix_displs(rank == 0 ? size : 0);
    std::vector<int> global_matrix;
    if (rank == 0) {
        int total_ints = 0;
        for (int i = 0; i < size; ++i) {
            matrix_counts[i] = all_local_k[i] * V;
            matrix_displs[i] = total_ints;
            total_ints += matrix_counts[i];
        }
        global_matrix.resize(total_ints);
    }

    const int local_matrix_count = local_k * V;
    MPI_Gatherv(data_or_null(local_matrix), local_matrix_count, MPI_INT,
                data_or_null(global_matrix), rank == 0 ? matrix_counts.data() : nullptr,
                rank == 0 ? matrix_displs.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_cp_end = MPI_Wtime();

    // Fase 7: escritura del CSV solo en rank 0.
    double t_io_start = MPI_Wtime();
    int write_failed = 0;
    if (rank == 0) {
        std::string csv;
        csv.reserve(static_cast<size_t>(k) * V * 4 + 100000);

        // Header con vocabulario global.
        csv.append("book_id");
        for (const auto &w : global_vocab) {
            csv.push_back(',');
            csv.append(w);
        }
        csv.push_back('\n');

        for (int i = 0; i < k; ++i) {
            append_int(csv, i);
            for (int j = 0; j < V; ++j) {
                csv.push_back(',');
                append_int(csv, global_matrix[static_cast<size_t>(i) * V + j]);
            }
            csv.push_back('\n');
        }

        std::ofstream out(output_file, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "[MPI] No se pudo abrir " << output_file
                      << " para escritura\n";
            write_failed = 1;
        } else {
            out.write(csv.data(), static_cast<std::streamsize>(csv.size()));
            if (!out.good()) {
                std::cerr << "[MPI] Error escribiendo " << output_file << "\n";
                write_failed = 1;
            }
        }
    }

    MPI_Bcast(&write_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_io_end = MPI_Wtime();

    curl_global_cleanup();
    double t_end = MPI_Wtime();

    const double dl_local = t_dl_end - t_dl_start;
    const double tokenize_local = t_tokenize_end - t_cp_start;
    const double vocab_local = t_vocab_end - t_tokenize_end;
    const double matrix_local = t_cp_end - t_vocab_end;
    const double compute_local = t_cp_end - t_cp_start;
    const double io_local = t_io_end - t_io_start;
    const double total_local = t_end - t_start;

    double dl_max = 0.0;
    double tokenize_max = 0.0;
    double vocab_max = 0.0;
    double matrix_max = 0.0;
    double compute_max = 0.0;
    double io_max = 0.0;
    double total_max = 0.0;

    // MPI_Reduce con MPI_MAX reporta el tiempo paralelo observado: aunque otros
    // ranks acaben antes, todos esperan al mas lento en los colectivos/barreras.
    MPI_Reduce(&dl_local, &dl_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tokenize_local, &tokenize_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&vocab_local, &vocab_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&matrix_local, &matrix_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_local, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&io_local, &io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "----------------------------------------\n";
        std::cout << "[MPI] Tiempo descarga:    " << dl_max << " s\n";
        std::cout << "[MPI] Tiempo tokenize:    " << tokenize_max << " s\n";
        std::cout << "[MPI] Tiempo vocab(comm): " << vocab_max << " s\n";
        std::cout << "[MPI] Tiempo matriz:      " << matrix_max << " s\n";
        std::cout << "[MPI] Tiempo computo:     " << compute_max << " s\n";
        std::cout << "[MPI] Tiempo io:          " << io_max << " s\n";
        std::cout << "[MPI] Tiempo total:       " << total_max << " s\n";
        std::cout << "[MPI] Vocabulario:        " << V << " palabras\n";
        std::cout << "[MPI] Matriz:             " << k << " x " << V << "\n";
        std::cout << "[MPI] CSV:                " << output_file << "\n";
    }

    // MPI_Finalize: cierra el entorno MPI. Despues de esto no se puede
    // llamar a ninguna funcion MPI_*.
    MPI_Finalize();
    return write_failed ? 1 : 0;
}
