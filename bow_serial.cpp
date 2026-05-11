// =============================================================================
// bow_serial.cpp  (versión optimizada)
// -----------------------------------------------------------------------------
// Versión SERIAL del proyecto Bag of Words.
// Sirve como baseline para calcular el speed-up de la versión MPI.
//
// Compilación:
//   g++ -O3 -march=native -std=c++17 -o bow_serial bow_serial.cpp -lcurl
//
// Uso:
//   ./bow_serial <urls.txt> <output.csv> [cache_dir]
//
//   - urls.txt:   URLs de Project Gutenberg (una por línea).
//   - output.csv: matriz BoW de salida.
//   - cache_dir:  opcional. Si se da, los libros se guardan en disco y se
//                 reusan en corridas sucesivas. Permite separar el tiempo
//                 de descarga del tiempo de cómputo en los benchmarks.
// =============================================================================

#include "bow_common.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unordered_set>

using clk = std::chrono::high_resolution_clock;
inline double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0]
                  << " <urls.txt> <output.csv> [cache_dir]\n";
        return 1;
    }
    std::string cache_dir = (argc >= 4) ? argv[3] : "";

    curl_global_init(CURL_GLOBAL_DEFAULT);
    auto t_start = clk::now();

    // Fase 1: leer URLs
    std::vector<std::string> urls = read_urls(argv[1]);
    int k = static_cast<int>(urls.size());
    std::cout << "[Serial] Procesando " << k << " libros"
              << (cache_dir.empty() ? " (sin cache)" : " (cache=" + cache_dir + ")")
              << "\n";

    // -------------------------------------------------------------------------
    // Fase 2: Descarga (medida aparte para separarla del cómputo)
    // -------------------------------------------------------------------------
    auto t_dl_start = clk::now();
    std::vector<std::string> raw_texts(k);
    for (int i = 0; i < k; ++i) {
        auto td0 = clk::now();
        raw_texts[i] = download_url_cached(urls[i], cache_dir);
        auto td1 = clk::now();
        std::cout << "[Serial] (" << (i + 1) << "/" << k << ") dl="
                  << secs(td0, td1) << "s " << urls[i] << "\n";
    }
    auto t_dl_end = clk::now();

    // -------------------------------------------------------------------------
    // Fase 3: Cómputo (tokenización + construcción de matriz)
    // -------------------------------------------------------------------------
    auto t_cp_start = clk::now();
    std::vector<std::unordered_map<std::string, int>> book_counts(k);
    std::unordered_set<std::string> vocab_set;
    vocab_set.reserve(50000);

    for (int i = 0; i < k; ++i) {
        auto tt0 = clk::now();
        book_counts[i] = tokenize_and_count_fast(raw_texts[i]);
        auto tt1 = clk::now();
        for (const auto& kv : book_counts[i]) vocab_set.insert(kv.first);
        std::cout << "[Serial] (" << (i + 1) << "/" << k << ") tk="
                  << secs(tt0, tt1) << "s unique=" << book_counts[i].size() << "\n";
    }
    // Ordenar lex al final: más rápido que std::set durante la inserción.
    std::vector<std::string> vocab(vocab_set.begin(), vocab_set.end());
    std::sort(vocab.begin(), vocab.end());
    int V = static_cast<int>(vocab.size());

    // Índice: palabra -> columna
    std::unordered_map<std::string, int> word_to_idx;
    word_to_idx.reserve(static_cast<size_t>(V) * 2);
    for (int j = 0; j < V; ++j) word_to_idx[vocab[j]] = j;

    // Matriz row-major
    std::vector<int> matrix(static_cast<size_t>(k) * V, 0);
    for (int i = 0; i < k; ++i) {
        for (const auto& kv : book_counts[i]) {
            auto it = word_to_idx.find(kv.first);
            if (it != word_to_idx.end())
                matrix[static_cast<size_t>(i) * V + it->second] = kv.second;
        }
    }
    auto t_cp_end = clk::now();

    // -------------------------------------------------------------------------
    // Fase 4: Escritura del CSV (buffer en memoria + un solo write)
    // -------------------------------------------------------------------------
    auto t_io_start = clk::now();
    std::string csv;
    csv.reserve(static_cast<size_t>(k) * V * 4 + 100000);  // estimación holgada
    csv.append("book_id");
    for (const auto& w : vocab) { csv.push_back(','); csv.append(w); }
    csv.push_back('\n');
    for (int i = 0; i < k; ++i) {
        append_int(csv, i);
        for (int j = 0; j < V; ++j) {
            csv.push_back(',');
            append_int(csv, matrix[static_cast<size_t>(i) * V + j]);
        }
        csv.push_back('\n');
    }
    std::ofstream out(argv[2], std::ios::binary);
    out.write(csv.data(), csv.size());
    out.close();
    auto t_io_end = clk::now();

    curl_global_cleanup();
    auto t_end = clk::now();

    // -------------------------------------------------------------------------
    // Reporte
    // -------------------------------------------------------------------------
    double t_dl    = secs(t_dl_start, t_dl_end);
    double t_cp    = secs(t_cp_start, t_cp_end);
    double t_io    = secs(t_io_start, t_io_end);
    double t_total = secs(t_start,   t_end);

    std::cout << "----------------------------------------\n";
    std::cout << "[Serial] Tiempo descarga:  " << t_dl    << " s\n";
    std::cout << "[Serial] Tiempo computo:   " << t_cp    << " s\n";
    std::cout << "[Serial] Tiempo io:        " << t_io    << " s\n";
    std::cout << "[Serial] Tiempo total:     " << t_total << " s\n";
    std::cout << "[Serial] Vocabulario:      " << V << " palabras\n";
    std::cout << "[Serial] Matriz:           " << k << " x " << V << "\n";
    std::cout << "[Serial] CSV:              " << argv[2] << "\n";
    return 0;
}