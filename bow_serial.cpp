// =============================================================================
// bow_serial.cpp
// -----------------------------------------------------------------------------
// Version serial del proyecto Bag of Words. Sirve como baseline para comparar
// tiempos contra la version MPI.
// =============================================================================

#include "bow_common.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Alias corto para el reloj de alta resolucion.
using clk = std::chrono::high_resolution_clock;

inline double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <urls.txt> <output.csv> [cache_dir]\n";
        return 1;
    }

    const std::string urls_file = argv[1];
    const std::string output_file = argv[2];
    const std::string cache_dir = (argc >= 4) ? argv[3] : "";

    // libcurl exige init/cleanup globales para inicializar SSL, threads, etc.
    curl_global_init(CURL_GLOBAL_DEFAULT);
    auto t_start = clk::now();

    // Fase 1: lectura y validacion de URLs.
    std::vector<std::string> urls = read_urls(urls_file);
    const int k = static_cast<int>(urls.size());
    if (k == 0) {
        std::cerr << "[Serial] No hay URLs validas en " << urls_file << "\n";
        curl_global_cleanup();
        return 1;
    }

    std::cout << "[Serial] Procesando " << k << " libros"
              << (cache_dir.empty() ? " (sin cache)" : " (cache=" + cache_dir + ")")
              << "\n";

    // Fase 2: descarga o lectura desde cache.
    auto t_dl_start = clk::now();
    std::vector<std::string> raw_texts(k);
    bool download_ok = true;
    for (int i = 0; i < k; ++i) {
        std::cout << "[Serial] (" << (i + 1) << "/" << k << ") " << urls[i] << "\n";
        raw_texts[i] = download_url_cached(urls[i], cache_dir);
        if (raw_texts[i].empty()) {
            std::cerr << "[Serial] Descarga vacia para URL " << (i + 1) << ": "
                      << urls[i] << "\n";
            download_ok = false;
        }
    }
    auto t_dl_end = clk::now();

    if (!download_ok) {
        curl_global_cleanup();
        return 1;
    }

    // Fase 3: tokenizacion, vocabulario global y matriz BoW.
    auto t_cp_start = clk::now();

    // book_counts[i]  -> hashmap palabra -> frecuencia para el libro i.
    std::vector<std::unordered_map<std::string, int>> book_counts(k);

    // vocab_set       -> union de todas las palabras vistas en cualquier libro.
    // Usamos un set hash, luego lo ordenamos al final (mas barato que std::set).
    std::unordered_set<std::string> vocab_set;
    vocab_set.reserve(50000);

    for (int i = 0; i < k; ++i) {
        auto tt0 = clk::now();
        book_counts[i] = tokenize_and_count_fast(raw_texts[i]);
        for (const auto &kv : book_counts[i])
            vocab_set.insert(kv.first);
    }

    std::vector<std::string> vocab(vocab_set.begin(), vocab_set.end());
    std::sort(vocab.begin(), vocab.end());
    const int V = static_cast<int>(vocab.size());

    std::unordered_map<std::string, int> word_to_idx;
    word_to_idx.reserve(static_cast<size_t>(V) * 2);
    for (int j = 0; j < V; ++j)
        word_to_idx[vocab[j]] = j;

    std::vector<int> matrix(static_cast<size_t>(k) * V, 0);
    for (int i = 0; i < k; ++i) {
        for (const auto &kv : book_counts[i]) {
            auto it = word_to_idx.find(kv.first);
            if (it != word_to_idx.end()) {
                matrix[static_cast<size_t>(i) * V + it->second] = kv.second;
            }
        }
    }
    auto t_cp_end = clk::now();

    // Fase 4: escritura del CSV en un solo bloque.
    auto t_io_start = clk::now();
    std::string csv;
    csv.reserve(static_cast<size_t>(k) * V * 4 + 100000);
    csv.append("book_id");
    for (const auto &w : vocab) {
        csv.push_back(',');
        csv.append(w);
    }
    csv.push_back('\n');

    for (int i = 0; i < k; ++i) {
        append_int(csv, i);
        for (int j = 0; j < V; ++j) {
            csv.push_back(',');
            append_int(csv, matrix[static_cast<size_t>(i) * V + j]);
        }
        csv.push_back('\n');
    }

    std::ofstream out(output_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[Serial] No se pudo abrir " << output_file << " para escritura\n";
        curl_global_cleanup();
        return 1;
    }
    out.write(csv.data(), static_cast<std::streamsize>(csv.size()));
    if (!out.good()) {
        std::cerr << "[Serial] Error escribiendo " << output_file << "\n";
        curl_global_cleanup();
        return 1;
    }
    out.close();
    auto t_io_end = clk::now();

    curl_global_cleanup();
    auto t_end = clk::now();

    const double t_dl = secs(t_dl_start, t_dl_end);
    const double t_cp = secs(t_cp_start, t_cp_end);
    const double t_io = secs(t_io_start, t_io_end);
    const double t_total = secs(t_start, t_end);

    std::cout << "----------------------------------------\n";
    std::cout << "[Serial] Tiempo descarga:  " << t_dl << " s\n";
    std::cout << "[Serial] Tiempo computo:   " << t_cp << " s\n";
    std::cout << "[Serial] Tiempo io:        " << t_io << " s\n";
    std::cout << "[Serial] Tiempo total:     " << t_total << " s\n";
    std::cout << "[Serial] Vocabulario:      " << V << " palabras\n";
    std::cout << "[Serial] Matriz:           " << k << " x " << V << "\n";
    std::cout << "[Serial] CSV:              " << output_file << "\n";
    return 0;
}
