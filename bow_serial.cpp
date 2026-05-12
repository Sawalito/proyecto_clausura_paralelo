// =============================================================================
// bow_serial.cpp
// -----------------------------------------------------------------------------
// Version SERIAL del proyecto Bag of Words.
//
// Que hace:
//   1. Lee una lista de URLs (libros de Project Gutenberg).
//   2. Descarga cada libro (con cache opcional en disco).
//   3. Tokeniza y cuenta palabras de cada libro.
//   4. Construye una matriz BoW (filas = libros, columnas = palabras del
//      vocabulario global, celdas = frecuencia).
//   5. Escribe la matriz como CSV.
//
// Este binario sirve de BASELINE para calcular el speed-up de bow_mpi:
//   speed-up = T_serial / T_mpi
//
// Compilacion (ver Makefile):
//   g++ -O3 -march=native -std=c++17 -o bow_serial bow_serial.cpp -lcurl
//
// Uso:
//   ./bow_serial <urls.txt> <output.csv> [cache_dir]
//
//   - urls.txt:   archivo con URLs de Project Gutenberg (una por linea).
//   - output.csv: ruta del CSV de salida.
//   - cache_dir:  opcional. Si se da, los libros se guardan en disco la
//                 primera vez y se reusan en corridas sucesivas. Permite
//                 separar el tiempo de descarga del tiempo de computo
//                 en los benchmarks.
// =============================================================================

#include "bow_common.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unordered_set>

// Alias corto para el reloj de alta resolucion.
using clk = std::chrono::high_resolution_clock;

// Diferencia entre dos timestamps, en segundos (double).
inline double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    // ---------- Validacion de argumentos ----------
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0]
                  << " <urls.txt> <output.csv> [cache_dir]\n";
        return 1;
    }
    std::string cache_dir = (argc >= 4) ? argv[3] : "";

    // libcurl exige init/cleanup globales para inicializar SSL, threads, etc.
    curl_global_init(CURL_GLOBAL_DEFAULT);
    auto t_start = clk::now();

    // =========================================================================
    // FASE 1: Leer URLs del archivo
    // =========================================================================
    std::vector<std::string> urls = read_urls(argv[1]);
    int k = static_cast<int>(urls.size());  // numero total de libros
    std::cout << "[Serial] Procesando " << k << " libros"
              << (cache_dir.empty() ? " (sin cache)" : " (cache=" + cache_dir + ")")
              << "\n";

    // =========================================================================
    // FASE 2: Descarga
    // -------------------------------------------------------------------------
    // La medimos aparte porque es I/O dominado por la red (o por lecturas de
    // disco si hay cache). En el benchmark queremos poder reportar speed-up
    // CPU "puro", asi que separamos descarga de computo.
    // =========================================================================
    auto t_dl_start = clk::now();
    std::vector<std::string> raw_texts(k);  // crudos: HTML/TXT tal como viene
    for (int i = 0; i < k; ++i) {
        auto td0 = clk::now();
        raw_texts[i] = download_url_cached(urls[i], cache_dir);
        auto td1 = clk::now();
        std::cout << "[Serial] (" << (i + 1) << "/" << k << ") dl="
                  << secs(td0, td1) << "s " << urls[i] << "\n";
    }
    auto t_dl_end = clk::now();

    // =========================================================================
    // FASE 3: Computo (tokenizacion + construccion de la matriz)
    // =========================================================================
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
        auto tt1 = clk::now();

        // Acumular el vocabulario global con las palabras unicas del libro.
        for (const auto& kv : book_counts[i]) vocab_set.insert(kv.first);

        std::cout << "[Serial] (" << (i + 1) << "/" << k << ") tk="
                  << secs(tt0, tt1) << "s unique=" << book_counts[i].size() << "\n";
    }

    // Ordenar el vocabulario lexicograficamente para que el CSV de salida
    // sea deterministico (y comparable bit-a-bit con la salida del MPI).
    std::vector<std::string> vocab(vocab_set.begin(), vocab_set.end());
    std::sort(vocab.begin(), vocab.end());
    int V = static_cast<int>(vocab.size());  // tamanio del vocabulario

    // Indice palabra -> columna. Lookup O(1) cuando llenamos la matriz.
    std::unordered_map<std::string, int> word_to_idx;
    word_to_idx.reserve(static_cast<size_t>(V) * 2);  // load factor ~0.5
    for (int j = 0; j < V; ++j) word_to_idx[vocab[j]] = j;

    // Matriz row-major plana: fila i, columna j -> matrix[i*V + j].
    // Aplanarla nos da contigüidad de memoria (mejor cache).
    std::vector<int> matrix(static_cast<size_t>(k) * V, 0);
    for (int i = 0; i < k; ++i) {
        for (const auto& kv : book_counts[i]) {
            auto it = word_to_idx.find(kv.first);
            if (it != word_to_idx.end())
                matrix[static_cast<size_t>(i) * V + it->second] = kv.second;
        }
    }
    auto t_cp_end = clk::now();

    // =========================================================================
    // FASE 4: Escritura del CSV
    // -------------------------------------------------------------------------
    // Estrategia: armar TODO el CSV en un std::string en memoria y hacer un
    // unico out.write() al final. Esto es mucho mas rapido que escribir
    // celda-por-celda con operator<< porque:
    //   - operator<< consulta locale en cada llamada.
    //   - El kernel hace 1 write() en lugar de millones.
    // =========================================================================
    auto t_io_start = clk::now();
    std::string csv;

    // Estimamos un tamanio holgado para evitar realloc durante el llenado:
    // ~4 chars por celda + cabecera ~100 KB para el header con todas las
    // palabras del vocabulario.
    csv.reserve(static_cast<size_t>(k) * V * 4 + 100000);

    // ---- Header: "book_id,palabra1,palabra2,...,palabraV\n"
    csv.append("book_id");
    for (const auto& w : vocab) { csv.push_back(','); csv.append(w); }
    csv.push_back('\n');

    // ---- Filas: "i,c0,c1,...,cV\n"
    for (int i = 0; i < k; ++i) {
        append_int(csv, i);
        for (int j = 0; j < V; ++j) {
            csv.push_back(',');
            append_int(csv, matrix[static_cast<size_t>(i) * V + j]);
        }
        csv.push_back('\n');
    }

    // Un solo write binario: el CSV ya viene con sus saltos de linea.
    std::ofstream out(argv[2], std::ios::binary);
    out.write(csv.data(), csv.size());
    out.close();
    auto t_io_end = clk::now();

    curl_global_cleanup();
    auto t_end = clk::now();

    // =========================================================================
    // Reporte de tiempos
    // =========================================================================
    double t_dl    = secs(t_dl_start, t_dl_end);
    double t_cp    = secs(t_cp_start, t_cp_end);
    double t_io    = secs(t_io_start, t_io_end);
    double t_total = secs(t_start,    t_end);

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
