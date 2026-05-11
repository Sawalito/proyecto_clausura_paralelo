// =============================================================================
// bow_common.hpp
// -----------------------------------------------------------------------------
// Utilidades compartidas por la version serial y la version MPI:
//   - lectura robusta de URLs
//   - descarga HTTP con libcurl
//   - cache local para reproducibilidad de benchmarks
//   - limpieza de metadata de Project Gutenberg
//   - tokenizacion y conteo rapido de palabras
//   - serializacion simple de vector<string> para MPI
// =============================================================================

#ifndef BOW_COMMON_HPP
#define BOW_COMMON_HPP

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <curl/curl.h>

inline char to_lower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

// Criterio pedagogico del proyecto: una palabra es una secuencia de letras
// ASCII y apostrofes. Se conservan palabras de longitud >= 2 y tambien "a"/"i".
inline bool is_word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '\'';
}

inline std::string trim_copy(const std::string &s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

inline int count_http_occurrences(const std::string &s) {
    int count = 0;
    size_t pos = 0;
    while ((pos = s.find("http", pos)) != std::string::npos) {
        ++count;
        pos += 4;
    }
    return count;
}

inline bool looks_like_url(const std::string &s) {
    return s.rfind("https://", 0) == 0 || s.rfind("http://", 0) == 0;
}

// libcurl invoca este callback por bloques. El buffer crece en memoria y luego
// el llamador decide si lo procesa o lo escribe al cache.
inline size_t bow_write_callback(void *contents, size_t size, size_t nmemb,
                                 std::string *userp) {
    const size_t bytes = size * nmemb;
    userp->append(static_cast<char *>(contents), bytes);
    return bytes;
}

// Descarga sincrona con libcurl. No desactiva verificacion SSL; si hay un
// problema de certificados es mejor reportarlo que ocultarlo en un benchmark.
inline std::string download_url(const std::string &url) {
    CURL *curl = curl_easy_init();
    std::string response;
    if (!curl) {
        std::cerr << "[curl] No se pudo inicializar libcurl\n";
        return response;
    }

    response.reserve(1 << 20);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, bow_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "BoW-MPI/2.0");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "[curl] Error descargando " << url << ": "
                  << curl_easy_strerror(res) << "\n";
        return "";
    }

    if (http_code >= 400) {
        std::cerr << "[curl] HTTP " << http_code << " descargando " << url << "\n";
        return "";
    }

    if (response.empty()) {
        std::cerr << "[curl] Respuesta vacia descargando " << url << "\n";
    }
    return response;
}

inline std::string stable_url_hash(const std::string &url) {
    unsigned long long hash = 1469598103934665603ull; // FNV-1a 64-bit
    for (unsigned char c : url) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return std::to_string(hash);
}

inline std::string url_to_cache_path(const std::string &url,
                                     const std::string &cache_dir) {
    return cache_dir + "/" + stable_url_hash(url) + ".cache";
}

inline std::string read_file_binary(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return "";
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string make_temp_cache_path(const std::string &path) {
    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    const auto ticks =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return path + ".tmp." + std::to_string(ticks) + "." + std::to_string(rng());
}

// Cache con escritura en dos pasos: primero un temporal unico, luego rename al
// nombre final. Esto reduce carreras cuando varios ranks MPI piden la misma URL.
inline std::string download_url_cached(const std::string &url,
                                       const std::string &cache_dir) {
    if (cache_dir.empty())
        return download_url(url);

    const std::string path = url_to_cache_path(url, cache_dir);
    std::string cached = read_file_binary(path);
    if (!cached.empty())
        return cached;

    std::string content = download_url(url);
    if (content.empty())
        return content;

    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (ec) {
        std::cerr << "[cache] No se pudo crear " << cache_dir << ": " << ec.message()
                  << "\n";
        return content;
    }

    // Otro proceso pudo haber llenado el cache mientras descargabamos.
    cached = read_file_binary(path);
    if (!cached.empty())
        return cached;

    const std::string tmp_path = make_temp_cache_path(path);
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "[cache] No se pudo abrir temporal " << tmp_path
                      << " para escritura\n";
            return content;
        }
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!out.good()) {
            std::cerr << "[cache] Error escribiendo temporal " << tmp_path << "\n";
            std::filesystem::remove(tmp_path, ec);
            return content;
        }
    }

    if (std::filesystem::exists(path)) {
        std::filesystem::remove(tmp_path, ec);
        cached = read_file_binary(path);
        return cached.empty() ? content : cached;
    }

    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        // Si fallo porque otro rank gano la carrera, usar el archivo final.
        std::filesystem::remove(tmp_path, ec);
        cached = read_file_binary(path);
        if (!cached.empty())
            return cached;
        std::cerr << "[cache] No se pudo mover temporal a cache final: " << ec.message()
                  << "\n";
    }
    return content;
}

// Project Gutenberg agrega encabezados y pies legales. Quitarlos evita que el
// vocabulario mida boilerplate en vez del texto literario.
inline std::string strip_gutenberg_metadata(const std::string &text) {
    static const std::string start_tag = "*** START OF";
    static const std::string end_tag = "*** END OF";

    std::string body = text;
    size_t s = text.find(start_tag);
    if (s != std::string::npos) {
        size_t nl = text.find('\n', s);
        if (nl != std::string::npos)
            body = text.substr(nl + 1);
    }
    size_t e = body.find(end_tag);
    if (e != std::string::npos)
        body = body.substr(0, e);
    return body;
}

// Tokenizador rapido:
//   1. Recorre el texto una sola vez con punteros.
//   2. Salta separadores como espacios, puntuacion y digitos.
//   3. Acumula letras/apostrofes en minusculas ASCII.
//   4. Incrementa el conteo en unordered_map.
//
// Evitar std::regex aqui es intencional: para textos grandes, el scanner manual
// reduce overhead y mantiene clara la unidad de trabajo paralelizable.
inline std::unordered_map<std::string, int>
tokenize_and_count_fast(const std::string &raw_text) {
    std::unordered_map<std::string, int> counts;
    counts.reserve(20000);

    std::string body = strip_gutenberg_metadata(raw_text);
    const char *p = body.data();
    const char *end = p + body.size();

    std::string token;
    token.reserve(64);

    while (p < end) {
        while (p < end && !is_word_char(static_cast<unsigned char>(*p)))
            ++p;

        token.clear();
        while (p < end && is_word_char(static_cast<unsigned char>(*p))) {
            token.push_back(to_lower_ascii(*p));
            ++p;
        }

        if (token.size() > 1 || token == "a" || token == "i") {
            ++counts[token];
        }
    }
    return counts;
}

// Lee una URL por linea. Ignora comentarios y lineas vacias, recorta espacios
// al inicio/final y advierte lineas sospechosas sin detener todo el programa.
inline std::vector<std::string> read_urls(const std::string &filename) {
    std::vector<std::string> urls;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[urls] No se pudo abrir " << filename << "\n";
        return urls;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim_copy(line);
        if (line.empty() || line[0] == '#')
            continue;

        const int http_count = count_http_occurrences(line);
        if (http_count > 1) {
            std::cerr << "[urls] Linea " << line_number
                      << " parece contener mas de una URL: " << line << "\n";
            continue;
        }

        if (!looks_like_url(line)) {
            std::cerr << "[urls] Linea " << line_number
                      << " no parece una URL HTTP/HTTPS: " << line << "\n";
            continue;
        }

        if (line.find_first_of(" \t") != std::string::npos) {
            std::cerr << "[urls] Linea " << line_number
                      << " contiene espacios internos; se ignora: " << line << "\n";
            continue;
        }

        urls.push_back(line);
    }
    return urls;
}

inline std::vector<char> serialize_strings(const std::vector<std::string> &v) {
    size_t total = 0;
    for (const auto &s : v)
        total += s.size() + 1;

    std::vector<char> buf;
    buf.reserve(total);
    for (const auto &s : v) {
        buf.insert(buf.end(), s.begin(), s.end());
        buf.push_back('\0');
    }
    return buf;
}

inline std::vector<std::string> deserialize_strings(const std::vector<char> &buf) {
    std::vector<std::string> out;
    out.reserve(buf.size() / 8);

    std::string cur;
    cur.reserve(64);
    for (char c : buf) {
        if (c == '\0') {
            out.push_back(std::move(cur));
            cur.clear();
            cur.reserve(64);
        } else {
            cur.push_back(c);
        }
    }
    return out;
}

inline void append_int(std::string &s, int n) {
    if (n == 0) {
        s.push_back('0');
        return;
    }

    char buf[12];
    int i = 0;
    while (n > 0) {
        buf[i++] = static_cast<char>('0' + (n % 10));
        n /= 10;
    }
    while (i > 0)
        s.push_back(buf[--i]);
}

#endif // BOW_COMMON_HPP
