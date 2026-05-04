// =============================================================================
// bow_common.hpp  (versión optimizada)
// -----------------------------------------------------------------------------
// Mejoras vs versión inicial:
//   - tokenize_and_count_fast:  scan manual con isalpha-equivalente. Reemplaza
//                               std::regex (que era ~10-30x más lento).
//   - to_lower_ascii:           inline, evita el lookup de locale de std::tolower.
//   - std::unordered_map:       conteo O(1) amortizado vs O(log n) de std::map.
//   - download_url_cached:      escribe/lee del disco para reproducibilidad de
//                               benchmarks (separa el speed-up CPU del I/O).
// -----------------------------------------------------------------------------

#ifndef BOW_COMMON_HPP
#define BOW_COMMON_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <curl/curl.h>

// -----------------------------------------------------------------------------
// Conversión a minúsculas ASCII inline.
// std::tolower es sorprendentemente lento porque consulta el locale en cada
// llamada. Para texto inglés ASCII basta la rama branchless siguiente.
// -----------------------------------------------------------------------------
inline char to_lower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

// Una palabra es secuencia de [A-Za-z'] de longitud >= 2 (excepción 'a' / 'i').
inline bool is_word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '\'';
}

// -----------------------------------------------------------------------------
// libcurl: callback de escritura. NO se puede llamar `curl_write_callback`
// porque ese identificador ya está typedef-eado en <curl/curl.h>.
// -----------------------------------------------------------------------------
inline size_t bow_write_callback(void* contents, size_t size, size_t nmemb,
                                 std::string* userp) {
    userp->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

// Descarga sincrónica con libcurl. Sigue redirecciones (Gutenberg las usa).
inline std::string download_url(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string response;
    if (!curl) return response;

    response.reserve(1 << 20);  // ~1 MB inicial; libros típicos pesan 200 KB-2 MB

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, bow_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 BoW-MPI/2.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[curl] Error en " << url << ": "
                  << curl_easy_strerror(res) << std::endl;
    }
    curl_easy_cleanup(curl);
    return response;
}

// -----------------------------------------------------------------------------
// Descarga con caché en disco. Si cache_dir está vacío, se comporta como
// download_url. Si no, busca en disco; si no existe, descarga y guarda.
//
// Esto permite distinguir dos benchmarks:
//   - "frio":  primera corrida que descarga del CDN (mide red + cómputo).
//   - "tibio": corridas posteriores leen del disco (mide cómputo puro).
// El speed-up tibio es el realmente acotado por la Ley de Amdahl.
// -----------------------------------------------------------------------------
inline std::string url_to_cache_path(const std::string& url,
                                     const std::string& cache_dir) {
    std::hash<std::string> hasher;
    return cache_dir + "/" + std::to_string(hasher(url)) + ".cache";
}

inline std::string download_url_cached(const std::string& url,
                                       const std::string& cache_dir) {
    if (cache_dir.empty()) return download_url(url);

    std::string path = url_to_cache_path(url, cache_dir);
    std::ifstream in(path, std::ios::binary);
    if (in.is_open()) {
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    std::string content = download_url(url);
    if (!content.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        std::ofstream out(path, std::ios::binary);
        out << content;
    }
    return content;
}

// -----------------------------------------------------------------------------
// Elimina los headers/footers de Project Gutenberg para que el vocabulario
// no se contamine con boilerplate legal.
// -----------------------------------------------------------------------------
inline std::string strip_gutenberg_metadata(const std::string& text) {
    static const std::string start_tag = "*** START OF";
    static const std::string end_tag   = "*** END OF";

    std::string body = text;
    size_t s = text.find(start_tag);
    if (s != std::string::npos) {
        size_t nl = text.find('\n', s);
        if (nl != std::string::npos) body = text.substr(nl + 1);
    }
    size_t e = body.find(end_tag);
    if (e != std::string::npos) body = body.substr(0, e);
    return body;
}

// -----------------------------------------------------------------------------
// TOKENIZADOR RÁPIDO (reemplaza std::regex).
//
// Hace un solo pass por el texto. En cada posición:
//   1. Avanza por caracteres no-palabra (espacios, puntuación, dígitos).
//   2. Acumula caracteres-palabra en `token` aplicando to_lower_ascii inline.
//   3. Inserta o incrementa en el unordered_map.
//
// Reserves clave:
//   counts.reserve(20000): evita rehash durante la construcción. El vocabulario
//                          local de un libro de Shakespeare es 5-8 K palabras
//                          únicas.
//   token.reserve(64):     evita realloc durante el llenado (palabra promedio
//                          en inglés ~5 letras, máx ~30).
// -----------------------------------------------------------------------------
inline std::unordered_map<std::string, int>
tokenize_and_count_fast(const std::string& raw_text) {
    std::unordered_map<std::string, int> counts;
    counts.reserve(20000);

    std::string body = strip_gutenberg_metadata(raw_text);
    const char* p   = body.data();
    const char* end = p + body.size();

    std::string token;
    token.reserve(64);

    while (p < end) {
        // Saltar separadores
        while (p < end && !is_word_char(static_cast<unsigned char>(*p))) ++p;
        // Acumular palabra (con lowercasing inline)
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

// -----------------------------------------------------------------------------
// Lee URLs de un archivo (una por línea). Líneas vacías o que comienzan con
// '#' se ignoran.
// -----------------------------------------------------------------------------
inline std::vector<std::string> read_urls(const std::string& filename) {
    std::vector<std::string> urls;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "No pude abrir " << filename << std::endl;
        return urls;
    }
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        urls.push_back(line);
    }
    return urls;
}

// ----- Helpers de serialización para enviar/recibir vector<string> por MPI ---
inline std::vector<char> serialize_strings(const std::vector<std::string>& v) {
    size_t total = 0;
    for (const auto& s : v) total += s.size() + 1;
    std::vector<char> buf;
    buf.reserve(total);
    for (const auto& s : v) {
        buf.insert(buf.end(), s.begin(), s.end());
        buf.push_back('\0');
    }
    return buf;
}

inline std::vector<std::string> deserialize_strings(const std::vector<char>& buf) {
    std::vector<std::string> out;
    out.reserve(buf.size() / 8);  // estimación conservadora
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

// -----------------------------------------------------------------------------
// Conversión int -> string append directamente al buffer.
// std::ofstream << int es 5-10x más lento porque consulta locale en cada llamada.
// Esta versión maneja el caso n>=0 (suficiente para conteos de palabras).
// -----------------------------------------------------------------------------
inline void append_int(std::string& s, int n) {
    if (n == 0) { s.push_back('0'); return; }
    char buf[12];
    int i = 0;
    while (n > 0) { buf[i++] = '0' + (n % 10); n /= 10; }
    while (i > 0) s.push_back(buf[--i]);
}

#endif  // BOW_COMMON_HPP