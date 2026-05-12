// =============================================================================
// bow_common.hpp
// -----------------------------------------------------------------------------
// Header compartido entre la version serial (bow_serial.cpp) y la version
// paralela con MPI (bow_mpi.cpp). Define todas las utilidades comunes:
//
//   * Descarga HTTP con libcurl (con y sin cache en disco).
//   * Limpieza del boilerplate de Project Gutenberg.
//   * Tokenizacion rapida (scan manual, sin std::regex).
//   * Algoritmo LPT (Longest Processing Time) para balancear carga en MPI.
//   * Serializacion/deserializacion de vector<string> para enviarlos como
//     buffers planos por MPI_Bcast / MPI_Gatherv.
//
// Por que vive todo aqui en un .hpp y no en un .cpp:
//   - Las funciones son cortas e inline, asi el compilador puede inlinearlas
//     en los hot loops de ambos binarios.
//   - Evita tener que crear una librería (.a / .so) intermedia para un
//     proyecto académico de 3 archivos.
//
// Decisiones de rendimiento clave (vs una version "naive"):
//   - tokenize_and_count_fast : scan manual con isalpha-equivalente inline.
//                               Reemplaza std::regex, que era ~10-30x mas lento.
//   - to_lower_ascii          : inline, evita el lookup de locale de
//                               std::tolower (sorprendentemente costoso).
//   - std::unordered_map      : conteo O(1) amortizado vs O(log n) de std::map.
//   - download_url_cached     : escribe/lee del disco para reproducibilidad
//                               de benchmarks (separa el speed-up CPU del I/O).
// =============================================================================

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
// to_lower_ascii: convierte un caracter ASCII a minuscula sin pasar por locale.
//
// std::tolower es sorprendentemente lento porque consulta el locale en cada
// llamada (no se puede inlinear). Para texto ingles ASCII basta esta version
// branchless: si esta en el rango 'A'-'Z' le suma 32, si no, lo deja igual.
// -----------------------------------------------------------------------------
inline char to_lower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

// -----------------------------------------------------------------------------
// is_word_char: define que cuenta como "letra" para nuestro tokenizador.
//
// Aceptamos A-Z, a-z y apostrofe (para palabras como "don't", "it's").
// Los digitos y signos de puntuacion actuan como separadores.
// -----------------------------------------------------------------------------
inline bool is_word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '\'';
}

// -----------------------------------------------------------------------------
// bow_write_callback: callback que libcurl invoca cada vez que recibe un
// chunk de datos de la red. Simplemente apendea esos bytes al string final.
//
// Nota: NO se puede llamar `curl_write_callback` porque ese identificador
// ya esta typedef-eado dentro de <curl/curl.h>.
//
// Firma fija requerida por libcurl: (puntero, size, nmemb, userdata).
// Debe retornar la cantidad de bytes procesados; si retorna otra cosa,
// libcurl interpreta error y aborta la descarga.
// -----------------------------------------------------------------------------
inline size_t bow_write_callback(void* contents, size_t size, size_t nmemb,
                                 std::string* userp) {
    userp->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

// -----------------------------------------------------------------------------
// download_url: descarga sincrónica de una URL con libcurl.
//
// Opciones que configuramos:
//   FOLLOWLOCATION = 1   -> sigue redirecciones HTTP 301/302 (Gutenberg las usa
//                           para servir mirrors regionales).
//   USERAGENT            -> algunos servidores rechazan peticiones sin UA.
//   TIMEOUT = 120        -> cota maxima por descarga (libros grandes + red lenta).
//   SSL_VERIFYPEER = 0   -> desactiva verificacion de certificado SSL para que
//                           el codigo corra en entornos sin CA bundle. En
//                           produccion real esto NO se haria asi.
// -----------------------------------------------------------------------------
inline std::string download_url(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string response;
    if (!curl) return response;

    // Reserva inicial ~1 MB: los libros de Gutenberg pesan tipicamente
    // entre 200 KB y 2 MB. Evita realloc/copy durante el llenado.
    response.reserve(1 << 20);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, bow_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 BoW-MPI/2.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    // curl_easy_perform bloquea hasta que la descarga termine (o falle).
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[curl] Error en " << url << ": "
                  << curl_easy_strerror(res) << std::endl;
    }
    curl_easy_cleanup(curl);
    return response;
}

// -----------------------------------------------------------------------------
// url_to_cache_path: mapea una URL a un nombre de archivo deterministico.
//
// Hasheamos la URL (std::hash) para evitar lidiar con caracteres invalidos
// para nombres de archivo ('/', '?', ':' etc). El nombre resultante luce como
// ".bow_cache/1234567890.cache". El hash NO necesita ser criptograficamente
// fuerte, solo deterministico y bajo colision dentro del set de URLs.
// -----------------------------------------------------------------------------
inline std::string url_to_cache_path(const std::string& url,
                                     const std::string& cache_dir) {
    std::hash<std::string> hasher;
    return cache_dir + "/" + std::to_string(hasher(url)) + ".cache";
}

// -----------------------------------------------------------------------------
// cached_file_size: retorna el tamanio del archivo cacheado, o 0 si no existe.
//
// La version MPI lo usa para alimentar LPT (mejor reparto si conocemos los
// pesos antes de procesar). En la primera corrida (cache vacio) retorna 0
// y LPT cae automaticamente al fallback contiguo.
//
// Usa std::error_code en vez de excepciones porque file_size lanza si el
// archivo no existe -> aqui ese caso es esperado y no debe propagar excepcion.
// -----------------------------------------------------------------------------
inline size_t cached_file_size(const std::string& url,
                               const std::string& cache_dir) {
    if (cache_dir.empty()) return 0;
    std::error_code ec;
    auto sz = std::filesystem::file_size(url_to_cache_path(url, cache_dir), ec);
    if (ec) return 0;
    return static_cast<size_t>(sz);
}

// -----------------------------------------------------------------------------
// lpt_assign: asignacion LPT (Longest Processing Time).
//
// Problema: tenemos `k` libros con pesos `sizes[i]` y `q` procesos MPI;
// queremos repartir los libros entre procesos de modo que el proceso mas
// cargado termine lo antes posible (minimizar el makespan).
//
// Algoritmo LPT (heuristica clasica de scheduling):
//   1. Ordenar tareas por peso DESCENDENTE.
//   2. Asignar cada tarea, en orden, al proceso con MENOR carga acumulada.
//
// Garantia teorica: makespan_LPT <= (4/3 - 1/(3q)) * optimo.
// En la practica el factor real esta muy cerca de 1 cuando hay >>q tareas
// con tamanios variados (caso de Gutenberg).
//
// Caso especial: si total == 0 (no hay cache aun) caemos a un reparto
// contiguo equivalente al esquema "per_proc + remainder" tradicional;
// esto da el MISMO resultado que la version original cuando no hay info
// de pesos, asi la primera corrida (cache frio) sigue funcionando.
// -----------------------------------------------------------------------------
inline std::vector<int> lpt_assign(const std::vector<size_t>& sizes, int q) {
    int k = static_cast<int>(sizes.size());
    std::vector<int> owners(k, 0);  // owners[i] = rank al que va el libro i

    // Suma total para decidir si tenemos info de pesos.
    size_t total = 0;
    for (size_t s : sizes) total += s;

    // Fallback: sin pesos (cache vacio) o solo 1 proceso -> reparto contiguo.
    if (total == 0 || q <= 1) {
        int per_proc  = k / q;       // libros base por proceso
        int remainder = k % q;       // sobrantes a distribuir
        for (int r = 0, idx = 0; r < q; ++r) {
            // Los primeros `remainder` ranks reciben un libro extra.
            int lk = per_proc + (r < remainder ? 1 : 0);
            for (int j = 0; j < lk; ++j) owners[idx++] = r;
        }
        return owners;
    }

    // Paso 1: indices ordenados por tamanio DESC. Empate -> indice ASC
    // para que el resultado sea deterministico (mismo input -> mismo output
    // en TODOS los ranks; esencial porque cada rank corre LPT por separado).
    std::vector<int> idx(k);
    for (int i = 0; i < k; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (sizes[a] != sizes[b]) return sizes[a] > sizes[b];
        return a < b;
    });

    // Paso 2: greedy. Para cada libro (de mayor a menor) buscamos el rank
    // menos cargado y le asignamos el libro.
    std::vector<size_t> load(q, 0);  // carga acumulada por rank
    for (int i : idx) {
        // Linear scan: O(q). Para q <= 16 (caso tipico de benchmark) es
        // mas rapido que una priority_queue por el overhead constante.
        int r = 0;
        for (int j = 1; j < q; ++j) if (load[j] < load[r]) r = j;
        owners[i] = r;
        load[r] += sizes[i];
    }
    return owners;
}

// -----------------------------------------------------------------------------
// download_url_cached: descarga con caché en disco.
//
// Si cache_dir esta vacio, equivale a download_url (descarga siempre).
// Si cache_dir esta seteado:
//   - busca el archivo cacheado; si existe, lo lee y retorna.
//   - si no existe, descarga, guarda en disco y retorna.
//
// Esto nos permite hacer dos tipos de benchmark:
//   "frio":  primera corrida que paga la descarga del CDN (mide red + CPU).
//   "tibio": corridas posteriores leen del disco (mide casi solo CPU).
// El speed-up "tibio" es el que realmente esta acotado por la Ley de Amdahl.
// -----------------------------------------------------------------------------
inline std::string download_url_cached(const std::string& url,
                                       const std::string& cache_dir) {
    if (cache_dir.empty()) return download_url(url);

    std::string path = url_to_cache_path(url, cache_dir);

    // Intentamos leer del cache primero.
    std::ifstream in(path, std::ios::binary);
    if (in.is_open()) {
        std::ostringstream ss;
        ss << in.rdbuf();   // copia todo el contenido del archivo al stream
        return ss.str();
    }

    // Cache miss: descargamos y guardamos antes de retornar.
    std::string content = download_url(url);
    if (!content.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);  // mkdir -p
        std::ofstream out(path, std::ios::binary);
        out << content;
    }
    return content;
}

// -----------------------------------------------------------------------------
// strip_gutenberg_metadata: elimina los headers/footers que Project Gutenberg
// agrega a cada libro (licencia, info legal, instrucciones de uso).
//
// El cuerpo real del libro queda delimitado por marcas tipo:
//   *** START OF THE PROJECT GUTENBERG EBOOK ... ***
//   *** END   OF THE PROJECT GUTENBERG EBOOK ... ***
//
// Si no incluyeramos este filtrado, el vocabulario se contaminaria con
// palabras de boilerplate ("Project", "Gutenberg", "License", etc.) que
// aparecen en TODOS los libros y distorsionan analisis posteriores.
// -----------------------------------------------------------------------------
inline std::string strip_gutenberg_metadata(const std::string& text) {
    static const std::string start_tag = "*** START OF";
    static const std::string end_tag   = "*** END OF";

    std::string body = text;
    // Saltamos hasta despues del fin de linea de "*** START OF ... ***".
    size_t s = text.find(start_tag);
    if (s != std::string::npos) {
        size_t nl = text.find('\n', s);
        if (nl != std::string::npos) body = text.substr(nl + 1);
    }
    // Cortamos en "*** END OF" para descartar el footer.
    size_t e = body.find(end_tag);
    if (e != std::string::npos) body = body.substr(0, e);
    return body;
}

// -----------------------------------------------------------------------------
// tokenize_and_count_fast: tokenizador + contador de frecuencias en una pasada.
//
// Reemplaza una version anterior basada en std::regex que era ~10-30x mas
// lenta. La idea es hacer UN solo scan lineal del texto:
//   1. Avanzar por caracteres que NO son letra (separadores).
//   2. Acumular caracteres letra en `token`, aplicando lowercasing inline.
//   3. Cuando terminamos de leer una palabra, incrementamos counts[token].
//
// Filtros:
//   - Palabras de 1 letra se descartan, EXCEPTO "a" e "i" (palabras validas
//     del ingles). Esto elimina ruido de iniciales y artefactos OCR sin
//     perder palabras reales.
//
// Por que reservar de antemano:
//   counts.reserve(20000) -> evita rehash mientras llenamos. Un libro
//                            de Shakespeare tiene ~5-8 K palabras unicas.
//   token.reserve(64)     -> evita realloc al crecer el buffer. Palabra
//                            inglesa promedio ~5 letras, max ~30.
// -----------------------------------------------------------------------------
inline std::unordered_map<std::string, int>
tokenize_and_count_fast(const std::string& raw_text) {
    std::unordered_map<std::string, int> counts;
    counts.reserve(20000);

    std::string body = strip_gutenberg_metadata(raw_text);
    const char* p   = body.data();        // cursor
    const char* end = p + body.size();    // fin del buffer

    std::string token;
    token.reserve(64);

    while (p < end) {
        // 1) Saltar separadores (todo lo que no sea is_word_char).
        while (p < end && !is_word_char(static_cast<unsigned char>(*p))) ++p;

        // 2) Acumular la palabra actual letra a letra, en minusculas.
        token.clear();
        while (p < end && is_word_char(static_cast<unsigned char>(*p))) {
            token.push_back(to_lower_ascii(*p));
            ++p;
        }

        // 3) Aceptar si tiene >= 2 letras, o si es "a"/"i".
        if (token.size() > 1 || token == "a" || token == "i") {
            ++counts[token];  // operator[] inserta con 0 si no existe
        }
    }
    return counts;
}

// -----------------------------------------------------------------------------
// read_urls: lee URLs de un archivo de texto (una por linea).
//
// Reglas:
//   - Si el argumento es directamente una URL (empieza con http:// o https://)
//     retornamos un vector con ese unico elemento. Util para tests rapidos.
//   - Lineas vacias y lineas que empiezan con '#' (comentarios) se ignoran.
//   - Se hace trim de '\r' (archivos con line endings de Windows) y espacios
//     finales para que el parseo sea robusto.
// -----------------------------------------------------------------------------
inline std::vector<std::string> read_urls(const std::string& filename) {
    std::vector<std::string> urls;

    // Caso especial: pasaron una URL directa en vez de un archivo.
    if (filename.rfind("http://", 0) == 0 || filename.rfind("https://", 0) == 0) {
        urls.push_back(filename);
        return urls;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "No pude abrir " << filename << std::endl;
        return urls;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Trim derecho: CR de Windows y espacios.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        urls.push_back(line);
    }
    return urls;
}

// -----------------------------------------------------------------------------
// serialize_strings / deserialize_strings: empaquetar un vector<string> en
// un buffer plano de chars para poder enviarlo por MPI.
//
// MPI solo sabe enviar tipos primitivos / arrays contiguos; no entiende
// std::vector<std::string> (cada string vive en un heap separado). El truco
// estandar es concatenar las strings separadas por '\0', mandar un solo
// MPI_CHAR array, y recortar en el receptor.
//
// Formato: "hola\0mundo\0...\0" (cada string termina en NUL).
// -----------------------------------------------------------------------------
inline std::vector<char> serialize_strings(const std::vector<std::string>& v) {
    // Reservamos exacto: sumamos longitudes + 1 NUL por string.
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
    // Estimacion conservadora: ~8 caracteres por palabra promedio.
    // El reserve no necesita ser exacto, solo evita rehash en push_back.
    out.reserve(buf.size() / 8);

    std::string cur;
    cur.reserve(64);
    for (char c : buf) {
        if (c == '\0') {
            // Fin de palabra: la movemos al vector (move evita la copia).
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
// append_int: convierte un entero >= 0 a su representacion decimal y lo
// apendea al string `s`.
//
// Por que no usar std::ofstream::operator<<(int):
//   - Cada operator<< consulta el locale para saber separadores de miles.
//   - Esa consulta es 5-10x mas lenta que esta conversion manual.
//   - Multiplicado por k*V escrituras (millones de enteros en un CSV grande),
//     el ahorro es notable.
//
// El algoritmo: extraer digitos de derecha a izquierda en un buffer chico
// y volcarlos en orden inverso al string final.
// -----------------------------------------------------------------------------
inline void append_int(std::string& s, int n) {
    if (n == 0) { s.push_back('0'); return; }
    char buf[12];          // suficiente para INT32_MAX (10 digitos + signo)
    int i = 0;
    while (n > 0) { buf[i++] = '0' + (n % 10); n /= 10; }
    while (i > 0) s.push_back(buf[--i]);
}

#endif  // BOW_COMMON_HPP
