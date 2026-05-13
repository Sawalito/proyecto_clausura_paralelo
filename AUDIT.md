# Auditoría de paralelización — `proyecto_clausura`

Revisión del pipeline MPI Bag-of-Words (`bow_mpi.cpp` + `bow_common.hpp`)
contrastado contra `bow_serial.cpp`. Foco: corrección paralela, condiciones
de carrera, determinismo de la salida y metodología de medición.

---

## 1. Modelo de paralelización (de un vistazo)

MPI usa **memoria distribuida**: cada rank es un proceso independiente con su
propio espacio de direcciones, así que las "data races" clásicas de threads
(OpenMP, `std::thread`, etc.) **no aplican entre ranks**. Lo que sí puede
fallar en MPI es:

- Operaciones colectivas mal pareadas (deadlock o tamaños inconsistentes).
- **Recursos compartidos fuera del modelo MPI** (sistema de archivos, `stdout`,
  variables globales de librerías como libcurl).
- Determinismo de la salida cuando depende del orden de iteración de
  contenedores no ordenados.
- Mediciones de tiempo contaminadas por desbalance entre ranks.

El veredicto adelantado: **el algoritmo en sí es correcto y produce el mismo
CSV que el serial**, pero hay **una condición de carrera real en el caché
compartido en disco** y varios problemas menores de metodología y robustez.

---

## 2. Cómo funciona la paralelización (paso a paso)

Pipeline dentro de `bow_mpi.cpp` (numerado igual que las fases del archivo):

1. **Lectura + `MPI_Bcast` de URLs** — `rank 0` lee `urls.txt`, serializa el
   vector de strings (`serialize_strings` con separador `\0`) y lo difunde a
   todos. Después todos tienen `urls` y `k = #URLs`.
2. **Asignación local determinista** — cada rank calcula su rango contiguo:
   ```
   per_proc  = k / size
   remainder = k % size
   local_k   = per_proc + (rank < remainder ? 1 : 0)
   start     = rank * per_proc + min(rank, remainder)
   ```
   El reparto es disjunto y cubre `[0, k)` exactamente, lo que es prerequisito
   para que el `MPI_Gatherv` final reconstruya las filas en orden.
3. **Descarga local** (`download_url_cached`) — cada rank descarga sus libros
   con libcurl. Si hay `cache_dir`, se intenta leer del disco antes de pegarle
   al CDN. *Aquí vive la condición de carrera; ver §3.1.*
4. **Tokenización local** — `tokenize_and_count_fast` produce un
   `unordered_map<string,int>` por libro y se acumula un `unordered_set` con
   el vocabulario local del rank.
5. **Vocabulario global** —
   - Cada rank serializa su `unordered_set` local con `serialize_strings`.
   - `MPI_Gather` para recoger los tamaños en `rc[]`.
   - `MPI_Gatherv` para recoger los buffers en `avb` (solo en `rank 0`).
   - `rank 0` deserializa, mete todo en otro `unordered_set` (unión) y luego
     hace `std::sort` para garantizar **orden lexicográfico determinista**.
   - `MPI_Bcast` de `V` (tamaño) y del buffer serializado del vocabulario
     ordenado.
6. **Construcción de filas locales** — cada rank construye una matriz
   `local_k × V` row-major usando un índice `unordered_map<string,int>` para
   pasar de palabra a columna en `O(1)`.
7. **`MPI_Gatherv` de la matriz** — `rank 0` arma `mc[i] = all_local_k[i] * V`
   y desplazamientos `md[i]`, y junta todo en `global_matrix`. Como los rangos
   `[start, start+local_k)` son contiguos y ordenados por rank, las filas
   quedan en el orden global correcto.
8. **Escritura del CSV** — solo `rank 0`: arma todo el CSV en un `std::string`
   y lo vuelca con un único `write`. Los otros ranks no tocan disco.

Mediciones: `MPI_Barrier` antes de las fases medidas, `MPI_Wtime` para tiempos
locales, `MPI_Reduce(..., MPI_MAX, 0, ...)` para reportar el tiempo del rank
más lento (el real para el wall-clock paralelo).

---

## 3. Errores y condiciones de carrera encontradas

### 3.1. 🔴 CRÍTICO — Race condition en el caché de disco (`download_url_cached`)

**Archivo:** `bow_common.hpp:92-112`

```cpp
inline std::string download_url_cached(const std::string& url,
                                       const std::string& cache_dir) {
    if (cache_dir.empty()) return download_url(url);

    std::string path = url_to_cache_path(url, cache_dir);
    std::ifstream in(path, std::ios::binary);
    if (in.is_open()) { /* leer y devolver */ }

    std::string content = download_url(url);
    if (!content.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        std::ofstream out(path, std::ios::binary);   // <-- carrera
        out << content;
    }
    return content;
}
```

**Problema:** la ruta del caché se calcula con `std::hash<std::string>(url)`.
Dos ranks distintos que descarguen la **misma URL** terminan apuntando al
**mismo archivo** en disco. Cuando arrancan en frío (sin cache):

1. Rank A y rank B hacen `ifstream` → ambos `is_open() == false`.
2. Ambos descargan vía libcurl.
3. Ambos abren `ofstream` sobre la misma ruta y escriben en paralelo.

Resultado posible: archivo truncado, contenido entrelazado, o un rank lee el
archivo a medio escribir en una corrida posterior. Si eso pasa, la
**tokenización produce conteos distintos** entre la versión serial y la
paralela y el CSV deja de coincidir.

**¿Pasa en este proyecto en concreto?** **Sí, está latente.** `urls.txt`
contiene URLs duplicadas a propósito (líneas 2/5, 3/6, 4/7), exactamente
el caso que dispara la carrera si el caché no está poblado y se corre MPI
directamente (sin pasar por el warm-up serial de `run_benchmark.sh`).

`run_benchmark.sh` enmascara el bug porque hace warm-up serial primero:
```bash
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    ./bow_serial "$URLS" /tmp/warmup.csv "$CACHE_DIR" > /dev/null
fi
```
Pero alguien que ejecute `mpirun -np 4 ./bow_mpi urls.txt out.csv .bow_cache`
sin warm-up cae directo en la carrera.

**Fix recomendado** (escritura atómica vía rename, que sí es atómico en POSIX
sobre el mismo filesystem):

```cpp
std::string content = download_url(url);
if (!content.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    // PID + hash hace única la ruta temporal por proceso
    std::string tmp = path + ".tmp." + std::to_string(::getpid());
    { std::ofstream out(tmp, std::ios::binary); out << content; }
    std::filesystem::rename(tmp, path, ec);  // atómico en POSIX
}
```

Alternativa más limpia: descargar solo en `rank 0` y `MPI_Bcast` el contenido
crudo. Eso elimina por completo el problema (y de paso evita pegarle al CDN
varias veces por la misma URL duplicada).

---

### 3.2. 🟠 Lectura potencialmente parcial del caché

**Archivo:** `bow_common.hpp:96-103`

Si una corrida previa **se interrumpió a la mitad** del `out << content`, el
archivo en disco queda truncado. La rama de caché lo lee como si fuera
válido y la tokenización produce un conteo incorrecto sin avisar.

Es la otra cara del mismo problema que §3.1. Se arregla igual: con escritura
atómica vía `tmp` + `rename`. Adicionalmente conviene guardar el tamaño
esperado o un checksum simple (CRC32 / longitud al inicio del archivo).

---

### 3.3. 🟠 `urls.txt` mal formado en la línea 7

**Archivo:** `urls.txt:7`

```
https://www.gutenberg.org/cache/epub/1524/pg1524.txthttps://www.gutenberg.org/cache/epub/100/pg100.txt
```

Dos URLs concatenadas sin `\n` entre ellas. `read_urls` no las separa, así
que se interpreta como **una sola URL** que libcurl resolverá como 404 (o
similar) y devolverá string vacío. La tokenización de un texto vacío es
benigna pero el conteo final será incorrecto.

No es un bug de paralelismo, pero contamina cualquier auditoría de
correctitud que compare `bow_serial.csv` vs `bow_mpi.csv`.

**Fix:** partir esa línea en dos.

---

### 3.4. 🟡 Salida `stdout` sin sincronizar entre ranks

**Archivo:** `bow_mpi.cpp:89-90`

```cpp
std::cout << "[Rank " << rank << "] libros [" << start
          << ", " << (start + local_k) << ")\n";
```

Todos los ranks escriben simultáneamente al mismo `stdout`. En MPI moderno
los flujos suelen estar buffereados por proceso, pero los `<<` encadenados
**no son atómicos**: la salida de varios ranks se entrelaza. No afecta la
salida del CSV, pero ensucia los logs y, en modo verbose, puede mezclar
mensajes de error.

Fix: imprimir desde `rank 0` después de un `MPI_Gather` de `start`/`local_k`,
o construir la línea completa con `oss << ...` y volcarla con un solo
`std::cout << oss.str()`.

---

### 3.5. 🟡 Métricas de fase contaminadas por desbalance de tokenización

**Archivo:** `bow_mpi.cpp:112-164`

```cpp
double t_cp_start    = MPI_Wtime();           // fase 4
// ... tokenización ...
double t_cp_local_end = MPI_Wtime();          // fin tokenize
// ... gather + bcast del vocabulario ...
double t_vocab_end   = MPI_Wtime();
```

Entre `t_cp_local_end` y la entrada al `MPI_Gather` no hay barrera. Si rank A
termina la tokenización en `t = 5` y rank B en `t = 10`, rank A llega antes
al `MPI_Gather` y se queda **bloqueado esperando a B dentro del colectivo**.
El intervalo `voc_local = t_vocab_end - t_cp_local_end` para rank A incluye
esos 5 s de espera por desbalance, y como después se reduce con `MPI_MAX`,
el `voc_max` reportado **sobreestima** el costo real de la comunicación de
vocabulario.

No es incorrección del cómputo, sí lo es de la metodología de medición.

Fix: `MPI_Barrier` justo antes de `t_cp_local_end`, o reportar también el
`MPI_MIN` y `MPI_AVG` por fase para que el desbalance se vea explícitamente.

---

### 3.6. 🟡 `std::error_code` silenciado en `create_directories`

**Archivo:** `bow_common.hpp:106-110`

```cpp
std::error_code ec;
std::filesystem::create_directories(cache_dir, ec);
std::ofstream out(path, std::ios::binary);
out << content;
```

Si `create_directories` falla (por permisos, disco lleno, etc.), `ec` queda
poblado pero se ignora. El `ofstream` falla silenciosamente y el caché no se
escribe. La función devuelve el contenido descargado, así que la corrida
actual funciona, pero **la siguiente corrida no encuentra nada en caché y
vuelve a descargar todo**, sin que nadie se entere.

Fix: chequear `ec` y `out.good()`; loggear si fallan.

---

### 3.7. 🟡 Colisión de hash en la ruta del caché

**Archivo:** `bow_common.hpp:86-90`

```cpp
return cache_dir + "/" + std::to_string(hasher(url)) + ".cache";
```

`std::hash<std::string>` no es resistente a colisiones (es un hash de tabla
hash, no criptográfico). Con 9 URLs es prácticamente imposible que choquen,
pero al escalar (cientos/miles de URLs) el riesgo crece. Si dos URLs colapsan
al mismo path, el segundo libro lee el contenido del primero.

Fix: usar la URL escapada como nombre (con `/` reemplazados por `_`) o un
hash más ancho como SHA-1 truncado.

---

### 3.8. 🟢 Cosas que están bien (y vale la pena documentar)

- **`MPI_Gatherv` usa argumentos no significativos en no-roots correctamente**
  (los buffers `dp`/`md` solo se llenan en rank 0 — el estándar MPI dice que
  son ignorados en los demás).
- **El orden lexicográfico del vocabulario está garantizado** por el
  `std::sort` en rank 0 (línea 153), así que el orden no determinista del
  `unordered_set` no se filtra al CSV.
- **El reparto de libros es disjunto y contiguo**, así que el `Gatherv`
  reconstruye `global_matrix` con el `book_id` global correcto sin
  reordenamiento posterior.
- **No hay deadlocks**: todos los colectivos están en el mismo orden en todos
  los ranks y los argumentos coinciden.
- **Caso `local_k == 0`** (cuando `k < size`): los buffers son vacíos, los
  colectivos los aceptan sin problema y la salida sigue siendo válida.

---

## 4. Resumen ejecutivo

| Severidad   | Problema                                              | Archivo / línea            | ¿Afecta corrección? |
|-------------|-------------------------------------------------------|----------------------------|---------------------|
| 🔴 Crítico  | Race en escritura concurrente de caché                | `bow_common.hpp:104-110`   | Sí (intermitente)   |
| 🟠 Alto     | Lectura de caché parcial tras crash                   | `bow_common.hpp:96-103`    | Sí                  |
| 🟠 Alto     | URL malformada en `urls.txt:7`                        | `urls.txt:7`               | Sí                  |
| 🟡 Medio    | `stdout` entrelazado entre ranks                      | `bow_mpi.cpp:89-90`        | No                  |
| 🟡 Medio    | Métricas de fase contaminadas por desbalance          | `bow_mpi.cpp:124,164`      | No (solo medición)  |
| 🟡 Medio    | `std::error_code` ignorado                            | `bow_common.hpp:106`       | Solo cache silencioso |
| 🟡 Medio    | Riesgo de colisión de hash al escalar                 | `bow_common.hpp:88`        | Sí (a escala)       |

**Veredicto.** El esquema MPI es correcto para el conjunto de datos actual y
produce salida determinista coincidente con el serial — siempre y cuando se
respete el camino de `run_benchmark.sh` que hace warm-up serial primero. El
único bug realmente disparable hoy es la **race del caché**, y se vuelve
seguro con un cambio puntual a escritura atómica (`tmp` + `rename`) o
moviendo la descarga a `rank 0`. El resto son endurecimientos de robustez y
metodología de medición.
