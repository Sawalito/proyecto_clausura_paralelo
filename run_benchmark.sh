#!/bin/bash
# =============================================================================
# run_benchmark.sh
# -----------------------------------------------------------------------------
# Mide el speed-up de bow_mpi vs bow_serial sobre una lista de URLs.
#
# Estrategia para que el benchmark sea reproducible:
#   1. Warm-up: corremos el serial UNA vez con un cache_dir para que TODOS
#               los libros queden en disco. Si el cache ya existe, se salta.
#   2. Baseline: corrida del serial CON cache. Asi medimos solo computo
#                (descarga se sirve del disco -> ~0 s).
#   3. Sweep: por cada q de la lista, corrida del MPI CON cache.
#   4. Reporta speed-up total y speed-up de computo (este es el acotado
#      por la Ley de Amdahl; el total puede dar superlineal porque la
#      descarga real se paralelizo).
#
# Uso:
#   bash run_benchmark.sh                       # sweep en {1,2,4,6,8}
#   bash run_benchmark.sh urls.txt              # archivo custom
#   bash run_benchmark.sh urls.txt "1 2 4 8"    # qs custom
#   bash run_benchmark.sh urls.txt "3"          # un solo q (ej. 3 procesos)
#
# Salida:
#   - bow_serial.csv          -> matriz BoW de referencia
#   - bow_mpi.csv             -> matriz BoW de la ultima corrida MPI
#   - benchmark_results.csv   -> tabla con tiempos y speed-ups por q
# =============================================================================

# set -e: abortar al primer error de comando (mejor que arrastrar fallas).
set -e

# Parametros con default:
#   $1 -> archivo de URLs (default: urls.txt)
#   $2 -> lista de procesos a probar entre comillas (default: "1 2 4 6 8")
URLS="${1:-urls.txt}"
QS="${2:-1 2 4 6 8}"
CACHE_DIR=".bow_cache"

# Validacion temprana: si no existe el archivo de URLs, salir con error.
[ ! -f "$URLS" ] && { echo "No existe $URLS"; exit 1; }

# extract: parsea una linea tipo "[Serial] Tiempo computo: 0.123 s" y
# extrae solamente el numero. Permite notacion cientifica (1.23e-04).
#   $1 = salida completa del binario
#   $2 = etiqueta del campo a buscar (e.g. "Tiempo computo")
extract() {
    grep "$2" <<< "$1" | grep -oE '[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?' | head -1
}

# extract_int: similar a extract pero captura un entero (sin punto decimal).
# Util para "Vocabulario: 49786 palabras".
extract_int() {
    grep "$2" <<< "$1" | grep -oE '[0-9]+' | head -1
}

echo "============================================="
echo " Benchmark Bag of Words: serial vs MPI"
echo "  URLs:  $URLS"
echo "  q:     $QS"
echo "  cache: $CACHE_DIR"
echo "============================================="

# ---------------------------------------------------------------------------
# WARM-UP: si el cache no existe o esta vacio, lo poblamos descargando todos
# los libros UNA vez. Si ya hay cache de una corrida previa, lo reusamos.
# La salida del warm-up no se reporta como medicion.
# ---------------------------------------------------------------------------
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo ""
    echo ">>> Warm-up (descarga inicial al cache)"
    ./bow_serial "$URLS" /tmp/warmup.csv "$CACHE_DIR" > /dev/null
    echo "  Cache poblado: $(ls "$CACHE_DIR" | wc -l) archivos"
fi

# ---------------------------------------------------------------------------
# BASELINE SERIAL con cache (mide computo puro).
# Las tres ultimas lineas de la salida tienen los tiempos que nos importan.
# ---------------------------------------------------------------------------
echo ""
echo ">>> SERIAL (con cache, mide cómputo puro)"
SERIAL_OUT=$(./bow_serial "$URLS" bow_serial.csv "$CACHE_DIR")
echo "$SERIAL_OUT" | tail -7

# Extraccion de los tiempos del serial para usarlos como denominador del
# speed-up de cada q MPI.
T_SERIAL_DL=$(extract "$SERIAL_OUT" "Tiempo descarga")
T_SERIAL_CP=$(extract "$SERIAL_OUT" "Tiempo computo")
T_SERIAL_IO=$(extract "$SERIAL_OUT" "Tiempo io")
T_SERIAL_TT=$(extract "$SERIAL_OUT" "Tiempo total")
V_SERIAL=$(extract_int "$SERIAL_OUT" "Vocabulario")

echo ""
echo ">>> Vocabulario del serial: ${V_SERIAL} palabras unicas"

# ---------------------------------------------------------------------------
# SWEEP PARALELO: una corrida MPI por cada q de $QS.
# ---------------------------------------------------------------------------
RESULTS_CSV="benchmark_results.csv"
# Header del CSV de resultados (una fila por q).
echo "q,t_total_par,t_download_par,t_compute_par,t_io_par,t_total_serial,t_compute_serial,speedup_total,speedup_compute,eficiencia_compute,vocab_serial,vocab_par" > "$RESULTS_CSV"

# Tabla en pantalla para consumo humano.
printf "\n%-3s | %-9s | %-9s | %-9s | %-12s | %-13s | %-10s | %-7s\n" \
       "q" "T_tot(s)" "T_dl(s)" "T_cp(s)" "S_total" "S_compute" "Eff_compute" "Vocab"
printf -- "----+-----------+-----------+-----------+--------------+---------------+------------+--------\n"

for q in $QS; do
    # --oversubscribe permite que mpirun lance mas procesos que cores fisicos.
    # En equipos con pocos nucleos es necesario para probar q grandes.
    # 2>&1 fusiona stderr con stdout para no perder mensajes de error.
    MPI_OUT=$(mpirun --oversubscribe -np "$q" ./bow_mpi "$URLS" bow_mpi.csv "$CACHE_DIR" 2>&1)

    # Mismos campos que en el serial, pero para esta corrida paralela.
    T_DL=$(extract "$MPI_OUT" "Tiempo descarga")
    T_CP=$(extract "$MPI_OUT" "Tiempo computo")
    T_IO=$(extract "$MPI_OUT" "Tiempo io")
    T_TT=$(extract "$MPI_OUT" "Tiempo total")
    V_PAR=$(extract_int "$MPI_OUT" "Vocabulario")

    # Speed-ups: T_serial / T_mpi. bc -l para aritmetica flotante.
    S_TOTAL=$(echo   "scale=3; $T_SERIAL_TT / $T_TT" | bc -l)
    S_COMP=$(echo    "scale=3; $T_SERIAL_CP / $T_CP" | bc -l)
    # Eficiencia paralela: S_compute / q. Ideal=1 (escalamiento lineal).
    EFF_COMP=$(echo  "scale=3; $S_COMP / $q"        | bc -l)

    # Linea de tabla + linea de CSV.
    printf "%-3s | %-9s | %-9s | %-9s | %-12s | %-13s | %-10s | %-7s\n" \
           "$q" "$T_TT" "$T_DL" "$T_CP" "${S_TOTAL}x" "${S_COMP}x" "$EFF_COMP" "$V_PAR"
    echo "$q,$T_TT,$T_DL,$T_CP,$T_IO,$T_SERIAL_TT,$T_SERIAL_CP,$S_TOTAL,$S_COMP,$EFF_COMP,$V_SERIAL,$V_PAR" >> "$RESULTS_CSV"

    # Sanity check: la matriz MPI debe ser bit-a-bit igual a la serial.
    # Si difiere, hay un bug en el balanceo / reordenamiento.
    if ! diff -q bow_serial.csv bow_mpi.csv > /dev/null 2>&1; then
        echo "  ⚠  Las matrices difieren con q=$q"
    fi
done

echo ""
echo "============================================="
echo " Notas para el reporte:"
echo "   * S_total   incluye descarga: superlineal posible (no Amdahl)"
echo "   * S_compute es el speed-up CPU puro: SÍ acotado por Amdahl"
echo "   * Resultados completos en $RESULTS_CSV"
echo "============================================="
