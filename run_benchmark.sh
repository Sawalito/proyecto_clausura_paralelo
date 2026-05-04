#!/bin/bash
# =============================================================================
# run_benchmark.sh  (versión final)
# -----------------------------------------------------------------------------
# Estrategia:
#   1. Warm-up: una corrida del serial CON cache_dir para llenar el caché.
#   2. Baseline: corrida del serial CON cache (mide cómputo puro).
#   3. Sweep: por cada q, corrida del MPI CON cache.
#   4. Reporta speed-up para descarga, cómputo, y total por separado.
#
# Uso:
#   bash run_benchmark.sh                       # sweep en {1,2,4,6,8}
#   bash run_benchmark.sh urls.txt              # archivo custom
#   bash run_benchmark.sh urls.txt "1 2 4 8"    # qs custom
# =============================================================================

set -e

URLS="${1:-urls.txt}"
QS="${2:-1 2 4 6 8}"
CACHE_DIR=".bow_cache"

[ ! -f "$URLS" ] && { echo "No existe $URLS"; exit 1; }

extract() {  # $1 = output completo, $2 = etiqueta del campo (e.g. "Tiempo computo")
    grep "$2" <<< "$1" | grep -oE '[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?' | head -1
}

echo "============================================="
echo " Benchmark Bag of Words: serial vs MPI"
echo "  URLs:  $URLS"
echo "  q:     $QS"
echo "  cache: $CACHE_DIR"
echo "============================================="

# ----- Warm-up: descarga todo al cache (una sola vez) -----
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo ""
    echo ">>> Warm-up (descarga inicial al cache)"
    ./bow_serial "$URLS" /tmp/warmup.csv "$CACHE_DIR" > /dev/null
    echo "  Cache poblado: $(ls "$CACHE_DIR" | wc -l) archivos"
fi

# ----- Baseline serial CON cache (mide cómputo puro) -----
echo ""
echo ">>> SERIAL (con cache, mide cómputo puro)"
SERIAL_OUT=$(./bow_serial "$URLS" bow_serial.csv "$CACHE_DIR")
echo "$SERIAL_OUT" | tail -7

T_SERIAL_DL=$(extract "$SERIAL_OUT" "Tiempo descarga")
T_SERIAL_CP=$(extract "$SERIAL_OUT" "Tiempo computo")
T_SERIAL_IO=$(extract "$SERIAL_OUT" "Tiempo io")
T_SERIAL_TT=$(extract "$SERIAL_OUT" "Tiempo total")

# ----- Sweep paralelo -----
RESULTS_CSV="benchmark_results.csv"
echo "q,t_total_par,t_download_par,t_compute_par,t_io_par,t_total_serial,t_compute_serial,speedup_total,speedup_compute,eficiencia_compute" > "$RESULTS_CSV"

printf "\n%-3s | %-9s | %-9s | %-9s | %-12s | %-13s | %-10s\n" \
       "q" "T_tot(s)" "T_dl(s)" "T_cp(s)" "S_total" "S_compute" "Eff_compute"
printf -- "----+-----------+-----------+-----------+--------------+---------------+------------\n"

for q in $QS; do
    MPI_OUT=$(mpirun --oversubscribe -np "$q" ./bow_mpi "$URLS" bow_mpi.csv "$CACHE_DIR" 2>&1)

    T_DL=$(extract "$MPI_OUT" "Tiempo descarga")
    T_CP=$(extract "$MPI_OUT" "Tiempo computo")
    T_IO=$(extract "$MPI_OUT" "Tiempo io")
    T_TT=$(extract "$MPI_OUT" "Tiempo total")

    S_TOTAL=$(echo   "scale=3; $T_SERIAL_TT / $T_TT" | bc -l)
    S_COMP=$(echo    "scale=3; $T_SERIAL_CP / $T_CP" | bc -l)
    EFF_COMP=$(echo  "scale=3; $S_COMP / $q"        | bc -l)

    printf "%-3s | %-9s | %-9s | %-9s | %-12s | %-13s | %-10s\n" \
           "$q" "$T_TT" "$T_DL" "$T_CP" "${S_TOTAL}x" "${S_COMP}x" "$EFF_COMP"
    echo "$q,$T_TT,$T_DL,$T_CP,$T_IO,$T_SERIAL_TT,$T_SERIAL_CP,$S_TOTAL,$S_COMP,$EFF_COMP" >> "$RESULTS_CSV"

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