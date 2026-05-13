#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh
# -----------------------------------------------------------------------------
# Compara bow_serial contra bow_mpi usando cache local para reducir ruido de red.
#
# Uso:
#   bash run_benchmark.sh
#   bash run_benchmark.sh urls.txt
#   bash run_benchmark.sh urls.txt "1 2 4 8"
#
# OpenMPI puede necesitar --oversubscribe en laptops. Se puede cambiar con:
#   MPIRUN_EXTRA_ARGS="" bash run_benchmark.sh
# =============================================================================

set -euo pipefail

# Parametros con default:
#   $1 -> archivo de URLs (default: urls.txt)
#   $2 -> lista de procesos a probar entre comillas (default: "1 2 4 6 8")
URLS="${1:-urls.txt}"
QS="${2:-1 2 4 6 8}"
CACHE_DIR="${CACHE_DIR:-.bow_cache}"
RESULTS_DIR="${RESULTS_DIR:-Results}"
SERIAL_CSV="$RESULTS_DIR/bow_serial.csv"
MPI_CSV="$RESULTS_DIR/bow_mpi.csv"
RESULTS_CSV="$RESULTS_DIR/benchmark_results.csv"
MPIRUN_EXTRA_ARGS="${MPIRUN_EXTRA_ARGS:---oversubscribe}"
WARMUP_CSV="${TMPDIR:-/tmp}/bow_warmup_$$.csv"

trap 'rm -f "$WARMUP_CSV"' EXIT

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Falta la dependencia: $1"
}

need_executable() {
    [ -x "$1" ] || fail "No existe o no es ejecutable: $1. Ejecuta make all."
}

extract_time() {
    local output="$1"
    local label="$2"
    grep -F "$label" <<< "$output" \
        | grep -oE '[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' \
        | head -1 || true
}

require_time() {
    local value="$1"
    local name="$2"
    [ -n "$value" ] || fail "No se pudo extraer el tiempo: $name"
}

ratio() {
    local numerator="$1"
    local denominator="$2"
    local positive
    positive=$(echo "$denominator > 0" | bc -l)
    if [ "$positive" != "1" ]; then
        echo "nan"
    else
        echo "scale=6; $numerator / $denominator" | bc -l
    fi
}

[ -f "$URLS" ] || fail "No existe el archivo de URLs: $URLS"
mkdir -p "$RESULTS_DIR"

need_command mpirun
need_command bc
need_command diff
need_executable ./bow_serial
need_executable ./bow_mpi

echo "============================================="
echo " Benchmark Bag of Words: serial vs MPI"
echo "  URLs:  $URLS"
echo "  q:     $QS"
echo "  cache: $CACHE_DIR"
echo "  out:   $RESULTS_DIR"
echo "============================================="

echo ""
echo ">>> Warm-up: asegura cache local"
if ! ./bow_serial "$URLS" "$WARMUP_CSV" "$CACHE_DIR" >/dev/null; then
    fail "El warm-up serial fallo; revisa URLs, red o libcurl."
fi
cache_files=$(find "$CACHE_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
echo "  Cache disponible: $cache_files archivos"

echo ""
echo ">>> SERIAL (con cache)"
if ! SERIAL_OUT=$(./bow_serial "$URLS" "$SERIAL_CSV" "$CACHE_DIR" 2>&1); then
    echo "$SERIAL_OUT"
    fail "La corrida serial fallo."
fi
echo "$SERIAL_OUT" | tail -7

T_SERIAL_DOWNLOAD=$(extract_time "$SERIAL_OUT" "Tiempo descarga")
T_SERIAL_COMPUTE=$(extract_time "$SERIAL_OUT" "Tiempo computo")
T_SERIAL_IO=$(extract_time "$SERIAL_OUT" "Tiempo io")
T_SERIAL_TOTAL=$(extract_time "$SERIAL_OUT" "Tiempo total")
require_time "$T_SERIAL_DOWNLOAD" "serial descarga"
require_time "$T_SERIAL_COMPUTE" "serial computo"
require_time "$T_SERIAL_IO" "serial io"
require_time "$T_SERIAL_TOTAL" "serial total"

echo "q,T_total,T_download,T_tokenize,T_vocab_comm,T_matrix,T_compute,T_io,T_serial_total,T_serial_compute,speedup_total,speedup_compute,efficiency_compute" > "$RESULTS_CSV"

printf "\n%-3s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-12s | %-10s\n" \
       "q" "T_total" "T_down" "T_tok" "T_vocab" "T_matrix" "T_comp" "S_comp" "Eff"
printf -- "----+------------+------------+------------+------------+------------+------------+--------------+------------\n"

for q in $QS; do
    echo ""
    echo ">>> MPI q=$q"
    if ! MPI_OUT=$(mpirun $MPIRUN_EXTRA_ARGS -np "$q" ./bow_mpi "$URLS" "$MPI_CSV" "$CACHE_DIR" 2>&1); then
        echo "$MPI_OUT"
        fail "La corrida MPI fallo para q=$q."
    fi

    T_DOWNLOAD=$(extract_time "$MPI_OUT" "Tiempo descarga")
    T_TOKENIZE=$(extract_time "$MPI_OUT" "Tiempo tokenize")
    T_VOCAB=$(extract_time "$MPI_OUT" "Tiempo vocab(comm)")
    T_MATRIX=$(extract_time "$MPI_OUT" "Tiempo matriz")
    T_COMPUTE=$(extract_time "$MPI_OUT" "Tiempo computo")
    T_IO=$(extract_time "$MPI_OUT" "Tiempo io")
    T_TOTAL=$(extract_time "$MPI_OUT" "Tiempo total")

    require_time "$T_DOWNLOAD" "MPI descarga q=$q"
    require_time "$T_TOKENIZE" "MPI tokenize q=$q"
    require_time "$T_VOCAB" "MPI vocab q=$q"
    require_time "$T_MATRIX" "MPI matriz q=$q"
    require_time "$T_COMPUTE" "MPI computo q=$q"
    require_time "$T_IO" "MPI io q=$q"
    require_time "$T_TOTAL" "MPI total q=$q"

    if ! diff -q "$SERIAL_CSV" "$MPI_CSV" >/dev/null 2>&1; then
        fail "MPI no produjo el mismo CSV que serial para q=$q."
    fi

    SPEEDUP_TOTAL=$(ratio "$T_SERIAL_TOTAL" "$T_TOTAL")
    SPEEDUP_COMPUTE=$(ratio "$T_SERIAL_COMPUTE" "$T_COMPUTE")
    EFFICIENCY_COMPUTE=$(ratio "$SPEEDUP_COMPUTE" "$q")

    printf "%-3s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-12sx | %-10s\n" \
           "$q" "$T_TOTAL" "$T_DOWNLOAD" "$T_TOKENIZE" "$T_VOCAB" \
           "$T_MATRIX" "$T_COMPUTE" "$SPEEDUP_COMPUTE" "$EFFICIENCY_COMPUTE"

    echo "$q,$T_TOTAL,$T_DOWNLOAD,$T_TOKENIZE,$T_VOCAB,$T_MATRIX,$T_COMPUTE,$T_IO,$T_SERIAL_TOTAL,$T_SERIAL_COMPUTE,$SPEEDUP_TOTAL,$SPEEDUP_COMPUTE,$EFFICIENCY_COMPUTE" >> "$RESULTS_CSV"
done

echo ""
echo "============================================="
echo " Resultados completos: $RESULTS_CSV"
echo " Validacion: cada corrida MPI produjo el mismo CSV que serial."
echo " speedup_total = T_serial_total / T_parallel_total"
echo " speedup_compute = T_serial_compute / T_parallel_compute"
echo " efficiency_compute = speedup_compute / q"
echo "============================================="
