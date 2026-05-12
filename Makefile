# =============================================================================
# Makefile  -  compila los dos binarios y orquesta los benchmarks
# =============================================================================
# Targets principales:
#   make            -> compila bow_serial y bow_mpi
#   make serial     -> solo bow_serial
#   make mpi        -> solo bow_mpi
#   make benchmark  -> corre run_benchmark.sh con un solo q (NPROC, default 4)
#   make sweep      -> corre run_benchmark.sh con la curva {1,2,4,6,8}
#   make clean      -> borra binarios y CSVs generados
#   make cache_clean-> borra el cache de libros descargados (.bow_cache)
#
# Variables sobreescribibles desde la linea de comandos:
#   make benchmark NPROC=3            -> benchmark con 3 procesos
#   make benchmark URLS=otra.txt      -> usar otra lista de URLs
#
# Flags de compilacion (vienen por que):
#   -O3              optimizaciones agresivas (vectorizacion, inlining).
#                    Es lo que nos da gran parte de la ganancia del tokenizador.
#   -march=native    usa el set de instrucciones del CPU local (AVX, AVX2,
#                    AVX-512 si lo soporta). Los binarios resultantes NO son
#                    portables a otra maquina con CPU mas viejo, pero esto
#                    es un benchmark local asi que no importa.
#   -std=c++17       usamos std::filesystem (necesita C++17 minimo).
#   -Wall -Wextra    todos los warnings utiles activados.
#   -DOMPI_SKIP_MPICXX  desactiva los bindings deprecados de C++ de OpenMPI
#                       (eliminan los warnings de op_inln.h al compilar).
# =============================================================================

CXX        = g++
MPICXX     = mpic++
CXXFLAGS   = -O3 -march=native -std=c++17 -Wall -Wextra
MPICXXFLAGS = $(CXXFLAGS) -DOMPI_SKIP_MPICXX
LDLIBS     = -lcurl

# Defaults para los targets de benchmark. Se pueden sobreescribir:
#   make benchmark NPROC=3 URLS=tests/short.txt
NPROC ?= 4
URLS  ?= urls.txt

# Targets que NO son archivos (evita conflictos si existieran nombres iguales).
.PHONY: all serial mpi clean benchmark sweep cache_clean

all: bow_serial bow_mpi

serial: bow_serial
mpi: bow_mpi

# Regla de compilacion del serial. $@ = nombre del target, $< = primer prereq.
bow_serial: bow_serial.cpp bow_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# Regla MPI. mpic++ es un wrapper sobre g++ que inyecta -I/-L/-lmpi.
bow_mpi: bow_mpi.cpp bow_common.hpp
	$(MPICXX) $(MPICXXFLAGS) -o $@ $< $(LDLIBS)

# Benchmark con un solo q (pasa "$(NPROC)" como QS al script).
benchmark: all
	bash run_benchmark.sh $(URLS) $(NPROC)

# Sweep: genera la curva de speed-up vs numero de procesos.
sweep: all
	bash run_benchmark.sh $(URLS) "1 2 4 6 8"

# Limpia ejecutables y outputs intermedios; NO toca el cache de libros.
clean:
	rm -f bow_serial bow_mpi *.o bow_serial.csv bow_mpi.csv \
	      benchmark_results.csv

# Borra el cache de descargas. Forzara redescarga real en la siguiente corrida.
cache_clean:
	rm -rf .bow_cache
