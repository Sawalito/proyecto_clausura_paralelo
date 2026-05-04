# =============================================================================
# Makefile (versión optimizada)
# =============================================================================
# Flags clave:
#   -O3            optimizaciones agresivas (vectorización, inlining)
#   -march=native  usa instrucciones específicas del CPU local (AVX, etc.)
#   -DOMPI_SKIP_MPICXX  desactiva los bindings C++ deprecados de OpenMPI
#                       (eliminan los warnings de op_inln.h)
# =============================================================================

CXX        = g++
MPICXX     = mpic++
CXXFLAGS   = -O3 -march=native -std=c++17 -Wall -Wextra
MPICXXFLAGS = $(CXXFLAGS) -DOMPI_SKIP_MPICXX
LDLIBS     = -lcurl

NPROC ?= 4
URLS  ?= urls.txt

.PHONY: all serial mpi clean benchmark sweep cache_clean

all: bow_serial bow_mpi

serial: bow_serial
mpi: bow_mpi

bow_serial: bow_serial.cpp bow_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

bow_mpi: bow_mpi.cpp bow_common.hpp
	$(MPICXX) $(MPICXXFLAGS) -o $@ $< $(LDLIBS)

# Benchmark con un solo q
benchmark: all
	bash run_benchmark.sh $(URLS) $(NPROC)

# Sweep sobre múltiples q (genera curva de speed-up)
sweep: all
	bash run_benchmark.sh $(URLS) "1 2 4 6 8"

clean:
	rm -f bow_serial bow_mpi *.o bow_serial.csv bow_mpi.csv \
	      benchmark_results.csv

cache_clean:
	rm -rf .bow_cache