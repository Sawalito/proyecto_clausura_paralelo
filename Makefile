# =============================================================================
# Makefile
# =============================================================================

CXX ?= g++
MPICXX ?= mpic++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
MPICXXFLAGS ?= $(CXXFLAGS) -DOMPI_SKIP_MPICXX
LDLIBS ?= -lcurl

# Defaults para los targets de benchmark. Se pueden sobreescribir:
#   make benchmark NPROC=3 URLS=tests/short.txt
NPROC ?= 4
URLS ?= urls.txt
QS ?= 1 2 4 6 8
CACHE_DIR ?= .bow_cache
RESULTS_DIR ?= Results
MPIRUN_EXTRA_ARGS ?= --oversubscribe

SOURCES = bow_common.hpp bow_serial.cpp bow_mpi.cpp

.PHONY: all serial mpi clean benchmark sweep cache_clean test validate check-deps format

all: bow_serial bow_mpi

serial: bow_serial

mpi: bow_mpi

# Regla de compilacion del serial. $@ = nombre del target, $< = primer prereq.
bow_serial: bow_serial.cpp bow_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# Regla MPI. mpic++ es un wrapper sobre g++ que inyecta -I/-L/-lmpi.
bow_mpi: bow_mpi.cpp bow_common.hpp
	$(MPICXX) $(MPICXXFLAGS) -o $@ $< $(LDLIBS)

benchmark: all
	bash run_benchmark.sh $(URLS) "$(NPROC)"

sweep: all
	bash run_benchmark.sh $(URLS) "$(QS)"

test: validate

validate: all
	mkdir -p $(RESULTS_DIR)
	./bow_serial $(URLS) $(RESULTS_DIR)/bow_serial.csv $(CACHE_DIR)
	mpirun $(MPIRUN_EXTRA_ARGS) -np $(NPROC) ./bow_mpi $(URLS) $(RESULTS_DIR)/bow_mpi.csv $(CACHE_DIR)
	diff -q $(RESULTS_DIR)/bow_serial.csv $(RESULTS_DIR)/bow_mpi.csv

check-deps:
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Falta $(CXX)"; exit 1; }
	@command -v $(MPICXX) >/dev/null 2>&1 || { echo "Falta $(MPICXX)"; exit 1; }
	@command -v mpirun >/dev/null 2>&1 || { echo "Falta mpirun"; exit 1; }
	@command -v bash >/dev/null 2>&1 || { echo "Falta bash"; exit 1; }
	@command -v bc >/dev/null 2>&1 || { echo "Falta bc"; exit 1; }
	@command -v diff >/dev/null 2>&1 || { echo "Falta diff"; exit 1; }
	@printf '#include <curl/curl.h>\n' | $(CXX) -x c++ -std=c++17 -E - >/dev/null 2>&1 || { echo "Faltan headers de desarrollo de libcurl"; exit 1; }
	@echo "Dependencias basicas disponibles."

format:
	@command -v clang-format >/dev/null 2>&1 || { echo "Falta clang-format"; exit 1; }
	clang-format -i $(SOURCES)

# Limpia ejecutables y outputs intermedios; NO toca el cache de libros.
clean:
	rm -f bow_serial bow_mpi *.o $(RESULTS_DIR)/*.csv

# Borra el cache de descargas. Forzara redescarga real en la siguiente corrida.
cache_clean:
	rm -rf .bow_cache
