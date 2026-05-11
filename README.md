# Bag of Words con C++ y MPI

Proyecto final para Computo Paralelo y en la Nube. El objetivo es construir una
matriz Bag of Words a partir de libros de Project Gutenberg y comparar una
implementacion serial contra una implementacion paralela con MPI.

El proyecto conserva dos ejecutables:

- `bow_serial`: version serial usada como linea base.
- `bow_mpi`: version paralela que reparte libros entre procesos MPI.

Ambas versiones leen URLs desde `urls.txt`, descargan los textos, eliminan la
metadata legal de Project Gutenberg, tokenizan palabras, cuentan frecuencias y
generan un CSV deterministico con la misma matriz Bag of Words.

## Que es Bag of Words

Bag of Words (BoW) representa documentos como conteos de palabras. Cada fila de
la matriz corresponde a un libro y cada columna corresponde a una palabra del
vocabulario global. La celda `(i, j)` contiene cuantas veces aparece la palabra
`j` en el libro `i`.

Este modelo ignora el orden de las palabras, pero es util para estudiar
frecuencias, comparar documentos y medir el costo de procesamiento de texto.

## Estructura de archivos

- `bow_common.hpp`: funciones compartidas para leer URLs, descargar con libcurl,
  usar cache local, limpiar metadata de Gutenberg, tokenizar, contar palabras y
  serializar cadenas para MPI.
- `bow_serial.cpp`: pipeline serial con mediciones de descarga, computo, I/O y
  tiempo total.
- `bow_mpi.cpp`: pipeline paralelo con reparto de libros por rank, vocabulario
  global, recoleccion de matriz y mediciones por fase.
- `urls.txt`: lista de textos de Project Gutenberg. Acepta comentarios con `#`
  y lineas vacias.
- `run_benchmark.sh`: ejecuta warm-up de cache, baseline serial, corridas MPI y
  calcula speed-up/eficiencia.
- `Makefile`: reglas de compilacion, limpieza, benchmark y validacion.
- `Results/`: carpeta para CSVs generados; Git conserva solo `.gitkeep`.
- `.gitignore`: evita versionar ejecutables, CSVs, cache y temporales.
- `.clang-format`: estilo sugerido para mantener el codigo consistente.

## Dependencias

En Linux/WSL o un entorno compatible con Bash necesitas:

- `g++` con soporte C++17.
- `mpic++` y `mpirun` de OpenMPI o MPICH.
- `libcurl` y sus headers de desarrollo.
- `make`.
- `bash`.
- `bc`.
- `diff`.

En Ubuntu/WSL, una instalacion tipica seria:

```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev libcurl4-openssl-dev make bash bc diffutils
```

## Compilacion

Compilar ambos ejecutables:

```bash
make all
```

Compilar solo la version serial:

```bash
make serial
```

Compilar solo la version MPI:

```bash
make mpi
```

Revisar dependencias basicas:

```bash
make check-deps
```

## Ejecucion serial

```bash
mkdir -p Results
./bow_serial urls.txt Results/bow_serial.csv .bow_cache
```

Argumentos:

- `urls.txt`: archivo con una URL por linea.
- `Results/bow_serial.csv`: CSV de salida.
- `.bow_cache`: directorio opcional para guardar descargas y reutilizarlas.

Tambien puede correrse sin cache:

```bash
mkdir -p Results
./bow_serial urls.txt Results/bow_serial.csv
```

## Ejecucion MPI

```bash
mkdir -p Results
mpirun -np 4 ./bow_mpi urls.txt Results/bow_mpi.csv .bow_cache
```

Donde `-np 4` indica el numero de procesos MPI. El programa tambien funciona si
hay mas procesos que libros; algunos ranks simplemente no reciben documentos.

Para ver el reparto por rank de forma ordenada:

```bash
BOW_VERBOSE=1 mpirun -np 4 ./bow_mpi urls.txt Results/bow_mpi.csv .bow_cache
```

## Benchmark

Ejecutar el benchmark por defecto:

```bash
bash run_benchmark.sh
```

Usar otro archivo de URLs o una lista especifica de procesos:

```bash
bash run_benchmark.sh urls.txt "1 2 4 8"
```

El script:

1. Verifica dependencias y ejecutables.
2. Hace warm-up para poblar `.bow_cache`.
3. Corre la version serial con cache.
4. Corre la version MPI para cada `q`.
5. Compara `Results/bow_serial.csv` contra `Results/bow_mpi.csv`.
6. Guarda resultados en `Results/benchmark_results.csv`.

Columnas principales del CSV de benchmark:

- `q`: numero de procesos MPI.
- `T_total`: tiempo paralelo total.
- `T_download`: tiempo de descarga o lectura desde cache.
- `T_tokenize`: tokenizacion local.
- `T_vocab_comm`: union y distribucion del vocabulario global.
- `T_matrix`: construccion y recoleccion de la matriz.
- `T_compute`: computo paralelo total.
- `T_io`: escritura del CSV.
- `speedup_total`: `T_serial_total / T_parallel_total`.
- `speedup_compute`: `T_serial_compute / T_parallel_compute`.
- `efficiency_compute`: `speedup_compute / q`.

## Interpretacion de speed-up y eficiencia

El speed-up mide cuantas veces mas rapida fue la version paralela respecto a la
serial:

```text
speedup = tiempo_serial / tiempo_paralelo
```

La eficiencia mide que fraccion del potencial de `q` procesos se aprovecho:

```text
eficiencia = speedup / q
```

Para el reporte conviene separar:

- `speedup_total`: incluye descarga, cache, comunicacion, computo e I/O.
- `speedup_compute`: se enfoca en el trabajo de tokenizacion, vocabulario y
  matriz; suele ser mas representativo de la paralelizacion.

## Validacion de equivalencia

Despues de correr serial y MPI con el mismo `urls.txt`, compara los CSV:

```bash
diff -q Results/bow_serial.csv Results/bow_mpi.csv
```

Si no hay salida, los archivos son iguales. Tambien puedes usar:

```bash
make validate
```

El vocabulario se ordena lexicograficamente para que el CSV sea deterministico y
la comparacion sea valida.

## Cache y reproducibilidad

El directorio `.bow_cache/` guarda copias locales de los textos descargados.
Esto reduce ruido de red y hace que las mediciones sean mas reproducibles. El
cache no se versiona en Git: puede borrarse y reconstruirse con una corrida
serial o con el benchmark.

Para limpiar el cache:

```bash
make cache_clean
```

Nota: Project Gutenberg puede actualizar archivos, redirecciones o respuestas
HTTP. Para experimentos comparables, usa el mismo `urls.txt`, conserva el cache
durante el benchmark y reporta si la corrida fue con cache frio o cache tibio.

## Limitaciones

- El tokenizador esta pensado para textos en ingles ASCII; no modela acentos ni
  normalizacion Unicode.
- Bag of Words ignora orden, contexto y significado de las palabras.
- La descarga depende de red y disponibilidad de Project Gutenberg si el cache
  esta vacio.
- El CSV puede crecer mucho cuando aumenta el numero de libros o el vocabulario.
- La paralelizacion se hace por documento; si los libros tienen tamanos muy
  distintos puede haber desbalance entre ranks.

## Limpieza

```bash
make clean
make cache_clean
```

`make clean` elimina ejecutables y resultados generados. `make cache_clean`
elimina solamente el cache local de descargas.
