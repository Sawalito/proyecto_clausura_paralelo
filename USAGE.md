# USAGE

## Build

```bash
make           # builds bow_serial and bow_mpi
make serial    # only the serial baseline
make mpi       # only the MPI binary
make clean     # removes binaries, CSVs, benchmark_results.csv
make cache_clean   # removes .bow_cache/
```

Requires `g++`, `mpic++` (OpenMPI / MPICH), `libcurl-dev`.

Build flags: `-O3 -march=native -std=c++17` (+ `-DOMPI_SKIP_MPICXX` for MPI).

---

## Invocation cheat-sheet

Both binaries accept the same positional arguments:

```
./bow_serial         <url_source> <output.csv> [cache_dir]
mpirun -np <q> ./bow_mpi <url_source> <output.csv> [cache_dir]
```

| Position | Meaning |
|----------|---------|
| `<url_source>` | Either a path to a text file (one URL per line, `#` = comment) **or** a single URL starting with `http://` / `https://`. `read_urls` auto-detects which one. |
| `<output.csv>` | Output Bag-of-Words matrix. |
| `[cache_dir]`  | Optional. If supplied, downloads are persisted and reused — separates download time from compute time across runs. |

`<q>` (after `-np`) is the number of MPI ranks (processes) to launch.

Verbosity: set `BOW_VERBOSE=1` to re-enable the per-rank, per-book stdout
chatter (silenced by default so it doesn't drag wall-clock on small runs).

---

## Case 1 — single URL

`read_urls` recognises `http://` / `https://` prefixes and treats the string
as a one-URL list. No need to create a temporary file.

Serial:
```bash
./bow_serial https://www.gutenberg.org/cache/epub/1513/pg1513.txt out.csv .bow_cache
```

MPI with one rank (only sane choice — extra ranks would sit idle):
```bash
mpirun -np 1 ./bow_mpi https://www.gutenberg.org/cache/epub/1513/pg1513.txt out.csv .bow_cache
```

No cache directory (downloads every time):
```bash
./bow_serial https://www.gutenberg.org/cache/epub/1513/pg1513.txt out.csv
```

---

## Case 2 — multiple URLs from a file

Default URL list (`urls.txt`, 12 Gutenberg books):

```bash
./bow_serial urls.txt bow_serial.csv .bow_cache
mpirun -np 4 ./bow_mpi urls.txt bow_mpi.csv .bow_cache
```

Custom URL file:
```bash
mpirun -np 8 ./bow_mpi my_urls.txt my_out.csv .bow_cache
```

URL file format (`urls.txt`):
```
# comments start with '#'
https://www.gutenberg.org/cache/epub/1342/pg1342.txt
https://www.gutenberg.org/cache/epub/11/pg11.txt
# blank lines and \r are tolerated
```

---

## Case 3 — varying the number of cores / ranks

Replace the number after `-np` with the rank count you want:

```bash
mpirun -np 1 ./bow_mpi urls.txt out.csv .bow_cache   # MPI baseline
mpirun -np 2 ./bow_mpi urls.txt out.csv .bow_cache
mpirun -np 4 ./bow_mpi urls.txt out.csv .bow_cache
mpirun -np 8 ./bow_mpi urls.txt out.csv .bow_cache
```

If you request more ranks than physical cores, add `--oversubscribe`:

```bash
mpirun --oversubscribe -np 16 ./bow_mpi urls.txt out.csv .bow_cache
```

Bind ranks for stable timings (optional):
```bash
mpirun --bind-to core -np 4 ./bow_mpi urls.txt out.csv .bow_cache
```

When `q > k` (more ranks than URLs) some ranks get zero books — they pass
through the collectives with empty buffers, which is correct but pure
overhead. Choose `q <= k` for meaningful speedup.

---

## Case 4 — benchmark sweep (serial vs MPI, many q)

The script handles warm-up, serial baseline and the q-sweep in one go.

```bash
make sweep                                 # default qs: 1 2 4 6 8 on urls.txt
bash run_benchmark.sh                      # equivalent
bash run_benchmark.sh urls.txt             # custom URL file
bash run_benchmark.sh urls.txt "1 2 4"     # custom qs
bash run_benchmark.sh my_urls.txt "1 2 4 8 12 16"
```

Output:
- `bow_serial.csv` — baseline matrix written by the serial run.
- `bow_mpi.csv`    — overwritten by each q; the final value belongs to the
  last q in the sweep.
- `benchmark_results.csv` — one row per q with download / compute / total
  times and `speedup_total`, `speedup_compute`, `eficiencia_compute`.

After every q the script runs `diff -q bow_serial.csv bow_mpi.csv` and
prints a ⚠ warning if the matrices differ.

---

## Case 5 — re-running with cache vs cold

Cold (network-bound; download dominates):
```bash
make cache_clean
./bow_serial urls.txt out.csv .bow_cache    # warms .bow_cache while running
```

Warm (compute-bound; downloads come from disk, isolates the speedup that
Amdahl actually bounds):
```bash
./bow_serial urls.txt out.csv .bow_cache    # second run: instant download
mpirun -np 4 ./bow_mpi urls.txt out.csv .bow_cache
```

Skip the cache entirely (download each run):
```bash
./bow_serial urls.txt out.csv               # no third arg
```

---

## Case 6 — verifying serial vs MPI outputs match

Bit-for-bit identity is a hard requirement. Confirm with:

```bash
diff -q bow_serial.csv bow_mpi.csv && echo "OK: identical"
# or to inspect differences:
diff bow_serial.csv bow_mpi.csv | head
```

A non-empty diff means the parallel pipeline is wrong (vocab order, row
order, or counts).

---

## Case 7 — verbose per-rank / per-book output

Default mode is quiet (intentional — stdout from mpirun ranks serializes
and was costing ~20–30 ms on a sub-second workload). To see what each rank
is doing:

```bash
BOW_VERBOSE=1 mpirun -np 4 ./bow_mpi urls.txt out.csv .bow_cache
```

Sample output:
```
[Rank 0] libros {0,2}
[Rank 0] book 0 dl=0.0019s https://...pg1342.txt
[Rank 0] book 0 tk=0.0421s unique=4231
[Rank 2] libros {5,7,9}
...
```

The serial binary always prints per-book lines because there is no rank
synchronisation cost:
```
[Serial] (1/12) dl=0.0023s https://...pg1342.txt
[Serial] (1/12) tk=0.0418s unique=4231
```

To isolate just per-URL timings:
```bash
./bow_serial https://example.org/book.txt out.csv .bow_cache | grep -E "dl=|tk="
```

---

## Parallelization strategy

Distributed memory: every rank is its own OS process with its own heap.
There are no shared variables across ranks, so there are no data races
between ranks — synchronisation is all explicit MPI collectives.

Pipeline:

1. **Bcast URLs.** Rank 0 reads the URL source, serializes the vector
   (`\0`-delimited), broadcasts size + buffer. All ranks deserialize.
2. **Bcast cached file sizes.** Rank 0 `stat`s each cached file and
   broadcasts the size vector so every rank can locally compute the
   LPT assignment with the same input.
3. **LPT load balancing.** Every rank runs `lpt_assign(sizes, q)`
   identically: sort books by size descending, greedy-assign each one to
   the currently-least-loaded rank. Worst-case makespan
   `<= (4/3 − 1/(3q)) · optimal`. Crucial when one book dominates (e.g.
   pg100 Shakespeare at 5.4 MB) — naïve contiguous splitting would dump
   it on a single rank and serialise the whole run on that rank's
   tokenisation.
   - Fallback: if there is no cache (sizes all zero) or `q == 1`,
     `lpt_assign` returns a contiguous block partition.
4. **Local download.** Each rank fetches only its assigned books via
   libcurl (`download_url_cached` if a cache dir is given). Embarrassingly
   parallel; no MPI traffic.
5. **Local tokenisation.** `tokenize_and_count_fast` runs a single
   ASCII-aware pass per book and produces an `unordered_map<string,int>`.
   The union of those keys is the rank's local vocabulary set.
6. **Global vocabulary (Gather → dedup → Bcast).**
   - Each rank serialises its local vocab directly from the set
     (no intermediate vector).
   - `MPI_Gather` of buffer sizes → `MPI_Gatherv` of buffers to rank 0.
   - Rank 0 unions into one `unordered_set`, then `std::sort` for a
     deterministic CSV column order.
   - `MPI_Bcast` of `V` and the sorted buffer back to all ranks.
7. **Local row construction.** Each rank builds its `local_k × V`
   row-major slab using a `word → column` `unordered_map` (`O(1)`
   lookups).
8. **Matrix Gatherv + reorder.** Two `MPI_Gatherv` calls: one for each
   rank's `my_indices`, one for its row slab. Rank 0 reorders the
   gathered rows back into original `book_id` order so the CSV row order
   matches `urls.txt` regardless of which rank owned which book.
9. **CSV write (rank 0 only).** Whole CSV is built in a `std::string`
   buffer with `append_int` (locale-free fast int-to-string) and flushed
   in one `write` call.

### Timing methodology

- `MPI_Wtime` locally inside each rank.
- `MPI_Reduce(MPI_MAX, root=0)` per phase → the slowest rank dictates the
  wall-clock for that phase (the physically correct measure).
- The phase-alignment `MPI_Barrier`s that used to sit between download /
  compute / IO have been removed: with `MPI_MAX` reduction the total
  wall-clock is preserved without them, and dropping the sync points
  shaved ~10–20 ms on small workloads.
- Only the very first `MPI_Barrier` (before `t_start`) is kept, so all
  ranks share a clean zero.

### Determinism

- CSV columns: lexicographic (`std::sort` on rank 0 before Bcast).
- CSV rows: original `urls.txt` order (rank 0 reorders gathered rows by
  `my_indices`).
- Result: `bow_mpi.csv` is bit-for-bit identical to `bow_serial.csv` for
  any `q >= 1`. `run_benchmark.sh` enforces this every q.

### Speedup notes

- `speedup_compute = T_serial_compute / T_mpi_compute` is the
  Amdahl-bounded number — strict CPU work.
- `speedup_total` includes download/IO; can exceed `q` on a cold cache
  because downloads parallelise.
- `q = 1` necessarily runs *slightly* below 1.0x vs the no-MPI serial:
  `MPI_Init` / `MPI_Finalize` / collectives have a fixed cost even on a
  single rank. That's structural, not a bug.

---

## Files

| File              | Purpose                                            |
|-------------------|----------------------------------------------------|
| `bow_common.hpp`  | Tokenizer, libcurl + cache helpers, MPI string serialize, LPT load balancer. |
| `bow_serial.cpp`  | Serial baseline.                                   |
| `bow_mpi.cpp`     | MPI parallel pipeline.                             |
| `urls.txt`        | Default URL list (12 Gutenberg books).             |
| `Makefile`        | Build targets and benchmark shortcuts.             |
| `run_benchmark.sh`| Warm-up + serial baseline + parallel sweep.        |
| `benchmark_results.csv` | Per-q timings and speedups (written by `run_benchmark.sh`). |
| `AUDIT.md`        | Race-condition and correctness audit.              |
| `USAGE.md`        | This file.                                         |
