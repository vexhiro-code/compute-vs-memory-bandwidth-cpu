# Compute vs. Memory-Bandwidth Limits on a Commodity CPU

Reproducibility package for an Architecture of Computers student research project by Abderrahmene Gueblaoui and Ali Ait Saadi. The repository contains benchmark source code, unmodified raw CSV measurements, and a script that summarizes the measurements. Student IDs are intentionally kept in the manuscript rather than this public code package.

## Research question

For vector addition, dot product, and matrix multiplication, do measured performance results lie nearer the compute reference or the memory-bandwidth reference on one commodity CPU, and how does this vary with data size and thread count?

## Test machine and software

- CPU: Intel Core i5-8365U, 4 physical cores / 8 logical processors, 6 MB Intel Smart Cache
- Memory: 7.8 GB physical memory reported by Windows
- OS: Windows 11 Pro, build 26300
- C compiler: Microsoft C/C++ Optimizing Compiler 19.52.37029
- C build flags: `/O2 /openmp`; the separate AVX2/FMA reference uses `/O2 /openmp /arch:AVX2`
- Python: 3.14.7; NumPy 2.5.5; OpenBLAS 0.3.34.237.2; threadpoolctl 3.8.0
- The Windows power plan was set to Best Performance for plugged-in operation. Per-run CPU temperature, clock frequency, and background activity were not logged.

## What was measured

The raw folder contains 101 CSV files and 2,020 timed repetitions (20 per file). Main kernel sizes were 16,384, 131,072, 1,048,576, and 4,194,304 vector elements; matrix dimensions were 64, 256, 512, and 1,024. OpenMP thread counts were 1, 2, 4, and 8. NumPy vector addition was measured at one thread because it is a NumPy ufunc and does not use the BLAS thread pool. NumPy dot and matrix multiplication used `threadpoolctl` to request 1, 2, 4, or 8 OpenBLAS threads.

Each executable performs one warm-up before recording 20 repetitions. C timings use Windows `QueryPerformanceCounter`; Python timings use `time.perf_counter`. Correctness checks run on every timed repetition, but the checks themselves are outside the timed interval. The C matrix output array is reset outside the timed interval.

### Kernels and nominal arithmetic intensity

- **Vector addition:** `c[i] = a[i] + b[i]`; nominally 1 FLOP and 24 bytes (two 8-byte reads plus one 8-byte write) per element, or 0.0417 FLOP/byte.
- **Dot product:** `sum += a[i] * b[i]`; nominally 2 FLOPs and 16 bytes per element, or 0.125 FLOP/byte.
- **Matrix multiplication:** row-major `i-k-j` loop order, `c[i,j] += a[i,k] * b[k,j]`; OpenMP distributes rows with static scheduling. The C code uses a simple, unblocked implementation. The nominal intensity uses 2n^3 FLOPs divided by 24n^2 bytes (one read of each input matrix plus one write of the output), giving n/12 FLOP/byte. This is an idealized unique-array traffic estimate, not measured traffic; repeated cache-line movement, write allocation, and compiler behavior are not counted.
- **NumPy matrix multiplication:** `numpy.matmul(a, b, out=c)`, which can use the optimized OpenBLAS backend and its blocking/vectorized kernels. It is a reference implementation, not an algorithmically identical implementation of the C loop.
- **STREAM-style triad:** `c[i] = a[i] + 3*b[i]`, with nominal traffic of two reads and one write per element. This is an application-level bandwidth reference, not a hardware DRAM counter.
- **FP64 AVX2/FMA compute microbenchmark:** eight independent vector chains, each processing four doubles, with multiply and add counted separately. The reported FLOP rate is an estimated instruction-work rate for the main loop. It is a measured reference, not a guaranteed peak or a strict per-thread upper bound.

**Important measurement limit:** the project computes nominal bytes from each algorithm. It does not collect memory-controller counters, so actual bytes transferred to DRAM are unknown. Arithmetic intensity in the report is therefore described as nominal/algorithmic intensity, not observed DRAM arithmetic intensity. Similarity to a reference bandwidth is evidence for discussion, not proof of a specific bottleneck.

## Main observed results

All values below are medians of 20 repetitions. At 4,194,304 elements, C vector-add nominal bandwidth was 7.98, 7.59, 8.04, and 8.23 GB/s at 1, 2, 4, and 8 threads. The STREAM-style triad reference at the same size measured 7.85, 6.98, 7.98, and 8.24 nominal GB/s. C dot-product nominal bandwidth increased from 6.22 GB/s (1 thread) to 11.94 GB/s (8 threads). These are derived rates, not direct DRAM measurements.

For 1,024 x 1,024 matrix multiplication, C achieved 2.65, 3.85, 5.61, and 7.30 GFLOP/s at 1, 2, 4, and 8 threads. NumPy/OpenBLAS achieved 33.36, 58.09, 64.66, and 61.21 GFLOP/s at the same requested thread counts. The implementations differ substantially, so this comparison demonstrates the effect of optimized library kernels and thread settings; it does not isolate a single hardware factor.

The AVX2/FMA microbenchmark measured 25.77, 47.40, 55.99, and 83.42 GFLOP/s at 1, 2, 4, and 8 threads. Variability increased at 4 and 8 threads, and some NumPy matrix results exceed the same-thread FMA rate. Therefore the FMA microbenchmark is reported as a separate measured reference and is not treated as a strict roof for every kernel/thread configuration.

The complete per-file medians and quartiles are in `analysis/summary.csv`. The raw repetition-level data remain in `results/raw/`.

## Reproduce the analysis

From the repository root, run with Python 3:

```powershell
python src\summarize_results.py
```

This reads each CSV under `results/raw/` and writes `analysis/summary.csv` with the median, first quartile, third quartile, and repetition count for available throughput metrics. The summarizer uses only Python's standard library.

## Rebuild and run the C benchmarks

Open **Developer PowerShell for Visual Studio** and change to the repository directory. Example:

```powershell
cl /O2 /openmp src\vector_add_bench.c /Fe:vector_add_bench.exe
cl /O2 /openmp src\dot_product_bench.c /Fe:dot_product_bench.exe
cl /O2 /openmp src\matrix_multiply_bench.c /Fe:matrix_multiply_bench.exe
cl /O2 /openmp src\stream_triad_bench.c /Fe:stream_triad_bench.exe
cl /O2 /openmp /arch:AVX2 src\compute_ceiling_fma.c /Fe:compute_ceiling_fma.exe
```

Example run (elements, threads, repetitions):

```powershell
foreach ($n in @(16384, 131072, 1048576, 4194304)) {
  foreach ($t in @(1, 2, 4, 8)) {
    .\vector_add_bench.exe $n $t 20
  }
}
```

The C programs write new CSVs using the same filenames. Back up the existing raw results before rerunning, or use a separate copy of the repository to avoid overwriting the submitted data.

## Reproduce the NumPy reference

Install the recorded packages:

```powershell
python -m pip install -r requirements.txt
```

Example:

```powershell
python src\numpy_reference.py matrix_multiply 1024 4 20
```

Arguments are kernel, vector length or square matrix dimension, requested BLAS threads, and repetitions. NumPy vector addition is a ufunc and does not use OpenBLAS threading; the recorded vector-add baseline therefore uses one thread.

## Interpretation and limitations

The tests establish measured timings for these programs on this laptop under the recorded setup. They do not establish actual DRAM bytes, a stable hardware peak, or the cause of every performance gap. CPU frequency, temperature, cache traffic, operating-system scheduling, and background workload were not recorded per repetition. C matrix multiplication and NumPy/OpenBLAS matrix multiplication use different algorithms. The conclusions are limited to these kernels, sizes, implementations, and this single machine.

## Data and code

Raw measurements, benchmark sources, and analysis instructions are intended to be released with this project repository. Insert the final repository URL in the manuscript's Data and Code Availability section after the repository is published.
