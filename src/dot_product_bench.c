#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static double dot_product(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    int i;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (i = 0; i < (int)n; i++) sum += a[i] * b[i];
    return sum;
}

static int parse_positive(const char *text, unsigned long long *value)
{
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0) return 0;
    *value = parsed;
    return 1;
}

int main(int argc, char **argv)
{
    unsigned long long n_arg = 1000000, reps_arg = 20, threads_arg = 1;
    if ((argc > 1 && !parse_positive(argv[1], &n_arg)) ||
        (argc > 2 && !parse_positive(argv[2], &threads_arg)) ||
        (argc > 3 && !parse_positive(argv[3], &reps_arg)) || argc > 4 ||
        n_arg > 20000000 || threads_arg > 8 || reps_arg > 1000) {
        fprintf(stderr, "Usage: dot_product_bench.exe [elements<=20000000] [threads<=8] [repetitions<=1000]\n");
        return 2;
    }
    size_t n = (size_t)n_arg;
    int threads = (int)threads_arg, reps = (int)reps_arg;
    size_t bytes = n * sizeof(double);
    double *a = (double *)_aligned_malloc(bytes, 64);
    double *b = (double *)_aligned_malloc(bytes, 64);
    if (!a || !b) {
        fprintf(stderr, "Memory allocation failed.\n");
        _aligned_free(a); _aligned_free(b); return 1;
    }
    double expected = 0.0;
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)(i % 101) * 0.125;
        b[i] = (double)(i % 103) * 0.25;
        expected += a[i] * b[i];
    }
    omp_set_dynamic(0); omp_set_num_threads(threads);
    (void)dot_product(a, b, n); /* warm-up */

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    char csv_path[260];
    snprintf(csv_path, sizeof(csv_path), "results/raw/dot_product_n%llu_t%d.csv", n_arg, threads);
    FILE *csv = NULL;
    if (fopen_s(&csv, csv_path, "w") != 0 || !csv) {
        fprintf(stderr, "Cannot create %s. Run from the cpu-project folder.\n", csv_path);
        _aligned_free(a); _aligned_free(b); return 1;
    }
    fprintf(csv, "kernel,elements,threads,repetition,seconds,nominal_bytes,nominal_flops,nominal_gb_per_s,nominal_gflop_per_s,checksum\n");
    for (int rep = 1; rep <= reps; ++rep) {
        LARGE_INTEGER start, finish;
        QueryPerformanceCounter(&start);
        double result = dot_product(a, b, n);
        QueryPerformanceCounter(&finish);
        double seconds = (double)(finish.QuadPart - start.QuadPart) / (double)frequency.QuadPart;
        if (fabs(result - expected) > 1e-9 * fmax(1.0, fabs(expected))) {
            fprintf(stderr, "Incorrect dot product on repetition %d.\n", rep);
            fclose(csv); _aligned_free(a); _aligned_free(b); return 1;
        }
        double nominal_bytes = 2.0 * sizeof(double) * (double)n;
        double nominal_flops = 2.0 * (double)n;
        fprintf(csv, "dot_product,%llu,%d,%d,%.9f,%.0f,%.0f,%.9f,%.9f,%.12f\n",
                n_arg, threads, rep, seconds, nominal_bytes, nominal_flops,
                nominal_bytes / seconds / 1e9, nominal_flops / seconds / 1e9, result);
    }
    fclose(csv);
    printf("Correctness: PASS\nElements: %llu | threads: %d | repetitions: %d\n", n_arg, threads, reps);
    printf("CSV saved to: %s\n", csv_path);
    printf("Nominal traffic assumes one read of each input; it is not a hardware DRAM traffic counter.\n");
    _aligned_free(a); _aligned_free(b); return 0;
}
