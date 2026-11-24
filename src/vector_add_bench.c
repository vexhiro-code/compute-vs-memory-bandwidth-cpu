#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void vector_add(const double *a, const double *b, double *c, size_t n)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < (int)n; i++) {
        c[i] = a[i] + b[i];
    }
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
        (argc > 3 && !parse_positive(argv[3], &reps_arg)) || argc > 4) {
        fprintf(stderr, "Usage: vector_add_bench.exe [elements] [threads] [repetitions]\n");
        return 2;
    }
    if (n_arg > 20000000 || threads_arg > 8 || reps_arg > 1000) {
        fprintf(stderr, "Limits: elements <= 20000000, threads <= 8, repetitions <= 1000.\n");
        return 2;
    }

    size_t n = (size_t)n_arg;
    int threads = (int)threads_arg;
    int reps = (int)reps_arg;
    size_t bytes = n * sizeof(double);
    double *a = (double *)_aligned_malloc(bytes, 64);
    double *b = (double *)_aligned_malloc(bytes, 64);
    double *c = (double *)_aligned_malloc(bytes, 64);
    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed for %llu elements.\n", n_arg);
        _aligned_free(a); _aligned_free(b); _aligned_free(c);
        return 1;
    }

    double expected_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)(i % 1009) * 0.25;
        b[i] = (double)(i % 1013) * 0.50;
        expected_sum += a[i] + b[i];
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    vector_add(a, b, c, n); /* Warm-up; excluded from timing. */

    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency)) {
        fprintf(stderr, "High-resolution timer unavailable.\n");
        _aligned_free(a); _aligned_free(b); _aligned_free(c);
        return 1;
    }

    char csv_path[260];
    snprintf(csv_path, sizeof(csv_path), "results/raw/vector_add_n%llu_t%d.csv", n_arg, threads);
    FILE *csv = NULL;
    if (fopen_s(&csv, csv_path, "w") != 0 || !csv) {
        fprintf(stderr, "Cannot create %s. Run from the cpu-project folder and make sure results/raw exists.\n", csv_path);
        _aligned_free(a); _aligned_free(b); _aligned_free(c);
        return 1;
    }
    fprintf(csv, "kernel,elements,threads,repetition,seconds,nominal_bytes,nominal_flops,nominal_gb_per_s,nominal_gflop_per_s,checksum\n");

    const double nominal_bytes_per_element = 3.0 * sizeof(double);
    double checksum = 0.0;
    for (int rep = 1; rep <= reps; ++rep) {
        LARGE_INTEGER start, finish;
        QueryPerformanceCounter(&start);
        vector_add(a, b, c, n);
        QueryPerformanceCounter(&finish);

        double seconds = (double)(finish.QuadPart - start.QuadPart) / (double)frequency.QuadPart;
        checksum = 0.0;
        for (size_t i = 0; i < n; ++i) checksum += c[i];
        if (fabs(checksum - expected_sum) > 1e-9 * fmax(1.0, fabs(expected_sum))) {
            fprintf(stderr, "Incorrect output detected on repetition %d.\n", rep);
            fclose(csv); _aligned_free(a); _aligned_free(b); _aligned_free(c);
            return 1;
        }

        double nominal_bytes = nominal_bytes_per_element * (double)n;
        double nominal_flops = (double)n;
        double nominal_gbps = nominal_bytes / seconds / 1e9;
        double nominal_gflops = nominal_flops / seconds / 1e9;
        fprintf(csv, "vector_add,%llu,%d,%d,%.9f,%.0f,%.0f,%.9f,%.9f,%.12f\n",
                n_arg, threads, rep, seconds, nominal_bytes, nominal_flops,
                nominal_gbps, nominal_gflops, checksum);
    }
    fclose(csv);
    printf("Correctness: PASS\nElements: %llu | threads: %d | repetitions: %d\n", n_arg, threads, reps);
    printf("CSV saved to: %s\n", csv_path);
    printf("Nominal traffic assumes 2 reads + 1 write per element; it is not a hardware DRAM traffic counter.\n");

    _aligned_free(a); _aligned_free(b); _aligned_free(c);
    return 0;
}


