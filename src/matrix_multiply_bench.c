#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void matmul_ikj(const double *a, const double *b, double *c, int n)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = a[(size_t)i * n + k];
            for (int j = 0; j < n; j++) {
                c[(size_t)i * n + j] += aik * b[(size_t)k * n + j];
            }
        }
    }
}

static int parse_positive(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 || parsed > 1024) return 0;
    *value = (int)parsed;
    return 1;
}

int main(int argc, char **argv)
{
    int n = 256, threads = 1, reps = 20;
    if ((argc > 1 && !parse_positive(argv[1], &n)) ||
        (argc > 2 && !parse_positive(argv[2], &threads)) ||
        (argc > 3 && !parse_positive(argv[3], &reps)) || argc > 4 || threads > 8 || reps > 1000) {
        fprintf(stderr, "Usage: matrix_multiply_bench.exe [dimension<=1024] [threads<=8] [repetitions<=1000]\n");
        return 2;
    }
    size_t count = (size_t)n * (size_t)n, bytes = count * sizeof(double);
    double *a = (double *)_aligned_malloc(bytes, 64);
    double *b = (double *)_aligned_malloc(bytes, 64);
    double *c = (double *)_aligned_malloc(bytes, 64);
    double *expected = (double *)_aligned_malloc(bytes, 64);
    if (!a || !b || !c || !expected) {
        fprintf(stderr, "Memory allocation failed for %d x %d matrices.\n", n, n);
        _aligned_free(a); _aligned_free(b); _aligned_free(c); _aligned_free(expected); return 1;
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        a[(size_t)i*n+j] = (double)((i*3+j*7)%17) * 0.125;
        b[(size_t)i*n+j] = (double)((i*5+j*11)%19) * 0.125;
    }
    /* Reference calculation uses the same i-k-j ordering, serially, outside timing. */
    memset(expected, 0, bytes);
    for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) {
        double aik = a[(size_t)i*n+k];
        for (int j = 0; j < n; j++) expected[(size_t)i*n+j] += aik * b[(size_t)k*n+j];
    }
    omp_set_dynamic(0); omp_set_num_threads(threads);
    memset(c, 0, bytes); matmul_ikj(a, b, c, n); /* warm-up */

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    char csv_path[260];
    snprintf(csv_path, sizeof(csv_path), "results/raw/matrix_multiply_n%d_t%d.csv", n, threads);
    FILE *csv = NULL;
    if (fopen_s(&csv, csv_path, "w") != 0 || !csv) {
        fprintf(stderr, "Cannot create %s. Run from the cpu-project folder.\n", csv_path);
        _aligned_free(a); _aligned_free(b); _aligned_free(c); _aligned_free(expected); return 1;
    }
    fprintf(csv, "kernel,dimension,threads,repetition,seconds,nominal_bytes,nominal_flops,nominal_gb_per_s,nominal_gflop_per_s,checksum\n");
    for (int rep = 1; rep <= reps; ++rep) {
        memset(c, 0, bytes); /* Reset is deliberately outside the timed region. */
        LARGE_INTEGER start, finish;
        QueryPerformanceCounter(&start);
        matmul_ikj(a, b, c, n);
        QueryPerformanceCounter(&finish);
        double seconds = (double)(finish.QuadPart - start.QuadPart) / (double)frequency.QuadPart;
        double max_error = 0.0, checksum = 0.0;
        for (size_t q = 0; q < count; ++q) {
            double err = fabs(c[q] - expected[q]);
            if (err > max_error) max_error = err;
            checksum += c[q];
        }
        if (max_error > 1e-9) {
            fprintf(stderr, "Incorrect matrix result (max error %.12g) on repetition %d.\n", max_error, rep);
            fclose(csv); _aligned_free(a); _aligned_free(b); _aligned_free(c); _aligned_free(expected); return 1;
        }
        double nominal_bytes = 3.0 * (double)count * sizeof(double);
        double nominal_flops = 2.0 * (double)n * (double)n * (double)n;
        fprintf(csv, "matrix_multiply,%d,%d,%d,%.9f,%.0f,%.0f,%.9f,%.9f,%.12f\n",
                n, threads, rep, seconds, nominal_bytes, nominal_flops,
                nominal_bytes / seconds / 1e9, nominal_flops / seconds / 1e9, checksum);
    }
    fclose(csv);
    printf("Correctness: PASS\nDimension: %d x %d | threads: %d | repetitions: %d\n", n, n, threads, reps);
    printf("CSV saved to: %s\n", csv_path);
    printf("Algorithm: row-major i-k-j loops; parallelism over rows; output reset excluded from timing.\n");
    printf("Nominal bytes assume each input matrix and output are transferred once; not a DRAM counter.\n");
    _aligned_free(a); _aligned_free(b); _aligned_free(c); _aligned_free(expected); return 0;
}
