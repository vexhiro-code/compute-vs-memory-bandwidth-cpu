#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void compute_kernel(double *out, int n, int iterations)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        double x0 = 0.25 + (double)(i % 17) * 0.001;
        double x1 = 0.50 + (double)(i % 19) * 0.001;
        double x2 = 0.75 + (double)(i % 23) * 0.001;
        double x3 = 1.00 + (double)(i % 29) * 0.001;
        for (int k = 0; k < iterations; k++) {
            x0 = x0 * 1.0000001 + 0.0000001;
            x1 = x1 * 1.0000001 + 0.0000002;
            x2 = x2 * 1.0000001 + 0.0000003;
            x3 = x3 * 1.0000001 + 0.0000004;
        }
        out[i] = x0 + x1 + x2 + x3;
    }
}

static int parse_positive(const char *s, int *v, int limit)
{
    char *e = NULL; long x = strtol(s, &e, 10);
    if (e == s || *e != '\0' || x <= 0 || x > limit) return 0;
    *v = (int)x; return 1;
}

int main(int argc, char **argv)
{
    int n=16384, threads=1, iterations=1000, reps=20;
    if ((argc>1&&!parse_positive(argv[1],&n,1000000)) ||
        (argc>2&&!parse_positive(argv[2],&threads,8)) ||
        (argc>3&&!parse_positive(argv[3],&iterations,1000000)) ||
        (argc>4&&!parse_positive(argv[4],&reps,1000)) || argc>5) {
        fprintf(stderr,"Usage: compute_ceiling_bench.exe [elements<=1000000] [threads<=8] [inner_iterations<=1000000] [repetitions<=1000]\n");return 2;
    }
    double *out=(double*)_aligned_malloc((size_t)n*sizeof(double),64);
    if(!out){fprintf(stderr,"Allocation failed.\n");return 1;}
    omp_set_dynamic(0);omp_set_num_threads(threads);compute_kernel(out,n,iterations); /* warm-up */
    LARGE_INTEGER f;QueryPerformanceFrequency(&f);
    char path[260];snprintf(path,sizeof(path),"results/raw/compute_ceiling_n%d_i%d_t%d.csv",n,iterations,threads);
    FILE *csv=NULL;if(fopen_s(&csv,path,"w")!=0||!csv){fprintf(stderr,"Cannot create %s.\n",path);_aligned_free(out);return 1;}
    fprintf(csv,"kernel,elements,threads,inner_iterations,repetition,seconds,estimated_flops,estimated_gflop_per_s,checksum\n");
    for(int r=1;r<=reps;r++){
        LARGE_INTEGER s,e;QueryPerformanceCounter(&s);compute_kernel(out,n,iterations);QueryPerformanceCounter(&e);
        double sec=(double)(e.QuadPart-s.QuadPart)/(double)f.QuadPart,check=0.0;
        for(int i=0;i<n;i++){if(!isfinite(out[i])){fprintf(stderr,"Non-finite output.\n");fclose(csv);_aligned_free(out);return 1;}check+=out[i];}
        /* Four multiply-add recurrences per inner iteration (8 FLOPs), plus 3 final additions per element. */
        double flops=(double)n*(8.0*(double)iterations+3.0);
        fprintf(csv,"compute_ceiling,%d,%d,%d,%d,%.9f,%.0f,%.9f,%.12f\n",n,threads,iterations,r,sec,flops,flops/sec/1e9,check);
    }
    fclose(csv);printf("Correctness: PASS (finite output)\nElements: %d | threads: %d | inner iterations: %d | repetitions: %d\n",n,threads,iterations,reps);printf("CSV saved to: %s\n",path);printf("FP64 arithmetic microbenchmark; estimated FLOPs count each multiply and add separately.\n");
    _aligned_free(out);return 0;
}
