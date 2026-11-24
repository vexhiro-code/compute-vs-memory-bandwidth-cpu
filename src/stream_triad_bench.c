#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void stream_triad(const double *a, const double *b, double *c, size_t n)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < (int)n; i++) c[i] = a[i] + 3.0 * b[i];
}

static int parse_positive(const char *s, unsigned long long *v)
{
    char *e = NULL; unsigned long long x = strtoull(s, &e, 10);
    if (e == s || *e != '\0' || x == 0) return 0;
    *v = x; return 1;
}

int main(int argc, char **argv)
{
    unsigned long long n_arg = 4194304, reps_arg = 20, threads_arg = 1;
    if ((argc > 1 && !parse_positive(argv[1], &n_arg)) ||
        (argc > 2 && !parse_positive(argv[2], &threads_arg)) ||
        (argc > 3 && !parse_positive(argv[3], &reps_arg)) || argc > 4 ||
        n_arg > 20000000 || threads_arg > 8 || reps_arg > 1000) {
        fprintf(stderr, "Usage: stream_triad_bench.exe [elements<=20000000] [threads<=8] [repetitions<=1000]\n"); return 2;
    }
    size_t n = (size_t)n_arg, bytes = n * sizeof(double);
    int threads = (int)threads_arg, reps = (int)reps_arg;
    double *a=(double*)_aligned_malloc(bytes,64), *b=(double*)_aligned_malloc(bytes,64), *c=(double*)_aligned_malloc(bytes,64);
    if (!a || !b || !c) { fprintf(stderr,"Allocation failed.\n"); _aligned_free(a);_aligned_free(b);_aligned_free(c);return 1; }
    double expected=0.0;
    for(size_t i=0;i<n;i++){a[i]=(double)(i%997)*0.125;b[i]=(double)(i%991)*0.125;expected+=a[i]+3.0*b[i];}
    omp_set_dynamic(0);omp_set_num_threads(threads);stream_triad(a,b,c,n);
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    char path[260];snprintf(path,sizeof(path),"results/raw/stream_triad_n%llu_t%d.csv",n_arg,threads);
    FILE *csv=NULL;if(fopen_s(&csv,path,"w")!=0||!csv){fprintf(stderr,"Cannot create %s.\n",path);_aligned_free(a);_aligned_free(b);_aligned_free(c);return 1;}
    fprintf(csv,"kernel,elements,threads,repetition,seconds,nominal_bytes,nominal_flops,nominal_gb_per_s,nominal_gflop_per_s,checksum\n");
    for(int r=1;r<=reps;r++){
        LARGE_INTEGER s,e;QueryPerformanceCounter(&s);stream_triad(a,b,c,n);QueryPerformanceCounter(&e);
        double sec=(double)(e.QuadPart-s.QuadPart)/(double)f.QuadPart, check=0.0;
        for(size_t i=0;i<n;i++)check+=c[i];
        if(fabs(check-expected)>1e-9*fmax(1.0,fabs(expected))){fprintf(stderr,"Incorrect result at repetition %d.\n",r);fclose(csv);_aligned_free(a);_aligned_free(b);_aligned_free(c);return 1;}
        double nb=3.0*sizeof(double)*(double)n,nf=2.0*(double)n;
        fprintf(csv,"stream_triad,%llu,%d,%d,%.9f,%.0f,%.0f,%.9f,%.9f,%.12f\n",n_arg,threads,r,sec,nb,nf,nb/sec/1e9,nf/sec/1e9,check);
    }
    fclose(csv);printf("Correctness: PASS\nElements: %llu | threads: %d | repetitions: %d\n",n_arg,threads,reps);printf("CSV saved to: %s\n",path);printf("STREAM-style triad; nominal traffic is 2 reads + 1 write, not a measured DRAM counter.\n");
    _aligned_free(a);_aligned_free(b);_aligned_free(c);return 0;
}
