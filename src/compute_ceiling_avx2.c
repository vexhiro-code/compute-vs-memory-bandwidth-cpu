#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <immintrin.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static __m256d seed4(double base, double step)
{
    return _mm256_set_pd(base+3.0*step, base+2.0*step, base+step, base);
}

/* AVX2 FP64 workload: 8 independent vector chains, each 4 doubles wide. */
__declspec(noinline) static void avx2_compute_kernel(double *out, int blocks, int iterations)
{
    int i;
    const __m256d scale=_mm256_set1_pd(1.00000001);
    const __m256d b0=_mm256_set1_pd(1.0e-9), b1=_mm256_set1_pd(2.0e-9);
    const __m256d b2=_mm256_set1_pd(3.0e-9), b3=_mm256_set1_pd(4.0e-9);
    const __m256d b4=_mm256_set1_pd(5.0e-9), b5=_mm256_set1_pd(6.0e-9);
    const __m256d b6=_mm256_set1_pd(7.0e-9), b7=_mm256_set1_pd(8.0e-9);
    #pragma omp parallel for schedule(static)
    for(i=0;i<blocks;i++){
        __m256d x0=seed4(0.25+(double)(i%17)*0.001,0.0001);
        __m256d x1=seed4(0.50+(double)(i%19)*0.001,0.0002);
        __m256d x2=seed4(0.75+(double)(i%23)*0.001,0.0003);
        __m256d x3=seed4(1.00+(double)(i%29)*0.001,0.0004);
        __m256d x4=seed4(1.25+(double)(i%31)*0.001,0.0005);
        __m256d x5=seed4(1.50+(double)(i%37)*0.001,0.0006);
        __m256d x6=seed4(1.75+(double)(i%41)*0.001,0.0007);
        __m256d x7=seed4(2.00+(double)(i%43)*0.001,0.0008);
        for(int k=0;k<iterations;k++){
            x0=_mm256_add_pd(_mm256_mul_pd(x0,scale),b0);
            x1=_mm256_add_pd(_mm256_mul_pd(x1,scale),b1);
            x2=_mm256_add_pd(_mm256_mul_pd(x2,scale),b2);
            x3=_mm256_add_pd(_mm256_mul_pd(x3,scale),b3);
            x4=_mm256_add_pd(_mm256_mul_pd(x4,scale),b4);
            x5=_mm256_add_pd(_mm256_mul_pd(x5,scale),b5);
            x6=_mm256_add_pd(_mm256_mul_pd(x6,scale),b6);
            x7=_mm256_add_pd(_mm256_mul_pd(x7,scale),b7);
        }
        __m256d s=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0,x1),_mm256_add_pd(x2,x3)),
                                _mm256_add_pd(_mm256_add_pd(x4,x5),_mm256_add_pd(x6,x7)));
        double lanes[4];_mm256_storeu_pd(lanes,s);
        out[i]=lanes[0]+lanes[1]+lanes[2]+lanes[3];
    }
}

static double expected_block(int i,int iterations)
{
    double base[8]={0.25+(double)(i%17)*0.001,0.50+(double)(i%19)*0.001,
        0.75+(double)(i%23)*0.001,1.00+(double)(i%29)*0.001,
        1.25+(double)(i%31)*0.001,1.50+(double)(i%37)*0.001,
        1.75+(double)(i%41)*0.001,2.00+(double)(i%43)*0.001};
    double bias[8]={1e-9,2e-9,3e-9,4e-9,5e-9,6e-9,7e-9,8e-9};
    double sum=0.0;
    for(int chain=0;chain<8;chain++)for(int lane=0;lane<4;lane++){
        double x=base[chain]+(double)lane*(0.0001*(double)(chain+1));
        for(int k=0;k<iterations;k++)x=x*1.00000001+bias[chain];
        sum+=x;
    }
    return sum;
}

static int parse_positive(const char*s,int*v,int limit){char*e=NULL;long x=strtol(s,&e,10);if(e==s||*e!='\0'||x<=0||x>limit)return 0;*v=(int)x;return 1;}

int main(int argc,char**argv)
{
    int blocks=16384,threads=1,iterations=1000,reps=20;
    if((argc>1&&!parse_positive(argv[1],&blocks,1000000))||(argc>2&&!parse_positive(argv[2],&threads,8))||
       (argc>3&&!parse_positive(argv[3],&iterations,1000000))||(argc>4&&!parse_positive(argv[4],&reps,1000))||argc>5){
        fprintf(stderr,"Usage: compute_ceiling_avx2.exe [blocks<=1000000] [threads<=8] [inner_iterations<=1000000] [repetitions<=1000]\n");return 2;}
    double*out=(double*)_aligned_malloc((size_t)blocks*sizeof(double),64);if(!out){fprintf(stderr,"Allocation failed.\n");return 1;}
    omp_set_dynamic(0);omp_set_num_threads(threads);avx2_compute_kernel(out,blocks,iterations);
    LARGE_INTEGER f;QueryPerformanceFrequency(&f);
    char path[260];snprintf(path,sizeof(path),"results/raw/compute_ceiling_avx2_b%d_i%d_t%d.csv",blocks,iterations,threads);
    FILE*csv=NULL;if(fopen_s(&csv,path,"w")!=0||!csv){fprintf(stderr,"Cannot create %s.\n",path);_aligned_free(out);return 1;}
    fprintf(csv,"kernel,blocks,threads,inner_iterations,repetition,seconds,estimated_main_loop_flops,estimated_gflop_per_s,checksum\n");
    for(int r=1;r<=reps;r++){
        LARGE_INTEGER s,e;QueryPerformanceCounter(&s);avx2_compute_kernel(out,blocks,iterations);QueryPerformanceCounter(&e);
        double sec=(double)(e.QuadPart-s.QuadPart)/(double)f.QuadPart,check=0.0;
        for(int i=0;i<blocks;i++){if(!isfinite(out[i])){fprintf(stderr,"Non-finite output.\n");fclose(csv);_aligned_free(out);return 1;}check+=out[i];}
        double e0=expected_block(0,iterations),e1=expected_block(blocks-1,iterations);
        if(fabs(out[0]-e0)>1e-8*fmax(1.0,fabs(e0))||fabs(out[blocks-1]-e1)>1e-8*fmax(1.0,fabs(e1))){
            fprintf(stderr,"Scalar reference check failed.\n");fclose(csv);_aligned_free(out);return 1;}
        /* 8 vectors x 4 FP64 lanes x (multiply + add) per inner iteration. */
        double flops=(double)blocks*(double)iterations*64.0;
        fprintf(csv,"compute_ceiling_avx2,%d,%d,%d,%d,%.9f,%.0f,%.9f,%.12f\n",blocks,threads,iterations,r,sec,flops,flops/sec/1e9,check);
    }
    fclose(csv);printf("Correctness: PASS (finite output and scalar checks)\nBlocks: %d | threads: %d | inner iterations: %d | repetitions: %d\n",blocks,threads,iterations,reps);
    printf("CSV saved to: %s\n",path);printf("FP64 AVX2 microbenchmark; estimated FLOPs count the main loop, not initialization or final reduction.\n");
    _aligned_free(out);return 0;
}
