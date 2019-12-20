#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nvprof to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < N; tid+=stride)
    {
        c[tid] = a[tid] * 2 + b[tid];
    }
}
__global__ void init(int *a, int val, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < N; tid+=stride)
    {
        a[tid] = val;
    }
}
int main()
{
    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    int SMc = 1;
    cudaDeviceGetAttribute(&SMc, cudaDevAttrMultiProcessorCount, deviceId);
    int threads_per_block = 256;
    int number_of_blocks = 32 * SMc;
    int stride = threads_per_block * number_of_blocks;
    init<<<number_of_blocks, threads_per_block>>>(a,2,stride);
    init<<<number_of_blocks, threads_per_block>>>(b,1,stride);
    init<<<number_of_blocks, threads_per_block>>>(c,0,stride);
    cudaDeviceSynchronize();
    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c, stride );
    cudaDeviceSynchronize();

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
