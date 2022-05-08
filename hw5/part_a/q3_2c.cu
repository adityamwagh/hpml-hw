#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>

// function to add the elements of two arrays
__global__
void add(long long n, float* x, float* y)
{   int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char* argv[]) {

    // get array size in millions
    long long K = atoi(argv[1]) * 1000000;

    // defining constants for division
    const double BILLION = 1000000000.0;
    double execTime = 0;
    size_t size = K * sizeof(float);

    // initialize and populate arrays in host memory
    float* hx = (float*)malloc(K * sizeof(float));
    float* hy = (float*)malloc(K * sizeof(float));
    for (int i = 0; i < K; ++i) {
        hx[i] = 1.0f;
        hy[i] = 1.0f;
    }

    // Allocate vectors in device memory
    float *dx, *dy;
    cudaMalloc((void**)&dx, size);
    cudaMalloc((void**)&dy, size);

    // Copy vectors from host memory to device global memory
    cudaMemcpy(dx, hx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, size, cudaMemcpyHostToDevice);

    // define blocksize and number of threads/block
    int blockSize = 256;
    int numBlocks = (K + blockSize - 1) / blockSize;

    // start timer
    struct timespec start, end;

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    add <<<numBlocks, blockSize>>> (K, dx, dy);

    // stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);
    execTime = ((double)end.tv_sec - (double)start.tv_sec) + (((double)end.tv_nsec - (double)start.tv_nsec) / BILLION);
    
    cudaMemcpy(hy, dy, size, cudaMemcpyDeviceToHost);

    // free the memory
    cudaFree(dx);
    cudaFree(dy);
    free(hx);
    free(hy);

    std::cout << "Time to execute add: " << execTime  << "sec" << std::endl;
    std::cout << "Number of elements: " << argv[1]  << "M" << std::endl;
  
  return 0;

}