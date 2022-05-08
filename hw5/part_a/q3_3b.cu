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

    // Allocate vectors in unified memory
    float *x, *y;
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

    // allocate on host
    for (int i = 0; i < K; ++i) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }

    // define blocksize and number of threads/block
    int blockSize = 256;
    int numBlocks = 1;

    struct timespec start, end;

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    add <<<numBlocks, blockSize>>> (K, x, y);

    // stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);
    execTime = ((double)end.tv_sec - (double)start.tv_sec) + (((double)end.tv_nsec - (double)start.tv_nsec) / BILLION);
    cudaDeviceSynchronize();

    // free the memory
    cudaFree(x);
    cudaFree(y);

    std::cout << "Time to execute add: " << execTime  << "sec" << std::endl;
    std::cout << "Number of elements: " << argv[1]  << "M" << std::endl;
  
  return 0;

}