#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// function to add the elements of two arrays
__global__
void add(long long n, float* x, float* y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(int argc, char* argv[]) {

    // get array size in millions
    long long K = atoi(argv[1]) * 1000000;

    // defining constants for division
    const double BILLION = 1000000000.0;
    double execTime = 0;

    // initialize and populate arrays
    float* A = (float*)malloc(K * sizeof(float));
    float* B = (float*)malloc(K * sizeof(float));
    for (int i = 0; i < K; ++i) {
        hx[i] = 1.0f;
        hy[i] = 1.0f;
    }

    // define blocksize and number of threads/block
    int blockSize = 256;
    int numBlocks = 1;

    // initialize timers
    struct timespec start, end;

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // one block, 256 thread
    add << <numBlocks, blockSize >> > (K, dx, dy);

    // stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // compute time in seconds
    execTime = ((double)end.tv_sec - (double)start.tv_sec) + (((double)end.tv_nsec - (double)start.tv_nsec) / BILLION);

    // Check for errors (all values should be 2.0f)
    float maxError = 0.0f;
    for (int i = 0; i < K; i++)
        maxError = fmax(maxError, fabs(hx[i] - 2.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // show the time to execute
    printf("Execution Time: %f\n", execTime);

    // free the memory
    cudaFree(A);
    cudaFree(B);

    // return
    return 0;

}