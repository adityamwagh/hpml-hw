#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl_cblas.h>

float bdp(long N, float* pA, float* pB) {

  // value of dot product
  float R = cblas_sdot(N, pA, 1, pB, 1);

  // return the value of dot product
  return R;
}

int main(int argc, char* argv[]) {

  // get array size and number of iterations
  long int N = atoi(argv[1]);
  int iterations = atoi(argv[2]);

  // defining constants for division
  const double BILLION = 1000000000.0;

  // initialize and populate arrays
  float* A = malloc(N * sizeof(float));
  float* B = malloc(N * sizeof(float));
  float R = 0.0;
  for (int i = 0; i < N; ++i) {
    A[i] = 1.0;
    B[i] = 1.0;
  }

  // computations for time
  double times[iterations];
  double sum_of_times = 0.0;
  double avg_time = 0.0;

  // initialize timers
  struct timespec start, end;
  //  add check for iterations

  for (int i = 0; i < iterations; ++i) {

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    R = bdp(N, A, B);

    // stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // store time for each repetition
    times[i] = ((double)end.tv_sec - (double)start.tv_sec) + (((double)end.tv_nsec - (double)start.tv_nsec) / BILLION);

    // sum times for second half of iterations
    if (i >= (iterations / 2)) sum_of_times += times[i];
  }

  // computing average time for 2nd half of iterations
  if (iterations == 1) avg_time = times[0];
  else if (iterations > 1 && iterations % 2 == 0) avg_time = sum_of_times / (double)(iterations / 2);
  else if (iterations > 1 && iterations % 2 != 0) avg_time = sum_of_times / (double)(iterations / 2 + 1);

  // computations for bandwidth and flops
  double bandwidth = (((double)N / 5.0) * sizeof(float) * 10.0) / (avg_time * BILLION);
  double flops = (((double)N / 5.0) * 10.0) / (avg_time * BILLION);

  // print dot product
  printf("Dot Product: %f\n", R);

  // print output to screen
  printf("N: %4ld, <T>: %.6f sec, B: %.3f GB/sec, F: %.3f GFLOP/sec\n", N, avg_time, bandwidth, flops);

  // free the memory
  free(A);
  free(B);

  // return
  return 0;

}