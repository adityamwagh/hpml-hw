#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float* pA, float* pB) {

  // value of dot product
  float R = 0.0;

  // loop through arrays and compute dot product
  for (int i = 0; i < N; i += 4) {
    R += pA[i] * pB[i] + pA[i + 1] * pB[i + 1] + pA[i + 2] * pB[i + 2] + pA[i + 3] * pB[i + 3];
  }

  // return the value of dot product
  return R;
}

int main(int argc, char* argv[]) {

  // get array size and number of iterations
  long int N = atoi(argv[1]);
  int iterations = atoi(argv[2]);

  // defining constants for division
  const double GIGA = 1024 * 1024 * 1024;
  const double BILLION = 1000000000;

  // initialize and populate arrays
  float* A = malloc(N * sizeof(float));
  float* B = malloc(N * sizeof(float));
  float dot_product;
  for (int i = 0; i < N; ++i) {
    A[i] = 1.0;
    B[i] = 1.0;
  }
  // time measurement
  double times[iterations];

  // initialize timers
  struct timespec start, end;

  for (int i = 0; i < iterations; ++i) {

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    dot_product = dpunroll(N, A, B);

    // stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // store time for each repetition
    times[i] = ((double)end.tv_sec - (double)start.tv_sec) + (((double)end.tv_nsec - (double)start.tv_nsec) / BILLION);
  }

  // computations for time
  int sample_size = (iterations > 1) ? (iterations / 2) : 1;
  double sum_time = 0.0;
  double avg_time = 0.0;

  // calculate avg_time based upon the number of iterations
  if ((iterations % 2 != 0) && (iterations > 1)) {

    // compute sum of times for second half of iterations
    for (int i = sample_size; i < iterations; ++i) sum_time += times[i];
    avg_time = sum_time / (double)(sample_size + 1);
  }
  else {
    avg_time = times[0];
  }

  // computations for bandwidth and flops
  double bandwidth = ((double)(N / 4.) * 4. * 10. / GIGA) / avg_time;
  double flops = ((double)N * 8.) / avg_time;

  // print dot product
  printf("Dot Product: %f\n", dot_product);

  // print output to screen
  printf("N: %4ld, <T>: %.6f sec, B: %.3f GB/sec, F: %.3f FLOP/sec", N, avg_time, bandwidth, flops);

  // free the memory
  free(A);
  free(B);

  // return
  return 0;

}