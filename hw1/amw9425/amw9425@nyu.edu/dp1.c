#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dp1(long N, float* pA, float* pB) {

  // value of dot product
  float R = 0.0;

  // loop through arrays and compute dot product
  for (int i = 0; i < N; ++i) R += pA[i] * pB[i];

  // return the value of dot product
  return R;
}

int main(int argc, char* argv[]) {

  // get array size and number of repetitions
  long int N = atoi(argv[1]);
  int repetitions = atoi(argv[2]);

  // initialize and populate arrays
  float* A = (float*)malloc(N * sizeof(float));
  float* B = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    A[i] = 1.0;
    B[i] = 1.0;
  }

  // time measurement
  double times[repetitions];

  // initilize timers
  struct timespec start, end;

  for (int i = 0; i < repetitions; ++i) {

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    int dot_product = dp1(N, A, B);

    // stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);
  }

  // store time for each repetition
  for (int i = 0; i < repetitions; ++i) {
    times[i] = ((double)end.tv_sec - (double)start.tv_sec) * 1000000
      + ((double)end.tv_nsec - (double)start.tv_nsec) / 1000;
  }
  // print the time
  for (int i = 0; i < repetitions; ++i) {
    printf("Iteration %4d - N: %ld, <T>: %.11f uS \n", i + 1, N, times[i]);
  }

  // free the memory
  free(A);
  free(B);

  // return
  return 0;

}