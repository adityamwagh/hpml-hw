#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// function to add the elements of two arrays
void add(long long n, float *x, float *y) {
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(int argc, char *argv[]) {

  // get array size in millions
  long long K = atoi(argv[1]) * 1000000;

  // defining constants for division
  const double BILLION = 1000000000.0;
  double execTime = 0;

  // initialize and populate arrays on host memory
  float *hx = new float[K];
  float *hy = new float[K];
  for (int i = 0; i < K; ++i) {
    hx[i] = 1.0;
    hy[i] = 1.0;
  }

  // initialize timers
  struct timespec start, end;

  // start timer
  clock_gettime(CLOCK_MONOTONIC, &start);

  add(K, hx, hy);

  // stop timer
  clock_gettime(CLOCK_MONOTONIC, &end);

  // compute time in seconds
  execTime = ((double)end.tv_sec - (double)start.tv_sec) +
             (((double)end.tv_nsec - (double)start.tv_nsec) / BILLION);

  std::cout << "Execution Time: " << execTime << std::endl;

  // free the memory
  free(hx);
  free(hy);

  // return
  return 0;
}