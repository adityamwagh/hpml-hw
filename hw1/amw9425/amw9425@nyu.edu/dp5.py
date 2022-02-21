import sys
import time

import numpy as np


def dp(A, B):

    # value of dot product
    R = 0.0

    # loop through arrays and compute dot product
    R = np.dot(A, B)

    # return the value of dot product
    return R


def main():

    # get array size and number of iterations
    N = int(sys.argv[1])
    iterations = int(sys.argv[2])

    # defining constants for division
    BILLION = 1000000000.0

    # initialize and populate arrays
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    # metric measurement
    times = np.zeros(iterations, dtype=np.float32)
    avg_time = 0.0
    sum_of_times = 0.0

    for i in range(iterations):

        # start timer
        start = time.monotonic()

        R = dp(A, B)

        # stop timer
        end = time.monotonic()

        # compute and store time for each repetition
        times[i] = end - start

        if i >= (iterations // 2):
            sum_of_times += times[i]

    # computing average time for 2nd half of iterations
    if iterations == 1:
        avg_time = times[0]
    elif iterations > 1 and iterations % 2 == 0:
        avg_time = sum_of_times / (iterations // 2)
    elif iterations > 1:
        avg_time = sum_of_times / (iterations // 2 + 1)

    # b/w and flops calculation
    bandwidth = (N * np.float32().nbytes * 2) / (avg_time * BILLION)
    flops = (N * 2) / (avg_time * BILLION)

    # print dot product
    print(f"Dot Product: {R}")

    # print results to screen
    print(
        f"N: {N}, <T>: {avg_time: .6f} sec, B: {bandwidth: .3f} GB/sec, F: {flops: .3f} GFLOP/sec"
    )


if __name__ == "__main__":
    main()
