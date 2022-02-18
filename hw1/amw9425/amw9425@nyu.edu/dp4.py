import sys
import time

import numpy as np


def dp(N, A, B):

    # value of dot product
    R = 0.0

    # loop through arrays and compute dot product
    for i in range(N):
        R += A[i] * B[i]

    # return the value of dot product
    return R


def main():

    # get array size and number of iterations
    N = int(sys.argv[1])
    iterations = int(sys.argv[2])

    # defining constants for division
    GIGA = 1024.0 * 1024.0 * 1024.0
    BILLION = 1000000000.0

    # initialize and populate arrays
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    # metric measurement
    times = np.zeros(iterations, dtype=np.float32)
    avg_time = 0.0

    for i in range(iterations):

        # start timer
        start = time.monotonic()

        R = dp(N, A, B)

        # stop timer
        end = time.monotonic()

        # compute and store time for each repetition
        times[i] = end - start

        if i >= (iterations // 2):
            sum_of_times += times[i]

    # b/w and flops calculation
    bandwidth = (N * np.float32().nbytes * 2) / (avg_time * GIGA)
    flops = (N * 2) / (avg_time * BILLION)

    # print dot product
    print(f"Dot Product: {R}")

    # print results to screen
    print(
        f"N: {N}, <T>: {avg_time: .6f} sec, B: {bandwidth: .3f} GB/sec, F: {flops: .3f} FLOP/sec"
    )


if __name__ == "__main__":
    main()
