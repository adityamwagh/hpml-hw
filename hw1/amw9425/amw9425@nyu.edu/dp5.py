import sys
import time

import numpy as np


def main():

    # get array size and number of iterations
    N = int(sys.argv[1])
    iterations = int(sys.argv[2])

    # initialize and populate arrays
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    # metric measurement
    times = np.zeros(iterations, dtype=np.float32)
    avg_time = 0.0

    for i in range(iterations):

        # start timer
        start = time.monotonic()

        dot_product = np.dot(A, B)

        # stop timer
        end = time.monotonic()

        # compute and store time for each repetition
        times[i] = end - start

    # time calculation
    sample_size = (iterations // 2) if iterations > 1 else 1
    if iterations % 2 != 0:
        avg_time = sum(times[sample_size:]) / (sample_size + 1)
    else:
        avg_time = sum(times[sample_size:]) / sample_size

    # b/w and flops calculation
    bandwidth = (N * 4 * 2 / (1024.0 * 1024.0 * 1024.0)) / avg_time
    flops = N * (iterations / (1024.0 * 1024.0 * 1024.0)) / avg_time

    # print results to screen
    print(
        f"N: {N}, <T>: {avg_time: .6f} sec, B: {bandwidth: .3f} GB/sec, F: {flops: .3f} FLOP/sec"
    )


if __name__ == "__main__":
    main()
