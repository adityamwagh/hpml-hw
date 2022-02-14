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

    # get array size and number of repetitions
    N = int(sys.argv[1])
    repetitions = int(sys.argv[2])

    # initialize and populate arrays
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    # time measurement
    times = np.zeros(repetitions, dtype=np.float32)

    for i in range(repetitions):

        # start timer
        start = time.monotonic()

        dot_product = dp(N, A, B)

        # stop timer
        end = time.monotonic()

        # compute and store time for each repetition
        times[i] = end - start

    for item in times:
        print(item)
if __name__ == "__main__":
    main()
