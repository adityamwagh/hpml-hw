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

    size = len(sys.argv)
    N = sys.argv[1]
    repetitions = sys.argv[2]

    A = np.ones(N, dtype=np.float32)

if __name__== "__main__":
    main() 