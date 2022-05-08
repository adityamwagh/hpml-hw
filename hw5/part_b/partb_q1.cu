#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>

using namespace std::chrono;
__global__ void convolution_(double *matrix, double* mask, double *result, int H, int W, int C, int K) {

  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_r = row - 1;
  int start_c = col - 1;

  // Temp value for accumulating the result
  for(int k=0;k<K;k++){
    double temp = 0;
    for(int c=0;c<C;c++){
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if ((start_r + i) >= 0 && (start_r + i) < H) {
            if ((start_c + j) >= 0 && (start_c + j) < W) {
              temp += matrix[c* H * W + (start_r + i)*W+(start_c + j)] *
                      mask[k*C*3*3 + c*3*3 + (2-i) *3 + (2-j)];
            }
          }
        }
      }
    }
    result[k * H * W + row * W + col] = temp;
  }
}
  
int main() {
  int H = 1024, W = 1024, C = 3,K =64; 
  // I = (c,w,h), I_0 = (C,W+2,H+2)
  // int H = 3, W = 3, C = 2,K =5; // I = (c,w,h), I_0 = (C,W+2,H+2)

  int c,x,y,i,j,k;

  // Allocate the matrix and initialize it
  double *matrix = new double[H * W * C];
  double *result = new double[K * H * W];
  double * mask = new double[K * C * 3 * 3];
  for(c =0;c<C;c++){
    for(x=0;x<H;x++){
      for(y=0;y<W;y++){
        // if(x!=0 && x!=H+1 && y!=0 && y!=W+1) *(matrix + c * (H+2) * (W+2) + x * (W+2) + y) = c*(x+y);
        // else *(matrix + c * H * W + x * W + y) = 0;

        *(matrix + c * H * W + x * W + y) = c*(x+y);
      }
    }
  }


  // intialize mask 
  for(k=0;k<K;k++){
    for(c=0;c<C;c++){
      for(i=0;i<3;i++){
        for(j=0;j<3;j++){
          *(mask + k*C* 3 * 3 + c*3*3+ i * 3 + j) = (c+k) * (i+j);
        }
      }
    }
  }


  // Size of the mask,input and output in bytes
  size_t bytes_mat = (H+2) * (W+2) * C * sizeof(double);
  size_t bytes_out = K * H * W * sizeof(double);
  size_t bytes_mask = K * C * 9 * sizeof(double);

  // Allocate device memory
  double *d_matrix;
  double *d_result;
  double *d_mask;

  cudaMalloc(&d_matrix, bytes_mat);
  cudaMalloc(&d_mask, bytes_mask);
  cudaMalloc(&d_result, bytes_out);

  // Copy data to the device
  cudaMemcpy(d_matrix, matrix, bytes_mat, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, mask, bytes_mask, cudaMemcpyHostToDevice);

  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = H / 16;

  // Dimension launch arguments
  dim3 block_dim(THREADS,THREADS);
  dim3 grid_dim(BLOCKS,BLOCKS);

  auto start = high_resolution_clock::now();
  convolution_<<<grid_dim, block_dim>>>(d_matrix, d_mask,d_result, H,W,C,K);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);


  // Copy the result back to the CPU
  cudaMemcpy(result, d_result, bytes_out, cudaMemcpyDeviceToHost);

  double check_sum = 0.0;
  for(c =0;c<K;c++){
    for(x=0;x<H;x++){
      for(y=0;y<W;y++){
        check_sum+= *(result + c * H * W + x * W + y);
      }
    }
  }
  std::cout << check_sum << " ";
  std::cout << duration.count() << "\n";

  // Free the memory we allocated
  delete[] matrix;
  delete[] mask;
  delete[] result;

  cudaFree(d_matrix);
  cudaFree(d_mask);
  cudaFree(d_result);

  return 0;
}

