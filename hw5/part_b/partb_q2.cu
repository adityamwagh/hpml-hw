#include <iostream>
#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

__global__ void convolution_tile(double *I, double* M, double *P, int H, int W, int C,int K)
{
    /***** WRITE TO SHARED MEMORY *****/
    W = 18;
    __shared__ double N_ds[18][18][18];

    // First batch loading
    int dest = threadIdx.z + (threadIdx.y * 16) + (threadIdx.x * 16 * 16);
    int destTmp = dest;
    int destX = destTmp % W;
    destTmp = destTmp / W;
    int destY = destTmp % W;
    destTmp = destTmp / W;
    int destZ = destTmp;

    int srcZ = destZ + (blockIdx.z * 16) - 1;
    int srcY = destY + (blockIdx.y * 16) - 1;
    int srcX = destX + (blockIdx.x * 16) - 1;
    int src = srcX + (srcY * W) + (srcZ * H * W);

    if(srcZ >= 0 && srcZ < C && srcY >= 0 && srcY < H && srcX >= 0 && srcX < W)
        N_ds[destZ][destY][destX] = I[src];
    else
        N_ds[destZ][destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.x + (threadIdx.y * 16) + (threadIdx.z * 16 * 16) + 16 * 16 * 16;
    destTmp = dest;
    destX = destTmp % W;
    destTmp = destTmp / W;
    destY = destTmp % W;
    destTmp = destTmp / W;
    destZ = destTmp;

    srcZ = destZ + (blockIdx.z * 16) - 1;
    srcY = destY + (blockIdx.y * 16) - 1;
    srcX = destX + (blockIdx.x * 16) - 1;
    src = srcX + (srcY * W) + (srcZ * W * H);

    if(destZ < W)
    {
        if(srcZ >= 0 && srcZ < C && srcY >= 0 && srcY < H && srcX >= 0 && srcX < W)
            N_ds[destZ][destY][destX] = I[src];
        else
            N_ds[destZ][destY][destX] = 0;
    }
    __syncthreads();

    /***** Perform Convolution *****/
    int x,y,z,k;
    for(k = 0;k<K;k++){
        double sum = 0;
        for(z = 0; z < C; z++)
            for(y = 0; y < 3; y++)
                for(x = 0; x < 3; x++)
                    sum+= N_ds[threadIdx.z + z][threadIdx.y + y][threadIdx.x + x] * M[k*C*3*3+ (2-x) + ((2-y) * 3) + (z * 3 * 3)];
        y = threadIdx.y + (blockIdx.y * 16);
        x = threadIdx.x + (blockIdx.x * 16);
        if(y < H && x < W)
            P[x + (y * W) + (k * W * H)] = sum;
    }

    __syncthreads();

}

int main(int argc, char* argv[]){
    int H = 1024, W = 1024, C = 3,K =64; // I = (c,w,h), I_0 = (C,W+2,H+2)
  // int H = 3, W = 3, C = 2,K =5; // I = (c,w,h), I_0 = (C,W+2,H+2)

    int c,x,y,i,j,k;
    // Allocate the matrix and initialize it
    double *matrix = new double[H * W * C];
    double *result = new double[K * H * W];
    double * mask = new double[K * C * 3 * 3];
    for(c =0;c<C;c++){
        for(x=0;x<H;x++){
          for(y=0;y<W;y++){
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

    auto start = high_resolution_clock::now();
    dim3 dimBlock(16,16, 16);
    dim3 dimGrid((C + 16 - 1) / 16,(H + 16 - 1) / 16, (W + 16 - 1) / 16);
    convolution_tile<<<dimGrid, dimBlock>>>(d_matrix, d_mask,d_result, H,W,C,K);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

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