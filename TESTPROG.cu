#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "salloc_new.h"


__global__
void kernel1(Arena<8, int> a)
{
  int tid = threadIdx.x;
  a.chunks[tid].count = tid;   	
 // printf("count for chunk %d = %d\n",tid,a.chunks[tid].count);   	
}


__global__
void kernel2(Arena<8, int> a)
{
  int tid = threadIdx.x;
  printf("count for chunk %d = %d\n",tid,a.chunks[tid].count);   	
}

int main(int argc, char** argv)
{
  Arena<8,int> arena(8);
  kernel1<<<1,8>>>(arena);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();
  printf("kernel1 done.\n");
  kernel2<<<1,8>>>(arena);
   err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  printf("kernel2 done.\n");
  cudaDeviceSynchronize();
  return 0;
}
