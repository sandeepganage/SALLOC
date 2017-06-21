#include <stdio.h>
#include "salloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
__global__
void kernel(gpuVector<32,int> vec1) // sending the initial chunk. This will grow inside the kernel
{
 vec1.push_back(); 
}

int main(void) {
  gpuVector<32,int> vec1; // allocating the initial chunk alone. It will grow inside the kernel.
  malloc(sizeof(vec1));
  //cudaMalloc(&vec1, sizeof(int)*32); 
  kernel<<<1,10>>>(vec1);
  cudaDeviceSynchronize();
  return 0;
}
