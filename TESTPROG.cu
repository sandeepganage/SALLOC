#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "salloc_new.h"


__global__
void kernel1(Arena<8, int> a, Chunk<8, int> **v1,Chunk<8, int> **v2)
{
  int tid = threadIdx.x;
  a.chunks[tid].count = tid;   	
  printf("v1 = %p\n",*v1); // *v1 contains the starting address of v1 in the arena.
  printf("v2 = %p\n",*v2); // *v2 contains the starting address of v2 in the arena.
  
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
  Arena<8,int> arena(16);
  
  // create a vector in the arena
  Chunk<8, int> **d_v1; 
  Chunk<8, int> **d_v2; 
  d_v1 = arena.createVector(); // store the address of 
  d_v2 = arena.createVector(); // store the address of 
  //cudaError_t err = cudaGetLastError();
  //if (err != cudaSuccess)
  //  printf("Error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  arena.reserve(8, d_v1);
  kernel1<<<1,8>>>(arena,d_v1,d_v2);
  //cudaError_t err = cudaGetLastError();
  //if (err != cudaSuccess)
  //  printf("Error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();
  printf("kernel1 done.\n");
  kernel2<<<1,8>>>(arena);
  // err = cudaGetLastError();
  //if (err != cudaSuccess)
  //  printf("Error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
  printf("kernel2 done.\n");
  return 0;
}
