// This is the sample program to illustrate the use of SALLOC

#include <stdio.h>
#include <cuda.h>
#include "salloc.h"

#define CHUNK_SZ 16 // size of a chunk
#define CAP 128 // number of chunks in the arena

typedef int T1;  


__global__ void kernel1(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v2, T1* v3 )
{
  unsigned tid = threadIdx.x;
   a.push_back(v1,tid); printf("push_back in v1\n");
   a.push_back(v2,tid); printf("push_back in v2\n");

}

__global__ void kernel2(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v3)
{
  unsigned tid = threadIdx.x;
  printf("pop value from v1 = %d\n",a.pop_back(v1));
  a.push_back(v3,tid); printf("push_back in v3\n");
  printf("pop value from v3 = %d\n",a.pop_back(v3));
}


__global__ void kernel3(GPUArena<CHUNK_SZ,T1> a, T1* v1)
{
  unsigned tid = threadIdx.x;
  printf("global index of v1[%d] = %d\n",tid,a.getIndex(v1,tid));
  printf("Current size of v1: %d\n",a.vecSize(v1));
}


int main(int argc, char** argv)
{
  GPUArena<CHUNK_SZ, T1> arena(CAP); // creating an arena
  
  T1 * v1 = arena.createVector(); 
  T1 * v2 = arena.createVector(); 
  T1 * v3 = arena.createVector(); 
  kernel1<<<1,23>>>(arena, v1, v2, v3);
  kernel2<<<1,21>>>(arena, v1,v3);
  kernel3<<<1,8>>>(arena, v1);
  cudaDeviceSynchronize();
  return 0;
}
