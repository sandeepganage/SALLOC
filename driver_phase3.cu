// This is the driver program for SALLOC

#include <stdio.h>
#include <cuda.h>
#include "salloc_phase3.h"

#define CHUNK_SZ 32 // size of a chunk
#define CAP 16 // number of chunks in the chunk

typedef int T1; // 


__global__ void kernel(GPUArena<CHUNK_SZ,T1> a )

{
 a.get_new_chunk();
}

__global__ void kernel1(GPUArena<CHUNK_SZ,T1> a, GPUChunk<CHUNK_SZ,T1>* v)
{
  printf("%p\n",v); 
}

int main(int argc, char** argv)
{
  GPUArena<CHUNK_SZ, T1> arena(CAP);
  
/* This is the desired API for create vector  */
// GPUChunk<CHUNK_SZ, T1> * v1 = arena.createVector(); // returns the address of the next fully free chunk in arena(on GPU) and stores it on v1 on the CPU
  kernel<<<1,8>>>(arena); 
  cudaDeviceSynchronize();
  GPUChunk<CHUNK_SZ,T1> * v1 = arena.createVector();
  kernel1<<<1,1>>>(arena, v1);
  cudaDeviceSynchronize();
  return 0;
}
