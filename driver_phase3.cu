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

__global__ void kernel1(GPUArena<CHUNK_SZ,T1> a, T1* v)
{
  printf("%p\n",v); 
//  printf("%p\n",v->values);
//  printf("%p\n",v->nextFreeValue);
//  printf("%p\n",v->next);
//  printf("%p\n",v->prev);
//  printf("%p\n",v+1); 
  v[0]= 5; 
  //v->values[1] = 2; 
  printf("%d\n",v[0]); 
//  printf("%p\n",(v + sizeof(int))); 
}

__global__ void kernel2(GPUArena<CHUNK_SZ,T1> a, T1* v)
{
  printf("%p\n",v); 
//  printf("%p\n",v->values);
//  printf("%p\n",v->nextFreeValue);
//  printf("%p\n",v->next);
//  printf("%p\n",v->prev);
//  printf("%p\n",v+1); 
  v[1]= 2; 
  //v->values[1] = 2; 
  printf("%d\n",v[1]); 
//  printf("%p\n",(v + sizeof(int))); 
  printf("**********");
  a.getIndex(v, 1);
}
int main(int argc, char** argv)
{
  GPUArena<CHUNK_SZ, T1> arena(CAP);
  
/* This is the desired API for create vector  */
// GPUChunk<CHUNK_SZ, T1> * v1 = arena.createVector(); // returns the address of the next fully free chunk in arena(on GPU) and stores it on v1 on the CPU
  kernel<<<1,8>>>(arena); 
  cudaDeviceSynchronize();
  T1 * v1 = arena.createVector(); // 'v1' points to a chunk and not to the array inside the chunk.
  //GPUChunk<CHUNK_SZ,T1> * v1 = arena.createVector(); // 'v1' points to a chunk and not to the array inside the chunk.
  kernel1<<<1,1>>>(arena, v1);
  T1 * v2 = arena.createVector(); // we can have a parameter 'size' which can be set to CHUNK_SZ by default.
  kernel2<<<1,1>>>(arena, v2);
  cudaDeviceSynchronize();
  return 0;
}
