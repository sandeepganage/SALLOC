// This is the driver program for SALLOC

#include <stdio.h>
#include <cuda.h>
#include "salloc_phase3.h"

#define CHUNK_SZ 5 // size of a chunk
#define CAP 16 // number of chunks in the chunk

typedef int T1; // 


__global__ void kernel(GPUArena<CHUNK_SZ,T1> a )

{
 a.get_new_chunk();
}

__global__ void kernel1(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v2)
{
  unsigned tid = threadIdx.x;
  //printf("%p\n",v); 
//  printf("%p\n",v->values);
//  printf("%p\n",v->nextFreeValue);
//  printf("%p\n",v->next);
//  printf("%p\n",v->prev);
//  printf("%p\n",v+1); 
  a.push_back(v1,tid); printf("push_back in v1\n");
 //a.push_back(v2,tid); printf("push_back in v2\n");
 
  //v->values[1] = 2; 
//  printf("v[%d] = %d\n", tid,v[tid]); 
//  printf("%p\n",(v + sizeof(int))); 
}

__global__ void kernel2(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v2)
{
  unsigned tid = threadIdx.x;
//  printf("%p\n",v); 
//  printf("%p\n",v->values);
//  printf("%p\n",v->nextFreeValue);
//  printf("%p\n",v->next);
//  printf("%p\n",v->prev);
//  printf("%p\n",v+1); 
  a.pop_back(v1); printf("pop_back from v1\n");
 // a.pop_back(v2); printf("pop_back from v2\n");
//  printf("v[%d] = %d\n", tid,v[tid]); 
//  v[1]= 2; 
  //v->values[1] = 2; 
//  printf("%d\n",v[1]); 
//  printf("%p\n",(v + sizeof(int))); 
  //printf("**********\n");
//  a.getIndex(v, 1);
}


__global__ void kernel3(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v2)
{
  unsigned tid = threadIdx.x;
  a.push_back(v2,tid);
}

__global__ void kernel4(GPUArena<CHUNK_SZ,T1> a, T1* v1)
{
  unsigned tid = threadIdx.x;
  printf("global index of v1[%d] = %d\n",tid,a.getIndex(v1,tid));
}

int main(int argc, char** argv)
{
  GPUArena<CHUNK_SZ, T1> arena(CAP);
  
/* This is the desired API for create vector  */
// GPUChunk<CHUNK_SZ, T1> * v1 = arena.createVector(); // returns the address of the next fully free chunk in arena(on GPU) and stores it on v1 on the CPU
  //kernel<<<1,8>>>(arena); 
  //cudaDeviceSynchronize();
  T1 * v1 = arena.createVector(); // 'v1' points to a chunk and not to the array inside the chunk.
  //GPUChunk<CHUNK_SZ,T1> * v1 = arena.createVector(); // 'v1' points to a chunk and not to the array inside the chunk.
  T1 * v2 = arena.createVector(); // we can have a parameter 'size' which can be set to CHUNK_SZ by default.
  T1 * v3 = arena.createVector(); // we can have a parameter 'size' which can be set to CHUNK_SZ by default.
  kernel1<<<1,23>>>(arena, v1,v2);
  //kernel2<<<1,25>>>(arena, v1,v2);
  //kernel3<<<1,5>>>(arena, v1,v2);
  //kernel2<<<1,18>>>(arena, v1,v2);
  kernel4<<<1,25>>>(arena, v1);
  cudaDeviceSynchronize();
  return 0;
}
