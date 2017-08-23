// This is the driver program for SALLOC

#include <stdio.h>
#include <cuda.h>
#include "salloc.h"

#define CHUNK_SZ 50 // size of a chunk
#define CAP 1600 // number of chunks in the chunk

typedef char T1; // 


__global__ void kernel(GPUArena<CHUNK_SZ,T1> a )

{
 a.get_new_chunk();
}

__global__ void kernel1(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v2, T1* v3 )
{
  unsigned tid = threadIdx.x;
   a.push_back(v1,tid); printf("push_back in v1\n");
   a.push_back(v2,tid); printf("push_back in v2\n");

}

__global__ void kernel5(GPUArena<CHUNK_SZ,T1> a, T1* v2)
{
  unsigned tid = threadIdx.x;
   a.push_back(v2,tid); printf("push_back in v1\n");

}


__global__ void kernel6(GPUArena<CHUNK_SZ,T1> a, T1* v3)
{
  unsigned tid = threadIdx.x;
   a.push_back(v3,tid); printf("push_back in v1\n");

}

__global__ void kernel2(GPUArena<CHUNK_SZ,T1> a, T1* v1, T1* v3)
{
  unsigned tid = threadIdx.x;
  printf("pop value from v1 = %d\n",a.pop_back(v1)); //printf("pop_back from v1\n");
   a.push_back(v3,tid); printf("push_back in v3\n");
  printf("pop value from v3 = %d\n",a.pop_back(v3)); //printf("pop_back from v1\n");
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
  
  T1 * v1 = arena.createVector(); 
  T1 * v2 = arena.createVector(); 
  T1 * v3 = arena.createVector(); // we can have a parameter 'size' which can be set to CHUNK_SZ by default.
  kernel1<<<1,23>>>(arena, v1, v2, v3);
  //kernel5<<<1,20>>>(arena, v2);
  //kernel6<<<1,20>>>(arena, v3);
  kernel2<<<1,25>>>(arena, v1,v3);
//  kernel1<<<1,5>>>(arena, v1,v2,v3);
//  kernel2<<<1,10>>>(arena, v1,v2,v3);
//  kernel3<<<1,25>>>(arena, v1);
  kernel4<<<1,10>>>(arena, v1);
  cudaDeviceSynchronize();
  return 0;
}
