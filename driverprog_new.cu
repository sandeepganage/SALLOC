#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include "timer.h"
#include "salloc_new.h"


__device__ int *d_v1; // making the offsetted address of vector global, 
		     // makes the corresponding vector global 
__constant__ int x;

// TODO: alternative is to initialize the variable using cudaMalloc() and pass the
// offset to the kernel requiring it.

__global__ 
void kernel1(int *d_arr) 
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  //printf("tid:%d ; addr: %p ; offseted addr: %p\n",tid,d_arr,d_v1);
//  printf("tid = %d\n",tid); 	
  printf("tid: %d; offseted address seen: %p\n",tid,d_v1);
  printf("%d\n",d_arr[tid]);
  printf("%p\n",d_arr);
 // printf("%d\n",*d_v1);
}


int main(int argc, char** argv)
{

  //GPUArena<32,int> arena;
  //int *g = arena.create(8);

  //GPUArena<32,int> *arena = GPUArena<32,int>.create(8);


  int *d_arr, *h_arr, *d_brr;
  const int ARRAY_SIZE = 8;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
  cudaMalloc((void**)&d_brr, ARRAY_BYTES );
  cudaMalloc((void**)&d_arr, ARRAY_BYTES );
  h_arr = (int*)malloc(ARRAY_SIZE * sizeof(int));

  for(int i = 0; i < ARRAY_SIZE; i++)
  {
    h_arr[i] = 2*i +1;
  } 
  cudaMemcpy(d_arr,h_arr,ARRAY_BYTES,cudaMemcpyHostToDevice);

  void *addr  = 0;
  printf("addr: %p\n",addr);
  int offset = 5;
  cudaGetSymbolAddress(&addr,x);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  printf("addr: %p\n",addr);
  int* offset_addr = (int*)((int*)addr + offset);
 // pointer arithmetic.. type cast of addr to (int*) is important. Adding 5 to addr to skip 5 chunks of (int).
  printf("offseted addr: %p\n",offset_addr);
   

  //int *d_v1;
  //cudaMalloc((void**) &d_v1, sizeof(int));  
  //cudaMemcpy(d_v1,offset,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_v1, &offset_addr, sizeof(int *));   




  // create an arena
  //GPUArena.create<chunk_size,int>; 
  // should the data type "int" be changed to "void" so that 
  // chunk can hold any kind of data by typecasting it appropriately.
  // "void" can be typecast to any type. 

  // allocate a global vector (visible to all threads)
  //arena.newVector<int> v1; 
  /*This should declare a vector of int. 
   *cudaMalloc() a variable with the name "v1" and set the value 
   *of "v1" to the offset inside the arena. 
   * This way, the starting offset of "v1" inside arena will be
   * available to all threads.
   * When we perform push_back() from inside a kernel like so:
   * v1.push_back(val), it should push into the correct vector.
   * Likewise, if we write v2.push_back(val), it should also
   * push_back() into vector v2 and not v1. 
  */
 
  kernel1<<<1,8>>>(d_arr);
  cudaDeviceSynchronize();
  return 0;
}
