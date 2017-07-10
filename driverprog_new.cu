#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "salloc_new.h"


__device__ int *d_v1; // making the offseted address of vector global, 
		     // makes the corresponding vector global 

// TODO: alternative is to initialize the variable using cudaMalloc() and pass the
// offset to the kernel requiring it.

__global__ 
void kernel1() 
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  //printf("tid:%d ; addr: %p ; offseted addr: %p\n",tid,d_arr,d_v1);
//  printf("tid = %d\n",tid); 	
  printf("tid: %d; offseted address seen: %p\n",tid,d_v1);
}


int main(int argc, char** argv)
{

  int *d_arr;
  const int ARRAY_SIZE = 32;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
  cudaMalloc((void**)&d_arr, ARRAY_BYTES );
  
  void *addr;
  int offset = 5;
  cudaGetSymbolAddress(&addr,d_arr);
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
 
  kernel1<<<1,32>>>();
  cudaDeviceSynchronize();
  return 0;
}
