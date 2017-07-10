#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "salloc_new.h"

__global__ 
void kernel1() 
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
//  printf("tid = %d\n",tid); 	
}


int main(int argc, char** argv)
{

  // create an arena
  GPUArena.create<chunk_size,int>; 
  // should the data type "int" be changed to "void" so that 
  // chunk can hold any kind of data by typecasting it appropriately.
  // "void" can be typecast to any type. 

  // allocate a global vector (visible to all threads)
  arena.newVector<int> v1; 
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
