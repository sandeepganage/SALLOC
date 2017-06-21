/*Question: how do we allocate different vectors in the arena and how to distinguish between different vectors?*/


#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include "GPUArena.h"


__global__ void kernel(GPUArena<2,32,int> ga1)
{
   
   printf("number of elements in layer %d = %d\n",threadIdx.x,ga1.get_num_elements_per_layer(threadIdx.x)); // this works
}


int main(void)
{
  std::array<int,2> arr= {3, 4}; // number of elements in each layer
  GPUArena<2,32,int> ga1(2,arr); // creating an arena with proper initial paramerters.
  kernel<<<1,2>>>(ga1); // this works.
  return 0;

}

