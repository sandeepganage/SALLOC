
/*

This is the driver program to test SALLOC. 

In the current snippet, the goal is to perform push_back on a vector vec1
inside kernel1.

gpuVector is the custom implementation of vector.

*/

#include <iostream>
#include <vector>
#include <cuda.h>
#include "salloc.h"


__global__
void kernel1(gpuVector<unsigned> vec1) 
// how to declare a vector and specify the vector,to the kernel, on which push_back needs to be performed?
{
  unsigned tid  = threadIdx.x + blockIdx.x * blockDim.x;
  vec1.my_push_back(tid); // performing push_back on vec1.
}

int main(int argc, char** argv)
{
  gpuVector<unsigned> vec; // on host -- should it initialize the vector on the GPU?
  kernel1<<<1,10>>>(); //or  kernel1<<<1,10>>>(vec); -- should this make vec1 an alias of vec? 
  return 0;
}
