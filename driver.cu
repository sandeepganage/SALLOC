
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
// should vec1 be defined in another kernel?
// or should kernel1 have an empty argument list?
{
  unsigned tid  = threadIdx.x + blockIdx.x * blockDim.x;
  vec1.my_push_back(tid);
}

int main(int argc, char** argv)
{
  gpuVector vec; // on host
  kernel1 <<<1,10>>> (vec); // or kernel1 <<<1,10>>>();
  return 0;
}
