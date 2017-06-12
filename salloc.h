#ifndef SALLOC_H
#define SALLOC_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

template <int H, typename T> // H is a nontyped template parameter.
struct gpuVector
{
  T * chunkSize[H]; // fix the chunck size to 32 for now
  int nextFreeValue;
  gpuVector<H,T> *next;
   
};
 

#endif

