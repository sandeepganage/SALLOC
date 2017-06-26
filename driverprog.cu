#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include "salloc.h"
#include "timer.h"


#define CHUNK_SZ 32
#define NUM_LAYERS 8
#define CAPACITY 2048 // 2^11


__global__ void kernel(GPUArena<NUM_LAYERS,CHUNK_SZ,int> ga1)
{
   
   int x[5]= {10,11,12,13,14}; // each thread pushs back an arrary and not a scalar
   int _eleId = threadIdx.x; 
   int _layer = blockIdx.x;
   ga1.push_back(_layer,_eleId,x); 
//   GPUArenaIterator<32, int> iter = ga1.iterator(_layer,_eleId); // accessing and traversing an arena
//   int *y = iter.get_next(); // get the first vector in a chunk 
//   printf("y[%d] = %d \n",0,y[0]); // accessing an element in the vector in the chunk
//   int* z = iter.get_next();
//   printf("z[%d] = %d \n",0,z[0]);
//   int* k = iter.get_next();
//   printf("k[%d] = %d \n",0,k[0]);
}

int main(void)
{

  GpuTimer timer;
  std::array<int,NUM_LAYERS> arr= {256,256,256,256,256,256,256,256}; // number of elements in each layer
 // note: capacity = sum of no. of elements in each layer
  GPUArena<NUM_LAYERS,CHUNK_SZ,int> ga1(CAPACITY,arr); // creating an arena with proper initial paramerters.
  unsigned block_sz = 32;
  unsigned grid_sz =  NUM_LAYERS;
  timer.Start();
  kernel<<<grid_sz,block_sz>>>(ga1); // invoking the kernel where push_back() is performed..
  timer.Stop();
  cudaDeviceSynchronize();
  printf("Kernel code ran in: %f msecs.\n", timer.Elapsed());
  ga1.~GPUArena();
  return 0;

}

