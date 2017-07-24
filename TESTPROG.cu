#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "salloc_new.h"
#include "timer.h"

__global__
void kernel(Arena<32, int> a, Chunk<32, int> **v1,Chunk<32, int> **v2)
{
  int tid = threadIdx.x;
  a.chunks[tid].count = tid;   	
  // *v1 contains the starting address of v1 in the arena.
  // *v2 contains the starting address of v2 in the arena.
  a.push_back(tid,*v1); 
  a.push_back(tid,*v2); 
}


int main(int argc, char** argv)
{

  Arena<32,int> arena(64); // creating an arena with 64 chunks each of size 32.
  
  // create a vector in the arena
  Chunk<32, int> **d_v1; 
  Chunk<32, int> **d_v2; 
  d_v1 = arena.createVector(); // creating a vector and storing its starting address 
  d_v2 = arena.createVector();  
  cudaDeviceSynchronize();

  bool status = arena.reserve(8, d_v1); // reserve 8 chunks for vector d_v1
  if (status) printf("Reserve success !\n");
  kernel<<<8,32>>>(arena,d_v1,d_v2); // kernel launch
  cudaDeviceSynchronize();
  printf("kernel1 done. push_back() successful!\n");
  cudaDeviceSynchronize();
  return 0;
}
