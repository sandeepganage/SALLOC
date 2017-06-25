/*Question: how do we allocate different vectors in the arena and how to distinguish between different vectors?*/

/*Compare the code with malloc() in GPU kernel*/


#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include "GPUArena.h"
#include "timer.h"

__global__ void kernel(GPUArena<2,32,int> ga1)
{
   
   printf("number of elements in layer %d = %d\n",threadIdx.x,ga1.get_num_elements_per_layer(threadIdx.x)); // this works
   int x[5]= {10,11,12,13,14}; // each thread pushs back an arrary and not a scalar
   int u[3]= {201,202,203}; // each thread pushs back an arrary and not a scalar
   int v[3]= {310,31,123}; // each thread pushs back an arrary and not a scalar
   printf("hello1\n");
   ga1.push_back(0,1,x); 
   printf("hello2\n");
   ga1.push_back(0,1,u); 
   printf("hello3\n");
   ga1.push_back(0,1,v); 
   printf("hello4\n");
//   ga1.push_back(0,2,v); 
 //  printf("next free chunk = %d \n",*(ga1.nextFreeChunk_d));
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
 //  ga1.push_back(0,2,v); 
//   printf("x[%d] = %d\n",1,x[1]);
   GPUArenaIterator<32, int> iter = ga1.iterator(0,1); // accessing and traversing an arena
//   printf("iter.cursorInChunk = %d", iter.cursorInChunk);
   int *y = iter.get_next();
   printf("y[%d] = %d \n",0,y[0]);
//   printf("iter.cursorInChunk = %d", iter.cursorInChunk);
   int* z = iter.get_next();
   printf("z[%d] = %d \n",0,z[0]);
//   printf("iter.cursorInChunk = %d", iter.cursorInChunk);
   int* k = iter.get_next();
   printf("k[%d] = %d \n",0,k[0]);
}

//__global__ void kernel1(GPUArena<2,32,int> ga2)
//{
//   int v[2]= {310,31}; // each thread pushs back an arrary and not a scalar
//   ga2.push_back(0,2,v); 
//  
//   GPUArenaIterator<32, int> iter = ga2.iterator(0,2); // accessing and traversing an arena
////   printf("iter.cursorInChunk = %d", iter.cursorInChunk);
//   int *y = iter.get_next();
//   printf("y[%d] = %d \n",0,y[0]);
////   printf("iter.cursorInChunk = %d", iter.cursorInChunk);
// // int* z = iter.get_next();
//  // printf("z[%d] = %d \n",0,z[0]);
//  // printf("iter.cursorInChunk = %d", iter.cursorInChunk);
//  // int* k = iter.get_next();
//  // printf("k[%d] = %d \n",0,k[0]);
//}


int main(void)
{
  std::array<int,2> arr= {4, 4}; // number of elements in each layer
 // note: capacity = sum of no. of elements in each layer
  GPUArena<2,32,int> ga1(2,arr); // creating an arena with proper initial paramerters.
  //GPUChunk<32,int> gc;
  
  kernel<<<1,2>>>(ga1); // this works.
  cudaDeviceSynchronize();
  //kernel1<<<1,2>>>(ga1);
 // cudaDeviceSynchronize();
  ga1.~GPUArena();
  return 0;

}

