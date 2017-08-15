
/* TODOs:
 * 1. implement push_back()
 * 2. implement pop_back()
 * 3. implement the API: arena.createVector([size]) -- initial size of the vector is an optional parameter.
 *    Support for multiple vectors on the arena.
 * 4. implement getIndex(): maps the index of a vector to the corrensponding index on the arena. 
 * */

/**
 * GOAL-1: implement support for multiple vectors on the arena
 */

#ifndef SALLOC_H
#define SALLOC_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

#define checkCudaError(val) __checkCudaError__ ( (val), #val, __FILE__, __LINE__ )

template <typename T>
inline void __checkCudaError__(T code, const char *func, const char *file, int line)
{
  if (code) {
    fprintf(stderr, "CUDA error at %s:%d: %s (code=%d)\n",
        file, line, cudaGetErrorString(code), (unsigned int)code);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#define checkLastCudaError() checkCudaError( (cudaGetLastError()) )

// Creating the arena
 
template <int CHUNK_SIZE, typename T>
class GPUChunk {
  public:
    T values[CHUNK_SIZE];
    int nextFreeValue;
    // required for push_back()
    GPUChunk<CHUNK_SIZE, T> *next; // pointer to the next chunk
    // required for pop_back()
    GPUChunk<CHUNK_SIZE, T> *prev; // pointer to the previous chunk
};


template <int CHUNK_SIZE, typename T>
class GPUArena {
private:
    // chunks points to the arena
    GPUChunk<CHUNK_SIZE, T> *chunks;
    //total number of chunks
    int capacity;
   
public: 
    int *nextFreeChunk_d; // points to the free chunk
    GPUArena(int _capacity)
      : capacity(_capacity)
    {
      //allocate the main arena storage and set everything to 0 (important
      //because the counters in each chunk must be )
      checkCudaError(cudaMalloc(&chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity));
      checkCudaError(cudaMemset(chunks, 0, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity));

      checkCudaError(cudaMalloc(&nextFreeChunk_d, sizeof(int)));
      checkCudaError(cudaMemset(nextFreeChunk_d, 0, sizeof(int)));

  }


 // expose a new chunk from the arena to the user program
  __device__ 
  GPUChunk<CHUNK_SIZE,T> * get_new_chunk()
  {
      int id = atomicAdd(nextFreeChunk_d, 1);

      if(id >= capacity) {
        printf("GPUArena out of capacity\n");
        assert(false);
        return NULL;
      }
      return &chunks[id]; // returns the address of the new chunk
   }

 // defining createVector() function 
 /**
 * The  functionality of the function will be to store the address of the starting chunk of the 
 * vector (on GPU) to a variable in CPU. This variable will be passed to the kernel(s) that would 
 * use this vector. 
 * */
 // createVector() has to be host function since it is always invoked from the CPU.
   GPUChunk<CHUNK_SIZE,T> * createVector()
  {
    /** Allocate all memory in CPU using malloc() and the corresponding memory 
      * in GPU using cudaMalloc() 
      * Also, do not forget to allocate space for both **v and *v separately
      * or else we will get segmentation fault. 
      * **/
    GPUChunk<CHUNK_SIZE,T> ** d_v; // a variable on the CPU stack to store the address of address of chunk on GPU.
    GPUChunk<CHUNK_SIZE,T> ** h_v; // a variable, on the stack, to store the address of address of chunk on CPU.
    h_v = (GPUChunk<CHUNK_SIZE,T>**)malloc(sizeof(GPUChunk<CHUNK_SIZE,T>*)); // alocating a block on the heap that can hold the address of a chunk. h_v points to this block on the heap.  
    checkCudaError(cudaMalloc(&h_v[0],sizeof(GPUChunk<CHUNK_SIZE,T>))); // allocating a pointer to a chunk on the GPU and storing its address in hv[0] on the CPU. 
    checkCudaError(cudaMalloc(&d_v, sizeof(GPUChunk<CHUNK_SIZE,T> *))); // allocating a pointer to a pointer on GPU. 
    
   // now h_v is a pointer to a pointer and d_v is also a pointer to a pointer on the CPU and GPU respectively.
   // d_v[0] can be allocated the address of a chunk now.

    int h_count; // a host variable to store the current value of nextFreeChunk_d

    
     /** copying of nextFreeChunk_d to h_count is working fine. **/
    checkCudaError(cudaMemcpy(&h_count, nextFreeChunk_d, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("h_count = %d\n",h_count);

   /* Do stuff (i.e. compute the address of the chunk )*/
   
   // copy the starting address of "chunks" in h_v[0], then compute the address of chunk using the value of h_count. After having computed the value, copy h_v to d_v 

    checkCudaError(cudaMemcpy(d_v, h_v, sizeof(GPUChunk<CHUNK_SIZE,T>*), cudaMemcpyHostToDevice));


   /* Copy the d_v to h_v  (if required) like so */
   /**
    checkCudaError(cudaMemcpy(h_v, d_v, sizeof(GPUChunk<CHUNK_SIZE,T>*), cudaMemcpyDeviceToHost));
   **/





    // address of the chunk (on the GPU) = starting address of arena + nextFreeChunk_d * sizeof(GPUChunk<..>)
    
/**
    GPUChunk<CHUNK_SIZE, T>* addr = chunk + h_count; // adding an offset to starting address of arena (making use of pointer arithmetic) [might be erroronous] 
    
   *v = addr;
    
**/    
    
  return *h_v; // dummy return 
  }  
 
};

#endif

