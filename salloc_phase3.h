
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
    GPUChunk<CHUNK_SIZE,T> ** d_v; // a variable to store the address of chunk on GPU.
    
    checkCudaError(cudaMalloc(&d_v, sizeof(T *)));
    
  }  
 
};

#endif

