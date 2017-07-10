
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






//template <int CHUNK_SIZE, typename T>
//class GPUChunk {
//  public:
//    T values[CHUNK_SIZE];
//    int nextFreeValue;
//    GPUChunk<CHUNK_SIZE, T> *next;
//
//    __device__
//    int num_values_in_chunk() {
//      if(nextFreeValue > CHUNK_SIZE) {
//        return CHUNK_SIZE;
//      } else {
//        return nextFreeValue;
//      }
//    }
//
//    __device__
//    bool push_back(T value) {
//      int id = atomicAdd(&nextFreeValue, 1);
//      if(id < CHUNK_SIZE) {
//        printf("push_back succeeded!\n");
//        values[id] = value;
//        return true;
//      } else {
//        return false;
//      }
//    }
//
//    __device__
//    T get_element_at(int i) {
//      return values[i];
//    }
//};
//
//template <int CHUNK_SIZE, typename T> 
//class GPUArenaIterator { // iterator for the arena
//  private:
//    GPUChunk<CHUNK_SIZE, T> *currentChunk;
//    int cursorInChunk;
//
//  public:
//    __device__
//    GPUArenaIterator(GPUChunk<CHUNK_SIZE, T> *head_chunk) {
//      currentChunk = head_chunk;
//      if(currentChunk != NULL) {  
//        cursorInChunk = currentChunk->num_values_in_chunk() - 1; /* num_values_in_chunk() may be 0 as well  */
//      }
//    }
//
//    __device__
//    bool has_next() {
//      return currentChunk != NULL && (cursorInChunk >= 0 || currentChunk->next != NULL);
//    }
//
//    __device__ 
//    T * get_next() {   
//      if(cursorInChunk < 0) {
//        //No more elements left in chunk, go to next chunk
//        currentChunk = currentChunk->next;
//        cursorInChunk = currentChunk->num_values_in_chunk() - 1;
//      }
//      return currentChunk->get_element_at(cursorInChunk--);
//    }
//};
//
//template <int CHUNK_SIZE, typename T>
//__global__
//void init_mappings_kernel(GPUChunk<CHUNK_SIZE, T> **mappings, GPUChunk<CHUNK_SIZE, T> *startOfChunks, int offset, int numElements) {
//  for(int mySlot = threadIdx.x + blockIdx.x * blockDim.x; mySlot < numElements; mySlot += gridDim.x * blockDim.x) {
//    mappings[mySlot] = startOfChunks + offset + mySlot;
//  }
//};
//
//template <int NumLayers, int CHUNK_SIZE, typename T>
//class GPUArena {
//  private:
//    //# of chunks per layer 
//    int numElementsPerLayer[NumLayers];
//    //a map from an element id (per layer) to the head of the chunk linked list that stores the values
//    GPUChunk<CHUNK_SIZE, T> **mappingIdToCurrentChunk[NumLayers];
//    //the shared chunks
//    GPUChunk<CHUNK_SIZE, T> *chunks;
//    //total number of chunks
//    int capacity;
//    //shared cursor to indicate the next free chunk
//    //next free chunk does not start out as 0 but every element in every layer
//    //by default gets a chunk
//  public:
//    int *nextFreeChunk_d;
//
//    GPUArena(int _capacity, std::array<int, NumLayers> pNumElementsPerLayer)
//      : capacity(_capacity)
//    {
//      //allocate the main arena storage and set everything to 0 (important
//      //because the counters in each chunk must be )
//      checkCudaError(cudaMalloc(&chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity));
//      checkCudaError(cudaMemset(chunks, 0, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity));
//      checkCudaError(cudaMalloc(&nextFreeChunk_d, sizeof(int)));
//      checkCudaError(cudaMemset(nextFreeChunk_d, 0, sizeof(int)));
//
//      int offset = 0;
//      for(int layer = 0; layer < NumLayers; layer++) {
//        numElementsPerLayer[layer] = pNumElementsPerLayer[layer];
//        //each element implicitly gets its own initial chunk
//        size_t mapSizeInBytes = sizeof(GPUChunk<CHUNK_SIZE, T>*) * numElementsPerLayer[layer];
//        checkCudaError(cudaMalloc(&mappingIdToCurrentChunk[layer], mapSizeInBytes));
//
//        init_mappings_kernel<<<8, 32>>>(mappingIdToCurrentChunk[layer], chunks, offset, numElementsPerLayer[layer]);
//        checkLastCudaError();
//        cudaDeviceSynchronize();
//        checkLastCudaError();
//        offset += numElementsPerLayer[layer];
//      }
//      checkCudaError(cudaMemcpy(nextFreeChunk_d, &offset, sizeof(int), cudaMemcpyHostToDevice));
//    }
//
//    ~GPUArena() {
//      for(int layer = 0; layer < NumLayers; layer++)
//        cudaFree(mappingIdToCurrentChunk[layer]);
//      cudaFree(nextFreeChunk_d);
//      cudaFree(chunks);
//    }
//
//    __device__
//    int get_num_elements_per_layer(int layer) {
//      return numElementsPerLayer[layer];
//    }
//
//    __device__
//    GPUChunk<CHUNK_SIZE, T>* get_new_chunk() {
//      int id = atomicAdd(nextFreeChunk_d, 1);
//
//      if(id >= capacity) {
//        printf("GPUArena out of capacity\n");
//        assert(false);
//        return NULL;
//      }
//      return &chunks[id];
//    }
//
//    __device__
//    GPUChunk<CHUNK_SIZE, T>* get_head_chunk(int layer, int elementId) {
//      return mappingIdToCurrentChunk[layer][elementId];
//    }
//
//    __device__
//    GPUArenaIterator<CHUNK_SIZE, T> iterator(int layer, int elementId) {
//      return GPUArenaIterator<CHUNK_SIZE, T>(get_head_chunk(layer, elementId));
//    }
//
//    __device__
//    void push_back(int layer, int elementId, T value) {
//
//      GPUChunk<CHUNK_SIZE, T> *currentChunk = get_head_chunk(layer, elementId);
//      assert(currentChunk);
//
//      while(true) {
//        bool status = currentChunk->push_back(value);
//        if(status == true) {
//          break;
//        } else {
//          //chunk is full.
//          GPUChunk<CHUNK_SIZE, T> *newChunk = get_new_chunk();
//          newChunk->next = currentChunk; //list
//          GPUChunk<CHUNK_SIZE, T> *oldChunk = (GPUChunk<CHUNK_SIZE, T>*)atomicCAS((unsigned long long int *)&mappingIdToCurrentChunk[layer][elementId], (unsigned long long int)currentChunk, (unsigned long long int)newChunk);
//          currentChunk = (oldChunk == currentChunk) ? newChunk : oldChunk;
//        }
//      }
//    }
//
//};

#endif

