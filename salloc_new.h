
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

// implementing with class

template<int CHUNK_SZ, typename T>
class Chunk {
   public:
     T values[CHUNK_SZ];
     int count;
     Chunk<CHUNK_SZ, T> *next;
   
   __device__
   int numElements() {
	if(count >= CHUNK_SZ)
           return CHUNK_SZ;
        else return count;
   }
  
  __device__ 
   T getElement(int i)
  {
    return values[i];
  }

  /* this is push_back into a chunk. 
   * It specifies the functionality once the chunk
   * to be pushed into is specified 
   */
  __device__
   bool push_back(T value) 
   {
     int id = atomicAdd(&count,1); // atomicAdd returns the oldVal. So id is count before incrementing  
     if(id < CHUNK_SZ) {// space found
       values[id] = value;
       return true; // written successfully in the current chunk    
     }
     
     else return false; 
   }
};

template<int CHUNK_SZ, typename T>
class Arena {
   private:
      int capacity;
   public:
       Chunk<CHUNK_SZ, T> *chunks;
       
	/*creating arena*/
       Arena(int _capacity) : capacity(_capacity)
       {
	  cudaMalloc(&chunks,sizeof(Chunk<CHUNK_SZ,int>) * capacity);
	  cudaMemset(chunks, 0,sizeof(Chunk<CHUNK_SZ, int>) * capacity);
	}

       /* Get_head_chunk() -- get the address of the starting chunk for 
        * the vector concerned..
        * We can follow the link from this chunk to push_back into the 
        * other chunks.
        * */
       
        /* Get_new_chunk -- 
 	*  Expose a new chunk from the arena to the user program  
 	*  by incrementinga counter
 	* */



       /* push_back() -- push_back to the correct vector and chunk
       *  [determine the chunk to which push_back has to happen]
       *  Determine the condiiton for finding the new chunk etc.
       * */      
	
       
};



// implementing with struct

//template<size_t chunk_sz, typename T>
//struct GPUChunk {
//
//	 GPUChunk<chunk_sz, T> *next; // pointer to next chunk
//	 size_t count; // count the number of spaces free in chunk
//	 T values[chunk_sz]; // array to store the values. Each value is a scalar
//};
//
//
//template<size_t chunk_sz, typename T>
//struct GPUArena {
//
//	GPUChunk<chunk_sz, T> *chunks;
//
//	GPUArena<chunk_sz, T> *  create(size_t capacity) {
//	  checkCudaError(cudaMalloc((void**)&chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity ));
//	 return 
//	}
//	  
//};



//template<int CHUNK_SIZE, typename T>
//class GPUChunk {
// public:
//	 GPUChunk<CHUNK_SIZE, T> *next; // pointer to next chunk
//	 int count; // count the number of spaces free in chunk
//	 T values[CHUNK_SIZE]; // array to store the values. Each value is a scalar
//};
//
//template<int CHUNK_SIZE, typename T>
//class GPUArena {
// private:
//	int capacity;
//	GPUChunk<CHUNK_SIZE, T> *chunks; // array of chunks
//
// public:
// 	T* create(int _capacity) //: capacity(_capacity)
//	{
//	  capacity = _capacity;
//	  checkCudaError(cudaMalloc((void**)&chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity ));
//	  T* chunks_addr = chunks;
//	  return chunks_addr;
//	}
// 		
//};


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

