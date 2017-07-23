
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

//template <int CHUNK_SZ, typename T>
//__global__
//void kernel_set_offset(Chunk<CHUNK_SZ,T> **d_vec);

template<int CHUNK_SZ, typename T>
class Arena {
   private:
      const int capacity;
   public:
       Chunk<CHUNK_SZ,T> **addrOfChunk;
       int *chunkCount;
       Chunk<CHUNK_SZ, T> *chunks;
	/*creating arena*/
       Arena(int _capacity) : capacity(_capacity)
       {
	  cudaMalloc(&addrOfChunk,sizeof(Chunk<CHUNK_SZ,T>*));
	  cudaMalloc(&chunkCount,sizeof(int));
	  cudaMemset(chunkCount, 0,sizeof(int));
	  cudaMalloc(&chunks,sizeof(Chunk<CHUNK_SZ,T>) * capacity);
	  cudaMemset(chunks, 0,sizeof(Chunk<CHUNK_SZ, T>) * capacity);
	}

       /* Get_head_chunk() -- get the address of the starting chunk for 
        * the vector concerned..
        * We can follow the link from this chunk to push_back into the 
        * other chunks.
        * */
	

	/*get_current_chunk() -- return the address of the chunk to perform push_back() in*/
	
		       
        /* Get_new_chunk -- 
 	*  Expose a new chunk from the arena to the user program  
 	*  by incrementing a counter
 	* */


        __device__
	Chunk<CHUNK_SZ,T>* get_new_chunk()
	{
	  int id = atomicAdd(chunkCount,1);
	  if(id >= capacity)
		  return NULL;
	  atomicExch(addrOfChunk,&chunks[id]);
	  return &chunks[id];
	}


       /* push_back() -- push_back to the correct vector and chunk
       *  [determine the chunk to which push_back has to happen]
       *  Determine the condiiton for finding the new chunk etc.
       * */      
       
       __device__
       bool push_back(T element, Chunk<CHUNK_SZ, T> * headChunk) // headChunk is the address of the starting chunk for the vector to which push_back has to happen. 
								 // it will be a device variable of type pointer which will be specified by the user
       {
	  // invoke Chunk.push_back() appropriately 
	  // traverse the chunks to get to the correct chunk

	
	/* Various scenarios */
	/*
	 * 1. The head chunk of the vector has space -- soln: simply call push_back(val, head_chunk)
	 * 2. The head chunk of the vector is full and we want a new chunk for the vector to grow. 
	 * 	-- soln: go to the next empty chunk.. which can be found by the chunk_count (a variable that keeps track of the next free chunk)  
	 * 	         set the *next field of current Chunk to the new chunk 
	 * 3. There are multiple such vectors and we want that the elements of two different vectors do not go into the same chunk. 
	 * 	-- soln: starting at the head_chunk of the vector, traverse the chunks to get to the correct chunk and perform push back to it.
	 * 	         Alternatively, keep a permanent copy of the head chunk for each vector. Maintain another variable that stores the 
	 * 	         address of the chunk to push_back into for each vector. The value of this variable is updated each time the vector 
	 * 	         spills to a new chunk. This removes the overhead of having to traverse the linked list.
	 *
	 * Note: The updation of pointers when a new chunk is found can be done using atomicCAS().  
	 * */ 
	 
	 Chunk<CHUNK_SZ, T> * currentChunk = headChunk;
	 
	 while(true) // ensure that each chunk does a push_back()
	 {
		 bool status = currentChunk->push_back(element);
		 if(status) break; // push_back() successful. 

		 else 
		{

		  Chunk<CHUNK_SZ,T>* newChunk = get_new_chunk();
                  //atomicCAS(currentChunk->next,NULL, newChunk); // updating the pointer of the current chunk to point to the new chunk
		  if(currentChunk->next == NULL)
		  {
                  atomicCAS(currentChunk->next,NULL, newChunk); // updating the pointer of the current chunk to point to the new chunk
		  __threadfence(); // global barrier 
		  currentChunk = newChunk;
		  }
		  
		}
	 }

	}	


      bool reserve(int numChunks, Chunk<CHUNK_SZ,T> **head_chunk) // reserve numChunks chunks for the specified vector
      {
        //Chunk<CHUNK_SZ,T> * currentChunk = *head_chunk; 
	int h_numChunks = numChunks;
	//int * d_numChunks;
	//cudaMalloc((void**)&d_numChunks, sizeof(int));
	int h_chunkCount;
	cudaMemcpy(&h_chunkCount, chunkCount, sizeof(int), cudaMemcpyDeviceToHost);
	h_chunkCount += h_numChunks;

        if(capacity < h_chunkCount)
	  return false;	

	cudaMemcpy(chunkCount, &h_chunkCount, sizeof(int), cudaMemcpyHostToDevice);
        return true;

      } 

      
      Chunk<CHUNK_SZ,T>** createVector()
      {
	// create a pointer type variable on the GPU
	int h_chunkCount;
	//cudaMemcpy(&h_chunkCount, chunkCount, sizeof(int), cudaMemcpyDeviceToHost);
	//Chunk<CHUNK_SZ,T> ** d_vec;

	//cudaMalloc((void**)&d_vec, sizeof(Chunk<CHUNK_SZ, T>*)); // d_vec is a pointer to a vector

	//h_chunkCount++;
        //
	//cudaMemcpy(chunkCount, &h_chunkCount, sizeof(int), cudaMemcpyHostToDevice);
	Chunk<CHUNK_SZ,T> ** d_vec, d_v;
	cudaMalloc((void**)&d_vec, sizeof(Chunk<CHUNK_SZ, T>*)); // d_vec is a pointer to a vector
	cudaMalloc((void**)&d_v, sizeof(Chunk<CHUNK_SZ, T>)); // d_vec is a pointer to a vector
	cudaMemcpy(&h_chunkCount, chunkCount, sizeof(int), cudaMemcpyDeviceToHost);
	Chunk<CHUNK_SZ,T> ** h_vec = (Chunk<CHUNK_SZ, T>**)malloc(sizeof(Chunk<CHUNK_SZ, T>*));
        
	cudaMemcpy(h_vec, addrOfChunk, sizeof(Chunk<CHUNK_SZ, T>*), cudaMemcpyDeviceToHost);

	h_chunkCount++;
	cudaMemcpy(chunkCount, &h_chunkCount, sizeof(int), cudaMemcpyDeviceToHost);


	//printf("chunkCount = %d\n",h_chunkCount);
	//printf("h_vec = %p\n",*h_vec);
	//kernel_set_offset<<<1,1>>>(d_vec); // set the offset for the vector 
	cudaMemcpy(d_vec, h_vec, sizeof(Chunk<CHUNK_SZ, T>*), cudaMemcpyHostToDevice);
	return d_vec; //*d_vec;
      }

	
     
};

//template <int CHUNK_SZ, typename T>
//__global__
//void kernel_set_offset(Chunk<CHUNK_SZ,T> **d_vec)
//{
//  Arena<CHUNK_SZ,T> a;
//   *d_vec = a.get_new_chunk();
//}

//************************************************************************//

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

