
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
       //printf("push_back successful!\n");
       return true; // written successfully in the current chunk    
     }
     
     else return false; 
   }
};


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
	  *addrOfChunk = &chunks[id];
	  return &chunks[id];
	}


       /* push_back() -- push_back to the correct vector and chunk
       *  [determine the chunk to which push_back has to happen]
       *  Determine the condiiton for finding the new chunk etc.
       * */      
       
       __device__
       void push_back(T element, Chunk<CHUNK_SZ, T> * headChunk) // headChunk is the address of the starting chunk for the vector to which push_back has to happen. 
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
		 if(status == true)
                 {  
		   	break; // push_back() successful. 
                 }
		 else 
		{

		  Chunk<CHUNK_SZ,T>* newChunk = get_new_chunk();
		  if(currentChunk->next == nullptr)
		  {
                  //atomicCAS(currentChunk->next,nullptr, newChunk); // updating the pointer of the current chunk to point to the new chunk
                  currentChunk->next = newChunk;
		  __threadfence(); // global barrier 
		  currentChunk = newChunk;
		  }
		  
		}
	 }

	}	


      bool reserve(int numChunks, Chunk<CHUNK_SZ,T> **head_chunk) // reserve numChunks chunks for the specified vector
      {
	int h_numChunks = numChunks;
	int h_chunkCount;
	cudaMemcpy(&h_chunkCount, chunkCount, sizeof(int), cudaMemcpyDeviceToHost);
	h_chunkCount += h_numChunks;

        if(capacity < h_chunkCount)
	  return false;	

	cudaMemcpy(chunkCount, &h_chunkCount, sizeof(int), cudaMemcpyHostToDevice);
        return true;

      } 

      
      Chunk<CHUNK_SZ,T>** createVector()  // create a new vector
      {
	int h_chunkCount;

	Chunk<CHUNK_SZ,T> ** d_vec, d_v;
	cudaMalloc((void**)&d_vec, sizeof(Chunk<CHUNK_SZ, T>*)); // d_vec is a pointer to a vector
	cudaMalloc((void**)&d_v, sizeof(Chunk<CHUNK_SZ, T>)); 
	cudaMemcpy(&h_chunkCount, chunkCount, sizeof(int), cudaMemcpyDeviceToHost);
	Chunk<CHUNK_SZ,T> ** h_vec = (Chunk<CHUNK_SZ, T>**)malloc(sizeof(Chunk<CHUNK_SZ, T>*));
        
	cudaMemcpy(h_vec, addrOfChunk, sizeof(Chunk<CHUNK_SZ, T>*), cudaMemcpyDeviceToHost);

	h_chunkCount++;
	cudaMemcpy(chunkCount, &h_chunkCount, sizeof(int), cudaMemcpyDeviceToHost);


	cudaMemcpy(d_vec, h_vec, sizeof(Chunk<CHUNK_SZ, T>*), cudaMemcpyHostToDevice);
	return d_vec; // return the address of the starting chunk for new Vector
      }

	
     
};

#endif

