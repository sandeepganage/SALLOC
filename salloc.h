
/**
 * SALLOC: An arena allocator for SIMT architectures in CUDA.
 *
 * @author Somesh Singh, IIT Madras 
 *
 **/

/* Features of SALLOC:
 * 1. createVector() -- Support for multiple vectors on the arena.
 * 2. push_back() on vector
 * 3. pop_back() from vector
 * 4. getIndex(): maps the index of a vector to the corrensponding index on the arena. 
 * 5. vecSize(): get size of the specified vector
 * */

/* The memory allocator does not reclaim the chunks that are freed because of pop_back(). */


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


// Specifying the chunk
 
template <int CHUNK_SIZE, typename T>
class GPUChunk {
  public:
    T values[CHUNK_SIZE]; // holds the actual values in the vector in the chunk
    int nextFreeValue; // keep track of the elements in a chunk
    GPUChunk<CHUNK_SIZE, T> *next; // pointer to the next chunk
    GPUChunk<CHUNK_SIZE, T> *prev; // pointer to the previous chunk


 /*push_back() to a chunk*/
    __device__
    bool push_back(T value) {

   int t;
   t =  atomicAdd(&nextFreeValue,1);
   if(t < CHUNK_SIZE)
   {
      printf("push_back succeeded!\n"); 
      values[t] = value;
      return true;
   }
     
   else 
   {
      nextFreeValue = CHUNK_SIZE;
      return false;
   }
  }

   /* pop_back() from a chunk */
    __device__ bool  pop_back(T &tempVal) {

   int t;
   t =  atomicAdd(&nextFreeValue,-1);
   if(t  > 0)
   {
      printf("pop_back succeeded!\n");
      tempVal = values[t];
      return true; 
   }
     
   else 
   {
      nextFreeValue = 0;
      return false;
   }

    }

};

// Specifying the arena

template <int CHUNK_SIZE, typename T>
class GPUArena {
private:
    // "chunks" points to the arena
    GPUChunk<CHUNK_SIZE, T> *chunks;
    //total number of chunks
    int capacity;
   
public: 
    int *nextFreeChunk_d; // points to the free chunk
    int *xyz_d; //semaphore variable for arena. Can take values 0/1.    
    int *uvw_d; //semaphore variable for arena. Can take values 0/1.    

    GPUArena(int _capacity)
      : capacity(_capacity)
    {
      //allocate the main arena storage and set everything to 0 
      checkCudaError(cudaMalloc(&chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity));
      checkCudaError(cudaMemset(chunks, 0, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity));

      checkCudaError(cudaMalloc(&nextFreeChunk_d, sizeof(int)));
      checkCudaError(cudaMemset(nextFreeChunk_d, 0, sizeof(int)));

      checkCudaError(cudaMalloc(&xyz_d, sizeof(int)));
      checkCudaError(cudaMemset(xyz_d, 0, sizeof(int)));

      checkCudaError(cudaMalloc(&uvw_d, sizeof(int)));
      checkCudaError(cudaMemset(uvw_d, 0, sizeof(int)));
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

 
 /*
  *  Function signature:
  *  T* createVector(int size)
  *
  */

   T * createVector(int sz = CHUNK_SIZE) // optional size parameter that defaults to CHUNK_SIZE.
  {
    
     int h_count; // a host variable to store the current value of nextFreeChunk_d

    
     /** copying of nextFreeChunk_d to h_count **/
    checkCudaError(cudaMemcpy(&h_count, nextFreeChunk_d, sizeof(int), cudaMemcpyDeviceToHost));
    

   
   GPUChunk<CHUNK_SIZE,T> * offsettedAddr = chunks; // offsettedAddr has the starting address of the arena.
   
   // computing the address of the chunk in the arena (by adding offset*sizeof(GPUCHUNK<..>) to "chunks")
   offsettedAddr += h_count * sizeof(GPUChunk<CHUNK_SIZE, T>); // number of bytes, thus the offset

   ++h_count; // incrementing the value of nextFreeChunk so that it now points to a new chunk.
  // The current chunk is reserved for a vector.
  checkCudaError(cudaMemcpy(nextFreeChunk_d, &h_count, sizeof(int), cudaMemcpyHostToDevice));
  return (T*) offsettedAddr; // address on GPU returned
 // it is cast to (T*) so that it can be stepped in multiples of sizeof(T)

   // For future versions : h_count should be incremented by ceil(sz/CHUNK_SIZE) for reserve() operation. Also, a link between these chunks should be created.   
  }  


/*
 * getIndex will map the vector "index" to arena index.
 *
 * Function signature:
 * int getIndex(T* vector, int vectorIndex)
 *
 * if vector index is not found in vector, it returns -1.
 *
 *
 * */

__device__ int getIndex(T* vec, int vecIndex)
{
 GPUChunk<CHUNK_SIZE,T>* currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec;
  
 int vecIndexChunk = vecIndex/CHUNK_SIZE ; // determining the chunk id of the vector in which the specified index of the vector will reside 
 int temp = 0; // to count the chunk of the vector the thread is at
 while(currentChunk->next != NULL)
{
 if(temp == vecIndexChunk) break; // found the correct chunk
 temp++;
 currentChunk = currentChunk->next;
 if(temp == vecIndexChunk) break; // found the correct chunk
} 
 
 // at this point currentChunk either points to the last chunk of the vector or to the chunk containing the vecIndex
 if(vecIndex < vecSize(vec)) // the index of the vector is in the range
{
 // return the corresponding index of arena.
 // computing the corresponding index of arena
 int arenaIndex = ((currentChunk - chunks) * CHUNK_SIZE + vecIndex % CHUNK_SIZE );
 return arenaIndex;
} 
else // the specified vecIndex is not in vector vec;
{
 //assert(false);
 return -1; // dummy value. Signifies the index could not be found.
}

}


/*
 * push_back() pushes the specified element in a vector
 *
 * Function signature:
 *
 * void push_back(T* vector, T elementToPush)
 *
 * */


 __device__ void push_back(T* vec, T ele)
 {
   GPUChunk<CHUNK_SIZE,T>* currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec;
   

   while(true) 
  {
    bool status = currentChunk->push_back(ele); 
    if(status == true) // push_back succeded
	break;
    
    else // chunk is full 
  {
   /* case-1: the filled chunk does not have a link to a new chunk*/
   // one thread establishes a link to a new chunk, while rest of the threads follow the link.
   
   /* case-2: the filled chunk has a link to another chunk */
   // The threads only follow the link to the new chunk.

   if(currentChunk->next == NULL) // case-1
   {
     if(atomicCAS(xyz_d,0,1)==0)  // A single thread exectues this block
     {
       GPUChunk<CHUNK_SIZE,T> * newChunk = get_new_chunk();
       currentChunk->next = newChunk;
       newChunk->prev = currentChunk; // creating a doublely linked list to support pop_back();
       *xyz_d = 0;
     }
     while(*xyz_d == 1); // barrier for all threads performing a push_back().
    // invariant : next chunk should be available and link should be established to the next chunk before other threads try to 
    // push_back in the new chunk.
   
   /* To allow multiple vectors to push_back in parallel, each vector should be local to a vector and not common to the entire arena.*/
  
  
   }
   
   else if(currentChunk->next != NULL) //  case-2: the current chunk is a part of an existing vector spanning multiple chunks.
  {
    currentChunk = currentChunk->next;
  }
   
  }

  }

 }

/*
 * pop_back() pops an element from a vector
 *
 *
 * Function signature:
 *
 * T pop_back(T* vector)
 *
 * returns '-1 or garbage' if pop_back() fails.
 * */

 __device__ T pop_back(T* vec)
{

   GPUChunk<CHUNK_SIZE,T>* currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec;
   GPUChunk<CHUNK_SIZE,T> * parent;
 
  while(currentChunk->next != NULL) // getting to the last chunk of the vec
  {
   currentChunk = currentChunk->next;
  }
 parent = currentChunk->prev;
  
 T tempVal = -1; // local to thread, to be passed by reference to pop_back() in GPUChunk for setting it aptly.
 
  while(true)
 {
 
   if (currentChunk == (GPUChunk<CHUNK_SIZE,T>*) vec &&  ((GPUChunk<CHUNK_SIZE,T>*) vec) ->nextFreeValue <= 0) // the head chunk of vector is empty
  {
	printf("Vector EMPTY! Nothing to pop.\n");
   	break;
  }

  bool status = currentChunk->pop_back(tempVal);
  if (status == true)  {
   return tempVal; 
   }
  
 else  // the current chunk has no element
  {
     if(currentChunk->prev != NULL)
    {
     if(atomicCAS(uvw_d,0,1)==0)  // A single thread executes this block
     {
      parent = currentChunk->prev;
      currentChunk->prev = NULL; 
      parent->next = NULL; // delink currentChunk;
      *uvw_d = 0;
     }
     while(*uvw_d == 1); // barrier for all threads doing pop_back().
    } 
       
   else if(currentChunk->prev == NULL && parent != NULL) //  the thread is pointing to a node that is not a part of any vector
   {
    currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec; // starting over again since we are in the middle of nowhere.
   while(currentChunk->next != NULL) // getting to the last chunk of the vec
   {
     currentChunk = currentChunk->next;
   }
    
   }

  }  

 }
 return tempVal;
}

/*
 * vecSize() returns the size of the specified vector.
 *
 * Function signature:
 *
 * int vecSize(T* vec)
 *
 * It returns '0' if the vector is not present or is fully empty.
 *
 * */


__device__ 
int vecSize(T* vec)
{
   GPUChunk<CHUNK_SIZE,T>* currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec;
   if(vec == NULL) 
   {
     printf("Vector does not exist");
   }
   int count = 0;
	 while(currentChunk->next != NULL)
	{
	 currentChunk = currentChunk->next;
	 count++;
	} 
	   
       // at this point currentChunk is pointing to the last chunk of the vector.
       int vec_sz = (count * CHUNK_SIZE) + (currentChunk->nextFreeValue) ;    
       return vec_sz;
}

};

#endif



