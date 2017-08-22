
/* TODOs:
 * 1. implement push_back()
 * 2. implement pop_back()
 * 3. implement the API: arena.createVector([size]) -- initial size of the vector is an optional parameter.
 *    Support for multiple vectors on the arena.
 * 4. implement getIndex(): maps the index of a vector to the corrensponding index on the arena. 
 * 5. Implement an iterator.
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
    
    

    __device__
    bool push_back(T value) {
      int id = atomicAdd(&nextFreeValue, 1);
      if(id < CHUNK_SIZE) {
        printf("push_back succeeded!\n");
        //printf("threadId = %d, nextFreeValue = %d\n",threadIdx.x,id);
        values[id] = value;
        return true;
      } else {
        return false;
      }
    }

    __device__ bool pop_back() {

// FIXME: correct the indexing part. Do not allow extra threads to do an atomic decrement if they will fail the condition (id < 0)

/****************************************************************/
      int id = atomicAdd(&nextFreeValue,-1);
//      printf("id = %d\n",nextFreeValue);
      if(id > 0) {
        printf("pop_back succeeded!\n");
        //printf("threadId = %d, nextFreeValue = %d\n",threadIdx.x,id);
        //return values[id];
        return true;
      } else {
        return false;
       // return;
	
    }
/*****************************************************************/       
    }

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
    int *xyz_d; //semaphore variable for arena. Can take values 0/1.    
    int *uvw_d; //semaphore variable for arena. Can take values 0/1.    

    GPUArena(int _capacity)
      : capacity(_capacity)
    {
      //allocate the main arena storage and set everything to 0 (important
      //because the counters in each chunk must be )
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

 // __device__ 
 // GPUChunk<CHUNK_SIZE,T> * reclaim_chunk()
  __device__
   void reclaim_chunk()
  {
      printf("hello from reclaim_chunk()\n");
      int id = atomicAdd(nextFreeChunk_d,-1);

      if(id <= 0) {
        printf("GPUArena empty\n");
        assert(false);
       // return NULL;
      }
      //return &chunks[id]; // returns the address of the new chunk
   }
 


 // defining createVector() function 
 /**
 * The  functionality of the function will be to store the address of the starting chunk of the 
 * vector (on GPU) to a variable in CPU. This variable will be passed to the kernel(s) that would 
 * use this vector. 
 * */
 // createVector() has to be host function since it is always invoked from the CPU.


# if 0

  /* Version 1 */
  
   GPUChunk<CHUNK_SIZE,T> * createVector()
  {
    /** Allocate all memory in CPU using malloc() and the corresponding memory 
      * in GPU using cudaMalloc() 
      * Also, do not forget to allocate space for both **v and *v separately
      * or else we will get segmentation fault. 
      * **/
    GPUChunk<CHUNK_SIZE,T> ** d_v; // a variable on the CPU stack to store the address of address of chunk on GPU.
    GPUChunk<CHUNK_SIZE,T> ** h_v; // a variable, on the stack, to store the address of address of chunk on GPU.
    h_v = (GPUChunk<CHUNK_SIZE,T>**)malloc(sizeof(GPUChunk<CHUNK_SIZE,T>*)); // alocating a block on the heap that can hold the address of a chunk. h_v points to this block on the heap.  
    //after  the above line, h_v[0] is allocated. h_v[0] can hold an address of a chunk

    checkCudaError(cudaMalloc(&h_v[0],sizeof(GPUChunk<CHUNK_SIZE,T>))); // allocating a pointer to a chunk on the GPU and storing its address in hv[0] on the CPU. 
    checkCudaError(cudaMalloc(&d_v, sizeof(GPUChunk<CHUNK_SIZE,T> *))); // allocating a pointer to a pointer on GPU. 
    
   // now h_v is a pointer to a pointer and d_v is also a pointer to a pointer on the CPU and GPU respectively.
   // d_v[0] can be allocated the address of a chunk now.

    checkCudaError(cudaMemcpy(d_v, h_v, sizeof(GPUChunk<CHUNK_SIZE,T>*), cudaMemcpyHostToDevice)); // copying h_v[0] to d_v[0]


    // the only purpose h_v serves is to help allocate a pointer to a pointer on the GPU.
    // we need to allocate a separate variable to work on the CPU.
    // Check: does modifying the value of h_v[0] and copying it to d_v[0] work?
    
    
    
    
    int h_count; // a host variable to store the current value of nextFreeChunk_d

    
     /** copying of nextFreeChunk_d to h_count is working fine. **/
    checkCudaError(cudaMemcpy(&h_count, nextFreeChunk_d, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("h_count = %d\n",h_count);

   /* Do stuff (i.e. compute the address of the chunk )*/
   
   // copy the starting address of "chunks" in h_v[0], then compute the address of chunk using the value of h_count. After having computed the value, copy h_v to d_v 

   
   // The starting address of the array "chunks" (the arena) on the GPU is already available on the CPU, in the variable chunks. We can use it as it is.
   
   GPUChunk<CHUNK_SIZE,T> * offsettedAddr = chunks; // offsettedAddr has the starting address of the arena.
   
   // computing the address of the chunk in the arena (by adding offset*sizeof(GPUCHUNK<..>) to "chunks")
   offsettedAddr += h_count * sizeof(GPUChunk<CHUNK_SIZE, T>); // number of bytes, thus the offset
   
   // copy the offsettedAddr to d_v[0]
  checkCudaError(cudaMemcpy(d_v, &offsettedAddr, sizeof(GPUChunk<CHUNK_SIZE, T>*), cudaMemcpyHostToDevice)); 
  



   /* Copy the d_v to h_v  (if required) like so */
   /**
    checkCudaError(cudaMemcpy(h_v, d_v, sizeof(GPUChunk<CHUNK_SIZE,T>*), cudaMemcpyDeviceToHost));
   **/





    // address of the chunk (on the GPU) = starting address of arena + nextFreeChunk_d * sizeof(GPUChunk<..>)
    
/**
    GPUChunk<CHUNK_SIZE, T>* addr = chunk + h_count; // adding an offset to starting address of arena (making use of pointer arithmetic) [might be erroronous] 
    
   *v = addr;
    
**/    
    
  return offsettedAddr; // address on GPU returned
  }  
#endif


#if 1

  /* Version 2 */

   T * createVector(int sz = CHUNK_SIZE) // optional size parameter that defaults to CHUNK_SIZE.
   //GPUChunk<CHUNK_SIZE,T> * createVector()
  {
    /** Allocate all memory in CPU using malloc() and the corresponding memory 
      * in GPU using cudaMalloc() 
      * Also, do not forget to allocate space for both **v and *v separately
      * or else we will get segmentation fault. 
      * **/
  //  GPUChunk<CHUNK_SIZE,T> ** d_v; // a variable on the CPU stack to store the address of address of chunk on GPU.
  //  GPUChunk<CHUNK_SIZE,T> ** h_v; // a variable, on the stack, to store the address of address of chunk on GPU.

  //  h_v = (GPUChunk<CHUNK_SIZE,T>**)malloc(sizeof(GPUChunk<CHUNK_SIZE,T>*)); // alocating a block on the heap that can hold the address of a chunk. h_v points to this block on the heap.  
    //after  the above line, h_v[0] is allocated. h_v[0] can hold an address of a chunk

  //  checkCudaError(cudaMalloc(&h_v[0],sizeof(GPUChunk<CHUNK_SIZE,T>))); // allocating a pointer to a chunk on the GPU and storing its address in hv[0] on the CPU. 
  //  checkCudaError(cudaMalloc(&d_v, sizeof(GPUChunk<CHUNK_SIZE,T> *))); // allocating a pointer to a pointer on GPU. 
    
   // now h_v is a pointer to a pointer and d_v is also a pointer to a pointer on the CPU and GPU respectively.
   // d_v[0] can be allocated the address of a chunk now.

 //   checkCudaError(cudaMemcpy(d_v, h_v, sizeof(GPUChunk<CHUNK_SIZE,T>*), cudaMemcpyHostToDevice)); // copying h_v[0] to d_v[0]


    // the only purpose h_v serves is to help allocate a pointer to a pointer on the GPU.
    // we need to allocate a separate variable to work on the CPU.
    // Check: does modifying the value of h_v[0] and copying it to d_v[0] work?
    
    int h_count; // a host variable to store the current value of nextFreeChunk_d

    
     /** copying of nextFreeChunk_d to h_count is working fine. **/
    checkCudaError(cudaMemcpy(&h_count, nextFreeChunk_d, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("h_count = %d\n",h_count);

   /* Do stuff (i.e. compute the address of the chunk )*/
   
   // copy the starting address of "chunks" in h_v[0], then compute the address of chunk using the value of h_count. After having computed the value, copy h_v to d_v 

   
   // The starting address of the array "chunks" (the arena) on the GPU is already available on the CPU, in the variable chunks. We can use it as it is.
   
   GPUChunk<CHUNK_SIZE,T> * offsettedAddr = chunks; // offsettedAddr has the starting address of the arena.
   
   // computing the address of the chunk in the arena (by adding offset*sizeof(GPUCHUNK<..>) to "chunks")
   offsettedAddr += h_count * sizeof(GPUChunk<CHUNK_SIZE, T>); // number of bytes, thus the offset
   
   // copy the offsettedAddr to d_v[0]
 // checkCudaError(cudaMemcpy(d_v, &offsettedAddr, sizeof(GPUChunk<CHUNK_SIZE, T>*), cudaMemcpyHostToDevice)); 
  



   /* Copy the d_v to h_v  (if required) like so */
   /**
    checkCudaError(cudaMemcpy(h_v, d_v, sizeof(GPUChunk<CHUNK_SIZE,T>*), cudaMemcpyDeviceToHost));
   **/





    // address of the chunk (on the GPU) = starting address of arena + nextFreeChunk_d * sizeof(GPUChunk<..>)
    
/**
    GPUChunk<CHUNK_SIZE, T>* addr = chunk + h_count; // adding an offset to starting address of arena (making use of pointer arithmetic) [might be erroronous] 
    
   *v = addr;
    
**/    
 // h_count should be incremented by ceil(sz/CHUNK_SIZE). Also, a link between these should be created.   
  ++h_count; // incrementing the value of nextFreeChunk so that it now points to a new chunk.
  // in effect, the current chunk is reserved for a vector.
  checkCudaError(cudaMemcpy(nextFreeChunk_d, &h_count, sizeof(int), cudaMemcpyHostToDevice));
  return (T*) offsettedAddr; // address on GPU returned
 // it is cast to (T*) so that it can be stepped in multiples of sizeof(T)
  
  // TODO: done
  // here I am returning the starting address of the chunk. I should rather return the address of the the array inside so that the user is able to write things like 
  // v[0], v[1] etc.

  // Note: no need to do operator overloading. 
  // A hack to tackle reserve/initial size of vector
  // Since we know all chunks after the one being pointed to by   
  // nextFreeChunk_d are free, we need to increment the value of nextFreeChunk by
  // ceil(requested size / chunk_size ).
  // create a link between the chunks, if the vector spans multiple contiguous 
  // chubks too. 
  
  //TODO: Not Required
  //overload the '[]' operator to check for max_allowed size (specified size or filled up size). Throw an error if one is trying to access out of bound accesses.


  }  

 // getIndex will map the vector "index" to arena index.
 // usage:
 // int i = getIndex(v,index);
 // v[i]++;
  __device__ int 
  getIndex(T* v, int index) // here index is the vector index
{
  GPUChunk<CHUNK_SIZE,T> * v1 = (GPUChunk<CHUNK_SIZE,T> *) v;
 // printf("\nprint from getIndex\n ");
 // printf("%d\n",v[index]);
 // printf("%d\n", v1->values[index]);


 // invoke an iterator to go over the arena starting from the specified chunk and follow the links to access all elements of a vector. In the process compute the cunks
}

 __device__ void push_back(T* vec, T ele)
 {
   GPUChunk<CHUNK_SIZE,T>* currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec;
   

   while(true) 
  {
    bool status = currentChunk->push_back(ele); 
    if(status == true) // push_back succeded
{printf("nextFreeChunk = %d\n",*nextFreeChunk_d);

	break;}
    
    else // chunk is full 
  {
   /* case-1: the filled chunk does not have a link to a new chunk*/

   // one thread establishes a link to a new chunk, while rest of the threads follow the link.
   
   /* case-2: the filled chunk has a link to another chunk */
  // The threads only follow the link to the new chunk.

   if(currentChunk->next == NULL) // case-1
   {
     //printf("want new chunk\n");
     if(atomicCAS(xyz_d,0,1)==0) // atomicCAS will be executed by all threads entering the then block.
    // if multiple threads try to perform the CAS simultaneously, only one of them will succeed. The rest will go ahead and
    // execute the else branch.
     {
       GPUChunk<CHUNK_SIZE,T> * newChunk = get_new_chunk();
       printf("got new chunk for PUSH_BACK\n");
       currentChunk->next = newChunk;
       newChunk->prev = currentChunk; // creating a doublely linked list to support pop_back();
       //currentChunk->nextFreeValue = CHUNK_SIZE;
       *xyz_d = 0;
       //currentChunk = newChunk; // updating the currentChunk to newChunk;
     }
    // printf("before the while loop\n");
     while(*xyz_d == 1); // barrier for all threads.
    // invariant : next chunk should be available and link should be established to the next chunk before other threads try to 
    // push_back in the new chunk.
   
   /* To allow multiple vectors to push_back in parallel, each vector should be local to a vector and not common to the entire arena.*/
  
   // printf("after the while loop\n");   
  
   }
   
   else if(currentChunk->next != NULL) //  case-2: the current chunk is a part of an existing vector spanning multiple chunks.
  {
    currentChunk = currentChunk->next;
   //FIXED: complete the todo .. [current_chunk is currently not being set correctly]
   //printf("go to new chunk\n");

   // TODO: may not be required
   // iterate over the chunks of the vector and get to the chunk which is either not full or to the chunk for which current->next is NULL.
   // thereafter call the push_back() method in GPUChunk to push back the elements.
  }
   
  }


  }


 // __syncthreads();
  if(blockIdx.x == 0 && threadIdx.x ==0)
{
  GPUChunk<CHUNK_SIZE,T> * temp = (GPUChunk<CHUNK_SIZE,T> *)  vec;
  while(temp->next != NULL){ temp->nextFreeValue = CHUNK_SIZE; temp = temp->next;}
}
 }


 __device__ void pop_back(T* vec)
{
// FIXME: currently pop_back pops as many elements as requested even if the value is not there an then gets stuck 
   GPUChunk<CHUNK_SIZE,T>* currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec;
   /* TODO: traverse the arena to get to the last chunk of the current vector
 * which has atleast one element */
  GPUChunk<CHUNK_SIZE,T> * parent;
 
  while(currentChunk->next != NULL) // getting to the last chunk of the vec
  {
   currentChunk = currentChunk->next;
  }
 parent = currentChunk->prev;
     //printf("parent chunksize outside while = %d\n",parent->nextFreeValue);
  
  while(true)
 {
  //if(((GPUChunk<CHUNK_SIZE,T>*) vec)->nextFreeValue < 0)
  //  break;  
 
   if (currentChunk == (GPUChunk<CHUNK_SIZE,T>*) vec &&  ((GPUChunk<CHUNK_SIZE,T>*) vec) ->nextFreeValue <= -1)
   	break;
  bool status = currentChunk->pop_back();
  if (status == true)  {//printf("nextFreeChunk = %d\n",*nextFreeChunk_d);
   break;}
  
 else  // the current chunk has no element
  {
     //printf("parent chunksize = %d\n",parent->nextFreeValue);
     // The thread that pops the last element in the chunk should change the things.
     if(currentChunk->prev != NULL)
    {
     if(atomicCAS(uvw_d,0,1)==0)  // only one thread goes inside
     {
      printf("want new chunk for POP_BACK\n");
      // set the *prev of the current chunk and *next of the parent chunk to NULL 
      //GPUChunk<CHUNK_SIZE,T> * parent = currentChunk->prev;
      parent = currentChunk->prev;
      currentChunk->prev = NULL; // isolating that node since a chunk can only be a part of one vector.
      //currentChunk->next = NULL; // already null. that is why we stopped at this node. Otherwise we would have marched ahead in the linked list.
      parent->next = NULL; // isolating currentChunk;
     // currentChunk = parent;	
      // *uvw_d = 0; // to test

    //  if(currentChunk == &chunks[*nextFreeChunk_d - 1]) // the current chunk is the last exposed chunk from the arena, so we can reclaim it.
   //	reclaim_chunk();// call the routine "reclaim_chunk()"
       *uvw_d = 0;
     }
     while(*uvw_d == 1); // barrier for all threads.
     //currentChunk = parent;	
    } 
    //  FIXME: currentChunk not updated properly. 
       
      // set the *prev of the current chunk and *next of the parent chunk to NULL 
      // and set the current chunk to the previous chunk.
      // regarding the counter nextFreeChunk_d, reduce it only if the current  value is equal to the current chunk(i.e. the chunk being popped from happens to be the last exposed chunk in the arena). Otherwise not.
         
   else if(currentChunk->prev == NULL && parent != NULL) //  the thread is pointing to a node that is not a part of any vector
   {
    // set the current chunk correctly
    // look for a valid chunk of the vector again
    currentChunk = (GPUChunk<CHUNK_SIZE,T>*) vec; // starting over again since we are in the middle of nowhere.
   while(currentChunk->next != NULL) // getting to the last chunk of the vec
   {
     currentChunk = currentChunk->next;
   }
    //currentChunk = parent;
    
   }

//  if((currentChunk->prev == NULL) && (currentChunk->next == NULL) && (currentChunk ==  (GPUChunk<CHUNK_SIZE,T>*) vec)) // current chunk is the head chunk of the vector and is empty
//   {
//   // printf("hello\n");
//   // break;
//   }
  
//   else if(currentChunk->prev == NULL && parent == NULL)
//          break;

  }  

 }
}

#endif 

};

#endif

