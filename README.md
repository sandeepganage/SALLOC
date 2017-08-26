
# SALLOC: Smart allocators for SIMT architectures 

**SALLOC** is developed as part of the project "Smart Data Structures in CUDA" with _CERN-HSF_ for _GSoC 2017_.

## Features

- Thread-safe arena allocator for GPUs in CUDA.
- Supports thread-safe vector container in CUDA.
- Support for multiple vectors on the arena. These vectors are shared across threads.
- The vector container supports the following operations:
    1. push_back()
    2. pop_back()
    3. getIndex()
    4. vecSize() 

## Requirements

### Linux

- CUDA version >= 8.0
- gcc version >= 5.3.0

## Getting Started

### Getting the code

- clone the code into a directory called SALLOC
```
git clone --branch=master https://github.com/felicepantaleo/SALLOC
```
- The source for arena allocator SALLOC is present in the file "salloc.h"

- No installation or build is required. 

- Include "salloc.h" (with the proper path) in the CUDA code and use.

## Using SALLOC in your code

- SALLOC is tested on NVIDIA Pascal (GeForce GTX 1080) GPU.
- *driver.cu* contains a sample program that uses SALLOC.
    - compile *driver.cu* and execute
```    
nvcc driver.cu -std=c++11 -o driver

./driver
```

## Developers

- Felice Pantaleo (felice.pantaleo (at) cern.ch)
- Somesh Singh (somesh.singh (at) gmail.com)


 



There is a driver code named "driver.cu" which can be used to test the operations on the vector in the allocator.

The file 'driver.cu' can be compiled like so:
	nvcc driver.cu -std=c++11 -o driver
After compiling, it should be run with the command: 
	./driver	


Another code named "cuMalloc.cu" is present in the folder. This file contains code for allocating memory using 'malloc()' function, supported by CUDA 8, from inside the GPU kernel.  The performance of SALLOC is compared with malloc() using this code. 
'cuMalloc.cu' can be compiled like so:
	nvcc cuMalloc.cu -o cuMalloc
 and run with the command: 
	./cuMalloc
 
Currently, arena allocator SALLOC supports allocation of multiple vectors on the arena. 
These vectors are shared  across threads (ie. visible to all threads).
 
The vector container supports the following thread-safe operations:
1. push_back()
2. pop_back()
3. getIndex()
4. vecSize()

The details about these are mentioned in SALLOC-description.txt in this folder.
