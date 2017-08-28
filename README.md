
# SALLOC: Smart allocators for SIMT architectures 

**SALLOC** is a thread-safe [*arena allocator*](https://en.wikipedia.org/wiki/Region-based_memory_management) (memory manager) in CUDA.

*SALLOC* has been developed as part of the project "Smart Data Structures in CUDA" with _CERN-HSF_ for _GSoC 2017_.

## Features

- Thread-safe arena allocator for GPUs in CUDA.
- Supports thread-safe vector container in CUDA.
- Support for multiple vectors on the arena. These vectors are shared across threads.
- The vector container supports the following operations:
    1. push_back()
    2. pop_back()
    3. getIndex()
    4. vecSize() 

## Documentation

A detailed documentation on *SALLOC* can be found in [*SALLOC_description.txt*](SALLOC_description.txt).

## Requirements

### Linux

- CUDA version >= 8.0
- gcc version >= 5.3.0

## Getting Started

### Getting the code

- Clone the code into a directory called SALLOC
    ```
    git clone --branch=master https://github.com/felicepantaleo/SALLOC
    ```
- The source for arena allocator SALLOC is present in the file *salloc.h*

- No installation or build is required. 

### Using SALLOC in your code

- Include "salloc.h" (with the proper path) in the CUDA code and use.
- *driver.cu* contains a sample program that uses SALLOC.
    - Compile *driver.cu* and execute
        ```    
        nvcc driver.cu -std=c++11 -o driver
    
        ./driver
        ```
- SALLOC has been tested on NVIDIA Pascal (GeForce GTX 1080) GPU.

## TODOs

- Support **resize()** operation on *vector* container allocated on the arena.
- Support reclaiming of free chunks in the arena to better utilize the space on the arena.

## Developers

- Felice Pantaleo (felice.pantaleo (at) cern.ch)
- Somesh Singh (somesh.singh1992 (at) gmail.com)

