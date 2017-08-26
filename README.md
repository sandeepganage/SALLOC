
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

A detailed description about these can be found in *SALLOC-description.txt*

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
- The source for arena allocator SALLOC is present in the file *salloc.h*

- No installation or build is required. 

### Using SALLOC in your code

- Include "salloc.h" (with the proper path) in the CUDA code and use.
- *driver.cu* contains a sample program that uses SALLOC.
    - compile *driver.cu* and execute
```    
nvcc driver.cu -std=c++11 -o driver

./driver
```
- SALLOC has been tested on NVIDIA Pascal (GeForce GTX 1080) GPU.

## Developers

- Felice Pantaleo (felice.pantaleo (at) cern.ch)
- Somesh Singh (somesh.singh1992 (at) gmail.com)

