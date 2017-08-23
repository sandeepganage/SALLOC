#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
__global__ void mallocTest()
{
    int* ptr = (int*)malloc(sizeof(int));
    free(ptr);
}

int main()
{
    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    GpuTimer timer;
    timer.Start();
    mallocTest<<<8, 32>>>();
    timer.Stop();
    cudaDeviceSynchronize();
     printf("Kernel code ran in %f msecs.\n", timer.Elapsed());
    return 0; 
}

