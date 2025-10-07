#include <iostream>
#include <cuda_runtime.h>

// This is the kernel function that runs on the GPU
__global__ void helloFromGPU() {
    printf("Hello World from Thread %d, Block %d, BlockDim %d\n", 
            threadIdx.x, blockIdx.x, blockDim.x);
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    // Launch 2 blocks with 4 threads each
    helloFromGPU<<<2, 4>>>();

    cudaDeviceSynchronize();   //this is used to wait for gpu to complete its execution

    return 0;
}


// Output:
// Hello World from CPU!
// Hello World from Thread 0, Block 1, BlockDim 4
// Hello World from Thread 1, Block 1, BlockDim 4
// Hello World from Thread 2, Block 1, BlockDim 4
// Hello World from Thread 3, Block 1, BlockDim 4
// Hello World from Thread 0, Block 0, BlockDim 4
// Hello World from Thread 1, Block 0, BlockDim 4
// Hello World from Thread 2, Block 0, BlockDim 4
// Hello World from Thread 3, Block 0, BlockDim 4