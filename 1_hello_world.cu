#include <iostream>
#include <cuda_runtime.h>

// __global__ = kernel function, runs on GPU, called from CPU
// Each thread executes this function independently
__global__ void helloFromGPU() {
    // threadIdx.x: Thread's index within its block (0 to blockDim.x-1)
    // blockIdx.x: Block's index within the grid (0 to gridDim.x-1)  
    // blockDim.x: Total threads per block (set in <<<>>> launch)
    
    printf("Hello World from Thread %d, Block %d, BlockDim %d\n", 
            threadIdx.x, blockIdx.x, blockDim.x);
    
    // Total thread ID = blockIdx.x * blockDim.x + threadIdx.x
    // Example: Block 1, Thread 2, blockDim=4 → ID = 1*4+2 = 6
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    // Launch configuration: <<<blocks, threads_per_block>>>
    // 2 blocks × 4 threads = 8 total threads
    // Each thread executes helloFromGPU() independently
    helloFromGPU<<<2, 4>>>();

    // CRITICAL: GPU execution is asynchronous!
    // Without this, main() might exit before GPU finishes
    cudaDeviceSynchronize();   // CPU waits for all GPU threads to complete

    return 0;
}

// Output Analysis:
// - Order is NOT guaranteed (threads execute in parallel)
// - Block 1 might finish before Block 0 (blocks run independently)
// - Within a block, threads may print in any order
// 
// Why 8 separate printf outputs?
/// → 2 blocks × 4 threads = 8 independent executions
//
// Architecture insight:
// - If GPU has 2 SMs, each block runs on one SM
// - Each SM divides 4 threads into 1 warp (needs 32 threads for full warp)
// - Underutilized! For real work, use 256+ threads per block

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