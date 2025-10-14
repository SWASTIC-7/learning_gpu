#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Must be power of 2 for this simple version

// Parallel Reduction: Sum all elements of an array
// Classic problem: N elements → 1 result
// Challenge: How to combine results from many threads?
//
// Strategy: Tree-based reduction
// Step 1: N/2 threads add pairs → N/2 results
// Step 2: N/4 threads add pairs → N/4 results
// ...until 1 result remains

__global__ void reduceSum(float *input, float *output, int n) {
    // Shared memory: visible to all threads in this block
    // Much faster than global memory (30 cycles vs 400)
    __shared__ float sdata[N];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 1: Load data from global to shared memory
    // Coalesced access: consecutive threads → consecutive addresses
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();  // Wait for all threads to load data
    
    // Step 2: Tree reduction in shared memory
    // Iteration 1: threads 0-511 add pairs → 512 values
    // Iteration 2: threads 0-255 add pairs → 256 values
    // ...
    // Iteration 10: thread 0 adds last pair → 1 value
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Ensure all additions complete before next iteration
    }
    
    // Step 3: Thread 0 writes block's result to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
    
    // Architecture insight:
    // - Only first 's' threads are active each iteration → warp divergence!
    // - Better approach: Use warp-level primitives (__shfl_down_sync)
    // - This version is simple but not optimal
}

int main() {
    const int numBlocks = 4;
    const int threadsPerBlock = 256;
    const int totalThreads = numBlocks * threadsPerBlock;
    
    float *h_input = new float[totalThreads];
    float *h_output = new float[numBlocks];
    
    // Initialize: array of ones → sum should be totalThreads
    for (int i = 0; i < totalThreads; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalThreads * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, totalThreads * sizeof(float), cudaMemcpyHostToDevice);
    
    // First reduction: N elements → numBlocks partial sums
    reduceSum<<<numBlocks, threadsPerBlock>>>(d_input, d_output, totalThreads);
    
    // Second reduction: numBlocks → 1 (could do on CPU for small numBlocks)
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float finalSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        finalSum += h_output[i];
    }
    
    printf("Sum of %d elements: %.1f (expected: %d)\n", 
           totalThreads, finalSum, totalThreads);
    
    // Performance analysis:
    // - Shared memory: ~30 cycle latency
    // - Each thread does log2(blockDim) additions
    // - Total operations: N/2 + N/4 + ... + 1 = N-1 (same as sequential!)
    // - Speedup comes from parallelism, not reduced work
    //
    // Bottlenecks:
    // 1. __syncthreads() stalls all threads
    // 2. Warp divergence (half threads idle each iteration)
    // 3. Bank conflicts in shared memory (can be avoided with careful indexing)
    
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}

// Optimization roadmap:
// → Use warp shuffle instead of shared memory for last 32 elements
// → Sequential addressing to avoid bank conflicts
// → Multiple elements per thread to reduce kernel launch overhead
// → See NVIDIA's reduction sample for production code
