#include <stdio.h>
#include <cuda_runtime.h>

#define N 100

// __global__ = runs on GPU, called from CPU
// __device__ = runs on GPU, called from GPU
// __host__ = runs on CPU, called from CPU (default)

__global__ void fun(int *a){
    // Each thread adds 10 to one array element
    // threadIdx.x ranges from 0 to N-1 (since we launch N threads)
    // 
    // Memory access pattern: COALESCED
    // Thread 0 accesses a[0], Thread 1 → a[1], etc.
    // Consecutive threads → consecutive memory = fast!
    a[threadIdx.x] += 10;
}

int main(){
    // Step 1: Allocate CPU (host) memory
    int a[N];
    int *d_a;  // 'd_' prefix = device pointer (convention)
    
    for(int i=0;i<N;i++) a[i] = i;  // a = [0,1,2,...,99]

    // Step 2: Allocate GPU (device) memory
    // cudaMalloc allocates GLOBAL memory on GPU
    // Global memory: ~400-800 cycle latency, 4-24 GB size
    cudaMalloc((void**)&d_a, N*sizeof(int));
    
    // Step 3: Copy data CPU → GPU (via PCIe bus)
    // This is SLOW (~10 GB/s vs ~900 GB/s GPU memory bandwidth)
    // Minimize these transfers!
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Step 4: Launch kernel
    // <<<1, N>>> = 1 block, N threads
    // All N threads run in parallel (grouped into ⌈N/32⌉ warps)
    fun<<<1, N>>>(d_a); 
    
    // Step 5: Copy result GPU → CPU
    cudaMemcpy(a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Step 6: Free GPU memory (prevents memory leak)
    cudaFree(d_a);

    // Print result: should be [10,11,12,...,109]
    for(int i=0;i<N;i++) printf("%d ", a[i]);
    printf("\n");
    
    // Memory Hierarchy Used:
    // 1. Registers: threadIdx.x stored in register
    // 2. Global Memory: array 'a' accessed (slowest)
    // 3. L2 Cache: automatic caching helps repeated access
    //
    // Optimization opportunity:
    // → Each thread only accesses memory once (good!)
    // → Could use shared memory if threads needed to communicate
    
    return 0;
}

// Performance Notes:
// - cudaMemcpy time >> kernel execution time for small N
// - Rule of thumb: Transfer once, compute much
// - For N=100: kernel ~1 microsecond, memcpy ~10 microseconds