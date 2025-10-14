#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// BLOG POST 6: "Taming the GPU: Optimization and Bottleneck Hunting"
// ============================================================================
//
// THE PERFORMANCE PYRAMID:
// =======================
//
// Level 1: CORRECTNESS (Get it working)
// Level 2: BASIC OPTIMIZATION (Obvious wins)
// Level 3: PROFILING (Find the real bottleneck)
// Level 4: ADVANCED OPTIMIZATION (Squeeze out last 10%)
//
// Rule: "Premature optimization is the root of all evil" - Donald Knuth
// → Profile first, optimize second!

// ============================================================================
// OPTIMIZATION 1: MEMORY COALESCING
// ============================================================================
//
// COALESCING: When consecutive threads access consecutive memory addresses
// GPU can combine multiple requests into one transaction = FAST
//
// BAD (Strided Access):
// Thread 0 → a[0], Thread 1 → a[1024], Thread 2 → a[2048]
// → Each thread triggers separate memory transaction
//
// GOOD (Coalesced Access):
// Thread 0 → a[0], Thread 1 → a[1], Thread 2 → a[2]
// → All 32 threads (warp) serviced in one transaction
//
// RULE: Make sure threadIdx.x corresponds to consecutive memory

// Example: Matrix Transpose (demonstrates coalescing issues)

#define TILE_DIM 32
#define BLOCK_ROWS 8

// NAIVE VERSION: Causes bank conflicts
__global__ void transposeNaive(float *out, const float *in, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < width && y < height) {
        // Read coalesced, write strided (BAD for output)
        out[x * height + y] = in[y * width + x];
    }
    
    // Problem: Writing to out[x * height + y] is strided
    // Thread 0 writes to out[0], Thread 1 writes to out[height]
    // → height stride between consecutive threads
}

// OPTIMIZED VERSION: Use shared memory to avoid bank conflicts
__global__ void transposeOptimized(float *out, const float *in, int width, int height) {
    // Shared memory tile: +1 to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Read from global memory (coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    
    // Transpose coordinates for output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write to global memory (now coalesced!)
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
    
    // KEY INSIGHT: +1 in tile dimension avoids bank conflicts
    // Without +1: tile[0][0], tile[1][0], tile[2][0] → same bank
    // With +1: Different threads access different banks
}

// ============================================================================
// OPTIMIZATION 2: WARP DIVERGENCE REDUCTION
// ============================================================================
//
// DIVERGENCE: When threads in a warp take different execution paths
// GPU must execute BOTH paths sequentially = 2× slower
//
// DETECTING DIVERGENCE:
// - if-else where condition varies within warp
// - Early loop exits that differ by thread
// - Different-length loops per thread

// Example: Parallel reduction with divergence

// BAD: Divergence on every iteration
__global__ void reduceBad(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // DIVERGENT: Threads split into active/inactive
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // Only some threads active
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// GOOD: Minimized divergence
__global__ void reduceGood(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // NON-DIVERGENT: First threads always active together
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {  // First 's' threads active
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// BEST: Use warp-level primitives (no divergence!)
__global__ void reduceWarpShuffle(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (i < n) ? input[i] : 0.0f;
    
    // Warp-level reduction (last 32 elements)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Only one thread per warp writes
    if (tid % 32 == 0) {
        __shared__ float warp_results[8];  // Assuming 256 threads = 8 warps
        warp_results[tid / 32] = val;
        __syncthreads();
        
        // Final reduction across warps
        if (tid < 8) {
            val = warp_results[tid];
            for (int offset = 4; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xff, val, offset);
            }
            if (tid == 0) output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// OPTIMIZATION 3: OCCUPANCY TUNING
// ============================================================================
//
// OCCUPANCY: Percentage of maximum threads that can run on SM
// Higher occupancy → better latency hiding → better performance
//
// FACTORS LIMITING OCCUPANCY:
// 1. Threads per block (too few = low occupancy)
// 2. Registers per thread (too many = fewer blocks fit)
// 3. Shared memory per block (too much = fewer blocks fit)
// 4. Max blocks per SM (hardware limit)
//
// TOOL: CUDA Occupancy Calculator
// nvcc --ptxas-options=-v shows register usage
//
// EXAMPLE: Register pressure

// BAD: Uses too many registers
__global__ void registerHeavy(float *data, int n) {
    // Many local variables = many registers
    float temp1, temp2, temp3, temp4, temp5;
    float temp6, temp7, temp8, temp9, temp10;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Complex computation
        temp1 = data[idx] * 1.1f;
        temp2 = data[idx] * 2.2f;
        // ... many operations ...
        data[idx] = temp1 + temp2;
    }
    // Compiler may spill to local memory (slow!)
}

// GOOD: Reuse registers
__global__ void registerEfficient(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = data[idx];
        temp = temp * 1.1f;  // Reuse 'temp'
        temp = temp + temp * 2.2f;
        data[idx] = temp;
    }
}

// ============================================================================
// OPTIMIZATION 4: SHARED MEMORY BANK CONFLICTS
// ============================================================================
//
// SHARED MEMORY BANKS: 32 banks, 4-byte width
// If multiple threads access same bank → serialized (conflict)
//
// NO CONFLICT: Different threads access different banks
// CONFLICT: Thread 0→bank 0, Thread 1→bank 0 (2-way conflict = 2× slower)

__global__ void bankConflictDemo() {
    __shared__ float shared[32][32];  // BAD: Causes conflicts
    __shared__ float shared_padded[32][33];  // GOOD: Padding avoids conflicts
    
    int tid = threadIdx.x;
    
    // CONFLICT: Consecutive threads access same bank
    float val = shared[0][tid];  // All threads access row 0, different columns
    // thread 0 → bank 0, thread 1 → bank 1, ... (NO CONFLICT)
    
    // CONFLICT: Strided access
    val = shared[tid][0];  // All threads access column 0, different rows
    // thread 0 → bank 0, thread 1 → bank 0, ... (32-WAY CONFLICT!)
    
    // FIX: Use padding to shift banks
    val = shared_padded[tid][0];  // Now different banks due to +1 padding
}

// ============================================================================
// PROFILING DEMO: MATRIX MULTIPLICATION VARIANTS
// ============================================================================

// Version 1: Naive (baseline)
__global__ void matmulNaive(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Version 2: Shared memory (from 5_shared_memory.cu)
__global__ void matmulShared(float *C, float *A, float *B, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + 15) / 16; t++) {
        if (row < N && (t * 16 + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && (t * 16 + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 16; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// ============================================================================
// MAIN: PROFILING COMPARISON
// ============================================================================

int main() {
    printf("=== GPU Optimization & Profiling Demo ===\n\n");
    
    // DEMO 1: Memory Coalescing (Transpose)
    printf("DEMO 1: Memory Coalescing\n");
    printf("---------------------------\n");
    
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    float *h_matrix = (float*)malloc(bytes);
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_matrix[i] = (float)i;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_matrix, bytes, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((WIDTH + TILE_DIM - 1) / TILE_DIM, 
                 (HEIGHT + TILE_DIM - 1) / TILE_DIM);
    
    // Benchmark naive version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        transposeNaive<<<gridDim, blockDim>>>(d_out, d_in, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    
    // Benchmark optimized version
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        transposeOptimized<<<gridDim, blockDim>>>(d_out, d_in, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_optimized;
    cudaEventElapsedTime(&ms_optimized, start, stop);
    
    printf("Naive transpose:     %.2f ms (avg)\n", ms_naive / 100);
    printf("Optimized transpose: %.2f ms (avg)\n", ms_optimized / 100);
    printf("Speedup:             %.2fx\n\n", ms_naive / ms_optimized);
    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_matrix);
    
    // DEMO 2: Warp Divergence (Reduction)
    printf("DEMO 2: Warp Divergence\n");
    printf("------------------------\n");
    
    const int N = 1 << 20;  // 1M elements
    bytes = N * sizeof(float);
    
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, 1024 * sizeof(float));  // Max blocks
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    
    // Benchmark bad reduction
    cudaEventRecord(start);
    reduceBad<<<blocks, threads>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);
    
    // Benchmark good reduction
    cudaEventRecord(start);
    reduceGood<<<blocks, threads>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_optimized, start, stop);
    
    printf("Divergent reduction:     %.3f ms\n", ms_naive);
    printf("Non-divergent reduction: %.3f ms\n", ms_optimized);
    printf("Speedup:                 %.2fx\n\n", ms_naive / ms_optimized);
    
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    
    // DEMO 3: Matrix Multiplication
    printf("DEMO 3: Shared Memory (Matrix Multiplication)\n");
    printf("----------------------------------------------\n");
    
    const int MAT_SIZE = 1024;
    bytes = MAT_SIZE * MAT_SIZE * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    dim3 matBlock(16, 16);
    dim3 matGrid((MAT_SIZE + 15) / 16, (MAT_SIZE + 15) / 16);
    
    // Naive version
    cudaEventRecord(start);
    matmulNaive<<<matGrid, matBlock>>>(d_C, d_A, d_B, MAT_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);
    
    // Shared memory version
    cudaEventRecord(start);
    matmulShared<<<matGrid, matBlock>>>(d_C, d_A, d_B, MAT_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_optimized, start, stop);
    
    printf("Naive matmul:  %.2f ms (%.2f GFLOPS)\n", 
           ms_naive, (2.0 * MAT_SIZE * MAT_SIZE * MAT_SIZE) / (ms_naive * 1e6));
    printf("Shared matmul: %.2f ms (%.2f GFLOPS)\n", 
           ms_optimized, (2.0 * MAT_SIZE * MAT_SIZE * MAT_SIZE) / (ms_optimized * 1e6));
    printf("Speedup:       %.2fx\n\n", ms_naive / ms_optimized);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // PROFILING INSTRUCTIONS
    printf("=== PROFILING TOOLS ===\n");
    printf("----------------------\n\n");
    
    printf("1. NVPROF (Legacy, CUDA 10 and earlier):\n");
    printf("   nvprof ./program\n");
    printf("   nvprof --metrics achieved_occupancy ./program\n\n");
    
    printf("2. NSIGHT SYSTEMS (Timeline profiling):\n");
    printf("   nsys profile -o report ./program\n");
    printf("   nsys-ui report.qdrep  # View GUI\n\n");
    
    printf("3. NSIGHT COMPUTE (Kernel profiling):\n");
    printf("   ncu --set full -o report ./program\n");
    printf("   ncu-ui report.ncu-rep  # View GUI\n\n");
    
    printf("4. COMPILE WITH PROFILING INFO:\n");
    printf("   nvcc --ptxas-options=-v program.cu\n");
    printf("   → Shows register usage, shared memory usage\n\n");
    
    printf("=== OPTIMIZATION CHECKLIST ===\n");
    printf("------------------------------\n");
    printf("☐ 1. Profile first (find actual bottleneck)\n");
    printf("☐ 2. Check memory access patterns (coalescing)\n");
    printf("☐ 3. Minimize warp divergence\n");
    printf("☐ 4. Use shared memory for reused data\n");
    printf("☐ 5. Avoid shared memory bank conflicts\n");
    printf("☐ 6. Maximize occupancy (balance registers/shared mem)\n");
    printf("☐ 7. Minimize CPU↔GPU transfers\n");
    printf("☐ 8. Use CUDA libraries when possible (cuBLAS, Thrust)\n");
    printf("☐ 9. Profile again (verify improvement)\n\n");
    
    printf("=== KEY METRICS TO WATCH ===\n");
    printf("----------------------------\n");
    printf("• Achieved Occupancy (target: >50%%)\n");
    printf("• Memory Throughput (GB/s)\n");
    printf("• Warp Execution Efficiency (target: >90%%)\n");
    printf("• Global Memory Load Efficiency\n");
    printf("• Shared Memory Bank Conflicts\n");
    printf("• Register usage per thread\n\n");
    
    return 0;
}

// ============================================================================
// ADVANCED OPTIMIZATION TECHNIQUES
// ============================================================================
//
// 1. INSTRUCTION-LEVEL PARALLELISM (ILP):
//    Process multiple elements per thread to hide latency
//    Example: Load 4 elements, compute all 4, store all 4
//
// 2. LOOP UNROLLING:
//    #pragma unroll
//    Reduces loop overhead, increases ILP
//
// 3. CONSTANT MEMORY:
//    __constant__ for read-only data accessed by all threads
//    Cached per-SM, broadcasts to all threads in warp
//
// 4. TEXTURE MEMORY:
//    2D spatial locality, automatic interpolation
//    Good for image processing
//
// 5. PINNED (PAGE-LOCKED) MEMORY:
//    cudaMallocHost() for faster CPU↔GPU transfers
//    But limited resource (use sparingly)
//
// 6. STREAMS & ASYNC OPERATIONS:
//    Overlap computation with memory transfers
//    Multiple kernels running concurrently
//
// 7. MULTI-GPU:
//    cudaSetDevice() to switch between GPUs
//    NCCL library for GPU-to-GPU communication
//
// 8. KERNEL FUSION:
//    Combine multiple small kernels into one
//    Reduces kernel launch overhead
//
// 9. ATOMIC OPERATIONS:
//    atomicAdd() etc. for thread-safe updates
//    But causes serialization (use sparingly)
//
// 10. COOPERATIVE GROUPS:
//     Flexible thread synchronization beyond blocks
//     Grid-wide sync, warp-level primitives
