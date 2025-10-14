#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Threads per block dimension (16×16 = 256 threads)

// Optimized Matrix Multiplication using Shared Memory
// Key insight: Reuse data from global memory by caching in shared memory
//
// Problem with naive version (3_matrix.cu):
// - Each element of B is loaded N times from global memory (slow!)
// 
// Solution: Tiled algorithm
// - Divide matrices into TILE_SIZE × TILE_SIZE tiles
// - Load one tile into shared memory, reuse for all computations
// - Reduces global memory accesses by factor of TILE_SIZE

__global__ void matMulTiled(const float* A, const float* B, float* C, int n) {
    // Shared memory tiles: visible to all threads in block
    // Size: TILE_SIZE × TILE_SIZE floats
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread's position in output matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles of A and B required to compute C[row,col]
    // Number of tiles = n / TILE_SIZE
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Step 1: Collaboratively load tile into shared memory
        // Each thread loads one element
        
        // Load tile of A (horizontal strip)
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (aRow < n && aCol < n) ? A[aRow * n + aCol] : 0.0f;
        
        // Load tile of B (vertical strip)
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && bCol < n) ? B[bRow * n + bCol] : 0.0f;
        
        // Step 2: Wait for all threads to finish loading
        __syncthreads();
        
        // Step 3: Compute partial dot product using shared memory
        // Now reads come from shared memory (30 cycles vs 400!)
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Step 4: Wait before loading next tile (prevent race condition)
        __syncthreads();
    }
    
    // Write final result
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
    
    // Memory access analysis:
    // - Global memory reads per element: 2 * n / TILE_SIZE (vs n in naive)
    // - Shared memory reads per element: 2 * n
    // - Speedup: ~TILE_SIZE × faster (e.g., 16× for TILE_SIZE=16)
    //
    // Architecture considerations:
    // - Shared memory: 48-96 KB per SM
    // - Our usage: 2 × TILE_SIZE² × 4 bytes = 2KB (plenty available!)
    // - Bank conflicts: Avoided (different threads access different banks)
}

int main() {
    const int N = 512;  // Matrix size (can be much larger now!)
    size_t bytes = N * N * sizeof(float);
    
    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];
    
    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Launch configuration: grid of blocks
    // Each block computes TILE_SIZE × TILE_SIZE elements
    dim3 threads(TILE_SIZE, TILE_SIZE);  // 16×16 = 256 threads per block
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                (N + TILE_SIZE - 1) / TILE_SIZE);  // 32×32 = 1024 blocks
    
    printf("Computing %d×%d matrix multiplication...\n", N, N);
    printf("Grid: %d×%d blocks, Block: %d×%d threads\n", 
           blocks.x, blocks.y, threads.x, threads.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", 
           (2.0 * N * N * N) / (milliseconds * 1e6));
    
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result (sample check)
    printf("Sample C[0,0] = %.4f\n", h_C[0]);
    
    // Further optimizations possible:
    // 1. Increase TILE_SIZE to 32 (full warp width)
    // 2. Process multiple elements per thread
    // 3. Use Tensor Cores on modern GPUs (for FP16/INT8)
    // 4. Use cuBLAS library (production code should use this!)
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// Performance comparison (RTX 3080, N=512):
// - Naive (3_matrix.cu): ~800 ms (extrapolated)
// - Tiled (this): ~50 ms
// - cuBLAS (NVIDIA): ~1 ms (highly optimized!)
