#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Small size for clarity (4x4 matrices)

// Matrix multiplication: C = A × B
// C[i,j] = Σ(A[i,k] * B[k,j]) for k=0 to N-1
//
// Parallelization strategy: Each thread computes ONE element of C
// Thread (row, col) computes C[row, col]

__global__ void matMulSimple(const float* A, const float* B, float* C, int n) {
    // 2D thread indexing (within block)
    // For <<<1, dim3(4,4)>>>: threadIdx.x ∈ [0,3], threadIdx.y ∈ [0,3]
    int row = threadIdx.y;  // Which row of C to compute
    int col = threadIdx.x;  // Which column of C to compute
    
    // Compute dot product of A's row with B's column
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        // Row-major layout: element [i,j] is at index i*n + j
        // A[row,k]: traverse row horizontally
        // B[k,col]: traverse column vertically
        sum += A[row * n + k] * B[k * n + col];
    }
    
    C[row * n + col] = sum;
    
    // Memory access pattern analysis:
    // - A: Coalesced (threads in same row access consecutive elements)
    // - B: Strided (threads in same col access elements n apart)
    // - C: Coalesced (each thread writes once to unique location)
    //
    // Problem: Each element of B is loaded n times (once per thread in that column)
    // Solution: Use shared memory (see 5_shared_memory.cu)
}

int main() {
    float h_A[N*N], h_B[N*N], h_C[N*N];
    
    // Initialize test matrices
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = i + 1;         // A = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        h_B[i] = (i % N) + 1;   // B = [1,2,3,4] repeated 4 times
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch 4x4 threads = 16 threads total
    // dim3(x, y, z) creates 3D dimensions
    dim3 threads(N, N);  // 4 threads in x, 4 in y = 4×4 grid
    
    // <<<1, threads>>> = 1 block with N×N threads
    // Each thread computes one element: C[threadIdx.y][threadIdx.x]
    matMulSimple<<<1, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result matrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%6.1f ", h_C[i*N + j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    // Limitations of this naive approach:
    // 1. Only works for N ≤ 32 (max threads per block dimension)
    // 2. Each element of B loaded from global memory N times
    // 3. No shared memory usage
    //
    // For larger matrices: need multiple blocks + shared memory
    // See: 5_shared_memory.cu for optimized version
    
    return 0;
}