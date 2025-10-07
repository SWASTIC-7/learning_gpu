// matmul_cuda.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>



const int TILE_WIDTH = 16; // common tile size. 16x16 threads per block.

// ---------------------- Naive kernel ---------------------------------
// Each thread computes one element C[row, col] by iterating over K.
__global__ void matMulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) // A: MxK, B: KxN, C: MxN
{
    // map a 2D grid/block to a matrix coordinate
    int row = blockIdx.y * blockDim.y + threadIdx.y; // global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // global column index

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col]; // row-major indexing
        C[row * N + col] = sum;
    }
}

// ---------------------- Tiled (shared-memory) kernel -----------------
// Uses tiles of A and B loaded into shared memory to reuse global loads.
__global__ void matMulTiled(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    // 2D thread/block
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Compute the global row & column this thread will write
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Accumulator
    float Cvalue = 0.0f;

    // Number of tiles along K dimension
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // Shared memory tile for A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int t = 0; t < numTiles; ++t) {
        // Global coordinates of the elements to load into the tile
        int Arow = row;
        int Acol = t * TILE_WIDTH + tx; // different threads have different tx -> contiguous A cols
        int Brow = t * TILE_WIDTH + ty; // different threads have different ty -> contiguous B rows
        int Bcol = col;

        // Load with bound checks
        As[ty][tx] = (Arow < M && Acol < K) ? A[Arow * K + Acol] : 0.0f;
        Bs[ty][tx] = (Brow < K && Bcol < N) ? B[Brow * N + Bcol] : 0.0f;

        // Wait until all threads finished loading the tile
        __syncthreads();

        // Multiply the two tiles (inner product for tile columns/rows)
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // Wait before the next tile overwrites As/Bs
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// ---------------------- Host helpers ---------------------------------
void cpuMatMul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
               int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

bool compareResults(const std::vector<float>& A, const std::vector<float>& B, int size, float eps=1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(A[i] - B[i]) > eps) {
            std::cerr << "Mismatch at " << i << ": CPU=" << A[i] << " GPU=" << B[i] << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    // dimensions: C(MxN) = A(MxK) * B(KxN)
    int M = 512, K = 512, N = 512; // default sizes (adjustable)
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }

    std::cout << "Matrix sizes: A(" << M << "x" << K << "), B(" << K << "x" << N << "), C(" << M << "x" << N << ")\n";
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    std::vector<float> h_A(sizeA), h_B(sizeB), h_C_cpu(sizeC), h_C_gpu(sizeC);

    // Initialize with deterministic values for reproducibility
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = float((i % 13) - 6) * 0.001f + 1.0f;
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = float((i % 7) - 3) * 0.002f + 2.0f;

    // CPU reference (for verification) - WARNING: O(N^3) time; pick moderate sizes
    std::cout << "Computing CPU reference...\n";
    cpuMatMul(h_A, h_B, h_C_cpu, M, N, K);

    // Device allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    // Launch tiled kernel with 2D blocks and grids
    dim3 block(TILE_WIDTH, TILE_WIDTH); // threads per block (tx,ty)
    dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y ); // blocks to cover matrix C

    std::cout << "Launching tiled kernel with block " << block.x << "x" << block.y
              << " and grid " << grid.x << "x" << grid.y << "\n";

    // runtime timing using events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Device kernel time: " << ms << " ms\n";

    // verify
    bool ok = compareResults(h_C_cpu, h_C_gpu, (int)sizeC);
    std::cout << "Verification: " << (ok ? "PASS" : "FAIL") << "\n";

    // cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ok ? 0 : 1;
}
