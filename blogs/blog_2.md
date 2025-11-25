# Blog 2: Understanding GPU Memory - The Highway Analogy

> *"GPUs are fast not because they do things quickly, but because they do many things at once. But there's a catch: they need the right fuel delivered the right way."*

Welcome back! In [Blog 1](./blog_1.md), we learned about GPU architecture - threads, warps, blocks, and grids. Now let's understand **why some GPU programs are 100× faster than others**, even when doing the same work.

**Spoiler:** It's all about **memory access patterns**.

---

## 🎯 What You'll Learn

By the end of this blog, you'll understand:
- How GPU memory works (host vs device)
- What "coalescing" means and why it matters
- The difference between fast and slow memory patterns
- How to write efficient GPU code

**Prerequisites:** Basic C programming, understanding of arrays, Blog 1 concepts.

---

## Part 1: The Two Worlds - CPU and GPU Memory

### The Problem: GPUs Don't Share Your Computer's RAM

Imagine you're working on homework at your desk (CPU), but all your textbooks are in a library across town (GPU). You can't just reach over and grab a book - you need to:
1. Request the book (send a message)
2. Wait for someone to find it
3. Wait for them to drive it to you
4. Finally use the book

This is **exactly** how CPU and GPU memory works!

```
Your Desk (CPU)          Library (GPU)
┌──────────┐            ┌──────────┐
│  RAM     │ ←──────→   │  VRAM    │
│ 16-32 GB │  PCIe Bus  │  4-24 GB │
└──────────┘  (slow!)   └──────────┘
```

**Key Terms:**
- **Host** = CPU and its RAM
- **Device** = GPU and its VRAM (Video RAM / Global Memory)
- **PCIe Bus** = The "highway" connecting them (~16 GB/s - sounds fast, but GPU needs 900 GB/s!)

---


**Note:** Transfer data once, do lots of computation, then transfer back.

---

### 📝 Example: Adding 10 to Every Number

Let's see the complete flow with `2_memory.cu`:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

#define N 100

// Kernel: Each thread adds 10 to one element
__global__ void addTen(int *arr) {
    int idx = threadIdx.x;  // Thread 0, 1, 2, ..., 99
    arr[idx] += 10;
}

int main() {
    // Step 1: Create data on CPU
    int h_arr[N];  // 'h_' = host
    for (int i = 0; i < N; i++) {
        h_arr[i] = i;  // [0, 1, 2, ..., 99]
    }
    
    // Step 2: Allocate GPU memory
    int *d_arr;  // 'd_' = device
    cudaMalloc(&d_arr, N * sizeof(int));
    
    // Step 3: Copy to GPU
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Step 4: Run kernel (100 threads)
    addTen<<<1, N>>>(d_arr);
    
    // Step 5: Copy back to CPU
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Step 6: Clean up
    cudaFree(d_arr);
    
    // Result: [10, 11, 12, ..., 109]
    printf("First 5 results: %d %d %d %d %d\n", 
           h_arr[0], h_arr[1], h_arr[2], h_arr[3], h_arr[4]);
    
    return 0;
}
```

**What's happening inside the GPU:**
- Thread 0 does: `arr[0] += 10`
- Thread 1 does: `arr[1] += 10`
- ...all at the same time!
- Thread 99 does: `arr[99] += 10`

**Visualization:**
```
Before:  [0] [1] [2] [3] ... [99]
          ↓   ↓   ↓   ↓       ↓
Threads:  0   1   2   3  ...  99  (all running simultaneously)
          ↓   ↓   ↓   ↓       ↓
After:   [10][11][12][13]...[109]
```

---

## Part 2: Memory Coalescing - The Highway Analogy

### The Setup: Why Some Memory Access is Fast

Imagine GPU memory as a **multi-lane highway** with toll booths. Here's the catch:

**The GPU can only request memory in "bus-sized chunks"** - like a toll booth that only serves entire buses, not individual passengers.

**Bus capacity:** 32 consecutive memory locations (one per warp thread)

Now let's see three scenarios...

---

### Scenario 1: Coalesced Access - Everyone Gets On the Same Bus ✅

**The ideal case:** 32 threads want 32 consecutive memory locations.

```cuda
__global__ void coalesced(float *data) {
    int idx = threadIdx.x;  // Thread 0, 1, 2, ..., 31
    float value = data[idx]; // Thread i reads data[i]
}
```

**What happens:**
```
Memory:  [0][1][2][3]...[31][32][33]...
          ↓  ↓  ↓  ↓     ↓
Threads:  0  1  2  3 ... 31  (warp 0)
```

**Result:** GPU loads **ONE bus** (128 bytes) and delivers to all 32 threads. **Efficient!**

**Real-world analogy:** 32 students boarding a school bus parked at their school. One trip!

**Performance:** ~900 GB/s (maximum speed)

---

### Scenario 2: Strided Access - Everyone Skips Seats ⚠️

**The problem:** Threads want every 4th memory location.

```cuda
__global__ void strided(float *data) {
    int idx = threadIdx.x * 4;  // Thread 0 → 0, Thread 1 → 4, etc.
    float value = data[idx];
}
```

**What happens:**
```
Memory:  [0][1][2][3][4][5][6][7][8]...
          ↓        ↓        ↓
Threads:  0        1        2  ...
```

**Result:** GPU loads **FOUR buses** but only uses 1/4 of each. **Wasteful!**

**Real-world analogy:** 32 students scattered across 4 different bus routes. Need 4 bus trips!

**Performance:** ~200-400 GB/s (2-4× slower)

**When this happens:**
- Accessing every Nth element (`arr[i * N]`)
- Column-major access in row-major storage
- Structure of Arrays (SoA) with wrong indexing

---

### Scenario 3: Random Access - Everyone Takes Different Buses ❌

**The nightmare:** Threads want completely random locations.

```cuda
__global__ void random(float *data, int *indices) {
    int idx = threadIdx.x;
    int location = indices[idx];  // Could be anywhere!
    float value = data[location];
}
```

**What happens:**
```
Memory:  [0][1][2][3]...[99][100]...[573]...[1042]...
          ↓              ↓          ↓        ↓
Threads:  5              2          0        1  ...
```

**Result:** GPU might need **32 separate bus trips** (one per thread). **Disaster!**

**Real-world analogy:** 32 students living in random neighborhoods needing individual pickups. 32 trips!

**Performance:** ~50-100 GB/s (8-16× slower)

**When this happens:**
- Graph algorithms (neighbors are scattered)
- Hash tables
- Pointer chasing
- Random indexing

---

### Performance Comparison Table

| Pattern | Buses Needed | Efficiency | Speed | Example |
|---------|-------------|-----------|--------|---------|
| **Coalesced** | 1 | 100% | 900 GB/s | `arr[threadIdx.x]` |
| **Strided (4)** | 4 | 25% | 225 GB/s | `arr[threadIdx.x * 4]` |
| **Strided (32)** | 32 | 3% | 30 GB/s | `arr[threadIdx.x * 32]` |
| **Random** | 1-32 | 3-50% | 50 GB/s | `arr[randomIdx[threadIdx.x]]` |

---

## Part 3: Real Example - Matrix Multiplication

Let's see coalescing in action with `3_matrix.cu`:

### The Problem: Multiply Two Matrices

```
Matrix A (4×4)    Matrix B (4×4)    Matrix C (4×4)
┌──────────┐     ┌──────────┐     ┌──────────┐
│ 1  2  3  4│  ×  │1  2  3  4│  =  │?  ?  ?  ?│
│ 5  6  7  8│     │1  2  3  4│     │?  ?  ?  ?│
│ 9 10 11 12│     │1  2  3  4│     │?  ?  ?  ?│
│13 14 15 16│     │1  2  3  4│     │?  ?  ?  ?│
└──────────┘     └──────────┘     └──────────┘
```

**Formula for C[row, col]:**
```
C[row, col] = A[row, 0] × B[0, col] 
            + A[row, 1] × B[1, col]
            + A[row, 2] × B[2, col]
            + A[row, 3] × B[3, col]
```

---

### Naive Implementation

```cuda
__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    // Each thread computes one element of C
    int row = threadIdx.y;  // Which row (0-3)
    int col = threadIdx.x;  // Which column (0-3)
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        // Read one element from A's row
        // Read one element from B's column
        sum += A[row * N + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

// Launch: 16 threads (4×4 grid)
dim3 threads(4, 4);
matMulNaive<<<1, threads>>>(d_A, d_B, d_C, 4);
```

**Memory Layout in RAM** (row-major order):
```
A: [A00, A01, A02, A03, A10, A11, A12, A13, ...]
B: [B00, B01, B02, B03, B10, B11, B12, B13, ...]
```

---

### Analyzing Memory Access Patterns

**Reading from A:**
```
Thread (0,0) reads: A[0*4+0], A[0*4+1], A[0*4+2], A[0*4+3]
Thread (0,1) reads: A[0*4+0], A[0*4+1], A[0*4+2], A[0*4+3]  ← Same as above!
Thread (0,2) reads: A[0*4+0], A[0*4+1], A[0*4+2], A[0*4+3]  ← Same again!
```

**Analysis:** ✅ **COALESCED** - threads in same row read consecutive A elements.

**Reading from B:**
```
Thread (0,0) reads: B[0*4+0], B[1*4+0], B[2*4+0], B[3*4+0]
                    ↓         ↓         ↓         ↓
Memory indices:     0,        4,        8,        12  ← Stride of 4!

Thread (0,1) reads: B[0*4+1], B[1*4+1], B[2*4+1], B[3*4+1]
                    ↓         ↓         ↓         ↓
Memory indices:     1,        5,        9,        13  ← Stride of 4!
```

**Analysis:** ⚠️ **STRIDED** - threads access B with stride of N (4). Not coalesced!

---

### The Problem Visualized

**Ideal coalesced access** (threads read consecutive memory):
```
Memory:  [B00][B01][B02][B03][B10][B11]...
          ↓    ↓    ↓    ↓    
Threads:  T0   T1   T2   T3   ← ONE bus trip
```

**What actually happens** (threads jump around):
```
Memory:  [B00][B01][B02][B03][B10][B11][B12][B13][B20]...
          ↓                   ↓                   ↓
Threads:  T0                  T0                  T0  ← MULTIPLE trips
```

**Performance impact:**
- Each element of B is loaded from global memory **N times** (once per thread in that row)
- For N=512: Each B element loaded **512 times**!
- Memory bandwidth wasted: ~75% efficiency loss

---

### The Solution Preview: Shared Memory (Blog 3)

The fix is to use **shared memory** - a small, fast cache shared by all threads in a block:

**Strategy:**
1. Load a **tile** of B into shared memory (coalesced access - all threads cooperate)
2. All threads read from shared memory (32× faster!)
3. Reuse the cached data for multiple computations

```cuda
__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float Bs[16][16];  // Shared memory tile
    
    // Step 1: Load tile cooperatively (coalesced!)
    Bs[threadIdx.y][threadIdx.x] = B[...];  // Each thread loads one element
    __syncthreads();  // Wait for everyone
    
    // Step 2: Compute using shared memory (fast!)
    for (int k = 0; k < 16; k++) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];  // From shared memory!
    }
}
```

**Result:** 16× speedup by avoiding repeated global memory access!

---

## Part 4: Practical Guidelines

### ✅ How to Write Coalesced Code

**Rule 1:** Make `threadIdx.x` correspond to the fastest-changing index

```cuda
// GOOD: threadIdx.x → array index
int idx = blockIdx.x * blockDim.x + threadIdx.x;
data[idx] = ...;  // Thread 0 → data[0], Thread 1 → data[1], etc.

// BAD: threadIdx.x → slow-changing dimension
int idx = blockIdx.x * blockDim.x + threadIdx.y;  // Wrong!
data[idx] = ...;
```

**Rule 2:** For 2D arrays, make `threadIdx.x` correspond to columns

```cuda
// GOOD: Row-major, threadIdx.x = column
int idx = row * WIDTH + threadIdx.x;
data[idx] = ...;  // Consecutive threads → consecutive columns

// BAD: Column-major style in row-major storage
int idx = threadIdx.x * WIDTH + col;  // Strided!
data[idx] = ...;
```

**Rule 3:** Prefer Structure of Arrays (SoA) over Array of Structures (AoS)

```cuda
// BAD: Array of Structures (AoS)
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
Particle particles[1000];
// Access: particles[threadIdx.x].x → Strided by sizeof(Particle)

// GOOD: Structure of Arrays (SoA)
struct Particles {
    float x[1000];
    float y[1000];
    float z[1000];
};
Particles p;
// Access: p.x[threadIdx.x] → Coalesced!
```

---

### 🔍 How to Check If Your Code is Coalesced

**Method 1: CUDA Profiler**

```bash
nvcc 2_memory.cu -o memory
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg ./memory
```

**Look for:**
- `Global Memory Load Efficiency` → Should be > 80%
- `Coalescing Efficiency` → Should be > 90%

**Method 2: Mental Check**

Ask yourself:
1. Do consecutive threads (`threadIdx.x`, `threadIdx.x + 1`) access consecutive memory?
2. Is the stride between accesses = 1 element?

If yes → Coalesced ✅  
If no → Problem ❌

---

### 📊 Performance Impact Examples

**Test case:** Process 1 million floats

```cuda
// Test 1: Coalesced (stride = 1)
__global__ void test1(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2.0f;
}
// Time: 0.1 ms, Bandwidth: 800 GB/s

// Test 2: Strided (stride = 4)
__global__ void test2(float *data) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    data[idx] *= 2.0f;
}
// Time: 0.4 ms, Bandwidth: 200 GB/s (4× slower!)

// Test 3: Random access
__global__ void test3(float *data, int *indices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[indices[idx]] *= 2.0f;
}
// Time: 2.0 ms, Bandwidth: 40 GB/s (20× slower!)
```

---

## Part 5: Debugging Memory Issues

### Common Mistakes and Fixes

**Mistake 1: Wrong dimension for threadIdx**

```cuda
// WRONG
__global__ void bad(float *data, int width) {
    int idx = threadIdx.y * width + threadIdx.x;  // Swapped!
    data[idx] = ...;
}

// CORRECT
__global__ void good(float *data, int width) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * width + col;  // threadIdx.x = column
    data[idx] = ...;
}
```

**Mistake 2: Forgetting to check bounds**

```cuda
// WRONG
__global__ void bad(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = ...;  // What if idx >= n?
}

// CORRECT
__global__ void good(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // Always check!
        data[idx] = ...;
    }
}
```

**Mistake 3: Reusing device pointer as host pointer**

```cuda
// WRONG
float *d_data;
cudaMalloc(&d_data, 100 * sizeof(float));
d_data[0] = 1.0f;  // Segmentation fault! Can't access device memory from host

// CORRECT
float *h_data = new float[100];
float *d_data;
cudaMalloc(&d_data, 100 * sizeof(float));
h_data[0] = 1.0f;  // OK - host memory
cudaMemcpy(d_data, h_data, 100 * sizeof(float), cudaMemcpyHostToDevice);
```

---

## 🎓 Summary: What We Learned

### Key Concepts

1. **Two Memory Worlds:**
   - Host (CPU) and Device (GPU) have separate memory
   - Need explicit copying between them (slow!)
   - Rule: Copy once, compute lots, copy back

2. **Memory Coalescing:**
   - GPU loads memory in chunks (buses) of 32 elements
   - **Coalesced**: Consecutive threads → consecutive memory = 1 bus ✅
   - **Strided**: Threads skip elements = multiple buses ⚠️
   - **Random**: Unpredictable = many buses ❌

3. **Performance Impact:**
   - Coalesced: ~900 GB/s
   - Strided: ~200-400 GB/s (2-4× slower)
   - Random: ~50-100 GB/s (8-16× slower)

4. **How to Write Fast Code:**
   - Make `threadIdx.x` the fastest-changing index
   - Check: Do consecutive threads access consecutive memory?
   - Use profiler to verify coalescing efficiency

---

### Connection to Code Files

**`2_memory.cu`** demonstrates:
- ✅ Perfect coalescing (thread i → array[i])
- CPU↔GPU memory transfers
- Performance of simple parallel operations

**`3_matrix.cu`** demonstrates:
- ✅ Coalesced access (matrix A)
- ⚠️ Strided access (matrix B)
- Why naive matrix multiplication is slow
- Motivation for shared memory (Blog 3!)

---

## 🏋️ Practice Exercises

### Exercise 1: Identify the Pattern

For each kernel, determine if access is coalesced:

```cuda
// A)
__global__ void kernelA(float *data) {
    int idx = threadIdx.x;
    data[idx] = idx;
}

// B)
__global__ void kernelB(float *data, int stride) {
    int idx = threadIdx.x * stride;
    data[idx] = idx;
}

// C)
__global__ void kernelC(float *data, int width) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    data[col * width + row] = row + col;
}
```

**Answers:**
- A: ✅ Coalesced (consecutive threads → consecutive memory)
- B: ⚠️ Strided (depends on `stride` value)
- C: ❌ Strided (column-major in row-major storage)

---

### Exercise 2: Fix the Code

This kernel has poor coalescing. Fix it!

```cuda
// BEFORE (slow)
__global__ void transpose(float *in, float *out, int N) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    out[col * N + row] = in[row * N + col];  // Write is strided!
}

// Your task: Rewrite for better coalescing
// Hint: Think about which dimension should be threadIdx.x
```

---

### Exercise 3: Predict Performance

Rank these from fastest to slowest:

```cuda
// Version 1
data[threadIdx.x] = ...;

// Version 2
data[threadIdx.x * 2] = ...;

// Version 3
data[threadIdx.x * 32] = ...;

// Version 4
data[random_indices[threadIdx.x]] = ...;
```

**Answer:** 1 > 2 > 3 > 4 (coalesced > stride-2 > stride-32 > random)

---

## 🔗 What's Next?

In **Blog 3**, we'll learn about:
- **Shared memory**: The secret weapon for 10-100× speedups
- **Tiled algorithms**: Breaking big problems into cache-friendly chunks
- **Synchronization**: Coordinating threads without race conditions
- **Optimized matrix multiplication**: From 800ms to 50ms to 1ms!

**Files to explore:**
- `4_reduction.cu` - Parallel sum with shared memory
- `5_shared_memory.cu` - Fast matrix multiplication

---

## 📚 Additional Resources

**Official Documentation:**
- [CUDA C Programming Guide - Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [CUDA Best Practices - Coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

**Video Tutorials:**
- [NVIDIA: Memory Coalescing Explained](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Parallel Programming Concepts (Coursera)](https://www.coursera.org/learn/parprog1)

**Interactive Tools:**
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
- [Nsight Compute Profiler](https://developer.nvidia.com/nsight-compute)

---

## 💡 Key Takeaway

> *"Memory coalescing is the difference between a sports car stuck in traffic (strided access) and a sports car on an empty highway (coalesced access). Both can go fast, but only one actually does."*

The GPU has incredible computing power, but it's starved for data. **Coalesced memory access is how you feed the beast.**

---

**Next:** [Blog 3: Shared Memory and Tiling](./blog_3.md) →

**Previous:** [Blog 1: GPU Architecture Fundamentals](./blog_1.md) ←

---

*Questions? Found a typo? Open an issue on GitHub or reach out!*
