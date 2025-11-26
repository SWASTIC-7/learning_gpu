# Blog 2: Understanding GPU Memory

> *"GPUs are fast not because they do things quickly, but because they do many things at once. But there's a catch: they need the right fuel delivered the right way."*

Welcome back! In [Blog 1](./blog_1.md), we learned about GPU architecture - threads, warps, blocks, and grids. Now let's understand **why some GPU programs are 100× faster than others**, even when doing the same work.


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

GPU don't share CPU's RAM, infact they have their own memory

```
(CPU)                    (GPU)
┌──────────┐            ┌──────────┐
│  RAM     │ ←──────→   │  VRAM    │
│ 16-32 GB │  PCIe Bus  │  4-24 GB │
└──────────┘  (slow!)   └──────────┘
```

---


**Note:** But the cost of sending data from CPU memory to GPU memory is high. Therefore you should, transfer data once, do lots of computation, then transfer back.

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

**Result:** GPU loads **ONE bus** (128 bytes) and delivers to all 32 threads(4 bytes each thread). **Efficient!**

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


**When this happens:**
- Accessing every Nth element (`arr[i * N]`)
- Column-major access in row-major storage
- Structure of Arrays (SoA) with wrong indexing

---

### Scenario 3: Random Access - Everyone Takes Different Buses

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


**When this happens:**
- Graph algorithms (neighbors are scattered)
- Hash tables
- Pointer chasing
- Random indexing

---

### Performance Comparison Table

| Pattern | Buses Needed | Efficiency | Max Speed | Example |
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
note memory store the elements by flatening the matrix
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
---

To fix this:  we will discuss in future series


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


---
