# Blog 3: Shared Memory and Parallel Patterns - The GPU's Secret Weapon

> *"Shared memory is like having a whiteboard in a meeting room. Instead of everyone running to the library (global memory) for every fact, you write important stuff on the board once, and everyone in the room can see it instantly."*

Welcome back! In [Blog 2](./blog_2.md), we learned why memory coalescing matters. Now we'll learn the **single most important optimization technique for GPUs**: **shared memory**.

By the end of this blog, you'll understand how to achieve **10-100× speedups** through clever memory management.

---

## 🎯 What You'll Learn

- **Shared memory**: The fastest memory accessible to all threads in a block
- **Parallel reduction**: Combining results from thousands of threads
- **Tiled algorithms**: Breaking big problems into cache-friendly chunks
- **Synchronization**: Coordinating threads without race conditions
- **Bank conflicts**: A hidden performance killer and how to avoid it

**Prerequisites:** Blog 1 (GPU architecture), Blog 2 (memory coalescing), basic understanding of matrix multiplication.

---

## Part 1: The Memory Hierarchy Revisited

### Why Global Memory Alone Isn't Enough

Remember from Blog 2: Global memory is **slow** (400-800 cycles latency). Even with perfect coalescing, we're bottlenecked by bandwidth.

**The problem:** Many algorithms need to **reuse** data. For example, in matrix multiplication:
- Element `B[k][col]` is needed by **all threads computing column `col`**
- With naive approach: Each thread loads it separately from global memory
- For N=1024: Each element loaded **1024 times**!

**The solution:** **Shared memory** - a small, fast cache shared by all threads in a block.

---

### Memory Hierarchy Comparison

| Memory | Latency | Bandwidth | Size | Scope | Hardware |
|--------|---------|-----------|------|-------|----------|
| **Registers** | 1 cycle | N/A | 64 KB/SM | Per-thread | Ultra-fast |
| **Shared Memory** | ~30 cycles | ~1500 GB/s | 48-96 KB/SM | Per-block | **Fast** ✓ |
| **L1 Cache** | ~30 cycles | Auto | 16-128 KB/SM | Per-SM | Automatic |
| **L2 Cache** | ~200 cycles | Auto | 4-6 MB | GPU-wide | Automatic |
| **Global Memory** | ~400-800 cycles | ~900 GB/s | 4-24 GB | GPU-wide | **Slow** |
| **Host Memory** | ~100,000 cycles | ~16 GB/s | 16-128+ GB | CPU | **Very slow** |

**Key insight:** Shared memory is **~13× faster** than global memory in latency, but only **48-96 KB** per SM!

---

### Shared Memory: The Meeting Room Whiteboard

**Analogy:**

Imagine you're working on a group project with 255 classmates (a block of threads):

**Without shared memory (global memory only):**
- Everyone keeps running to the library (global memory) every time they need a fact
- Even if multiple people need the same book, each makes a separate trip
- Lots of time wasted walking

**With shared memory:**
- You have a whiteboard in your meeting room
- One person fetches the book from library and writes key facts on the whiteboard
- Everyone in the room can see it instantly
- **Much faster!**

**Critical constraint:** Only people in **your meeting room** (your thread block) can see your whiteboard. Other meeting rooms (other blocks) have their own whiteboards.

---

## Part 2: Parallel Reduction - Combining Results from Many Threads

### The Problem: Sum 1 Million Numbers

**Sequential CPU code:**
```c
float sum = 0;
for (int i = 0; i < 1000000; i++) {
    sum += array[i];
}
```

**Challenge on GPU:** Each thread computes one partial result. How do we combine them into one final answer?

**Answer:** Tree-based reduction using shared memory.

---

### Tree-Based Reduction Strategy

**Visual representation:**

```
Input: [1] [2] [3] [4] [5] [6] [7] [8]  (8 elements)

Step 1: Threads 0-3 add pairs
        [3]     [7]     [11]    [15]     (4 elements)
         ↑       ↑       ↑       ↑
       1+2     3+4     5+6     7+8

Step 2: Threads 0-1 add pairs
        [10]            [26]             (2 elements)
         ↑               ↑
        3+7           11+15

Step 3: Thread 0 adds final pair
        [36]                             (1 element)
         ↑
       10+26
```

**Key observation:** 
- Each step, **half** the threads work
- Total steps: log₂(N) = 10 for N=1024
- Total operations: N-1 (same as sequential, but **parallel**!)

---

### Code Walkthrough: `4_reduction.cu`

#### Step 1: Load Data into Shared Memory

```cuda
__global__ void reduceSum(float *input, float *output, int n) {
    // Declare shared memory - visible to all threads in block
    __shared__ float sdata[256];  // 256 threads per block
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads ONE element from global memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    
    // CRITICAL: Wait for ALL threads to finish loading
    __syncthreads();
```

**What's happening:**
- **`__shared__`**: Declares shared memory array (fast, block-scoped)
- Each thread copies its element: `global → shared`
- **`__syncthreads()`**: Synchronization barrier (explained below)

**Memory traffic:**
- Global memory reads: N (coalesced, one per thread)
- Shared memory writes: N (fast!)

---

#### Step 2: Tree Reduction

```cuda
    // Reduction loop: s = stride between elements to add
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Add element at distance 's' to current element
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Wait before next iteration
    }
```

**Step-by-step execution (blockDim.x = 8):**

```
Initial: sdata = [1, 2, 3, 4, 5, 6, 7, 8]
         tid =    0  1  2  3  4  5  6  7

Iteration 1: s = 4
  Active threads: 0, 1, 2, 3 (tid < 4)
  Thread 0: sdata[0] += sdata[4] → sdata[0] = 1+5 = 6
  Thread 1: sdata[1] += sdata[5] → sdata[1] = 2+6 = 8
  Thread 2: sdata[2] += sdata[6] → sdata[2] = 3+7 = 10
  Thread 3: sdata[3] += sdata[7] → sdata[3] = 4+8 = 12
  Result: sdata = [6, 8, 10, 12, 5, 6, 7, 8]

Iteration 2: s = 2
  Active threads: 0, 1 (tid < 2)
  Thread 0: sdata[0] += sdata[2] → sdata[0] = 6+10 = 16
  Thread 1: sdata[1] += sdata[3] → sdata[1] = 8+12 = 20
  Result: sdata = [16, 20, 10, 12, 5, 6, 7, 8]

Iteration 3: s = 1
  Active threads: 0 (tid < 1)
  Thread 0: sdata[0] += sdata[1] → sdata[0] = 16+20 = 36
  Result: sdata = [36, 20, 10, 12, 5, 6, 7, 8]
```

**Final answer:** `sdata[0] = 36` (sum of 1+2+3+4+5+6+7+8)

---

#### Step 3: Write Result

```cuda
    // Only thread 0 writes final result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

**Why only thread 0?** After reduction, `sdata[0]` contains the sum for this block.

---

### Understanding `__syncthreads()`

**What it does:** Makes all threads in a block **wait** until everyone reaches this point.

**Why we need it:**

```cuda
// WITHOUT sync - WRONG!
sdata[tid] = input[i];
// Some fast threads might proceed before slow threads finish writing
if (tid < s) {
    sdata[tid] += sdata[tid + s];  // Might read uninitialized data!
}

// WITH sync - CORRECT!
sdata[tid] = input[i];
__syncthreads();  // Everyone waits here until all writes complete
if (tid < s) {
    sdata[tid] += sdata[tid + s];  // Safe: all data is ready
}
```

**Analogy:** Like a teacher saying "Everyone finish writing your answer before we share with the group."

**Rules:**
1. **All threads in block must reach it** (no conditional syncs in divergent code)
2. **Cannot sync across blocks** (blocks are independent)
3. **Deadlock risk** if not all threads reach the barrier

---

### Performance Analysis: Reduction

**Comparison (summing 1 million floats):**

| Version | Time | Speedup | Notes |
|---------|------|---------|-------|
| **CPU Sequential** | 3.0 ms | 1× | Single core, no cache misses |
| **GPU Naive (global)** | 0.5 ms | 6× | Atomic adds (serialized) |
| **GPU Reduction (shared)** | 0.03 ms | **100×** | Tree reduction, shared memory |

**Why so fast?**
- Shared memory: 13× faster than global
- Parallel execution: log₂(N) steps instead of N
- Coalesced memory access

---

## Part 3: Tiled Matrix Multiplication - The Killer Application

### The Problem Revisited

From Blog 2, we saw that naive matrix multiplication has **strided access** for matrix B:

```cuda
// Thread (row, col) computes C[row][col]
for (int k = 0; k < N; k++) {
    sum += A[row][k] * B[k][col];  // B access is strided!
}
```

**Memory inefficiency:**
- Each element of B is loaded **N times** from global memory
- For N=1024: Each element loaded 1024 times!
- Total wasted bandwidth: ~1024× more than necessary

---

### Tiled Algorithm Strategy

**Key idea:** Divide matrices into **tiles** (small sub-matrices), process one tile at a time, caching it in shared memory.

**Visual representation (4×4 matrices, 2×2 tiles):**

```
Matrix A (4×4):          Matrix B (4×4):          Matrix C (4×4):
┌───────┬───────┐       ┌───────┬───────┐       ┌───────┬───────┐
│ A00   │ A01   │       │ B00   │ B01   │       │ C00   │ C01   │
│   A0  │   A1  │   ×   │   B0  │   B1  │   =   │   C0  │   C1  │
├───────┼───────┤       ├───────┼───────┤       ├───────┼───────┤
│ A10   │ A11   │       │ B10   │ B11   │       │ C10   │ C11   │
│   A2  │   A3  │       │   B2  │   B3  │       │   C2  │   C3  │
└───────┴───────┘       └───────┴───────┘       └───────┴───────┘

Computing tile C00:
C00 = A0 × B0 + A1 × B2
      ↓       ↓     ↓       ↓
   Load to   From  Load to From
   shared    shared shared shared
```

**Algorithm steps:**
1. Load tile of A into shared memory (collaborative load by all threads)
2. Load tile of B into shared memory
3. **Synchronize** (wait for all loads to complete)
4. Compute partial results using **shared memory** (fast!)
5. Repeat for next tile
6. Write final result to C

---

### Code Walkthrough: `5_shared_memory.cu`

#### Step 1: Declare Shared Memory

```cuda
__global__ void matMulTiled(const float* A, const float* B, float* C, int n) {
    // Shared memory tiles for this block
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // 16×16 floats
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // This thread's position in output matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
```

**Memory allocation:**
- `As` and `Bs`: Each 16×16×4 = 1024 bytes = 1 KB
- Total: 2 KB per block (well within 48-96 KB limit)
- Allocated once per block, reused for all tiles

---

#### Step 2: Tile Loop

```cuda
    // Loop over tiles: need n/TILE_SIZE tiles to span the full dot product
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
```

**Number of iterations:** ⌈N / TILE_SIZE⌉
- For N=1024, TILE_SIZE=16: 64 iterations
- Each iteration processes one "chunk" of the dot product

---

#### Step 3: Collaborative Loading

```cuda
        // Load tile of A (each thread loads ONE element)
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (aRow < n && aCol < n) 
            ? A[aRow * n + aCol] 
            : 0.0f;
        
        // Load tile of B
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && bCol < n) 
            ? B[bRow * n + bCol] 
            : 0.0f;
        
        // CRITICAL: Wait for all threads to finish loading
        __syncthreads();
```

**What's happening:**
- **256 threads** (16×16 block) load **256 elements** in parallel
- Each thread: loads 1 element of A, 1 element of B
- **Coalesced access:** All threads in a row load consecutive memory
- **After sync:** Both tiles fully populated in shared memory

**Memory traffic per tile:**
- Global reads: 2 × TILE_SIZE² = 2×256 = 512 elements
- Shared reads: (will happen in next step)

---

#### Step 4: Compute Using Shared Memory

```cuda
        // Compute partial dot product using SHARED memory (fast!)
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Wait before loading next tile
        __syncthreads();
    }
```

**Key insight:** All reads are from **shared memory** (30 cycles), not global (400+ cycles)!

**Per-thread work:**
- 16 reads from `As` (shared memory)
- 16 reads from `Bs` (shared memory)
- 16 multiply-adds
- Total: 48 shared memory accesses vs 32 global accesses in naive version

---

#### Step 5: Write Result

```cuda
    // Write final sum to global memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}
```

---

### Memory Access Comparison

**Naive version (3_matrix.cu):**
```
For each C element (N iterations):
  Read 1 element from A (global)  → N reads per C element
  Read 1 element from B (global)  → N reads per C element
Total global reads per C element: 2N
```

**Tiled version (5_shared_memory.cu):**
```
For each tile (N/TILE_SIZE iterations):
  Load entire tile to shared (256 reads, collaborative)
  Read TILE_SIZE times from shared memory
Total global reads per C element: 2N/TILE_SIZE
```

**Speedup in global memory traffic:** TILE_SIZE× (16× for TILE_SIZE=16)

---

### Performance Comparison

**Test case:** 1024×1024 matrix multiplication on RTX 3080

| Version | Time (ms) | GFLOPS | Memory Traffic | Notes |
|---------|-----------|--------|----------------|-------|
| **Naive CPU** | 2000 | 1 | N/A | Single-threaded |
| **GPU Naive** | 50 | 43 | 2GB | From `3_matrix.cu` |
| **GPU Tiled** | 3 | 700 | 125MB | **16× less memory!** |
| **cuBLAS** | 1 | 2000 | Optimal | NVIDIA-optimized |

**Why tiled is faster:**
- 16× reduction in global memory traffic
- Better cache utilization (L1, L2)
- Higher arithmetic intensity (compute/memory ratio)

---

## Part 4: Synchronization Deep Dive

### `__syncthreads()` - The Thread Barrier

**Purpose:** Ensure all threads in a block reach a synchronization point before any proceed.

**Use cases:**
1. **After loading shared memory** (wait for all data to arrive)
2. **After computing with shared memory** (wait before reusing space)
3. **Multi-stage algorithms** (ensure stage N completes before stage N+1)

---

### Synchronization Example

```cuda
__shared__ float data[256];

// Stage 1: Load data
data[tid] = input[tid];
__syncthreads();  // Wait here!

// Stage 2: Use data (safe - all writes complete)
output[tid] = data[tid] + data[(tid+1) % 256];
__syncthreads();  // Wait again!

// Stage 3: Reuse shared memory for something else
data[tid] = output[tid] * 2;
```

**Without syncs:** Race conditions! Thread T might read `data[T+1]` before thread T+1 writes it.

---

### Common Mistakes

**Mistake 1: Conditional sync (DEADLOCK!)**

```cuda
// WRONG - Some threads never reach sync
if (tid < 128) {
    data[tid] = ...;
    __syncthreads();  // Only half threads reach here → DEADLOCK!
}
```

**Fix:** Ensure all threads hit the sync

```cuda
// CORRECT
data[tid] = (tid < 128) ? ... : 0;
__syncthreads();  // All threads reach here
```

---

**Mistake 2: Sync in divergent loop**

```cuda
// WRONG
for (int i = 0; i < degree[tid]; i++) {  // Different nodes, different degrees
    data[tid] += neighbor[i];
    __syncthreads();  // Threads loop different amounts → DEADLOCK!
}
```

**Fix:** Sync outside loop or ensure all threads loop same amount

```cuda
// CORRECT
int maxDegree = ...; // All threads use same value
for (int i = 0; i < maxDegree; i++) {
    if (i < degree[tid]) {
        data[tid] += neighbor[i];
    }
    __syncthreads();  // All threads sync each iteration
}
```

---

**Mistake 3: Syncing across blocks**

```cuda
// WRONG - Cannot sync across blocks!
__global__ void kernel(...) {
    // Do work in block 0
    __syncthreads();  // Only syncs within block, not across blocks!
    // Do more work depending on block 0's results
}
```

**Fix:** Use multiple kernel launches or cooperative groups (advanced)

```cuda
// CORRECT
kernel1<<<...>>>(...);  // All blocks do part 1
cudaDeviceSynchronize();  // CPU waits for all blocks
kernel2<<<...>>>(...);  // All blocks do part 2
```

---

## Part 5: Shared Memory Bank Conflicts

### What Are Memory Banks?

Shared memory is divided into **32 banks** (one per warp thread), each 4 bytes wide.

**Ideal scenario:** Each thread in a warp accesses a **different bank** → all 32 accesses happen simultaneously.

**Bank conflict:** Multiple threads access the **same bank** → accesses are **serialized**.

---

### Bank Layout

```
Address:  0   4   8   12  16  20  24  28  32  36  ...
Bank:     0   1   2   3   4   5   6   7   8   9   ...

Address 0, 32, 64, 96 → all Bank 0
Address 4, 36, 68, 100 → all Bank 1
```

**Formula:** Bank = (Address / 4) % 32

---

### Example: No Conflict

```cuda
__shared__ float data[32];

// Thread i accesses data[i]
float val = data[threadIdx.x];

// Thread 0 → data[0] → Bank 0
// Thread 1 → data[1] → Bank 1
// ...
// Thread 31 → data[31] → Bank 31
// → No conflict, all simultaneous!
```

---

### Example: 2-Way Conflict

```cuda
__shared__ float data[64];

// Thread i accesses data[i * 2]  (stride of 2)
float val = data[threadIdx.x * 2];

// Thread 0 → data[0] → Bank 0
// Thread 1 → data[2] → Bank 2
// ...
// Thread 16 → data[32] → Bank 0 (CONFLICT with thread 0!)
// → 2-way conflict, 2× slower
```

---

### Example: 32-Way Conflict (Worst Case)

```cuda
__shared__ float matrix[32][32];

// All threads access column 0
float val = matrix[threadIdx.x][0];

// Thread 0 → matrix[0][0] → Address 0 → Bank 0
// Thread 1 → matrix[1][0] → Address 32 → Bank 0
// ...
// All 32 threads → Bank 0!
// → 32-way conflict, 32× slower (serialized)
```

---

### The Padding Trick

**Problem:** Accessing a column of a 2D array causes bank conflicts.

**Solution:** Add 1 extra column (padding) to shift banks.

```cuda
// BAD: 32×32 array
__shared__ float bad[32][32];
float val = bad[threadIdx.x][0];  // All access bank 0

// GOOD: 32×33 array (padded)
__shared__ float good[32][33];
float val = good[threadIdx.x][0];

// Now:
// Thread 0 → good[0][0] → Address 0 → Bank 0
// Thread 1 → good[1][0] → Address 33 → Bank 1 (shifted!)
// Thread 2 → good[2][0] → Address 66 → Bank 2
// → No conflict!
```

**Why it works:** Extra column shifts each row's starting bank.

**Cost:** 1 extra column = 32 floats = 128 bytes (negligible)

---

### Real Example: Matrix Transpose

```cuda
// In 5_shared_memory.cu and 9_optimization_profiling.cu

// WITHOUT padding (bank conflicts)
__shared__ float tile[TILE_DIM][TILE_DIM];

// WITH padding (no conflicts)
__shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 is the trick!
```

**Performance impact:**
- Without padding: ~150 GB/s
- With padding: ~600 GB/s
- **4× speedup from one character!**

---

## Part 6: Advanced Optimization Techniques

### Warp-Level Primitives (Warp Shuffle)

**Problem:** `__syncthreads()` synchronizes entire block (256 threads), but we often only need to sync within a warp (32 threads).

**Solution:** Warp shuffle instructions - exchange data between threads **without shared memory**!

```cuda
// Old way (shared memory + sync)
__shared__ float sdata[256];
sdata[tid] = value;
__syncthreads();
float neighbor = sdata[tid + 1];

// New way (warp shuffle, no sync needed!)
float neighbor = __shfl_down_sync(0xffffffff, value, 1);
```

**Advantages:**
- No shared memory usage (save for other data)
- No synchronization overhead
- Faster (single instruction)

---

### Warp Shuffle Reduction

From `9_optimization_profiling.cu`:

```cuda
__global__ void reduceWarpShuffle(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (i < n) ? input[i] : 0.0f;
    
    // Reduce within warp (32 threads) - no shared memory!
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
        // 0xffffffff = all 32 threads participate
        // offset = distance to fetch from
    }
    
    // val now contains warp's sum in thread 0 of each warp
    // ... (handle multiple warps)
}
```

**Shuffle execution:**

```
Initial: [1, 2, 3, 4, 5, 6, 7, 8] (showing 8 threads of 32)

offset=4:
  Thread 0 gets value from thread 4: 1+5=6
  Thread 1 gets value from thread 5: 2+6=8
  ...
  Result: [6, 8, 10, 12, 5, 6, 7, 8]

offset=2:
  Thread 0 gets value from thread 2: 6+10=16
  ...
  Result: [16, 20, 10, 12, 5, 6, 7, 8]

offset=1:
  Thread 0 gets value from thread 1: 16+20=36
  Result: [36, 20, 10, 12, 5, 6, 7, 8]
```

**Performance:** 2-3× faster than shared memory reduction for small arrays!

---

### Multiple Elements Per Thread

**Strategy:** Have each thread process multiple elements to increase arithmetic intensity.

```cuda
// Simple version: 1 element per thread
__global__ void process(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2;
}

// Optimized: 4 elements per thread
__global__ void process4(float *data, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < n) data[i] *= 2;
    if (i+1 < n) data[i+1] *= 2;
    if (i+2 < n) data[i+2] *= 2;
    if (i+3 < n) data[i+3] *= 2;
}
```

**Benefits:**
- Better instruction-level parallelism
- Fewer blocks needed (less kernel launch overhead)
- Better register utilization

**Tradeoff:** Might reduce occupancy (fewer blocks fit on SM)

---

### Loop Unrolling

**Compiler directive to unroll loops:**

```cuda
#pragma unroll
for (int i = 0; i < 16; i++) {
    sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
}

// Compiler expands to:
// sum += As[threadIdx.y][0] * Bs[0][threadIdx.x];
// sum += As[threadIdx.y][1] * Bs[1][threadIdx.x];
// ...
// sum += As[threadIdx.y][15] * Bs[15][threadIdx.x];
```

**Benefits:**
- Eliminates loop counter overhead
- More instruction-level parallelism
- Better register reuse

**Tradeoff:** More instructions → more registers → might lower occupancy

---

## Part 7: Putting It All Together

### Optimization Workflow

```
1. Write CORRECT code (naive is fine)
   ↓
2. Profile (find bottleneck)
   ↓
3. Is it memory-bound?
   → Add coalescing (Blog 2)
   → Add shared memory (this blog)
   ↓
4. Is it compute-bound?
   → Loop unrolling
   → Multiple elements per thread
   ↓
5. Is it synchronization-bound?
   → Reduce __syncthreads() calls
   → Use warp shuffles
   ↓
6. Profile again
   ↓
7. Repeat until satisfied
```

---

### Checklist: Is My Code Using Shared Memory Effectively?

- ☐ **Data reuse?** Same data accessed multiple times? → Use shared memory
- ☐ **Coalesced loads?** Load tiles collaboratively in coalesced pattern
- ☐ **Synchronized correctly?** `__syncthreads()` after load, before use
- ☐ **No bank conflicts?** Use padding (+1) for 2D arrays if needed
- ☐ **Right tile size?** Balance shared memory size vs parallelism
- ☐ **Within limits?** Shared memory < 48 KB per block (or < 96 KB if configured)

---

### Performance Expectations

| Optimization | Expected Speedup | When to Use |
|--------------|------------------|-------------|
| **Coalescing** | 2-4× | Always (first priority) |
| **Shared memory** | 10-20× | Data reused by multiple threads |
| **Tiling** | 10-100× | Matrix operations, stencils |
| **Warp shuffles** | 2-3× | Warp-level reductions |
| **Bank conflict fix** | 2-4× | Column access, transpose |

---

## Part 8: Common Patterns and Recipes

### Pattern 1: Parallel Reduction

**Use case:** Sum, max, min, etc. of array elements

**Recipe:**
```cuda
__shared__ float sdata[BLOCK_SIZE];

// Load
sdata[tid] = input[globalIdx];
__syncthreads();

// Reduce
for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// Write
if (tid == 0) output[blockIdx.x] = sdata[0];
```

---

### Pattern 2: Tiled Matrix Operation

**Use case:** Matrix multiply, convolution, stencils

**Recipe:**
```cuda
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

for (each tile) {
    // Collaborative load
    As[ty][tx] = A[...];
    Bs[ty][tx] = B[...];
    __syncthreads();
    
    // Compute using shared memory
    for (int k = 0; k < TILE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
}

C[...] = sum;
```

---

### Pattern 3: Prefix Sum (Scan)

**Use case:** Cumulative sum, stream compaction

**Recipe:**
```cuda
__shared__ float temp[BLOCK_SIZE];

temp[tid] = input[globalIdx];
__syncthreads();

// Up-sweep phase
for (int d = 0; d < log2(BLOCK_SIZE); d++) {
    if (tid % (1 << (d+1)) == 0) {
        temp[tid + (1<<(d+1))-1] += temp[tid + (1<<d)-1];
    }
    __syncthreads();
}

// Down-sweep phase (details omitted)
```

---

### Pattern 4: Histogram

**Use case:** Counting occurrences

**Recipe:**
```cuda
__shared__ int hist[NUM_BINS];

// Initialize shared histogram
if (tid < NUM_BINS) hist[tid] = 0;
__syncthreads();

// Accumulate
for (int i = tid; i < n; i += blockDim.x) {
    int bin = computeBin(data[i]);
    atomicAdd(&hist[bin], 1);  // Atomic for thread-safety
}
__syncthreads();

// Write to global
if (tid < NUM_BINS) {
    atomicAdd(&globalHist[tid], hist[tid]);
}
```

---

## 🎓 Summary

### What We Learned

1. **Shared Memory:**
   - 13× faster than global memory
   - Visible to all threads in a block (48-96 KB per block)
   - Perfect for data reuse and inter-thread communication

2. **Parallel Reduction:**
   - Tree-based algorithm: log₂(N) steps
   - Combines results from many threads efficiently
   - Foundation for sum, max, min, etc.

3. **Tiled Matrix Multiplication:**
   - Reduces global memory traffic by factor of TILE_SIZE
   - 10-100× speedup over naive version
   - Demonstrates power of shared memory reuse

4. **Synchronization:**
   - `__syncthreads()` ensures data consistency
   - Critical for correctness, not performance
   - Must be called by all threads in block (no conditionals!)

5. **Bank Conflicts:**
   - Accessing same bank from multiple threads = serialization
   - Fix with padding: `array[SIZE][SIZE+1]`
   - Can make 4× difference in performance

6. **Advanced Techniques:**
   - Warp shuffles: faster than shared memory for warp-level ops
   - Loop unrolling: better instruction-level parallelism
   - Multiple elements per thread: better arithmetic intensity

---

### Connection to Code Files

**`4_reduction.cu`** demonstrates:
- Shared memory allocation and usage
- `__syncthreads()` synchronization
- Tree-based reduction pattern
- Warp divergence issues

**`5_shared_memory.cu`** demonstrates:
- Tiled algorithm design
- Collaborative loading pattern
- Shared memory reuse (16× reduction in global memory traffic)
- Matrix multiplication optimization

Both show how to achieve **10-100× speedups** through shared memory!

---

## 🏋️ Practice Exercises

### Exercise 1: Optimize 2D Convolution

Given this naive 2D convolution kernel:

```cuda
__global__ void convolve2D(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (x+i >= 0 && x+i < width && y+j >= 0 && y+j < height) {
                sum += input[(y+j)*width + (x+i)] * kernel[i+1][j+1];
            }
        }
    }
    output[y*width + x] = sum;
}
```

**Task:** Optimize using shared memory. Each block should load a tile with halo region.

**Hints:**
- Tile size: 16×16, but load 18×18 (including borders)
- Use shared memory to cache input tile
- Reduce global memory reads by ~9×

---

### Exercise 2: Parallel Max Reduction

Modify the reduction kernel to find the **maximum** element instead of sum.

**Challenge:** How do you handle the identity element?

```cuda
// For sum: identity is 0
// For max: identity is -∞ (use -FLT_MAX)

__global__ void reduceMax(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // YOUR CODE HERE
}
```

---

### Exercise 3: Shared Memory Bank Conflict Detector

Write a kernel that measures bank conflicts:

```cuda
__global__ void testBankConflicts(float *output, int stride) {
    __shared__ float data[32 * 32];
    
    // Access with given stride
    int idx = threadIdx.x * stride;
    float val = data[idx];
    
    output[threadIdx.x] = val;
}
```

**Task:** 
- Run with stride = 1, 2, 4, 8, 16, 32
- Use `ncu` to measure bank conflicts
- Plot conflicts vs stride

---

## 🔗 What's Next?

In **Blog 4**, we'll apply everything we've learned to **graph algorithms**:
- Representing graphs on GPU (CSR format)
- Handling irregular memory patterns
- Watts-Strogatz network generation
- Dealing with warp divergence in real algorithms

**Files to explore:**
- `6_graph_representation.cu` - CSR format and basic graph operations
- `7_watts_strogatz.cu` - Small-world network generator
- `8_clustering_coefficient.cu` - Graph analysis

---

## 📚 Additional Resources

**Official CUDA Documentation:**
- [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)
- [Best Practices - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)

**Papers:**
- ["Optimizing Parallel Reduction in CUDA"](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) - NVIDIA
- ["Better Performance at Lower Occupancy"](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf) - Volkov, GTC 2010

**Videos:**
- [CUDA Parallel Reduction](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [Shared Memory Bank Conflicts](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

**Interactive Tools:**
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - Profile shared memory usage

---

## 💡 Key Takeaway

> *"Shared memory is the difference between a good GPU programmer and a great one. Master it, and you unlock 10-100× speedups. Ignore it, and you're leaving 90% of the GPU's potential untapped."*

The patterns you learned here (reduction, tiling, collaborative loading) appear in **every high-performance GPU application**. They're the foundation of:
- Deep learning frameworks (PyTorch, TensorFlow)
- Scientific computing (cuBLAS, cuFFT)
- Graphics engines (ray tracing, rasterization)
- Graph analytics (cuGraph, Gunrock)

**You now have the tools to write truly fast GPU code!**

---

**Next:** [Blog 4: Graph Algorithms on GPU](./blog_4.md) →

**Previous:** [Blog 2: Memory and Coalescing](./blog_2.md) ←

---

*Questions? Found a bug? Want to share your optimized kernel? Open an issue on GitHub!*
