# Blog 7: Mastering GPU Optimization - From Bottlenecks to Breakthroughs

> *"Optimization without profiling is like surgery without diagnosis. You might make the patient feel better, but you'll never cure the disease."*

Welcome to the final chapter! In [Blog 6](./blog_6.md), we analyzed small-world networks and computed clustering coefficients. Now we'll learn the most valuable skill in GPU programming: **systematic optimization**.

By the end of this blog, you'll know:
- How to find real bottlenecks (not imagined ones)
- The complete optimization workflow
- Advanced techniques for maximum performance
- Production-ready best practices

---

## 🎯 What You'll Learn

- **Profiling fundamentals**: Nsight Systems, Nsight Compute, metrics interpretation
- **The optimization hierarchy**: What to fix first, second, third
- **Memory optimization**: Coalescing, bank conflicts, bandwidth maximization
- **Compute optimization**: Warp divergence, occupancy, instruction-level parallelism
- **Advanced techniques**: Multi-GPU, streams, cooperative groups
- **Production mindset**: Maintainability vs performance tradeoffs

**Prerequisites:** All previous blogs, understanding of GPU architecture, experience running CUDA programs.

---

## Part 1: The Performance Pyramid

### The Four Levels of Optimization

```
         ╱╲         Level 4: Advanced (Last 10%)
        ╱  ╲        - Warp shuffles, async copies
       ╱    ╲       - Tensor cores, multi-GPU
      ╱──────╲      
     ╱ Level 3╲     Level 3: Profiling (Find Real Bottleneck)
    ╱  Memory  ╲    - Nsight tools, metrics analysis
   ╱────────────╲   - Identify memory vs compute bound
  ╱   Level 2    ╲  
 ╱   Coalescing   ╲ Level 2: Basic Optimizations (Obvious Wins)
╱──────────────────╲ - Memory coalescing, shared memory
╲   Level 1        ╱ - Avoid bank conflicts, basic occupancy
 ╲  Correctness   ╱  
  ╲──────────────╱   Level 1: Correctness (Get It Working)
   ╲────────────╱    - Correct results, no race conditions
    ╲──────────╱     - Proper synchronization
     ╲────────╱      
      ╲──────╱       Foundation: Understanding
       ╲────╱        - GPU architecture, CUDA API
        ╲──╱         - Memory hierarchy, execution model
         ╲╱
```

**Critical principle:** Don't skip levels!

1. **Correctness first**: A fast wrong answer is useless
2. **Easy optimizations**: Low-hanging fruit (coalescing, shared memory)
3. **Profile**: Find the REAL bottleneck (not what you think it is)
4. **Advanced techniques**: Squeeze out the last percentages

---

### The Optimization Workflow

```
┌──────────────────┐
│ 1. Write naive   │ → Correct but slow
│    implementation│
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ 2. Verify        │ → Test with small inputs
│    correctness   │   Compare with CPU/reference
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ 3. Profile       │ → Nsight Systems: timeline
│    (find         │   Nsight Compute: kernel details
│     bottleneck)  │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ 4. Is it memory  │ → Memory-bound: optimize access patterns
│    or compute    │   Compute-bound: optimize instructions
│    bound?        │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ↓         ↓
Memory      Compute
Bound       Bound
    │         │
    ↓         ↓
┌───────┐ ┌──────────┐
│Coalesce│ │Reduce    │
│Shared  │ │divergence│
│memory  │ │Increase  │
└───┬────┘ │occupancy │
    │      └────┬─────┘
    │           │
    └─────┬─────┘
          ↓
┌──────────────────┐
│ 5. Measure       │ → Did performance improve?
│    improvement   │   By how much?
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ 6. Repeat        │ → Find next bottleneck
│    (3-5) until   │   Iterate until satisfied
│    satisfied     │
└──────────────────┘
```

---

## Part 2: Profiling Tools

### Tool 1: Nsight Systems - The Timeline View

**Purpose:** See the big picture - what's happening when?

**What it shows:**
- CPU timeline (which CPU threads are active)
- GPU timeline (which kernels are running)
- Memory transfers (CPU↔GPU)
- Gaps (wasted time)

**How to use:**

```bash
# Generate report
nsys profile -o timeline ./my_program

# View in GUI
nsys-ui timeline.qdrep
```

**What to look for:**

```
Timeline visualization:

CPU Thread 1  ████░░░░████████░░░░░░████
              ↑   ↑   ↑       ↑     ↑
              │   │   │       │     └─ Another kernel launch
              │   │   │       └─────── cudaMemcpy (blocking!)
              │   │   └─────────────── Long kernel
              │   └─────────────────── Short gap (good)
              └─────────────────────── Kernel launch

GPU Timeline  ░░████░░░░██████░░░░████
              ↑ ↑   ↑   ↑     ↑   ↑
              │ │   │   │     │   └─ Kernel 3
              │ │   │   │     └───── Gap (GPU idle!)
              │ │   │   └─────────── Kernel 2
              │ │   └─────────────── Gap (GPU idle - BAD!)
              │ └───────────────────Kernel 1
              └─────────────────────GPU idle (waiting for data)

Memory        ░░██░░░░░░░░░░░░░░░░░░██
              ↑ ↑                   ↑
              │ └───────────────────── H→D transfer
              └─────────────────────── D→H transfer
```

**Red flags:**
- **Large gaps**: GPU sitting idle (underutilization)
- **Sequential execution**: Kernels not overlapping (no concurrency)
- **Frequent small transfers**: PCIe overhead dominates
- **Long cudaMemcpy**: Transfer time > compute time

---

### Tool 2: Nsight Compute - The Microscope

**Purpose:** Detailed analysis of a single kernel

**What it shows:**
- Memory bandwidth utilization
- Compute throughput
- Warp execution efficiency
- Occupancy
- Bottlenecks (memory vs compute)

**How to use:**

```bash
# Profile specific kernel
ncu --set full -o kernel_report ./my_program

# View in GUI
ncu-ui kernel_report.ncu-rep

# Get specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./my_program
```

**Key metrics explained:**

| Metric | What It Means | Target | Interpretation |
|--------|--------------|--------|----------------|
| **Achieved Occupancy** | % of max threads running | >50% | Low = wasted SM resources |
| **Memory Throughput** | GB/s utilized | >70% of peak | Low = memory bound but inefficient |
| **Compute Throughput** | % of peak FLOPS | >70% | Low = compute underutilized |
| **Warp Execution Efficiency** | % of threads doing useful work | >90% | Low = warp divergence |
| **Global Load Efficiency** | % of loaded bytes used | >80% | Low = poor coalescing |
| **Shared Bank Conflicts** | Conflicts per access | <1% | High = serialization |

**Example output:**

```
Kernel: matrixMultiply
==================================================
GPU: NVIDIA RTX 3080 (SM 8.6, 68 SMs)

Execution Time:          5.2 ms
Theoretical Peak:        0.8 ms
Efficiency:              15% ← BAD! Something's wrong

Memory Throughput:       850 GB/s
Memory Bandwidth:        936 GB/s (theoretical)
Memory Efficiency:       91% ← GOOD! Memory is fine

Compute Throughput:      50 GFLOPS
Compute Peak:           29,800 GFLOPS
Compute Efficiency:      0.2% ← TERRIBLE! Compute underutilized

Achieved Occupancy:      95% ← GOOD! Plenty of threads

Warp Execution Eff:      22% ← BAD! Warp divergence!
                         ↑
                         This is the bottleneck!

Diagnosis: Memory is efficient, occupancy is high,
           but warp divergence is killing performance.
Action: Reduce branching in kernel code.
```

---

### Tool 3: nvcc Compiler Feedback

**Purpose:** See register and memory usage at compile time

```bash
# Show resource usage
nvcc --ptxas-options=-v kernel.cu

# Output:
# ptxas info    : Used 32 registers, 1024 bytes smem
# ptxas info    : Function properties for matMul
# ptxas info    :     0 bytes stack frame, 0 bytes spill stores,
#                     0 bytes spill loads
```

**What to look for:**
- **High register count** (>64): Might limit occupancy
- **Spill stores/loads**: Registers spilled to slow local memory (BAD!)
- **Shared memory usage**: Must fit within 48-96 KB per SM

---

## Part 3: Memory Optimization Deep Dive

### Optimization 1: Memory Coalescing

**The Problem Visualized:**

```
Bad (Strided) Access:

Memory:  [0][1][2][3][4][5]...[31][32][33]...[63]
Thread:   0           0            0           0
          (iteration) (iteration) (iteration)

Thread 0 accesses: 0, 32, 64, 96, ...
Thread 1 accesses: 1, 33, 65, 97, ...
→ Stride of 32 → Poor locality → Multiple transactions

Good (Coalesced) Access:

Memory:  [0][1][2][3][4]...[31][32][33]...[63]
Thread:   0  1  2  3  4     31
          ↑────────────────↑
          One transaction!

Thread 0 accesses: 0, 1, 2, 3, ...
Thread 1 accesses: 32, 33, 34, 35, ...
→ Consecutive access → Single transaction per warp
```

**Case Study: Matrix Transpose**

From `9_optimization_profiling.cu`:

```cuda
// NAIVE: Strided writes (BAD)
__global__ void transposeNaive(float *out, const float *in, int W, int H) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < W && y < H) {
        // Read: in[y * W + x] → coalesced ✓
        // Write: out[x * H + y] → strided by H ✗
        out[x * H + y] = in[y * W + x];
    }
}
// Performance: 150 GB/s (16% of peak!)
```

**Why it's slow:**

```
Writing to out[x * H + y]:

Thread 0 writes to: out[0 * 4096 + 0] = out[0]
Thread 1 writes to: out[1 * 4096 + 0] = out[4096]
Thread 2 writes to: out[2 * 4096 + 0] = out[8192]
...
Stride = 4096 elements = 16 KB!
→ 32 separate memory transactions for one warp
```

**Optimized version:**

```cuda
// OPTIMIZED: Use shared memory as staging area
__global__ void transposeOptimized(float *out, const float *in, int W, int H) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 is crucial!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Read from global (coalesced)
    if (x < W && y < H) {
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    }
    __syncthreads();
    
    // Transpose coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write to global (now coalesced!)
    if (x < H && y < W) {
        // tile[threadIdx.x][threadIdx.y] already transposed
        out[y * H + x] = tile[threadIdx.x][threadIdx.y];
    }
}
// Performance: 850 GB/s (91% of peak!) → 5.6× faster!
```

**Why it works:**

1. **Read phase**: Coalesced read into shared memory
2. **Transpose**: Happens in shared memory (free!)
3. **Write phase**: Transposed coordinates = coalesced write

**Profile comparison:**

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    transpose_naive transpose_optimized

# Naive:    32 sectors/warp (bad)
# Optimized: 1 sector/warp (perfect!)
```

---

### Optimization 2: Shared Memory Bank Conflicts

**The Hardware:**

```
Shared memory is divided into 32 banks (4 bytes each)

Bank 0:  [0]   [32]  [64]  [96]  ...
Bank 1:  [1]   [33]  [65]  [97]  ...
Bank 2:  [2]   [34]  [66]  [98]  ...
...
Bank 31: [31]  [63]  [95]  [127] ...

Bank number = (address / 4) % 32
```

**Problem: Column Access**

```cuda
__shared__ float tile[32][32];

// All threads access column 0
float val = tile[threadIdx.x][0];

// Address computation:
// Thread 0: tile[0][0] → address 0 → Bank 0
// Thread 1: tile[1][0] → address 32 → Bank 0 (CONFLICT!)
// Thread 2: tile[2][0] → address 64 → Bank 0 (CONFLICT!)
// ...
// All 32 threads → Bank 0 → 32-way conflict!
// → Serialized access → 32× slower
```

**Solution: Padding**

```cuda
__shared__ float tile[32][33];  // +1 column

// Now addresses:
// Thread 0: tile[0][0] → address 0 → Bank 0
// Thread 1: tile[1][0] → address 33 → Bank 1 (no conflict!)
// Thread 2: tile[2][0] → address 66 → Bank 2 (no conflict!)
// ...
// Each thread → different bank → parallel access!
```

**Performance impact:**

```
Test: 10,000 column accesses

Without padding: 150 ms
With padding:     5 ms → 30× faster!

Cost: 32 floats × 4 bytes = 128 bytes (negligible)
```

**Profile it:**

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./program

# Without padding: 32,000 conflicts
# With padding:    0 conflicts
```

---

### Optimization 3: Occupancy Tuning

**What is occupancy?**

```
Occupancy = (Active warps per SM) / (Maximum warps per SM)

Example GPU: 2048 max threads per SM = 64 warps max

If only 32 warps are active:
  Occupancy = 32 / 64 = 50%
```

**Why it matters:**

**High occupancy** → More warps → Better latency hiding

```
Warp 0: ███░░░░░░░░░ (memory stall)
Warp 1:    ░░░███░░░░ (executing while Warp 0 stalls)
Warp 2:       ░░░░███░ (executing while others stall)
...

With 64 warps: Always ~32 warps ready to execute
With 10 warps: Frequent GPU idle time
```

**Limiting factors:**

1. **Threads per block** (set by programmer)
2. **Registers per thread** (compiler decides)
3. **Shared memory per block** (programmer allocates)

**Occupancy calculator:**

```python
# Example: RTX 3080
max_threads_per_sm = 2048
max_blocks_per_sm = 16
max_registers_per_sm = 65536
max_shared_mem_per_sm = 102400  # 100 KB

# Your kernel:
threads_per_block = 256
registers_per_thread = 40
shared_mem_per_block = 8192  # 8 KB

# Limiting factor calculation:
blocks_by_threads = max_threads_per_sm / threads_per_block = 8
blocks_by_registers = max_registers_per_sm / (threads_per_block * registers_per_thread)
                    = 65536 / (256 * 40) = 6.4 → 6 blocks
blocks_by_shared = max_shared_mem_per_sm / shared_mem_per_block
                 = 102400 / 8192 = 12.5 → 12 blocks
blocks_by_hardware = max_blocks_per_sm = 16

# Actual blocks per SM = min(8, 6, 12, 16) = 6 (limited by registers!)

# Occupancy = (6 blocks × 256 threads) / 2048 max = 75%
```

**Improving occupancy:**

```cuda
// BAD: Too many registers
__global__ void kernelBad(float *data) {
    float temp1, temp2, temp3, temp4, temp5;
    float temp6, temp7, temp8, temp9, temp10;
    // Compiler allocates 50+ registers → low occupancy
}

// GOOD: Reuse registers
__global__ void kernelGood(float *data) {
    float temp;  // Reused throughout
    temp = data[0] * 2.0f;
    temp = temp + data[1];
    // Compiler allocates 10 registers → high occupancy
}
```

**Trade-off:** Sometimes lower occupancy is faster if more registers enable better optimization!

```bash
# Profile occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./program

# Achieved occupancy: 48%
# If memory-bound, increasing occupancy won't help!
```

---

## Part 4: Warp Divergence Optimization

### Understanding Divergence

**The Problem:**

```cuda
__global__ void divergent(int *data, int threshold) {
    int idx = threadIdx.x;
    
    if (data[idx] > threshold) {
        // Path A: 20 instructions
        for (int i = 0; i < 10; i++) {
            data[idx] += i;
        }
    } else {
        // Path B: 5 instructions
        data[idx] = 0;
    }
}
```

**What happens in a warp:**

```
Warp with 32 threads:
  - 16 threads: data > threshold (take Path A)
  - 16 threads: data ≤ threshold (take Path B)

Execution timeline:
  Iteration 1: All 32 threads evaluate condition (1 cycle)
  Iteration 2-21: 16 threads execute Path A, 16 idle (20 cycles)
  Iteration 22-26: 16 threads execute Path B, 16 idle (5 cycles)
  
Total: 1 + 20 + 5 = 26 cycles
Best case (no divergence): 1 + max(20, 5) = 21 cycles
Worst case (all diverge): 1 + 20 + 5 = 26 cycles

Efficiency: 21 / 26 = 81% (lost 19% to divergence)
```

---

### Strategy 1: Eliminate Branches

**Before (branching):**

```cuda
__global__ void withBranch(float *data, int *mask) {
    int idx = threadIdx.x;
    
    if (mask[idx]) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
// Warp efficiency: ~50% (half threads diverge)
```

**After (predicated execution):**

```cuda
__global__ void noBranch(float *data, int *mask) {
    int idx = threadIdx.x;
    
    // Compiler generates predicated instruction (no branch!)
    data[idx] = mask[idx] ? (data[idx] * 2.0f + 1.0f) : data[idx];
}
// Warp efficiency: ~100% (no divergence)
```

**How it works:**

```
x86 CPU: if-else compiles to branch instruction (jump)
CUDA GPU: simple predicates compile to predicated instructions

Predicated instruction format:
  @p ADD r1, r2, r3  // Only execute if predicate p is true
  
All threads execute instruction, but only some commit results.
```

---

### Strategy 2: Restructure Loops

**Before (varying loop counts):**

```cuda
__global__ void varyingLoops(int *data, int *counts) {
    int idx = threadIdx.x;
    int count = counts[idx];  // Different for each thread!
    
    for (int i = 0; i < count; i++) {
        data[idx] += i;
    }
    // Threads finish at different times → divergence
}
```

**After (uniform loop count):**

```cuda
__global__ void uniformLoops(int *data, int *counts, int max_count) {
    int idx = threadIdx.x;
    int count = counts[idx];
    
    for (int i = 0; i < max_count; i++) {  // All threads same count
        if (i < count) {  // Predicated execution
            data[idx] += i;
        }
    }
    // All threads finish together → minimal divergence
}
```

**Trade-off:** More iterations, but better warp efficiency.

---

### Strategy 3: Warp-Level Primitives

**Replace shared memory with shuffle:**

```cuda
// OLD: Reduction with shared memory
__global__ void reduceShared(float *input, float *output) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // Many syncs, divergence in early iterations
}

// NEW: Reduction with warp shuffle
__global__ void reduceShuffle(float *input, float *output) {
    int tid = threadIdx.x;
    float val = input[tid];
    
    // Warp-level reduction (no shared memory, no sync!)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // No divergence, faster
    
    if (tid % 32 == 0) output[tid / 32] = val;
}
```

**Shuffle operations:**

```
__shfl_sync(mask, val, lane)        // Get value from specific lane
__shfl_up_sync(mask, val, delta)    // Get from lane-delta
__shfl_down_sync(mask, val, delta)  // Get from lane+delta
__shfl_xor_sync(mask, val, xor)     // Butterfly exchange

mask = 0xffffffff → all 32 threads participate
```

**Performance:**

```
Shared memory reduction:  8.5 ms
Shuffle reduction:        2.1 ms → 4× faster!
```

---

## Part 5: Advanced Optimization Techniques

### Technique 1: Instruction-Level Parallelism (ILP)

**Process multiple elements per thread:**

```cuda
// SIMPLE: 1 element per thread
__global__ void process1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrt(data[idx]) + 1.0f;
    }
}
// Memory: 1 load, 1 store per thread

// ILP: 4 elements per thread
__global__ void process4(float *data, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Unrolled loop
    if (idx < n) data[idx] = sqrt(data[idx]) + 1.0f;
    if (idx+1 < n) data[idx+1] = sqrt(data[idx+1]) + 1.0f;
    if (idx+2 < n) data[idx+2] = sqrt(data[idx+2]) + 1.0f;
    if (idx+3 < n) data[idx+3] = sqrt(data[idx+3]) + 1.0f;
}
// Memory: 4 loads, 4 stores (amortize overhead)
// Latency hiding: While waiting for load 1, compute loads 2-4
```

**Performance:**

```
Process1: 12.0 ms (memory-bound)
Process4:  3.5 ms → 3.4× faster!

Why? Better instruction scheduling, hide memory latency
```

---

### Technique 2: Streams and Concurrency

**Overlap computation with transfers:**

```cuda
// SEQUENTIAL: Slow
for (int i = 0; i < nStreams; i++) {
    cudaMemcpyAsync(d_data[i], h_data[i], size, H2D);
    kernel<<<grid, block>>>(d_data[i]);
    cudaMemcpyAsync(h_result[i], d_result[i], size, D2H);
}
// Timeline: H2D | Kernel | D2H | H2D | Kernel | D2H | ...

// CONCURRENT: Fast
cudaStream_t streams[nStreams];
for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int i = 0; i < nStreams; i++) {
    cudaMemcpyAsync(d_data[i], h_data[i], size, H2D, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(d_data[i]);
    cudaMemcpyAsync(h_result[i], d_result[i], size, D2H, streams[i]);
}

// Timeline: H2D_0 | Kernel_0 | D2H_0
//              H2D_1 | Kernel_1 | D2H_1  ← Overlapping!
//                 H2D_2 | Kernel_2 | D2H_2
```

**Requirements:**
- Pinned (page-locked) host memory: `cudaMallocHost()`
- Asynchronous operations: `cudaMemcpyAsync()`
- Multiple CUDA streams

**Speedup:** Up to 3× if perfect overlap (transfer + compute simultaneously)

---

### Technique 3: Cooperative Groups

**Flexible synchronization:**

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void flexibleSync(float *data) {
    // Traditional: Block-level sync only
    __syncthreads();  // All 256 threads wait
    
    // Cooperative groups: Arbitrary group sizes
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Warp-level sync (faster than block-level)
    int warp_sum = reduce(warp, data[threadIdx.x]);
    
    // Conditional groups
    auto active_threads = cg::coalesced_threads();
    active_threads.sync();  // Only sync threads in group
}
```

**Benefits:**
- More granular control
- Better performance
- Enables advanced patterns

---

### Technique 4: Tensor Cores (Ampere/Ada GPUs)

**Leverage hardware matrix multiplication:**

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void matmulTensorCore(half *A, half *B, float *C, int N) {
    // Declare 16×16 matrix fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Load 16×16 tiles
    wmma::load_matrix_sync(a_frag, A, N);
    wmma::load_matrix_sync(b_frag, B, N);
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Compute C = A × B using Tensor Cores
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}
```

**Performance:**

```
FP32 CUDA cores:     100 TFLOPS
FP16 Tensor cores:   300 TFLOPS → 3× faster!
INT8 Tensor cores:   600 TOPS   → 6× faster!
```

**Requirements:**
- Ampere GPU or newer (RTX 30xx+)
- Matrix dimensions multiple of 16
- FP16 or INT8 precision (mixed precision for FP32 output)

---

## Part 6: Production Best Practices

### Practice 1: Error Checking

**Always check CUDA errors:**

```cuda
// BAD: No error checking
cudaMalloc(&d_data, size);
kernel<<<grid, block>>>(d_data);
cudaMemcpy(h_data, d_data, size, D2H);

// GOOD: Comprehensive error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

CUDA_CHECK(cudaMalloc(&d_data, size));
kernel<<<grid, block>>>(d_data);
CUDA_CHECK(cudaGetLastError());  // Check kernel launch
CUDA_CHECK(cudaDeviceSynchronize());  // Check execution
CUDA_CHECK(cudaMemcpy(h_data, d_data, size, D2H));
```

---

### Practice 2: Use Libraries When Possible

**Don't reinvent the wheel:**

```cuda
// Custom matrix multiply: 500 lines, 100 GFLOPS
__global__ void myMatMul(...) { /* complex code */ }

// cuBLAS: 2 lines, 20,000 GFLOPS (200× faster!)
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
```

**Available libraries:**
- **cuBLAS**: Linear algebra
- **cuFFT**: Fast Fourier transforms
- **cuSPARSE**: Sparse matrices
- **Thrust**: STL-like algorithms
- **cuGraph**: Graph analytics
- **cuDNN**: Deep learning primitives

---

### Practice 3: Maintainability vs Performance

**Balance readability with speed:**

```cuda
// READABLE: Easy to understand
__global__ void processSimple(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrt(data[idx]) + 1.0f;
    }
}

// OPTIMIZED: 4× faster, but complex
__global__ void processOptimized(float *data, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 vals = *reinterpret_cast<float4*>(&data[idx]);
    vals.x = sqrt(vals.x) + 1.0f;
    vals.y = sqrt(vals.y) + 1.0f;
    vals.z = sqrt(vals.z) + 1.0f;
    vals.w = sqrt(vals.w) + 1.0f;
    *reinterpret_cast<float4*>(&data[idx]) = vals;
}
```

**Decision tree:**

```
Is this code:
  ├─ Called once per program? → Use simple version
  ├─ Called millions of times? → Use optimized version
  └─ Hot path (<5% of runtime)? → Profile first!
```

---

## 🎓 Summary

### What We Learned

1. **The Optimization Workflow:**
   - Correctness → Basic optimizations → Profile → Advanced techniques
   - Never skip profiling!
   - Fix the real bottleneck, not the imagined one

2. **Profiling Tools:**
   - Nsight Systems: Timeline view (CPU/GPU activity)
   - Nsight Compute: Kernel-level analysis (metrics)
   - nvcc: Compile-time resource usage

3. **Memory Optimizations:**
   - Coalescing: Consecutive threads → consecutive memory (5-10× faster)
   - Bank conflicts: Padding trick (+1 column) for 30× improvement
   - Occupancy: Balance registers, shared memory, threads

4. **Compute Optimizations:**
   - Warp divergence: Eliminate branches, use predicates (2× faster)
   - ILP: Multiple elements per thread (3-4× faster)
   - Warp shuffles: No shared memory needed (4× faster than traditional)

5. **Advanced Techniques:**
   - Streams: Overlap transfers with computation (3× speedup)
   - Cooperative groups: Flexible synchronization
   - Tensor cores: 3-6× faster for matrix ops

6. **Production Practices:**
   - Always check errors
   - Use libraries (cuBLAS 200× faster than custom code)
   - Balance performance with maintainability

---

### Key Takeaways

> *"The fastest GPU code is the code you don't write. Use libraries. Profile before optimizing. Optimize what matters."*

✅ **Profile-driven development**: Measure, don't guess  
✅ **Low-hanging fruit first**: Coalescing, shared memory (10× gains)  
✅ **Understand hardware**: Memory bandwidth vs compute throughput  
✅ **Library usage**: 100× faster than custom implementations  
✅ **Maintainability matters**: Fast code must also be correct and readable  

---

### Complete Optimization Checklist

**Level 1: Correctness** (Always)
- ☐ Verify output matches reference
- ☐ Check for race conditions
- ☐ Proper synchronization
- ☐ Error checking on all CUDA calls

**Level 2: Basic Optimizations** (Easy wins)
- ☐ Memory coalescing (consecutive thread → consecutive memory)
- ☐ Use shared memory for reused data
- ☐ Avoid shared memory bank conflicts (padding)
- ☐ Launch configuration (256-512 threads per block)
- ☐ Minimize CPU↔GPU transfers

**Level 3: Profiling** (Find bottleneck)
- ☐ Timeline analysis (Nsight Systems)
- ☐ Kernel analysis (Nsight Compute)
- ☐ Identify: memory-bound or compute-bound?
- ☐ Check: occupancy, coalescing efficiency, bank conflicts

**Level 4: Advanced** (Last percentages)
- ☐ Reduce warp divergence
- ☐ Instruction-level parallelism
- ☐ Warp-level primitives
- ☐ Streams and concurrency
- ☐ Consider libraries (cuBLAS, Thrust)

---

### Connection to Code Files

**`9_optimization_profiling.cu`** demonstrates:
- Memory coalescing (transpose example)
- Warp divergence (reduction variants)
- Bank conflicts (shared memory padding)
- Performance comparison framework
- Profiling command templates

**Shows:** Complete optimization pipeline from naive to production-ready!

---

## 🏋️ Practice Exercises

### Exercise 1: Profile and Optimize

Take any previous example (matrix multiply, clustering coefficient):

```bash
# Step 1: Profile
nsys profile -o baseline ./program

# Step 2: Identify bottleneck
ncu --set full -o details ./program

# Step 3: Apply one optimization
# (coalescing, shared memory, or divergence reduction)

# Step 4: Measure improvement
nsys profile -o optimized ./program

# Step 5: Compare
# Baseline vs Optimized speedup?
```

---

### Exercise 2: Fix This Kernel

This kernel is slow. Identify and fix the issues:

```cuda
__global__ void slow(float *output, float *input, int n) {
    int idx = threadIdx.x;  // BUG 1: What if n > blockDim.x?
    
    __shared__ float temp[256][256];  // BUG 2: Bank conflicts!
    
    temp[idx][0] = input[idx];
    __syncthreads();
    
    if (idx % 2 == 0) {  // BUG 3: Warp divergence!
        for (int i = 0; i < 100; i++) {
            output[idx] += temp[idx][0] * i;
        }
    }
}
```

**Answers:**
1. Use global index: `blockIdx.x * blockDim.x + threadIdx.x`
2. Add padding: `temp[256][257]`
3. Use predicated execution or restructure

---

### Exercise 3: Multi-GPU Implementation

Extend Watts-Strogatz generator to multiple GPUs:

```cuda
// GPU 0: Generate nodes 0-499
// GPU 1: Generate nodes 500-999

cudaSetDevice(0);
generateSubgraph<<<...>>>(d_edges0, 0, 500);

cudaSetDevice(1);
generateSubgraph<<<...>>>(d_edges1, 500, 1000);

// Merge results
// Handle cross-GPU edges
```

---

## 🔗 Conclusion: Your GPU Mastery Journey

You've completed the full learning path:

**Blog 1:** GPU architecture fundamentals  
**Blog 2:** Memory hierarchy and coalescing  
**Blog 3:** Shared memory and parallel patterns  
**Blog 4:** Graph representations and irregular workloads  
**Blog 5:** Watts-Strogatz small-world networks  
**Blog 6:** Clustering coefficient and network analysis  
**Blog 7:** Optimization and profiling mastery ← You are here!  

### What's Next?

**Immediate:**
- Profile your existing code
- Apply one optimization per iteration
- Measure every change

**Short-term:**
- Explore CUDA libraries (cuBLAS, Thrust, cuGraph)
- Learn PyTorch/TensorFlow custom CUDA ops
- Study production codebases (RAPIDS, Gunrock)

**Long-term:**
- Multi-GPU scaling (NCCL, MPI + CUDA)
- Advanced algorithms (GNNs, sparse ops, scientific computing)
- Contribute to open-source GPU projects

---

## 📚 Additional Resources

**Profiling Guides:**
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Metrics Reference](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

**Optimization Resources:**
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Optimizing CUDA Applications (GTC Talks)](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session)
- [GPU Optimization Book by Storti & Yurtoglu](https://www.elsevier.com/books/cuda-programming/storti/978-0-12-415933-4)

**Advanced Topics:**
- [Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [CUDA Streams and Concurrency](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- [Tensor Core Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)

**Community:**
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [r/CUDA](https://www.reddit.com/r/CUDA/)
- [GPU Mode Community](https://github.com/gpu-mode/)

---

## 💡 Final Wisdom

> *"Optimization is a journey, not a destination. Every hardware generation brings new capabilities. Every application has unique bottlenecks. Master the fundamentals, profile religiously, and never stop learning."*

You now have the tools to:
- ✅ Write correct CUDA code
- ✅ Identify performance bottlenecks
- ✅ Apply systematic optimizations
- ✅ Achieve 10-100× speedups
- ✅ Understand when to stop optimizing

**Congratulations on completing the GPU Programming Mastery series!** 🎉

Now go forth and parallelize the world! 🚀

---

**End of Blog Series**

**Start:** [Blog 1: GPU Fundamentals](./blog_1.md)

**Previous:** [Blog 6: Network Analysis](./blog_6.md)

---

*Questions? Optimized something cool? Share your journey!*
