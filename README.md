# Learning CUDA: Roadmap and File-by-File Guide

This repository is a guided learning path: fundamentals → GPU programming patterns → graph algorithms → Watts–Strogatz small-world networks → profiling → visualization. Each stage pairs an intuition-first blog-style explanation with one or more example .cu files that you can compile and run.

Quick goals
- Understand GPU architecture and execution model.
- Learn CUDA basics: kernels, memory, synchronization.
- Practice common parallel patterns: map, reduction, tiling, shared memory.
- Represent graphs on GPU (CSR), generate Watts–Strogatz networks, measure clustering.
- Profile and optimize kernels, then export data for Python visualization.

Roadmap (stages and files)

Stage 0 — Orientation
- Goal: high-level motivation and plan.
- Files: this README + blog_1.md (intro and "how GPUs differ").
- What you'll learn: why GPUs, how to structure a multi-file learning project.

Stage 1 — GPU fundamentals (how the machine is organized)
- Goal: map hardware concepts to software constructs.
- Files used:
  - 1_hello_world.cu — threads, blocks, grid, kernel launch, asynchronous execution.
    - Explanation: show how threadIdx/blockIdx map to work items; show cudaDeviceSynchronize() necessity.
    - Exercise: compile and run the program to observe prints from each thread.
  - blog_1.md — complementary blog-style conceptual notes.
- How the code teaches: you will see direct printf uses inside a kernel to illustrate parallelism and non-deterministic ordering.

Stage 2 — Memory & simple data-parallel kernels
- Goal: learn host/device memory, cudaMalloc/cudaMemcpy, coalescing basics.
- Files:
  - 2_memory.cu — host↔device transfers and a simple per-thread update demonstrating coalesced accesses.
  - 3_matrix.cu — naive 2D-indexed matrix multiplication: exposes strided access patterns and motivates shared memory.
- How the code teaches: step from vector-style coalesced access to 2D indexing and show why naive matrix multiply suffers memory inefficiency.

Stage 3 — Parallel patterns and shared memory
- Goal: reductions, tiling, and shared-memory optimizations.
- Files:
  - 4_reduction.cu — tree-based reduction with shared memory; explains __syncthreads and warp issues.
  - 5_shared_memory.cu — tiled matrix multiply using shared memory; shows tiling, bank conflict notes.
- How the code teaches: compare naive vs tiled implementations and reduction variants (shared vs warp primitives).

Stage 4 — Graph representations and irregular workloads
- Goal: store graphs for GPU, understand CSR and irregular memory access.
- Files:
  - 6_graph_representation.cu — CSR layout, degree computation, triangle counting demo; explains warp divergence and load balance challenges.
- How the code teaches: shows tradeoffs (adjacency matrix vs CSR), demonstrates why graphs are irregular and how to reason about performance.

Stage 5 — Watts–Strogatz model on GPU
- Goal: generate small-world networks in parallel; practice cuRAND and edge rewiring.
- Files:
  - 7_watts_strogatz.cu — ring lattice creation and parallel rewiring kernels; conversion to CSR for analysis.
  - 10_visualization_python.cu — included exporter to CSV and an embedded Python visualization script (visualize.py).
- How the code teaches: you will implement ring construction (thread-per-node), per-edge rewiring (thread-per-edge), and move data to CSR for analysis and visualization.

Stage 6 — Analysis: clustering & path lengths
- Goal: compute clustering coefficient and (optionally) path-length estimates.
- Files:
  - 8_clustering_coefficient.cu — per-node triangle counting, reduction to compute average clustering.
- How the code teaches: demonstrates the O(deg²) nature of naive algorithms, options for optimizations (sorting, hashing, matrix methods) and how to parallelize reductions.

Stage 7 — Optimization & profiling
- Goal: find bottlenecks, fix coalescing/warp divergence, tune occupancy.
- Files:
  - 9_optimization_profiling.cu — examples of coalescing (transpose), divergence (reductions), occupancy notes, and profiling hints.
- How the code teaches: run the demos, profile with Nsight/NSYS/NCU, iterate using the checklist in the file.

Stage 8 — Visualization & integration
- Goal: export GPU-generated graphs and visualize them with Python (NetworkX/Matplotlib) or WebGL.
- Files:
  - 10_visualization_python.cu — exporter + example Python plotting script.
- How the code teaches: generate CSVs from CUDA, then load and render with Python scripts; build phase-transition plots and panels.

How the series will explain each .cu file
- For every .cu file we will:
  1. Start with the mathematical or algorithmic goal.
  2. Map high-level operations to CUDA primitives (threads, memory spaces, sync).
  3. Explain the kernel launch geometry and per-thread responsibilities.
  4. Analyze memory access patterns and expected performance pitfalls.
  5. Show how to validate correctness and measure time with CUDA events.
  6. Suggest stepwise optimizations and how to profile the impact.

Getting started — compile and run 1_hello_world.cu
1. Ensure CUDA Toolkit and driver are installed:
   - nvcc --version
   - nvidia-smi
2. Compile and run:
```bash
nvcc c:\Users\incre\OneDrive\Desktop\gpu\learning_gpu\1_hello_world.cu -o hello
./hello
```
3. Expected behavior:
   - CPU prints "Hello World from CPU!"
   - Multiple prints from GPU threads (order non-deterministic).

Short build commands for the series
- Basic examples:
  - nvcc 1_hello_world.cu -o hello
  - nvcc 2_memory.cu -o memory
  - nvcc 3_matrix.cu -o matrix
- Advanced (needs cuRAND):
  - nvcc 6_graph_representation.cu -o graph -lcurand
  - nvcc 7_watts_strogatz.cu -o watts -lcurand
  - nvcc 8_clustering_coefficient.cu -o clustering -lcurand
  - nvcc 10_visualization_python.cu -o visualize -lcurand
- Profiling:
  - nsys profile -o report ./program
  - ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg ./program

Suggested learning path (hands-on)
1. Run and read 1_hello_world.cu and blog_1.md to internalize threads/warps/blocks.
2. Run 2_memory.cu and inspect cudaMalloc/cudaMemcpy behavior and coalescing.
3. Study 3_matrix.cu → then 5_shared_memory.cu to see tiled speedups.
4. Learn reduction patterns in 4_reduction.cu and optimize with warp primitives.
5. Move to graphs: run 6_graph_representation.cu and understand CSR layout.
6. Implement and run 7_watts_strogatz.cu to generate networks, convert to CSR, export edges.
7. Analyze networks with 8_clustering_coefficient.cu.
8. Profile and optimize using 9_optimization_profiling.cu.
9. Export and visualize using 10_visualization_python.cu + provided Python script.

Notes and best practices
- Profile before optimizing. Use Nsight Systems / Nsight Compute.
- Keep kernels simple; prefer library primitives (cuBLAS/cuGraph) for production.
- Always bound-check thread indices (if (tid < n) ...).
- Prefer coalesced memory access; use shared memory to transform patterns.

Where to go next
- Implement BFS on GPU for average path length (next exercise).
- Replace CPU-side edge-list→CSR conversion with a GPU-based builder.
- Experiment with larger N/K and measure scaling and memory limits.

This README is the master roadmap and index into the code examples. Each .cu file contains inline comments explaining the kernel-level mapping between algorithm and GPU architecture — read them while running the examples to solidify learning.

## Resources
[blog](https://yangyangli.top/posts/022-cuda-configuration-for-rust-and-cpp/)
[eunomia blogs](https://eunomia.dev/others/cuda-tutorial/)
[resource stream](https://github.com/gpu-mode/resource-stream)