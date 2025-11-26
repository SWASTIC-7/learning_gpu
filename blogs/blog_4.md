# Blog 4: Graph Algorithms on GPU - Taming Irregular Data

> *"Regular algorithms are like marching soldiers—everyone moves in lockstep. Graph algorithms are like a crowd at a festival—everyone moves at their own pace, in their own direction. And that's exactly what makes them interesting... and challenging for GPUs."*

Welcome back! In [Blog 3](./blog_3.md), we mastered shared memory and parallel patterns on regular, structured data (arrays, matrices). Now we tackle the **irregular, unpredictable world of graphs**.

By the end of this blog, you'll understand why graph algorithms are the **"final boss"** of GPU programming—and how to beat them!

---

## 🎯 What You'll Learn

- **Graph theory basics**: Vertices, edges, paths, triangles
- **Graph representations**: Adjacency matrix vs CSR (Compressed Sparse Row)
- **Why graphs are hard for GPUs**: Irregular memory access, warp divergence, load imbalance
- **Real implementations**: Degree computation, triangle counting, clustering coefficient
- **Performance tradeoffs**: When to use GPU vs CPU for graphs

**Prerequisites:** Blogs 1-3 (GPU architecture, memory coalescing, shared memory), basic graph theory (nodes and edges).

---

## Part 1: What Are Graphs?

### The Universal Data Structure

**Graphs** model relationships between objects. They appear everywhere:

| Domain | Nodes (Vertices) | Edges (Links) |
|--------|-----------------|---------------|
| **Social Networks** | People | Friendships, follows |
| **Internet** | Routers, servers | Network cables |
| **Biology** | Neurons | Synapses |
| **Transportation** | Cities | Roads, flights |
| **Chemistry** | Atoms | Chemical bonds |
| **Recommendation** | Users, items | Purchases, ratings |

**Formal definition:** A graph G = (V, E) where:
- V = set of vertices (nodes)
- E = set of edges (connections between nodes)

---

### Example: Small Social Network

```
Alice --- Bob
  |        |
  |        |
Carol --- Dave

Graph G:
V = {Alice, Bob, Carol, Dave}
E = {(Alice,Bob), (Alice,Carol), (Bob,Dave), (Carol,Dave)}
```

**Numbered version:**
```
0 --- 1
|     |
3 --- 2

V = {0, 1, 2, 3}
E = {(0,1), (0,3), (1,2), (2,3)}
```

---

### Graph Properties

**Directed vs Undirected:**
```
Undirected (friendship):    Directed (following):
    0 ←→ 1                      0 → 1
                                ↑
                                2
```

**Sparse vs Dense:**
- **Sparse**: Few edges relative to possible edges (E ≈ 10×V)
  - Example: Social networks, web graphs
  - Most real-world graphs are sparse!
- **Dense**: Many edges (E ≈ V²)
  - Example: Complete graph, molecular structures

**Degree:**
- Number of connections a node has
- Example: Alice has degree 2 (connected to Bob and Carol)

**Triangle:**
- Three nodes all connected to each other
- Example: {Alice, Bob, Carol} if all three are friends
- Used to measure clustering

---

## Part 2: Representing Graphs in Memory

The **single most important decision** for GPU graph algorithms: how to store the graph in memory.

### Representation 1: Adjacency Matrix

**Idea:** N×N matrix where `M[i][j] = 1` if edge (i,j) exists.

**Example:** Our 4-node graph
```
     0  1  2  3
  ┌──────────────┐
0 │  0  1  0  1  │  ← Row 0: node 0's neighbors
1 │  1  0  1  0  │
2 │  0  1  0  1  │
3 │  1  0  1  0  │
  └──────────────┘
```

**Reading the matrix:**
- `M[0][1] = 1` → Edge (0,1) exists
- `M[0][2] = 0` → No edge between 0 and 2
- Row 0 = `[0,1,0,1]` → Node 0 connects to nodes 1 and 3

---

#### Adjacency Matrix: Pros and Cons

**✅ Advantages:**

1. **O(1) edge lookup:** "Does edge (u,v) exist?" → Just check `M[u][v]`
2. **Regular memory access:** Perfect for GPU coalescing
3. **Easy to parallelize:** Each thread processes one row
4. **Dense linear algebra:** Use cuBLAS, matrix multiplication tricks

**Example kernel:**
```cuda
__global__ void hasEdge(int *matrix, int u, int v, int N, bool *result) {
    *result = matrix[u * N + v];  // Single memory access!
}
```

**❌ Disadvantages:**

1. **O(N²) memory:** 
   - 1 million nodes = 1 TB of memory (!!)
   - Even if graph has only 10 million edges
2. **Wastes memory:** Most real graphs are sparse
   - Facebook: ~5000 friends / 3 billion users = 0.0002% density
   - Matrix would be 99.9998% zeros!
3. **Bandwidth waste:** Loading mostly zeros from memory

**When to use:** Dense graphs (E > 0.1×N²) or small N (< 10,000)

---

### Representation 2: Adjacency List (Edge List)

**Idea:** For each node, store a list of its neighbors.

**Example:**
```
Node 0: [1, 3]
Node 1: [0, 2]
Node 2: [1, 3]
Node 3: [0, 2]
```

**In memory (array of arrays):**
```
neighbors[0] = [1, 3]       // Node 0's neighbors
neighbors[1] = [0, 2]       // Node 1's neighbors
neighbors[2] = [1, 3]       // Node 2's neighbors
neighbors[3] = [0, 2]       // Node 3's neighbors
```

**Problem for GPU:** Variable-length arrays → hard to allocate and access efficiently!

---

### Representation 3: CSR (Compressed Sparse Row) ⭐

**The industry standard for sparse graphs on GPU.**

**Idea:** Flatten all neighbor lists into two arrays:
1. `offsets[N+1]`: Where each node's neighbors start
2. `edges[E]`: All neighbors, concatenated

**Example: Our 4-node graph**

```
neighbors[0] = [1, 3]  →  Start at index 0, length 2
neighbors[1] = [0, 2]  →  Start at index 2, length 2
neighbors[2] = [1, 3]  →  Start at index 4, length 2
neighbors[3] = [0, 2]  →  Start at index 6, length 2

CSR representation:
offsets = [0,  2,  4,  6,  8]
           ↑   ↑   ↑   ↑   ↑
          n0  n1  n2  n3  end

edges   = [1, 3,  0, 2,  1, 3,  0, 2]
           ↑────┘  ↑────┘  ↑────┘  ↑────┘
           node 0  node 1  node 2  node 3
```

**How to read CSR:**
```cuda
// Get neighbors of node i
int start = offsets[i];
int end = offsets[i + 1];
for (int j = start; j < end; j++) {
    int neighbor = edges[j];
    // Process neighbor
}
```

---



**When to use:** Sparse graphs (E << N²) - **most real-world cases**

---

### Visual Comparison

```
Graph:  0―1―2―3

Adjacency Matrix (16 integers):
[0 1 0 0]
[1 0 1 0]
[0 1 0 1]
[0 0 1 0]

CSR (4 nodes, 6 edges = 11 integers):
offsets = [0, 1, 3, 5, 6]
edges   = [1, 0, 2, 1, 3, 2]

Savings: 16 → 11 (31% reduction)
For larger sparse graphs: 90-99% reduction!
```

---

## Part 3: The Challenge - Why Graphs Are Hard for GPUs

Remember from Blog 1: **GPUs love regularity**. Coalesced memory, uniform workloads, predictable branches.

**Graphs are the opposite of regular:**

### Challenge 1: Irregular Memory Access

**Problem:** Neighbor lists are scattered in memory.

**Visual:**
```
Memory layout of edges array:
[1, 3, | 0, 2, | 1, 3, | 0, 2, | 5, 7, 9, 10, 11, | 4, ...]
 ↑────┘  ↑────┘  ↑────┘  ↑────┘  ↑─────────────┘   ↑
 Node 0  Node 1  Node 2  Node 3  Node 4 (degree 5) Node 5

When threads 0,1,2,3 access their neighbors:
Thread 0: edges[0,1]     → Memory locations 0-1
Thread 1: edges[2,3]     → Memory locations 2-3
Thread 2: edges[4,5]     → Memory locations 4-5
Thread 3: edges[6,7]     → Memory locations 6-7
Thread 4: edges[8-12]    → Memory locations 8-12  (longer!)
```

**Result:** 
- Not coalesced! (threads access non-consecutive memory)
- Different threads load different amounts
- Cache thrashing (poor locality)

**Compare to matrix operations:**
```cuda
// Regular matrix access (coalesced!)
float val = matrix[row * N + threadIdx.x];  // Consecutive columns

// Graph access (scattered!)
int neighbor = edges[offsets[node] + k];  // Random locations
```

---

### Challenge 2: Warp Divergence

**Problem:** Different nodes have different degrees → threads diverge.



**What happens in a warp:**
```
Iteration 1:  All 32 threads active ✓
Iteration 2:  All 32 threads active ✓
Iteration 3:  Only threads with degree ≥ 3 active (some idle) ⚠️
...
Iteration 100: Only thread 16 active (31 threads idle!) ❌
```

**Performance impact:** Warp executes at the speed of the **slowest thread**.

**Compare to regular algorithms:**
```cuda
// Regular: All threads do same work
for (int i = 0; i < 1000; i++) {  // Fixed loop count
    sum += data[i];
}

// Graph: Different threads do different work
for (int i = start; i < end; i++) {  // end varies by node!
    sum += edges[i];
}
```

---

### Challenge 3: Load Imbalance

**Problem:** High-degree nodes dominate execution time.

**Real-world example: Twitter graph**
There is very famous example where in social media platfroms, where general users have an average of 100 followers while celebrities have followers in millions, when you will try to run same code for all members parallely, final time = time taken by celeberity user with most followers

### Challenge 4: Random Memory Access

**Problem:** Following edges jumps around memory unpredictably.

**Example: Triangle counting**
```cuda
// For each pair of my neighbors, check if they're connected
for (int i = start; i < end; i++) {
    int u = edges[i];  // My first neighbor
    for (int j = i+1; j < end; j++) {
        int v = edges[j];  // My second neighbor
        
        // Are u and v connected? Must check u's neighbor list
        int u_start = offsets[u];  // ← Random jump in memory!
        int u_end = offsets[u+1];
        
        for (int k = u_start; k < u_end; k++) {
            if (edges[k] == v) {  // ← Another random jump!
                triangle_count++;
            }
        }
    }
}
```

**Memory access pattern:**
```
Thread 0 → offsets[5] → edges[120-150] → offsets[37] → edges[890-920]
Thread 1 → offsets[12] → edges[230-245] → offsets[2] → edges[4-8]
...
Completely random, unpredictable, uncacheable!
```

---
With all the problems in hand, lets try to see our implementation

## Part 4: Code Walkthrough - `6_graph_representation.cu`

Let's see these challenges in action!

### Setup: Generate Random Graph (CPU)

```cuda
void generateErdosRenyiCPU(int n, float p, Graph* g) {
    // Erdős-Rényi: Each possible edge exists with probability p
    // Expected edges: n×(n-1)/2 × p
    
    // Step 1: Generate edges using adjacency matrix (easier)
    bool* adj_matrix = (bool*)calloc(n * n, sizeof(bool));
    int edge_count = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((float)rand() / RAND_MAX < p) {
                adj_matrix[i * n + j] = true;
                adj_matrix[j * n + i] = true;  // Undirected
                edge_count += 2;  // Count both directions
            }
        }
    }
}
```

**Why on CPU?** Random number generation is sequential here. For production, use cuRAND on GPU.

---

### Convert to CSR Format

```cuda
    // Step 2: Count degree of each node
    int *degree = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n * n; i++) {
        if (adj_matrix[i]) degree[i / n]++;
    }
    
    // Step 3: Build offsets array (prefix sum of degrees)
    int *offsets = (int*)malloc((n + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < n; i++) {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    // offsets[i] = starting index of node i's neighbors
    // offsets[n] = total number of edges
    
    // Step 4: Fill edges array
    int *edges = (int*)malloc(edge_count * sizeof(int));
    int *current_pos = (int*)calloc(n, sizeof(int));  // Track position per node
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (adj_matrix[i * n + j]) {
                int pos = offsets[i] + current_pos[i];
                edges[pos] = j;
                current_pos[i]++;
            }
        }
    }
```

**Result:** Two arrays `offsets` and `edges` ready for GPU!

---

### Kernel 1: Compute Degrees (Simple)

```cuda
__global__ void computeDegrees(const int* offsets, int* degrees, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        // Degree = number of neighbors = difference in offsets
        degrees[node] = offsets[node + 1] - offsets[node];
    }
}
```



---

### Kernel 2: Print Neighbors (Debugging)

```cuda
__global__ void printNeighbors(const int* offsets, const int* edges, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = offsets[node];
        int end = offsets[node + 1];
        int degree = end - start;
        
        printf("Node %d (degree %d): ", node, degree);
        for (int i = start; i < end; i++) {  // ← Loop length varies!
            printf("%d ", edges[i]);
        }
        printf("\n");
    }
}
```

here you can observe warp divergence


---

### Kernel 3: Count Triangles (Complex)

```cuda
__global__ void countTriangles(const int* offsets, const int* edges, 
                                int* triangles, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int count = 0;
        int start = offsets[node];
        int end = offsets[node + 1];
        
        // For each pair of neighbors (u, v)
        for (int i = start; i < end; i++) {
            int u = edges[i];
            
            for (int j = i + 1; j < end; j++) {
                int v = edges[j];
                
                // Check if u and v are connected
                int u_start = offsets[u];  // ← Random memory access!
                int u_end = offsets[u + 1];
                
                for (int k = u_start; k < u_end; k++) {
                    if (edges[k] == v) {  // ← Another random access!
                        count++;
                        break;
                    }
                }
            }
        }
        
        triangles[node] = count;
    }
}
```

**Analysis:**

❌❌❌ **Triple whammy:**

1. **Extreme divergence:** 
   - Low-degree node (degree 2): ~2 comparisons
   - High-degree node (degree 100): ~10,000 comparisons
   - Ratio: 5000× difference!

2. **Random access:**
   - `offsets[u]` where `u` is unpredictable
   - `edges[k]` where `k` depends on `u`
   - Zero spatial locality

3. **O(degree³) complexity:**
   - Nested loops over neighbors
   - For degree d: d² pairs × d checks = d³ operations
   - Degree 100 node: 1 million operations!

**Performance:** ~20-50 GB/s (20-40× slower than optimal)

---

## Part 5: Fighting Back - Optimization Strategies

### Strategy 1: Sort Adjacency Lists

**Problem:** Linear search for edge existence: O(degree)

**Solution:** Sort each neighbor list, use binary search: O(log degree)

```cuda
// Before: Linear search
for (int k = u_start; k < u_end; k++) {
    if (edges[k] == v) return true;  // O(degree)
}

// After: Binary search (if sorted)
int left = u_start, right = u_end;
while (left < right) {
    int mid = (left + right) / 2;
    if (edges[mid] == v) return true;
    else if (edges[mid] < v) left = mid + 1;
    else right = mid;
}
// O(log degree) - 10× faster for degree 1000!
```

**Tradeoff:** Must sort during construction (one-time cost)

---

### Strategy 2: Load Balancing

**Problem:** High-degree nodes dominate execution time

**Solution 1: Virtual Warps** (assign multiple threads to one node)

```cuda
// Instead of: 1 thread per node
int node = threadIdx.x;

// Use: Multiple threads per high-degree node
int node = threadIdx.x / THREADS_PER_NODE;
int local_tid = threadIdx.x % THREADS_PER_NODE;

// Each thread processes subset of neighbors
int start = offsets[node] + local_tid;
int end = offsets[node + 1];
int stride = THREADS_PER_NODE;

for (int i = start; i < end; i += stride) {
    process(edges[i]);
}
```

**Solution 2: Binning** (sort nodes by degree, process similar degrees together)

```cuda
// Bin 1: degree 1-10    (most nodes)     → 256 threads/block
// Bin 2: degree 11-100  (some nodes)     → 128 threads/block
// Bin 3: degree 101+    (few nodes)      → 32 threads/block + virtual warps
```

---

### Strategy 3: Hash Table for Edge Lookup

**Problem:** O(degree) linear search for edge existence

**Solution:** Build hash table of edges for O(1) lookup

```cuda
// Preprocessing: Build hash table
__global__ void buildEdgeHash(const int* offsets, const int* edges,
                               int* hash_table, int table_size) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    // For each edge (u, v), insert hash(u,v) → 1
}

// Triangle counting: Fast lookup
__global__ void countTrianglesHash(const int* offsets, const int* edges,
                                    const int* hash_table, int* triangles) {
    // ...
    if (hashLookup(hash_table, u, v)) {  // O(1) instead of O(degree)!
        triangle_count++;
    }
}
```

**Tradeoff:** Extra memory for hash table, collisions possible

---

### Strategy 4: Matrix Multiplication Approach

**Key insight:** Triangles = non-zero entries in A² where A = adjacency matrix

```cuda
// Triangle count for node i = (A² ∘ A)[i,i] / 2
// Where ∘ = element-wise multiplication

// Step 1: Compute C = A × A (using cuBLAS or custom tiled kernel)
// Step 2: Count non-zeros where both A[i,j] and C[i,j] are non-zero
```

**Advantage:** Leverage optimized matrix multiply (cuBLAS)

**Disadvantage:** Requires dense matrix (O(N²) memory) or sparse BLAS

---

### Strategy 5: Reducing Warp Divergence

**Technique: Work-item packing**

```cuda
// Bad: Each thread processes one node (divergent)
__global__ void processNodes(const int* offsets, const int* edges) {
    int node = threadIdx.x;
    int start = offsets[node];
    int end = offsets[node + 1];
    
    for (int i = start; i < end; i++) {  // Diverges!
        process(edges[i]);
    }
}

// Better: Each warp processes one node (less divergence)
__global__ void processNodesWarp(const int* offsets, const int* edges) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    int node = warp_id;
    int start = offsets[node];
    int end = offsets[node + 1];
    
    // All threads in warp work on same node's neighbors
    for (int i = start + lane; i < end; i += 32) {
        process(edges[i]);  // Same loop count for all threads in warp!
    }
}
```




---

## Part 7: Real-World Graph Algorithms

### Algorithm 1: Breadth-First Search (BFS)

**Use case:** Shortest paths, connected components, web crawling

**GPU strategy: Frontier-based**

```cuda
// Level-synchronous BFS
bool *visited;      // Size N
int *frontier;      // Current level
int *next_frontier; // Next level

// Iteration k: Process all nodes at distance k
__global__ void bfsKernel(const int* offsets, const int* edges,
                          bool* visited, int* frontier, int* next_frontier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier_size) {
        int node = frontier[idx];
        
        // Explore all neighbors
        int start = offsets[node];
        int end = offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = edges[i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                // Add to next frontier (use atomics or prefix sum)
                atomicAdd(&next_frontier_size, 1);
                next_frontier[next_frontier_size-1] = neighbor;
            }
        }
    }
}
```

**Challenges:**
- Frontier size varies by level
- Atomic operations for next frontier (serialization)
- Load imbalance (some nodes have many unvisited neighbors)

**Performance:** 100-1000× faster than CPU for large graphs

---

### Algorithm 2: PageRank

**Use case:** Google search, influence ranking, importance scores

**GPU strategy: Iterative matrix-vector multiply**

```cuda
// PageRank formula: PR(i) = (1-d)/N + d × Σ PR(j)/degree(j)
//                                       j→i

__global__ void pageRankKernel(const int* offsets, const int* edges,
                                const float* pr_old, float* pr_new,
                                const int* degrees, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        float sum = 0.0f;
        
        // Sum contributions from incoming edges
        int start = offsets[node];
        int end = offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = edges[i];
            sum += pr_old[neighbor] / degrees[neighbor];
        }
        
        pr_new[node] = 0.15f / num_nodes + 0.85f * sum;
    }
}
```

**Advantages:**
- Regular memory access (read `pr_old`, write `pr_new`)
- Converges in 20-50 iterations
- Embarrassingly parallel (no synchronization needed)

**Performance:** 10-100× faster than CPU

---

### Algorithm 3: Community Detection (Label Propagation)

**Use case:** Find clusters in social networks

**GPU strategy: Iterative label propagation**

```cuda
__global__ void labelPropagationKernel(const int* offsets, const int* edges,
                                        const int* labels, int* new_labels) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Count label frequency among neighbors
    int label_count[MAX_LABELS] = {0};
    
    int start = offsets[node];
    int end = offsets[node + 1];
    
    for (int i = start; i < end; i++) {
        int neighbor = edges[i];
        int label = labels[neighbor];
        label_count[label]++;
    }
    
    // Adopt most common neighbor label
    int max_count = 0, best_label = labels[node];
    for (int l = 0; l < MAX_LABELS; l++) {
        if (label_count[l] > max_count) {
            max_count = label_count[l];
            best_label = l;
        }
    }
    
    new_labels[node] = best_label;
}
```

---

## Part 8: Advanced Topics

### Topic 1: Dynamic Graphs

**Challenge:** Add/remove edges during computation

**Solution:** Use dynamic data structures
- Adjacency list with gaps and compaction
- Hash-based edge storage
- Versioned snapshots

**Library:** [Hornet](https://github.com/hornet-gt/hornet) (dynamic graph on GPU)

---

### Topic 2: Multi-GPU Graph Processing

**Challenge:** Graph doesn't fit in one GPU's memory

**Solution:** Partition graph across GPUs
- Edge cuts (minimize cross-GPU edges)
- Vertex cuts (replicate vertices, partition edges)
- Communication via NVLink or network

**Libraries:** 
- [Gunrock](https://gunrock.github.io/) (single-GPU)
- [CuGraph](https://github.com/rapidsai/cugraph) (multi-GPU)

---

### Topic 3: Graph Neural Networks (GNNs)

**Idea:** Learn node representations via message passing

```cuda
// GNN layer: Aggregate neighbor features
__global__ void gnnAggregation(const int* offsets, const int* edges,
                                const float* features, float* new_features) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Average neighbor features
    float sum[FEATURE_DIM] = {0};
    int count = 0;
    
    int start = offsets[node];
    int end = offsets[node + 1];
    
    for (int i = start; i < end; i++) {
        int neighbor = edges[i];
        for (int f = 0; f < FEATURE_DIM; f++) {
            sum[f] += features[neighbor * FEATURE_DIM + f];
        }
        count++;
    }
    
    // Normalize and apply activation
    for (int f = 0; f < FEATURE_DIM; f++) {
        new_features[node * FEATURE_DIM + f] = sum[f] / count;
    }
}
```

**Libraries:** PyTorch Geometric, DGL (Deep Graph Library)

---

