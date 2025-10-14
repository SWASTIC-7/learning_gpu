#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// BLOG POST 3: "How to Represent Graphs for GPUs"
// ============================================================================
//
// GRAPH THEORY PRIMER:
// -------------------
// A graph G = (V, E) consists of:
// - V: Set of vertices (nodes) - think social network users, neurons, atoms
// - E: Set of edges (links) - connections between vertices
//
// Example: 4-node graph
//    0 --- 1
//    |     |
//    3 --- 2
//
// Adjacency: nodes 0-1, 0-3, 1-2, 2-3 are connected
//
// TWO MAIN REPRESENTATIONS:
// -------------------------

// 1. ADJACENCY MATRIX (Dense Representation)
// ------------------------------------------
// An N×N matrix where M[i][j] = 1 if edge (i,j) exists
//
// Pros:
// + O(1) edge lookup: "Does edge (u,v) exist?" → check M[u][v]
// + Regular memory access (good for GPU coalescing)
// + Easy to parallelize: each thread processes one row
//
// Cons:
// - O(N²) memory usage (huge for large sparse graphs!)
// - Wastes memory: most real-world networks are sparse (edges << N²)
//
// Example: The 4-node graph above as matrix
//     0  1  2  3
// 0 [ 0  1  0  1 ]
// 1 [ 1  0  1  0 ]
// 2 [ 0  1  0  1 ]
// 3 [ 1  0  1  0 ]
//
// Memory: 16 integers (4×4) to store 4 edges

// 2. ADJACENCY LIST (Sparse Representation) - CSR FORMAT
// -------------------------------------------------------
// Compressed Sparse Row (CSR) - industry standard for sparse graphs
//
// Three arrays:
// - offsets[N+1]: Starting index of each node's neighbors
// - edges[E]: Flattened list of all edges
// - (optional) weights[E]: Edge weights
//
// Example: Same 4-node graph
// offsets = [0, 2, 4, 6, 8]  ← node 0's neighbors start at index 0
// edges   = [1, 3, 0, 2, 1, 3, 0, 2]  ← node 0 connects to [1,3]
//
// To find node i's neighbors: edges[offsets[i] : offsets[i+1]]
// - Node 0: edges[0:2] = [1, 3]
// - Node 1: edges[2:4] = [0, 2]
// - Node 2: edges[4:6] = [1, 3]
// - Node 3: edges[6:8] = [0, 2]
//
// Pros:
// + O(N + E) memory (optimal for sparse graphs!)
// + Real-world networks: E ≈ 10×N (sparse), not N²
//
// Cons:
// - Irregular memory access (neighbors of node i not contiguous with node i+1)
// - Warp divergence: different nodes have different degrees
// - Edge lookup O(degree) instead of O(1)
//
// GPU CHALLENGES:
// - Threads processing adjacent nodes access non-adjacent memory
// - Load balancing: high-degree nodes take longer
// - Solution: Use CSR for storage, design algorithms to minimize divergence

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Graph stored in CSR format
struct Graph {
    int num_nodes;      // Number of vertices |V|
    int num_edges;      // Number of edges |E| (undirected: count each edge once)
    int *offsets;       // Size: num_nodes+1 (device pointer)
    int *edges;         // Size: num_edges (device pointer)
};

// ============================================================================
// ERDŐS-RÉNYI RANDOM GRAPH GENERATOR (CPU VERSION)
// ============================================================================
// G(n, p) model: n nodes, each possible edge exists with probability p
// Used as baseline to compare with Watts-Strogatz
//
// Properties:
// - Expected degree: (n-1) × p
// - Random → low clustering, short path length
// - NOT a small-world (we'll fix this with Watts-Strogatz!)

void generateErdosRenyiCPU(int n, float p, Graph* g) {
    // Count edges first (need to know size for CSR)
    int edge_count = 0;
    bool* adj_matrix = (bool*)calloc(n * n, sizeof(bool));
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((float)rand() / RAND_MAX < p) {
                adj_matrix[i * n + j] = true;
                adj_matrix[j * n + i] = true;  // Undirected
                edge_count += 2;  // Count both directions
            }
        }
    }
    
    printf("Generated %d edges (expected: ~%.0f)\n", 
           edge_count / 2, n * (n-1) / 2.0 * p);
    
    // Build CSR representation
    int* h_offsets = (int*)malloc((n + 1) * sizeof(int));
    int* h_edges = (int*)malloc(edge_count * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < n; i++) {
        h_offsets[i] = offset;
        for (int j = 0; j < n; j++) {
            if (adj_matrix[i * n + j]) {
                h_edges[offset++] = j;
            }
        }
    }
    h_offsets[n] = offset;
    
    // Copy to GPU
    cudaMalloc(&g->offsets, (n + 1) * sizeof(int));
    cudaMalloc(&g->edges, edge_count * sizeof(int));
    cudaMemcpy(g->offsets, h_offsets, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g->edges, h_edges, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    
    g->num_nodes = n;
    g->num_edges = edge_count;
    
    free(adj_matrix);
    free(h_offsets);
    free(h_edges);
}

// ============================================================================
// GPU KERNELS: BASIC GRAPH OPERATIONS
// ============================================================================

// Kernel 1: Compute degree of each node
// Degree = number of neighbors
__global__ void computeDegrees(const int* offsets, int* degrees, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        // Degree = number of edges = difference in offsets
        degrees[node] = offsets[node + 1] - offsets[node];
    }
    
    // Memory access: Coalesced for offsets
    // - Thread 0 reads offsets[0], offsets[1]
    // - Thread 1 reads offsets[1], offsets[2]
    // - Adjacent threads read adjacent memory = fast!
}

// Kernel 2: Print neighbors of each node (debugging)
__global__ void printNeighbors(const int* offsets, const int* edges, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = offsets[node];
        int end = offsets[node + 1];
        int degree = end - start;
        
        printf("Node %d (degree %d): ", node, degree);
        for (int i = start; i < end; i++) {
            printf("%d ", edges[i]);
        }
        printf("\n");
    }
    
    // Warp divergence alert!
    // - Nodes with different degrees loop different amounts
    // - Within a warp, some threads finish early and idle
    // - Unavoidable for irregular graphs
}

// Kernel 3: Count triangles (for clustering coefficient later)
// Triangle: three nodes all connected to each other
// Used to measure clustering in networks
__global__ void countTriangles(const int* offsets, const int* edges, 
                                int* triangles, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int count = 0;
        int start = offsets[node];
        int end = offsets[node + 1];
        
        // For each pair of neighbors (u, v) of this node
        for (int i = start; i < end; i++) {
            int u = edges[i];
            for (int j = i + 1; j < end; j++) {
                int v = edges[j];
                
                // Check if u and v are connected (forms triangle)
                int u_start = offsets[u];
                int u_end = offsets[u + 1];
                
                for (int k = u_start; k < u_end; k++) {
                    if (edges[k] == v) {
                        count++;
                        break;
                    }
                }
            }
        }
        
        triangles[node] = count;
    }
    
    // This is O(degree³) per node - expensive!
    // Better algorithms exist (matrix multiplication approach)
    // But demonstrates graph traversal on GPU
}

// ============================================================================
// MAIN: DEMO
// ============================================================================

int main() {
    printf("=== Graph Representations on GPU ===\n\n");
    
    const int N = 10;        // Small graph for demonstration
    const float P = 0.3;     // 30%% edge probability
    
    // Generate random graph
    printf("Generating Erdős-Rényi graph: %d nodes, p=%.2f\n", N, P);
    Graph g;
    generateErdosRenyiCPU(N, P, &g);
    
    // Allocate output arrays
    int *d_degrees, *d_triangles;
    cudaMalloc(&d_degrees, N * sizeof(int));
    cudaMalloc(&d_triangles, N * sizeof(int));
    
    // Compute degrees
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    computeDegrees<<<blocks, threads>>>(g.offsets, d_degrees, N);
    
    // Print graph structure
    printf("\nGraph structure:\n");
    printNeighbors<<<blocks, threads>>>(g.offsets, g.edges, N);
    cudaDeviceSynchronize();
    
    // Count triangles
    countTriangles<<<blocks, threads>>>(g.offsets, g.edges, d_triangles, N);
    
    // Copy results back
    int *h_degrees = (int*)malloc(N * sizeof(int));
    int *h_triangles = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_degrees, d_degrees, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_triangles, d_triangles, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Statistics
    printf("\nDegree distribution:\n");
    int total_degree = 0;
    for (int i = 0; i < N; i++) {
        printf("Node %d: degree=%d, triangles=%d\n", i, h_degrees[i], h_triangles[i]);
        total_degree += h_degrees[i];
    }
    printf("\nAverage degree: %.2f\n", total_degree / (float)N);
    
    // Cleanup
    cudaFree(g.offsets);
    cudaFree(g.edges);
    cudaFree(d_degrees);
    cudaFree(d_triangles);
    free(h_degrees);
    free(h_triangles);
    
    printf("\n=== KEY TAKEAWAYS ===\n");
    printf("1. CSR format: O(N+E) memory vs O(N²) for matrix\n");
    printf("2. Irregular access patterns cause warp divergence\n");
    printf("3. Graph algorithms inherently have load imbalance\n");
    printf("4. Next: Watts-Strogatz model for small-world networks!\n");
    
    return 0;
}

// ============================================================================
// PERFORMANCE CONSIDERATIONS
// ============================================================================
//
// Memory Bandwidth:
// - Adjacency matrix: predictable, coalesced, but wastes bandwidth
// - CSR: unpredictable, but only loads necessary data
// - For sparse graphs (E << N²): CSR wins
//
// Warp Divergence:
// - Unavoidable in graph algorithms (different degrees)
// - Mitigation: Sort nodes by degree, process similar degrees together
// - Advanced: Use warp-centric programming (distribute work within warps)
//
// Load Balancing:
// - Some nodes have degree 2, others degree 1000
// - Solution: Dynamic parallelism or virtual warp assignment
// - Assign multiple threads to high-degree nodes
//
// Cache Efficiency:
// - Neighbor lists are not locality-friendly
// - Reordering nodes (graph coloring, BFS order) can help
// - Shared memory cannot help much (neighbor sets too large)
//
// When to use each representation:
// - Matrix: Dense graphs (E > 0.1×N²), or need O(1) edge queries
// - CSR: Sparse graphs (most real-world networks)
// - Hybrid: Store high-degree nodes differently than low-degree
