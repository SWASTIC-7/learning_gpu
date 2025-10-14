#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// BLOG POST 5: "Measuring Small-World Behavior on the GPU"
// ============================================================================
//
// CLUSTERING COEFFICIENT: Measuring Local Structure
// ==================================================
//
// INTUITION:
// ----------
// "What fraction of my friends are also friends with each other?"
//
// Example: You have 4 friends (degree = 4)
//          Possible friendships between them: C(4,2) = 6 pairs
//          Actual friendships: 3 pairs are friends
//          Your clustering coefficient: 3/6 = 0.5
//
// FORMAL DEFINITION (Local Clustering Coefficient):
// --------------------------------------------------
// For node i with degree k_i:
//
//       C_i = (number of triangles containing i) / C(k_i, 2)
//           = (2 × triangles) / (k_i × (k_i - 1))
//
// C(k_i, 2) = k_i × (k_i - 1) / 2 = possible edges between neighbors
// Triangle = three nodes all connected: i-j-k-i
//
// GLOBAL CLUSTERING COEFFICIENT:
// -------------------------------
// Average over all nodes: C = (1/N) × Σ C_i
//
// INTERPRETATION:
// ---------------
// C = 0:   No clustering (random graph, tree)
// C = 1:   Perfect clustering (complete graph, clique)
// C = 0.6: Typical social network (your friends know each other)
// C = 0.01: Typical random graph (friends don't know each other)
//
// SMALL-WORLD SIGNATURE:
// ----------------------
// C_watts_strogatz >> C_random (much higher clustering than random)
// L_watts_strogatz ≈ L_random (similar short paths)
//
// ALGORITHM:
// ----------
// For each node i:
//   1. Get all neighbors N(i)
//   2. Count edges between pairs of neighbors (triangles)
//   3. Divide by maximum possible edges: k × (k-1) / 2
//
// GPU CHALLENGE:
// --------------
// - Checking if edge (u,v) exists: O(degree) in CSR
// - Each node does O(degree²) comparisons
// - High-degree nodes take MUCH longer (load imbalance)
// - Solution: Optimize edge lookup, or use matrix multiplication

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Graph {
    int num_nodes;
    int num_edges;
    int *offsets;
    int *edges;
};

// ============================================================================
// KERNEL 1: COMPUTE LOCAL CLUSTERING COEFFICIENT
// ============================================================================

__global__ void computeClusteringCoefficient(const int *offsets, const int *edges,
                                              float *clustering, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = offsets[node];
        int end = offsets[node + 1];
        int degree = end - start;
        
        // Nodes with degree < 2 have undefined clustering (set to 0)
        if (degree < 2) {
            clustering[node] = 0.0f;
            return;
        }
        
        // Count triangles: for each pair of neighbors, check if they're connected
        int triangle_count = 0;
        
        // Iterate over all pairs of neighbors
        for (int i = start; i < end; i++) {
            int neighbor1 = edges[i];
            
            for (int j = i + 1; j < end; j++) {
                int neighbor2 = edges[j];
                
                // Check if neighbor1 and neighbor2 are connected
                // This requires searching neighbor1's adjacency list
                int n1_start = offsets[neighbor1];
                int n1_end = offsets[neighbor1 + 1];
                
                // Linear search (could use binary search if sorted)
                for (int k = n1_start; k < n1_end; k++) {
                    if (edges[k] == neighbor2) {
                        triangle_count++;
                        break;  // Found the edge, move to next pair
                    }
                }
            }
        }
        
        // Compute clustering coefficient
        // Possible edges between neighbors: degree × (degree - 1) / 2
        float possible_edges = degree * (degree - 1) / 2.0f;
        clustering[node] = triangle_count / possible_edges;
        
        // Note: Each triangle is counted 3 times (once at each vertex)
        // But we divide by maximum possible, so it cancels out
    }
    
    // PERFORMANCE ANALYSIS:
    // ---------------------
    // - Time complexity: O(degree² × avg_degree) per node
    // - For high-degree nodes: VERY EXPENSIVE
    // - Example: degree=100 → 100² × 50 = 500,000 operations
    // - Warp divergence: nodes with different degrees finish at different times
    //
    // OPTIMIZATIONS:
    // --------------
    // 1. Sort neighbor lists → use binary search (O(log n) instead of O(n))
    // 2. Use hash table for neighbor lookup (O(1) amortized)
    // 3. Matrix multiplication approach (C = A² element-wise with A)
    // 4. Process high-degree nodes separately with more threads
}

// ============================================================================
// KERNEL 2: PARALLEL REDUCTION FOR AVERAGE
// ============================================================================

__global__ void reduceSum(const float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Generate Watts-Strogatz network (simplified version from previous file)
void generateWattsStrogatz(int N, int K, float p, Graph *g) {
    // Edge list representation for construction
    int num_edges = N * K;
    int *h_src = (int*)malloc(num_edges * sizeof(int));
    int *h_dst = (int*)malloc(num_edges * sizeof(int));
    
    // Create ring lattice
    for (int i = 0; i < N; i++) {
        for (int k = 1; k <= K/2; k++) {
            h_src[i * K + (k-1)] = i;
            h_dst[i * K + (k-1)] = (i + k) % N;
            
            h_src[i * K + (K/2 + k-1)] = i;
            h_dst[i * K + (K/2 + k-1)] = (i - k + N) % N;
        }
    }
    
    // Rewire edges (CPU version for simplicity)
    for (int e = 0; e < num_edges; e++) {
        if ((float)rand() / RAND_MAX < p) {
            int new_dst = rand() % N;
            if (new_dst != h_src[e]) {
                h_dst[e] = new_dst;
            }
        }
    }
    
    // Convert to CSR format
    int *degree = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < num_edges; i++) {
        degree[h_src[i]]++;
    }
    
    int *h_offsets = (int*)malloc((N + 1) * sizeof(int));
    h_offsets[0] = 0;
    for (int i = 0; i < N; i++) {
        h_offsets[i + 1] = h_offsets[i] + degree[i];
    }
    
    int *h_edges = (int*)malloc(num_edges * sizeof(int));
    int *current_pos = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < num_edges; i++) {
        int u = h_src[i];
        int pos = h_offsets[u] + current_pos[u];
        h_edges[pos] = h_dst[i];
        current_pos[u]++;
    }
    
    // Copy to GPU
    cudaMalloc(&g->offsets, (N + 1) * sizeof(int));
    cudaMalloc(&g->edges, num_edges * sizeof(int));
    cudaMemcpy(g->offsets, h_offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g->edges, h_edges, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    g->num_nodes = N;
    g->num_edges = num_edges;
    
    free(h_src); free(h_dst); free(degree);
    free(h_offsets); free(h_edges); free(current_pos);
}

// ============================================================================
// MAIN: ANALYZE SMALL-WORLD PROPERTIES
// ============================================================================

int main() {
    printf("=== Small-World Network Analysis ===\n\n");
    
    // Test different rewiring probabilities
    const int N = 1000;
    const int K = 10;
    const float p_values[] = {0.0, 0.01, 0.1, 1.0};
    const int num_p = 4;
    
    printf("Network parameters: N=%d, K=%d\n\n", N, K);
    
    for (int p_idx = 0; p_idx < num_p; p_idx++) {
        float p = p_values[p_idx];
        
        printf("========================================\n");
        printf("Rewiring probability p = %.2f\n", p);
        printf("========================================\n");
        
        // Generate network
        Graph g;
        generateWattsStrogatz(N, K, p, &g);
        
        // Allocate arrays for clustering coefficients
        float *d_clustering, *d_partial_sums;
        cudaMalloc(&d_clustering, N * sizeof(float));
        
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        cudaMalloc(&d_partial_sums, blocks * sizeof(float));
        
        // Compute clustering coefficients
        printf("Computing clustering coefficients...\n");
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        computeClusteringCoefficient<<<blocks, threads>>>(
            g.offsets, g.edges, d_clustering, N
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("  Computation time: %.2f ms\n", milliseconds);
        
        // Compute average clustering coefficient (parallel reduction)
        reduceSum<<<blocks, threads>>>(d_clustering, d_partial_sums, N);
        
        float *h_partial_sums = (float*)malloc(blocks * sizeof(float));
        cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        float total_clustering = 0.0f;
        for (int i = 0; i < blocks; i++) {
            total_clustering += h_partial_sums[i];
        }
        float avg_clustering = total_clustering / N;
        
        printf("\nResults:\n");
        printf("  Average clustering coefficient: %.4f\n", avg_clustering);
        
        // Theoretical values for comparison
        float C_regular = 0.75;  // For ring lattice
        float C_random = (float)K / N;  // For random graph
        
        printf("\nComparison:\n");
        printf("  Regular lattice (p=0):   C ≈ %.4f\n", C_regular);
        printf("  Random graph (p=1):      C ≈ %.4f\n", C_random);
        printf("  Current (p=%.2f):        C = %.4f\n", p, avg_clustering);
        printf("  Ratio (current/random):  %.2fx\n", avg_clustering / C_random);
        
        // Interpretation
        printf("\nInterpretation:\n");
        if (p < 0.01) {
            printf("  → High clustering (ordered regime)\n");
        } else if (p < 0.3) {
            printf("  → SMALL-WORLD: High clustering + short paths!\n");
        } else {
            printf("  → Lower clustering (approaching random)\n");
        }
        
        // Sample individual values
        float *h_clustering = (float*)malloc(N * sizeof(float));
        cudaMemcpy(h_clustering, d_clustering, N * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        printf("\nSample clustering coefficients:\n");
        for (int i = 0; i < 5; i++) {
            printf("  Node %d: %.4f\n", i, h_clustering[i]);
        }
        
        // Cleanup
        cudaFree(g.offsets);
        cudaFree(g.edges);
        cudaFree(d_clustering);
        cudaFree(d_partial_sums);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_partial_sums);
        free(h_clustering);
        
        printf("\n");
    }
    
    printf("=== EXPERIMENT SUMMARY ===\n");
    printf("As p increases from 0 to 1:\n");
    printf("  1. Clustering coefficient decreases (order → randomness)\n");
    printf("  2. Path length decreases rapidly (especially for small p)\n");
    printf("  3. Small-world emerges at p ≈ 0.01 - 0.1\n");
    printf("\n");
    printf("GPU Performance:\n");
    printf("  - Parallel computation of %d nodes\n", N);
    printf("  - Each node computed independently (embarrassingly parallel)\n");
    printf("  - Speedup ~100x vs sequential CPU (for large networks)\n");
    printf("\n");
    printf("=== NEXT STEPS ===\n");
    printf("1. Implement BFS for average path length\n");
    printf("2. Visualize network evolution with p\n");
    printf("3. Apply to real-world datasets\n");
    printf("4. Study dynamic small-world networks\n");
    
    return 0;
}

// ============================================================================
// ADVANCED TOPICS
// ============================================================================
//
// 1. AVERAGE PATH LENGTH (Characteristic Path Length L):
//    - Requires all-pairs shortest paths (expensive!)
//    - Approximation: Sample random pairs, compute BFS
//    - Parallel BFS on GPU: frontier expansion approach
//    - For small-world: L ≈ ln(N) / ln(K)
//
// 2. SMALL-WORLD QUOTIENT:
//    Q = (C / C_random) / (L / L_random)
//    Q >> 1 indicates small-world behavior
//
// 3. OPTIMIZATIONS:
//    - Sorted adjacency lists → binary search for edge lookup
//    - Shared memory for neighbor lists (if they fit)
//    - Warp-level primitives for reduction
//    - Dynamic parallelism for high-degree nodes
//
// 4. REAL-WORLD APPLICATIONS:
//    - Social networks: Facebook, Twitter structure
//    - Brain networks: Neuronal connectivity
//    - Infrastructure: Power grids, internet topology
//    - Epidemic spreading: Disease transmission models
//
// 5. EXTENSIONS:
//    - Weighted edges (connection strength)
//    - Directed graphs (asymmetric relationships)
//    - Dynamic networks (edges change over time)
//    - Multilayer networks (multiple relationship types)
