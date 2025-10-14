#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

// ============================================================================
// BLOG POST 4: "Generating Small-World Networks on the GPU"
// ============================================================================
//
// THE WATTS-STROGATZ MODEL (1998)
// ================================
// 
// MOTIVATION: Real-world networks are neither random nor regular
// ------------
// - Social networks: You know your neighbors AND their friends (clustering)
// - But also: "6 degrees of separation" (short paths to everyone)
// - Random graphs: Short paths ✓, but NO clustering ✗
// - Regular lattices: High clustering ✓, but LONG paths ✗
// - Watts-Strogatz: BOTH! ✓✓ → "Small-World" property
//
// THE ALGORITHM:
// --------------
// Parameters: N (nodes), K (initial neighbors), p (rewiring probability)
//
// Step 1: Create RING LATTICE
//         Connect each node to K nearest neighbors (K/2 on each side)
//         Example: K=4, node 5 connects to [3, 4, 6, 7]
//
//         Visual (N=8, K=4):
//              0
//            /   \
//          7       1
//         |         |
//        6           2
//         |         |
//          5       3
//            \   /
//              4
//
//         High clustering (your friends know each other)
//         Long paths (need ~N/2 hops to reach opposite side)
//
// Step 2: REWIRE edges with probability p
//         For each edge (u, v):
//         - With probability p: disconnect v, reconnect to random node w
//         - With probability 1-p: keep edge as-is
//
//         Example: Node 0-1 edge rewired to 0-5 (random)
//              0 -----------> 5
//            /   \          /
//          7       1  -->  1
//                            
//         Creates "shortcuts" across the network
//
// Step 3: Result depends on p:
//         p = 0:   Regular ring (high clustering, long paths)
//         p = 0.01: Small-world! (high clustering, short paths) ← sweet spot
//         p = 1:   Random graph (low clustering, short paths)
//
// WHY IT MATTERS:
// ---------------
// - Models: Social networks, neural connections, internet topology
// - Applications: Disease spreading, information diffusion, resilience
// - Theory: Phase transition from order to randomness
//
// KEY METRICS:
// ------------
// 1. Clustering Coefficient (C): 
//    Fraction of your friends who are also friends with each other
//    C_lattice ≈ 0.75, C_random ≈ K/N → 0
//
// 2. Characteristic Path Length (L):
//    Average shortest path between all pairs
//    L_lattice ≈ N/(2K), L_random ≈ ln(N)/ln(K)
//
// Small-world: C ≈ C_lattice (high) AND L ≈ L_random (low)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Graph {
    int num_nodes;
    int num_edges;
    int *offsets;    // CSR format (device)
    int *edges;      // CSR format (device)
};

// Edge list representation (easier for construction)
struct EdgeList {
    int *src;        // Source nodes
    int *dst;        // Destination nodes
    int count;
};

// ============================================================================
// KERNEL 1: INITIALIZE RING LATTICE
// ============================================================================
// Each thread handles one node, creates K connections to nearest neighbors

__global__ void createRingLattice(int *src, int *dst, int N, int K) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < N) {
        // Each node connects to K/2 neighbors on each side
        int half_K = K / 2;
        
        // Create edges to K nearest neighbors
        for (int k = 1; k <= half_K; k++) {
            // Right neighbor (clockwise)
            int right = (node + k) % N;  // Wrap around with modulo
            int edge_idx = node * K + (k - 1);
            src[edge_idx] = node;
            dst[edge_idx] = right;
            
            // Left neighbor (counter-clockwise)
            int left = (node - k + N) % N;  // +N to handle negative
            edge_idx = node * K + (half_K + k - 1);
            src[edge_idx] = node;
            dst[edge_idx] = left;
        }
    }
    
    // Memory access: Each thread writes to its own region (coalesced)
    // No conflicts between threads
    // Work per thread: O(K) operations (balanced, K is small)
}

// ============================================================================
// KERNEL 2: REWIRE EDGES
// ============================================================================
// Each thread processes one edge, rewires with probability p

__global__ void rewireEdges(int *src, int *dst, int num_edges, int N, 
                            float p, unsigned long long seed) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx < num_edges) {
        // Initialize random number generator (one per thread)
        curandState state;
        curand_init(seed, edge_idx, 0, &state);
        
        // Decide whether to rewire this edge
        float random_val = curand_uniform(&state);  // Random in [0, 1)
        
        if (random_val < p) {
            // REWIRE: Keep source, pick new random destination
            int u = src[edge_idx];
            int v = dst[edge_idx];
            
            // Pick random node (but not self, and not duplicate)
            int new_v;
            int attempts = 0;
            bool valid = false;
            
            // Try to find valid target (avoid self-loops and duplicates)
            while (!valid && attempts < 100) {
                new_v = curand(&state) % N;  // Random node
                
                // Check validity
                if (new_v != u) {
                    // In production: check for duplicate edges
                    // For simplicity: accept any non-self connection
                    valid = true;
                }
                attempts++;
            }
            
            if (valid) {
                dst[edge_idx] = new_v;
            }
            // If no valid target found, keep original edge
        }
        // Else: keep edge as-is (probability 1-p)
    }
    
    // WARP DIVERGENCE ALERT:
    // - Threads with random_val < p execute rewiring (complex path)
    // - Threads with random_val >= p do nothing (fast path)
    // - Within warp, both paths execute sequentially
    // - For p=0.01: only 1/100 edges rewire, lots of wasted cycles
    //
    // OPTIMIZATION: Could batch rewiring (separate kernel for edges to rewire)
    // But for simplicity, we accept some inefficiency
}

// ============================================================================
// HELPER: CONVERT EDGE LIST TO CSR
// ============================================================================

void edgeListToCSR(int *h_src, int *h_dst, int num_edges, int N, Graph *g) {
    // Count degree of each node
    int *degree = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < num_edges; i++) {
        degree[h_src[i]]++;
    }
    
    // Build offsets (prefix sum of degrees)
    int *offsets = (int*)malloc((N + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < N; i++) {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    
    // Fill edges array
    int *edges = (int*)malloc(num_edges * sizeof(int));
    int *current_pos = (int*)calloc(N, sizeof(int));
    
    for (int i = 0; i < num_edges; i++) {
        int u = h_src[i];
        int pos = offsets[u] + current_pos[u];
        edges[pos] = h_dst[i];
        current_pos[u]++;
    }
    
    // Copy to GPU
    cudaMalloc(&g->offsets, (N + 1) * sizeof(int));
    cudaMalloc(&g->edges, num_edges * sizeof(int));
    cudaMemcpy(g->offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g->edges, edges, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    g->num_nodes = N;
    g->num_edges = num_edges;
    
    free(degree);
    free(offsets);
    free(edges);
    free(current_pos);
}

// ============================================================================
// MAIN: GENERATE WATTS-STROGATZ NETWORK
// ============================================================================

int main() {
    printf("=== Watts-Strogatz Small-World Network Generator ===\n\n");
    
    // PARAMETERS
    const int N = 1000;      // Number of nodes
    const int K = 10;        // Each node connects to K nearest neighbors
    const float p = 0.05;    // Rewiring probability (5%%)
    
    printf("Parameters: N=%d, K=%d, p=%.3f\n", N, K, p);
    printf("Expected: High clustering + Short paths\n\n");
    
    if (K % 2 != 0 || K >= N) {
        printf("Error: K must be even and K < N\n");
        return 1;
    }
    
    // STEP 1: CREATE RING LATTICE ON GPU
    printf("Step 1: Creating ring lattice...\n");
    int num_edges = N * K;  // Each node has K edges
    
    int *d_src, *d_dst;
    cudaMalloc(&d_src, num_edges * sizeof(int));
    cudaMalloc(&d_dst, num_edges * sizeof(int));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    createRingLattice<<<blocks, threads>>>(d_src, d_dst, N, K);
    cudaDeviceSynchronize();
    printf("  Created %d edges (ring lattice)\n", num_edges);
    
    // STEP 2: REWIRE EDGES WITH PROBABILITY p
    printf("\nStep 2: Rewiring edges (p=%.3f)...\n", p);
    unsigned long long seed = time(NULL);
    
    int edge_blocks = (num_edges + threads - 1) / threads;
    rewireEdges<<<edge_blocks, threads>>>(d_src, d_dst, num_edges, N, p, seed);
    cudaDeviceSynchronize();
    printf("  Expected rewired edges: ~%d\n", (int)(num_edges * p));
    
    // STEP 3: CONVERT TO CSR FORMAT
    printf("\nStep 3: Converting to CSR format...\n");
    int *h_src = (int*)malloc(num_edges * sizeof(int));
    int *h_dst = (int*)malloc(num_edges * sizeof(int));
    
    cudaMemcpy(h_src, d_src, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dst, d_dst, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    
    Graph g;
    edgeListToCSR(h_src, h_dst, num_edges, N, &g);
    printf("  CSR graph ready on GPU\n");
    
    // STEP 4: ANALYZE NETWORK PROPERTIES
    printf("\n=== Network Properties ===\n");
    printf("Nodes: %d\n", g.num_nodes);
    printf("Edges: %d\n", g.num_edges);
    printf("Average degree: %.2f\n", (2.0 * g.num_edges) / g.num_nodes);
    
    // Sample some nodes
    printf("\nSample connections:\n");
    int *h_offsets = (int*)malloc((N + 1) * sizeof(int));
    int *h_edges = (int*)malloc(num_edges * sizeof(int));
    cudaMemcpy(h_offsets, g.offsets, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edges, g.edges, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 3; i++) {
        int start = h_offsets[i];
        int end = h_offsets[i + 1];
        printf("Node %d connects to: ", i);
        for (int j = start; j < end && j < start + 10; j++) {
            printf("%d ", h_edges[j]);
        }
        if (end - start > 10) printf("...");
        printf("\n");
    }
    
    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(g.offsets);
    cudaFree(g.edges);
    free(h_src);
    free(h_dst);
    free(h_offsets);
    free(h_edges);
    
    printf("\n=== SUCCESS ===\n");
    printf("Next: Compute clustering coefficient (8_clustering_coefficient.cu)\n");
    
    return 0;
}

// ============================================================================
// THEORETICAL ANALYSIS
// ============================================================================
//
// PHASE TRANSITION AS p VARIES:
// ------------------------------
// p = 0:     Regular lattice
//            C(0) ≈ 3/4 (high clustering)
//            L(0) ≈ N/(2K) (long paths)
//
// p → 0.01:  SMALL-WORLD REGIME ← sweet spot!
//            C(p) ≈ C(0) × (1-p)³ (still high!)
//            L(p) drops dramatically (shortcuts create fast paths)
//
// p → 1:     Random graph
//            C(1) ≈ K/N → 0 (low clustering)
//            L(1) ≈ ln(N)/ln(K) (short paths)
//
// Why small-world emerges:
// - First few rewired edges create "long-range shortcuts"
// - Drastically reduce average path length
// - But most edges still local → clustering remains high
// - Magic happens at p ≈ 0.01-0.1
//
// COMPUTATIONAL COMPLEXITY:
// -------------------------
// Sequential: O(NK) to build lattice + O(NK) to rewire = O(NK)
// Parallel GPU: 
//   - Build: O(K) per node × N nodes / P threads = O(NK/P)
//   - Rewire: O(1) per edge × NK edges / P threads = O(NK/P)
//   - Speedup: ~P (linear with thread count, up to memory bandwidth limit)
//
// For N=1M, K=10, P=10K threads: ~100× faster than CPU
