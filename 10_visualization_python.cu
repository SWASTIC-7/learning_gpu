#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// BLOG POST 7: "Visualizing GPU-Generated Small Worlds"
// ============================================================================
//
// GOAL: Bridge GPU computation with Python visualization
// APPROACH: Export graph data → Load in Python → Visualize with NetworkX
//
// WORKFLOW:
// 1. Generate Watts-Strogatz network on GPU (CUDA)
// 2. Export edge list to CSV file
// 3. Load in Python (NetworkX)
// 4. Compute metrics and visualize
// 5. Create interactive dashboard (varying p parameter)
//
// WHY THIS MATTERS:
// - GPU: Fast computation (millions of edges/second)
// - Python: Easy visualization, analysis, sharing
// - Together: Best of both worlds!

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Graph {
    int num_nodes;
    int num_edges;
    int *offsets;  // CSR format
    int *edges;
};

struct NetworkMetrics {
    float avg_clustering;
    float avg_path_length;
    int *degree_distribution;
    float diameter;
};

// ============================================================================
// WATTS-STROGATZ GENERATOR (Optimized Version)
// ============================================================================

__global__ void createRingLattice(int *src, int *dst, int N, int K) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < N) {
        int half_K = K / 2;
        for (int k = 1; k <= half_K; k++) {
            int idx_right = node * K + (k - 1);
            src[idx_right] = node;
            dst[idx_right] = (node + k) % N;
            
            int idx_left = node * K + (half_K + k - 1);
            src[idx_left] = node;
            dst[idx_left] = (node - k + N) % N;
        }
    }
}

__global__ void rewireEdges(int *src, int *dst, int num_edges, int N, 
                            float p, unsigned long long seed) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx < num_edges) {
        curandState state;
        curand_init(seed, edge_idx, 0, &state);
        
        if (curand_uniform(&state) < p) {
            int u = src[edge_idx];
            int new_v;
            int attempts = 0;
            
            do {
                new_v = curand(&state) % N;
                attempts++;
            } while (new_v == u && attempts < 100);
            
            if (new_v != u) {
                dst[edge_idx] = new_v;
            }
        }
    }
}

// ============================================================================
// CLUSTERING COEFFICIENT (Optimized with early termination)
// ============================================================================

__global__ void computeClusteringOptimized(const int *offsets, const int *edges,
                                           float *clustering, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = offsets[node];
        int end = offsets[node + 1];
        int degree = end - start;
        
        if (degree < 2) {
            clustering[node] = 0.0f;
            return;
        }
        
        int triangle_count = 0;
        
        // Optimized: Use sorted neighbor lists for binary search
        for (int i = start; i < end; i++) {
            int neighbor1 = edges[i];
            
            for (int j = i + 1; j < end; j++) {
                int neighbor2 = edges[j];
                
                // Check if edge exists
                int n1_start = offsets[neighbor1];
                int n1_end = offsets[neighbor1 + 1];
                
                // Linear search (could be binary search if sorted)
                for (int k = n1_start; k < n1_end; k++) {
                    if (edges[k] == neighbor2) {
                        triangle_count++;
                        break;
                    }
                    // Early termination if we've passed possible match
                    if (edges[k] > neighbor2) break;
                }
            }
        }
        
        float possible = degree * (degree - 1) / 2.0f;
        clustering[node] = triangle_count / possible;
    }
}

// ============================================================================
// DEGREE DISTRIBUTION
// ============================================================================

__global__ void computeDegreeDistribution(const int *offsets, int *degrees, 
                                          int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < num_nodes) {
        degrees[node] = offsets[node + 1] - offsets[node];
    }
}

// ============================================================================
// FILE EXPORT FUNCTIONS
// ============================================================================

void exportEdgeList(const char *filename, int *h_src, int *h_dst, int num_edges) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(f, "source,target\n");
    for (int i = 0; i < num_edges; i++) {
        fprintf(f, "%d,%d\n", h_src[i], h_dst[i]);
    }
    
    fclose(f);
    printf("Exported %d edges to %s\n", num_edges, filename);
}

void exportMetrics(const char *filename, float p, float avg_clustering, 
                   int *degree_dist, int num_nodes) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "p,avg_clustering,avg_degree\n");
    
    // Compute average degree
    int total_degree = 0;
    for (int i = 0; i < num_nodes; i++) {
        total_degree += degree_dist[i];
    }
    float avg_degree = total_degree / (float)num_nodes;
    
    fprintf(f, "%.4f,%.4f,%.2f\n", p, avg_clustering, avg_degree);
    
    fclose(f);
}

void exportDegreeDistribution(const char *filename, int *degree_dist, int num_nodes) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "node,degree\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(f, "%d,%d\n", i, degree_dist[i]);
    }
    
    fclose(f);
}

// ============================================================================
// HELPER: CSR CONVERSION
// ============================================================================

void edgeListToCSR(int *h_src, int *h_dst, int num_edges, int N, Graph *g) {
    int *degree = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < num_edges; i++) {
        degree[h_src[i]]++;
    }
    
    int *offsets = (int*)malloc((N + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < N; i++) {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    
    int *edges = (int*)malloc(num_edges * sizeof(int));
    int *current_pos = (int*)calloc(N, sizeof(int));
    
    for (int i = 0; i < num_edges; i++) {
        int u = h_src[i];
        int pos = offsets[u] + current_pos[u];
        edges[pos] = h_dst[i];
        current_pos[u]++;
    }
    
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
// MAIN: GENERATE DATA FOR VISUALIZATION
// ============================================================================

int main(int argc, char **argv) {
    printf("=== Watts-Strogatz Network Generator & Exporter ===\n\n");
    
    // Parameters (can be command-line args)
    const int N = 500;      // Network size (visualizable)
    const int K = 6;        // Average degree
    const int num_p_values = 10;
    
    printf("Network parameters:\n");
    printf("  Nodes (N): %d\n", N, K);
    printf("  Degree (K): %d\n", K);
    printf("  Testing p from 0.0 to 1.0\n\n");
    
    // Create output directory (you may need to create this manually)
    system("mkdir -p output");
    
    // Generate networks for different p values
    for (int p_idx = 0; p_idx < num_p_values; p_idx++) {
        float p = p_idx / (float)(num_p_values - 1);  // 0.0 to 1.0
        
        printf("----------------------------------------\n");
        printf("Generating network with p = %.3f\n", p);
        printf("----------------------------------------\n");
        
        // Step 1: Generate edge list
        int num_edges = N * K;
        int *d_src, *d_dst;
        cudaMalloc(&d_src, num_edges * sizeof(int));
        cudaMalloc(&d_dst, num_edges * sizeof(int));
        
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        
        createRingLattice<<<blocks, threads>>>(d_src, d_dst, N, K);
        cudaDeviceSynchronize();
        
        // Step 2: Rewire
        unsigned long long seed = time(NULL) + p_idx;
        int edge_blocks = (num_edges + threads - 1) / threads;
        rewireEdges<<<edge_blocks, threads>>>(d_src, d_dst, num_edges, N, p, seed);
        cudaDeviceSynchronize();
        
        // Step 3: Copy to host
        int *h_src = (int*)malloc(num_edges * sizeof(int));
        int *h_dst = (int*)malloc(num_edges * sizeof(int));
        cudaMemcpy(h_src, d_src, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dst, d_dst, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Step 4: Export edge list
        char filename[256];
        sprintf(filename, "output/edges_p%.3f.csv", p);
        exportEdgeList(filename, h_src, h_dst, num_edges);
        
        // Step 5: Convert to CSR for analysis
        Graph g;
        edgeListToCSR(h_src, h_dst, num_edges, N, &g);
        
        // Step 6: Compute metrics
        float *d_clustering;
        int *d_degrees;
        cudaMalloc(&d_clustering, N * sizeof(float));
        cudaMalloc(&d_degrees, N * sizeof(int));
        
        computeClusteringOptimized<<<blocks, threads>>>(
            g.offsets, g.edges, d_clustering, N
        );
        computeDegreeDistribution<<<blocks, threads>>>(
            g.offsets, d_degrees, N
        );
        cudaDeviceSynchronize();
        
        // Copy results
        float *h_clustering = (float*)malloc(N * sizeof(float));
        int *h_degrees = (int*)malloc(N * sizeof(int));
        cudaMemcpy(h_clustering, d_clustering, N * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_degrees, d_degrees, N * sizeof(int), 
                   cudaMemcpyDeviceToHost);
        
        // Compute average clustering
        float total_clustering = 0.0f;
        for (int i = 0; i < N; i++) {
            total_clustering += h_clustering[i];
        }
        float avg_clustering = total_clustering / N;
        
        printf("  Average clustering: %.4f\n", avg_clustering);
        
        // Step 7: Export metrics
        sprintf(filename, "output/metrics_p%.3f.csv", p);
        exportMetrics(filename, p, avg_clustering, h_degrees, N);
        
        sprintf(filename, "output/degrees_p%.3f.csv", p);
        exportDegreeDistribution(filename, h_degrees, N);
        
        // Cleanup
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(g.offsets);
        cudaFree(g.edges);
        cudaFree(d_clustering);
        cudaFree(d_degrees);
        free(h_src);
        free(h_dst);
        free(h_clustering);
        free(h_degrees);
        
        printf("  Exported data for p=%.3f\n\n", p);
    }
    
    printf("=== DATA GENERATION COMPLETE ===\n\n");
    printf("Files created in ./output/ directory:\n");
    printf("  - edges_p*.csv: Edge lists for each p value\n");
    printf("  - metrics_p*.csv: Network metrics\n");
    printf("  - degrees_p*.csv: Degree distributions\n\n");
    
    printf("Next steps:\n");
    printf("  1. Run Python visualization script: python visualize.py\n");
    printf("  2. Open interactive dashboard\n");
    printf("  3. Explore small-world transition!\n\n");
    
    return 0;
}

// ============================================================================
// PYTHON VISUALIZATION SCRIPT (Save as visualize.py)
// ============================================================================
/*
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
import glob

# Configuration
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def load_graph(p):
    """Load graph from CSV file"""
    filename = f'output/edges_p{p:.3f}.csv'
    df = pd.read_csv(filename)
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    return G

def load_metrics(p):
    """Load metrics from CSV"""
    filename = f'output/metrics_p{p:.3f}.csv'
    df = pd.read_csv(filename)
    return df.iloc[0]

def visualize_single_network(p):
    """Visualize a single network"""
    G = load_graph(p)
    metrics = load_metrics(p)
    
    plt.figure(figsize=(12, 10))
    
    # Layout
    if p < 0.01:
        pos = nx.circular_layout(G)  # Ring for p=0
    elif p > 0.9:
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # Random for p=1
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=100)  # Small-world
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue', 
                           edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    
    # Title with metrics
    plt.title(f'Watts-Strogatz Network (p={p:.3f})\n'
              f'Clustering: {metrics["avg_clustering"]:.4f} | '
              f'Avg Degree: {metrics["avg_degree"]:.2f}',
              fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'output/network_p{p:.3f}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved network visualization for p={p:.3f}')

def plot_phase_transition():
    """Plot clustering coefficient vs p (phase transition)"""
    p_vals = []
    clustering_vals = []
    
    for p in P_VALUES:
        try:
            metrics = load_metrics(p)
            p_vals.append(p)
            clustering_vals.append(metrics['avg_clustering'])
        except FileNotFoundError:
            continue
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_vals, clustering_vals, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Rewiring Probability (p)', fontsize=12)
    plt.ylabel('Average Clustering Coefficient', fontsize=12)
    plt.title('Small-World Phase Transition', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate regions
    plt.axvspan(0, 0.1, alpha=0.2, color='green', label='Small-World')
    plt.axvspan(0.8, 1.0, alpha=0.2, color='red', label='Random')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/phase_transition.png', dpi=150)
    plt.close()
    
    print('Saved phase transition plot')

def create_comparison_panel():
    """Create 3x3 panel showing different p values"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    p_samples = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    for i, p in enumerate(p_samples):
        if i >= len(axes):
            break
            
        try:
            G = load_graph(p)
            metrics = load_metrics(p)
            
            # Choose layout based on p
            if p < 0.01:
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            
            ax = axes[i]
            nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue',
                                   edgecolors='black', linewidths=0.3, ax=ax)
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.3, ax=ax)
            
            ax.set_title(f'p = {p:.2f}\nC = {metrics["avg_clustering"]:.3f}',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f'No data\nfor p={p:.2f}',
                        ha='center', va='center', fontsize=12)
            axes[i].axis('off')
    
    plt.suptitle('Small-World Network Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/comparison_panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print('Saved comparison panel')

def main():
    """Main visualization pipeline"""
    print("=== Watts-Strogatz Network Visualization ===\n")
    
    # Generate individual network visualizations
    print("Generating network visualizations...")
    for p in P_VALUES:
        try:
            visualize_single_network(p)
        except FileNotFoundError:
            print(f'  Skipping p={p:.3f} (no data)')
    
    # Generate phase transition plot
    print("\nGenerating phase transition plot...")
    plot_phase_transition()
    
    # Generate comparison panel
    print("\nGenerating comparison panel...")
    create_comparison_panel()
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print("Check the output/ directory for PNG files!")

if __name__ == '__main__':
    main()
*/
