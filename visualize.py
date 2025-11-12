import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

# Configuration
P_VALUES = [0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0]

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
    
    # Layout based on p value
    if p < 0.01:
        pos = nx.circular_layout(G)  # Ring for p=0
    elif p > 0.9:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)  # Random
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)  # Small-world
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue', 
                           edgecolors='black', linewidths=0.5, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Title with metrics
    plt.title(f'Watts-Strogatz Network (p={p:.3f})\n'
              f'Clustering: {metrics["avg_clustering"]:.4f} | '
              f'Avg Degree: {metrics["avg_degree"]:.2f}',
              fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'output/network_p{p:.3f}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved network visualization for p={p:.3f}')

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
    plt.plot(p_vals, clustering_vals, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Rewiring Probability (p)', fontsize=12)
    plt.ylabel('Average Clustering Coefficient', fontsize=12)
    plt.title('Small-World Phase Transition', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate regions
    plt.axvspan(0, 0.1, alpha=0.2, color='green', label='Small-World Regime')
    plt.axvspan(0.8, 1.0, alpha=0.2, color='red', label='Random Regime')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('output/phase_transition.png', dpi=150)
    plt.close()
    
    print('✓ Saved phase transition plot')

def create_comparison_panel():
    """Create 3×3 panel showing different p values"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    p_samples = [0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889]
    
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
                                   edgecolors='black', linewidths=0.3, ax=ax, alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.3, ax=ax, edge_color='gray')
            
            ax.set_title(f'p = {p:.2f}\nC = {metrics["avg_clustering"]:.3f}',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f'No data\nfor p={p:.2f}',
                        ha='center', va='center', fontsize=12)
            axes[i].axis('off')
    
    plt.suptitle('Small-World Network Evolution', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('output/comparison_panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print('✓ Saved comparison panel')

def plot_degree_distribution():
    """Plot degree distribution for different p values"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    p_samples = [0.0, 0.111, 0.556, 1.0]
    
    for i, p in enumerate(p_samples):
        try:
            filename = f'output/degrees_p{p:.3f}.csv'
            df = pd.read_csv(filename)
            
            ax = axes[i]
            ax.hist(df['degree'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Degree', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'p = {p:.2f}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f'No data for p={p:.2f}',
                        ha='center', va='center', fontsize=12)
    
    plt.suptitle('Degree Distribution Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/degree_distributions.png', dpi=150)
    plt.close()
    
    print('✓ Saved degree distribution plot')

def main():
    """Main visualization pipeline"""
    print("=" * 50)
    print("Watts-Strogatz Network Visualization")
    print("=" * 50)
    
    # Check if output directory exists
    if not os.path.exists('output'):
        print("Error: 'output' directory not found!")
        print("Please run the CUDA program first to generate data.")
        return
    
    # Generate individual network visualizations
    print("\n[1/4] Generating individual network visualizations...")
    for p in P_VALUES:
        try:
            visualize_single_network(p)
        except FileNotFoundError:
            print(f'  ⚠ Skipping p={p:.3f} (no data)')
    
    # Generate phase transition plot
    print("\n[2/4] Generating phase transition plot...")
    try:
        plot_phase_transition()
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Generate comparison panel
    print("\n[3/4] Generating comparison panel...")
    try:
        create_comparison_panel()
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Generate degree distribution plots
    print("\n[4/4] Generating degree distribution plots...")
    try:
        plot_degree_distribution()
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETE")
    print("=" * 50)
    print("Generated files in output/ directory:")
    print("  • network_p*.png - Individual network layouts")
    print("  • phase_transition.png - Clustering vs p plot")
    print("  • comparison_panel.png - 3×3 comparison grid")
    print("  • degree_distributions.png - Degree histograms")
    print("=" * 50)

if __name__ == '__main__':
    main()
