"""
Effective Connectivity Graph Visualization

Shows which neurons are co-active during processing of each digit class,
revealing functional connectivity patterns in the network.

Outputs PDF figures with modern styling.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from config import Config

from viz_utils import (
    setup_network, load_mnist_data, select_balanced_examples,
    load_assignments, get_digit_color,
    setup_modern_style, save_figure
)


def compute_coactivation_matrix(spike_indices_list, n_neurons, time_window=0.01):
    """Compute co-activation matrix from spike data.

    Neurons are considered co-active if they spike within the same time window.
    """
    coactivation = np.zeros((n_neurons, n_neurons))

    for spike_times, spike_indices in spike_indices_list:
        if len(spike_times) == 0:
            continue

        max_time = max(spike_times) + time_window
        n_bins = int(max_time / time_window) + 1

        for bin_idx in range(n_bins):
            t_start = bin_idx * time_window
            t_end = (bin_idx + 1) * time_window

            mask = (spike_times >= t_start) & (spike_times < t_end)
            active_neurons = np.unique(spike_indices[mask])

            for i in active_neurons:
                for j in active_neurons:
                    if i != j and i < n_neurons and j < n_neurons:
                        coactivation[i, j] += 1

    max_val = coactivation.max()
    if max_val > 0:
        coactivation = coactivation / max_val

    return coactivation


def create_connectivity_graph(coactivation, assignments, threshold=0.1, max_edges=500):
    """Create NetworkX graph from co-activation matrix."""
    n_neurons = len(assignments)

    G = nx.Graph()

    for i in range(n_neurons):
        G.add_node(i, assignment=int(assignments[i]))

    edges = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if coactivation[i, j] > threshold:
                edges.append((i, j, coactivation[i, j]))

    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:max_edges]

    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    return G


def create_degree_distribution(G, assignments):
    """Create node degree distribution analysis.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall degree distribution
    ax1 = axes[0]
    degrees = [G.degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax1.axvline(np.mean(degrees), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(degrees):.1f}')
    ax1.set_xlabel('Node Degree')
    ax1.set_ylabel('Count')
    ax1.set_title('Degree Distribution', fontweight='medium')
    ax1.legend(fontsize=9)

    # Degree by class
    ax2 = axes[1]
    class_degrees = []
    class_labels = []

    for cls in range(10):
        class_nodes = [n for n in G.nodes() if assignments[n] == cls]
        if class_nodes:
            degrees = [G.degree(n) for n in class_nodes]
            class_degrees.append(degrees)
            class_labels.append(str(cls))

    if class_degrees:
        bp = ax2.boxplot(class_degrees, labels=class_labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(get_digit_color(i))
            box.set_alpha(0.7)

    ax2.set_xlabel('Neuron Class')
    ax2.set_ylabel('Node Degree')
    ax2.set_title('Degree by Neuron Class', fontweight='medium')

    fig.suptitle('Network Connectivity Analysis', fontsize=14, fontweight='medium', y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Effective Connectivity')
    parser.add_argument('--examples-per-class', type=int, default=10,
                       help='Number of examples per class')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Co-activation threshold for edges')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Effective Connectivity Visualization")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True
    cfg._compute_derived_params()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    print(f"\nConfiguration:")
    print(f"  Examples per class: {args.examples_per_class}")
    print(f"  Edge threshold: {args.threshold}")

    # Load data
    print("\nLoading MNIST data...")
    images, labels = load_mnist_data(cfg, use_test_set=True)
    assignments = load_assignments(weight_path, cfg.n_e, cfg)

    classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))
    selected_indices, selected_labels = select_balanced_examples(
        labels, args.examples_per_class, classes, random_seed=42
    )

    # Set up network
    print("\nSetting up network...")
    defaultclock.dt = cfg.dt
    net, input_groups, neuron_groups, connections, spike_monitor_e, spike_monitor_input, n_input, n_e, brian_ns = \
        setup_network(cfg, custom_weight_path=weight_path)

    for name in cfg.input_population_names:
        input_groups[name+'e'].rates = 0*Hz
    net.run(0*ms, namespace=brian_ns)

    # Collect spike data
    print("\nRunning simulation...")
    all_spike_data = []

    for data_idx, label in tqdm(zip(selected_indices, selected_labels),
                                 total=len(selected_indices), desc="Processing"):
        current_data = images[data_idx]
        rates = current_data.reshape((cfg.n_input)) / 8. * cfg.input_intensity
        input_groups['Xe'].rates = rates * Hz

        start_time = float(net.t / second)
        net.run(cfg.single_example_time, namespace=brian_ns)

        spike_times = np.array(spike_monitor_e.t / second)
        spike_indices = np.array(spike_monitor_e.i)
        mask = spike_times >= start_time

        example_times = spike_times[mask] - start_time
        example_indices = spike_indices[mask]

        all_spike_data.append((example_times, example_indices))

        input_groups['Xe'].rates = 0*Hz
        net.run(cfg.resting_time, namespace=brian_ns)

    # Compute co-activation
    print("\nComputing co-activation matrix...")
    coactivation = compute_coactivation_matrix(all_spike_data, n_e)

    # Create graph
    G = create_connectivity_graph(coactivation, assignments,
                                  threshold=args.threshold, max_edges=500)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create visualization
    print("\nCreating visualization...")

    fig_degree = create_degree_distribution(G, assignments)

    # Save as PDF
    output_degree = save_figure(fig_degree, 'degree_distribution')

    print(f"\nSaved: {output_degree}")

    if args.show:
        fig_degree = create_degree_distribution(G, assignments)
        plt.show()


if __name__ == "__main__":
    main()
