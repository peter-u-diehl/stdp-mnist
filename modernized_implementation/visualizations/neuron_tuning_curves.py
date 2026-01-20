"""
Neuron Tuning Curves Visualization

Interactive plot showing each neuron's response profile across digit classes,
revealing selectivity patterns and class preferences.

Outputs PDF figures with modern styling.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from config import Config

from viz_utils import (
    setup_network, load_mnist_data, load_assignments,
    get_output_path, get_digit_color, DIGIT_COLORS,
    setup_modern_style, save_figure, create_figure
)


def compute_tuning_curves(spike_counts_per_class, n_e):
    """Compute tuning curves from spike count data.

    Args:
        spike_counts_per_class: Dict mapping class -> list of spike count arrays
        n_e: Number of excitatory neurons

    Returns:
        mean_responses: (n_classes, n_e) mean spike counts
        std_responses: (n_classes, n_e) std of spike counts
    """
    n_classes = len(spike_counts_per_class)
    mean_responses = np.zeros((n_classes, n_e))
    std_responses = np.zeros((n_classes, n_e))

    for cls in range(n_classes):
        if cls in spike_counts_per_class and len(spike_counts_per_class[cls]) > 0:
            counts = np.array(spike_counts_per_class[cls])
            mean_responses[cls] = np.mean(counts, axis=0)
            std_responses[cls] = np.std(counts, axis=0)

    return mean_responses, std_responses


def create_tuning_curve_heatmap(mean_responses, assignments):
    """Create heatmap of tuning curves (neurons x classes).

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    n_classes, n_e = mean_responses.shape

    # Sort neurons by assigned class
    sort_idx = np.argsort(assignments)
    sorted_responses = mean_responses[:, sort_idx].T  # (n_e, n_classes)

    fig, ax = plt.subplots(figsize=(8, 10))

    im = ax.imshow(sorted_responses, aspect='auto', cmap='viridis')

    # Add horizontal lines to separate class groups
    boundaries = []
    for cls in range(n_classes):
        count = np.sum(assignments[sort_idx] == cls)
        if len(boundaries) == 0:
            boundaries.append(count)
        else:
            boundaries.append(boundaries[-1] + count)

    for b in boundaries[:-1]:
        ax.axhline(y=b - 0.5, color='white', linewidth=0.8, alpha=0.7)

    # Add class labels on right
    cumsum = 0
    for cls in range(n_classes):
        count = np.sum(assignments[sort_idx] == cls)
        if count > 0:
            mid_pos = cumsum + count / 2
            ax.annotate(f'{cls}', xy=(1.02, mid_pos / n_e), xycoords='axes fraction',
                       fontsize=9, color=get_digit_color(cls), fontweight='bold',
                       va='center', ha='left')
        cumsum += count

    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Neuron (sorted by class)')
    ax.set_title('Neuron Tuning Curves', fontsize=14, fontweight='medium')
    ax.set_xticks(range(n_classes))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean Spike Count', fontsize=10)

    fig.tight_layout()
    return fig


def create_individual_tuning_curves(mean_responses, std_responses, assignments, top_k=20):
    """Create line plots of tuning curves for top neurons per class.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    n_classes, n_e = mean_responses.shape

    # Find top K most selective neurons (highest peak response)
    peak_responses = np.max(mean_responses, axis=0)
    top_neurons = np.argsort(peak_responses)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_classes)
    for neuron_idx in top_neurons:
        assigned_class = int(assignments[neuron_idx])
        color = get_digit_color(assigned_class)

        ax.plot(x, mean_responses[:, neuron_idx], '-o', color=color,
                linewidth=1.5, markersize=4, alpha=0.7,
                label=f'N{neuron_idx} (class {assigned_class})')
        ax.fill_between(x,
                       mean_responses[:, neuron_idx] - std_responses[:, neuron_idx],
                       mean_responses[:, neuron_idx] + std_responses[:, neuron_idx],
                       color=color, alpha=0.1)

    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Mean Spike Count')
    ax.set_title(f'Tuning Curves: Top {top_k} Most Selective Neurons', fontsize=14, fontweight='medium')
    ax.set_xticks(range(n_classes))

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Neuron Tuning Curves')
    parser.add_argument('--examples-per-class', type=int, default=20,
                       help='Number of examples per class to compute tuning')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top neurons to plot individually')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Neuron Tuning Curves Visualization")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True
    cfg._compute_derived_params()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    print(f"\nConfiguration:")
    print(f"  Examples per class: {args.examples_per_class}")

    # Load data
    print("\nLoading MNIST data...")
    images, labels = load_mnist_data(cfg, use_test_set=True)

    # Load assignments
    assignments = load_assignments(weight_path, cfg.n_e, cfg)

    # Set up network
    print("\nSetting up network...")
    defaultclock.dt = cfg.dt
    net, input_groups, neuron_groups, connections, spike_monitor_e, spike_monitor_input, n_input, n_e, brian_ns = \
        setup_network(cfg, custom_weight_path=weight_path)

    for name in cfg.input_population_names:
        input_groups[name+'e'].rates = 0*Hz
    net.run(0*ms, namespace=brian_ns)

    # Collect spike counts per class
    print("\nRunning simulations to compute tuning curves...")
    spike_counts_per_class = {i: [] for i in range(10)}

    classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))
    for cls in classes:
        class_indices = np.where(labels == cls)[0]
        sample_indices = np.random.choice(class_indices,
                                          min(args.examples_per_class, len(class_indices)),
                                          replace=False)

        for data_idx in tqdm(sample_indices, desc=f"Class {cls}", leave=False):
            current_data = images[data_idx]
            rates = current_data.reshape((cfg.n_input)) / 8. * cfg.input_intensity
            input_groups['Xe'].rates = rates * Hz

            start_time = float(net.t / second)
            net.run(cfg.single_example_time, namespace=brian_ns)

            # Count spikes
            spike_times = np.array(spike_monitor_e.t / second)
            spike_indices = np.array(spike_monitor_e.i)
            mask = spike_times >= start_time

            spike_counts = np.zeros(n_e)
            for idx in spike_indices[mask]:
                if idx < n_e:
                    spike_counts[idx] += 1

            spike_counts_per_class[cls].append(spike_counts)

            input_groups['Xe'].rates = 0*Hz
            net.run(cfg.resting_time, namespace=brian_ns)

    # Compute tuning curves
    print("\nComputing tuning curves...")
    mean_responses, std_responses = compute_tuning_curves(spike_counts_per_class, n_e)

    # Create visualizations
    print("\nCreating visualizations...")

    fig_heatmap = create_tuning_curve_heatmap(mean_responses, assignments)
    fig_lines = create_individual_tuning_curves(mean_responses, std_responses, assignments, args.top_k)

    # Save as PDFs
    output_heatmap = save_figure(fig_heatmap, 'tuning_curves_heatmap')
    output_lines = save_figure(fig_lines, 'tuning_curves_lines')

    print(f"\nSaved: {output_heatmap}")
    print(f"Saved: {output_lines}")

    if args.show:
        fig_heatmap = create_tuning_curve_heatmap(mean_responses, assignments)
        plt.show()


if __name__ == "__main__":
    main()
