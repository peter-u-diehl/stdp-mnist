"""
Spike Timing Precision Visualization

Analyze the temporal structure of spikes beyond just rates,
including first-spike latencies, inter-spike intervals, and timing precision.

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
    setup_network, load_mnist_data, select_balanced_examples,
    load_assignments, get_output_path, get_digit_color, DIGIT_COLORS,
    setup_modern_style, save_figure, create_figure
)


def compute_first_spike_latencies(spike_times, spike_indices, n_neurons):
    """Compute first spike latency for each neuron."""
    latencies = np.full(n_neurons, np.nan)

    for t, idx in zip(spike_times, spike_indices):
        if idx < n_neurons and np.isnan(latencies[idx]):
            latencies[idx] = t

    return latencies


def compute_inter_spike_intervals(spike_times, spike_indices, n_neurons):
    """Compute inter-spike intervals for each neuron."""
    isis = {i: [] for i in range(n_neurons)}
    last_spike = {i: None for i in range(n_neurons)}

    sorted_indices = np.argsort(spike_times)

    for idx in sorted_indices:
        t, neuron = spike_times[idx], spike_indices[idx]
        if neuron < n_neurons:
            if last_spike[neuron] is not None:
                isis[neuron].append(t - last_spike[neuron])
            last_spike[neuron] = t

    return isis


def create_latency_analysis(latencies_by_class, assignments):
    """Create first-spike latency analysis plot.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean latency per neuron class
    ax1 = axes[0]
    class_latencies = []
    class_labels = []

    for cls in range(10):
        # Get neurons assigned to this class
        class_neurons = np.where(assignments == cls)[0]

        # Collect all latencies for these neurons across examples
        all_latencies = []
        for latencies in latencies_by_class.values():
            for lat in latencies:
                for neuron in class_neurons:
                    if neuron < len(lat) and not np.isnan(lat[neuron]):
                        all_latencies.append(lat[neuron] * 1000)  # Convert to ms

        if all_latencies:
            class_latencies.append(all_latencies)
            class_labels.append(str(cls))

    if class_latencies:
        bp = ax1.boxplot(class_latencies, labels=class_labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(get_digit_color(i))
            box.set_alpha(0.7)

    ax1.set_xlabel('Neuron Class')
    ax1.set_ylabel('First-Spike Latency (ms)')
    ax1.set_title('First-Spike Latency by Neuron Class', fontweight='medium')

    # Latency distribution overall
    ax2 = axes[1]
    all_latencies = []
    for latencies_list in latencies_by_class.values():
        for lat in latencies_list:
            valid = lat[~np.isnan(lat)] * 1000  # Convert to ms
            all_latencies.extend(valid)

    if all_latencies:
        ax2.hist(all_latencies, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
        ax2.axvline(np.mean(all_latencies), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_latencies):.1f}ms')
        ax2.axvline(np.median(all_latencies), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(all_latencies):.1f}ms')
        ax2.legend(fontsize=9)

    ax2.set_xlabel('First-Spike Latency (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Overall Latency Distribution', fontweight='medium')

    fig.suptitle('First-Spike Latency Analysis', fontsize=14, fontweight='medium', y=1.02)
    fig.tight_layout()
    return fig


def create_isi_analysis(isis_by_class, assignments):
    """Create inter-spike interval analysis plot.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ISI distribution overall
    ax1 = axes[0]
    all_isis = []
    for isis_dict in isis_by_class.values():
        for isis in isis_dict:
            for neuron_isis in isis.values():
                all_isis.extend([isi * 1000 for isi in neuron_isis])  # Convert to ms

    if all_isis:
        ax1.hist(all_isis, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
        ax1.axvline(np.mean(all_isis), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_isis):.1f}ms')
        ax1.legend(fontsize=9)

    ax1.set_xlabel('Inter-Spike Interval (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title('ISI Distribution', fontweight='medium')

    # ISI CV (coefficient of variation) per neuron class
    ax2 = axes[1]
    class_cvs = []
    class_labels = []

    for cls in range(10):
        class_neurons = np.where(assignments == cls)[0]

        cvs = []
        for isis_dict in isis_by_class.values():
            for isis in isis_dict:
                for neuron in class_neurons:
                    if neuron in isis and len(isis[neuron]) > 1:
                        neuron_isis = np.array(isis[neuron])
                        cv = np.std(neuron_isis) / np.mean(neuron_isis)
                        if not np.isnan(cv) and not np.isinf(cv):
                            cvs.append(cv)

        if cvs:
            class_cvs.append(cvs)
            class_labels.append(str(cls))

    if class_cvs:
        bp = ax2.boxplot(class_cvs, labels=class_labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(get_digit_color(i))
            box.set_alpha(0.7)

    ax2.set_xlabel('Neuron Class')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('ISI Regularity by Neuron Class', fontweight='medium')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Poisson (CV=1)')
    ax2.legend(fontsize=9)

    fig.suptitle('Inter-Spike Interval Analysis', fontsize=14, fontweight='medium', y=1.02)
    fig.tight_layout()
    return fig


def create_spike_count_by_class(spike_counts_by_class):
    """Create spike count analysis.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean spike count per digit class (input)
    ax1 = axes[0]
    means = []
    stds = []
    for cls in sorted(spike_counts_by_class.keys()):
        counts = spike_counts_by_class[cls]
        if counts:
            means.append(np.mean(counts))
            stds.append(np.std(counts))
        else:
            means.append(0)
            stds.append(0)

    classes = sorted(spike_counts_by_class.keys())
    bars = ax1.bar(range(len(classes)), means, yerr=stds, capsize=3,
                  color=[get_digit_color(c) for c in classes], edgecolor='white', alpha=0.85)
    ax1.set_xlabel('Digit Class (Input)')
    ax1.set_ylabel('Mean Spike Count')
    ax1.set_title('Excitatory Spike Count by Input Digit', fontweight='medium')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes)

    # Distribution of spike counts
    ax2 = axes[1]
    all_counts = []
    for counts in spike_counts_by_class.values():
        all_counts.extend(counts)

    if all_counts:
        ax2.hist(all_counts, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax2.axvline(np.mean(all_counts), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_counts):.1f}')
        ax2.legend(fontsize=9)

    ax2.set_xlabel('Total Spike Count')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Spike Count Distribution', fontweight='medium')

    fig.suptitle('Spike Count Analysis', fontsize=14, fontweight='medium', y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Spike Timing Analysis')
    parser.add_argument('--examples-per-class', type=int, default=20,
                       help='Number of examples per class')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Spike Timing Analysis")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True
    cfg._compute_derived_params()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    print(f"\nConfiguration:")
    print(f"  Examples per class: {args.examples_per_class}")

    # Load data
    print("\nLoading data...")
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

    # Collect spike timing data
    print("\nRunning simulation...")
    latencies_by_class = {cls: [] for cls in classes}
    isis_by_class = {cls: [] for cls in classes}
    spike_counts_by_class = {cls: [] for cls in classes}

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

        # Compute metrics
        latencies = compute_first_spike_latencies(example_times, example_indices, n_e)
        isis = compute_inter_spike_intervals(example_times, example_indices, n_e)
        total_spikes = len(example_times)

        latencies_by_class[label].append(latencies)
        isis_by_class[label].append(isis)
        spike_counts_by_class[label].append(total_spikes)

        input_groups['Xe'].rates = 0*Hz
        net.run(cfg.resting_time, namespace=brian_ns)

    # Create visualizations
    print("\nCreating visualizations...")

    fig_latency = create_latency_analysis(latencies_by_class, assignments)
    fig_isi = create_isi_analysis(isis_by_class, assignments)
    fig_counts = create_spike_count_by_class(spike_counts_by_class)

    # Save as PDFs
    output_latency = save_figure(fig_latency, 'first_spike_latency')
    output_isi = save_figure(fig_isi, 'isi_analysis')
    output_counts = save_figure(fig_counts, 'spike_count_analysis')

    print(f"\nSaved: {output_latency}")
    print(f"Saved: {output_isi}")
    print(f"Saved: {output_counts}")

    if args.show:
        fig_latency = create_latency_analysis(latencies_by_class, assignments)
        plt.show()


if __name__ == "__main__":
    main()
