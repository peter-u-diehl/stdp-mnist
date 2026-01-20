"""
Separation Dynamics Visualization

Plot inter-class distance in neural state space over time,
showing when and how different digit classes become separable.

Outputs PDF figures with modern styling.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from config import Config

from viz_utils import (
    setup_network, load_mnist_data, select_balanced_examples,
    compute_activity_trajectory, get_output_path, get_digit_color, DIGIT_COLORS,
    setup_modern_style, save_figure, create_figure
)


def compute_class_centroids(trajectories, labels, n_bins):
    """Compute centroid trajectories for each class.

    Returns:
        centroids: Dict mapping class -> trajectory array (n_bins, n_dims)
    """
    unique_labels = sorted(set(labels))
    centroids = {}

    for label in unique_labels:
        label_trajectories = [t for t, l in zip(trajectories, labels) if l == label]
        if label_trajectories:
            centroids[label] = np.mean(label_trajectories, axis=0)

    return centroids


def compute_pairwise_distances(centroids, n_bins):
    """Compute pairwise distances between class centroids over time.

    Returns:
        distances: Dict mapping (class1, class2) -> distance array (n_bins,)
    """
    classes = sorted(centroids.keys())
    distances = {}

    for c1, c2 in combinations(classes, 2):
        dist_over_time = np.zeros(n_bins)
        for t in range(n_bins):
            dist_over_time[t] = np.linalg.norm(centroids[c1][t] - centroids[c2][t])
        distances[(c1, c2)] = dist_over_time

    return distances


def create_mean_separation_plot(pairwise_distances, n_bins):
    """Create plot showing mean inter-class distance over time.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    time_axis = np.arange(n_bins)

    # Compute mean and std across all pairs
    all_distances = np.array(list(pairwise_distances.values()))
    mean_distance = np.mean(all_distances, axis=0)
    std_distance = np.std(all_distances, axis=0)

    ax.plot(time_axis, mean_distance, 'b-', linewidth=2.5, label='Mean Distance')
    ax.fill_between(time_axis, mean_distance - std_distance, mean_distance + std_distance,
                   alpha=0.3, color='steelblue')

    # Mark key timepoints
    max_idx = np.argmax(mean_distance)
    ax.axvline(x=max_idx, color='red', linestyle='--', alpha=0.7, label=f'Max @ bin {max_idx}')

    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Mean Inter-Class Distance')
    ax.set_title('Average Class Separation Over Time', fontsize=14, fontweight='medium')
    ax.legend(fontsize=10)

    fig.tight_layout()
    return fig


def create_within_vs_between_class(trajectories, labels, n_bins):
    """Compare within-class vs between-class distances.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    unique_labels = sorted(set(labels))

    within_distances = []
    between_distances = []

    # Sample time points
    sample_times = [n_bins // 4, n_bins // 2, 3 * n_bins // 4]

    for t in sample_times:
        points = np.array([traj[t] for traj in trajectories])

        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                dist = np.linalg.norm(points[i] - points[j])
                if labels[i] == labels[j]:
                    within_distances.append(dist)
                else:
                    between_distances.append(dist)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, max(max(within_distances), max(between_distances)), 40)
    ax.hist(within_distances, bins=bins, alpha=0.7, label='Within-class',
           color='steelblue', edgecolor='white')
    ax.hist(between_distances, bins=bins, alpha=0.7, label='Between-class',
           color='coral', edgecolor='white')

    ax.axvline(np.mean(within_distances), color='steelblue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(between_distances), color='coral', linestyle='--', linewidth=2)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title('Within-Class vs Between-Class Distances', fontsize=14, fontweight='medium')
    ax.legend(fontsize=10)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Separation Dynamics')
    parser.add_argument('--examples-per-class', type=int, default=10,
                       help='Number of examples per class')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Separation Dynamics Visualization")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True
    cfg._compute_derived_params()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    TIME_BINS = 50

    print(f"\nConfiguration:")
    print(f"  Examples per class: {args.examples_per_class}")

    # Load data
    print("\nLoading MNIST data...")
    images, labels = load_mnist_data(cfg, use_test_set=True)

    classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))
    selected_indices, selected_labels = select_balanced_examples(
        labels, args.examples_per_class, classes, random_seed=42
    )

    # Set up network
    print("\nSetting up network...")
    defaultclock.dt = cfg.dt
    net, input_groups, neuron_groups, connections, spike_monitor_e, spike_monitor_input, n_input, n_e, brian_ns = \
        setup_network(cfg, custom_weight_path=weight_path)
    n_total = n_input + n_e

    for name in cfg.input_population_names:
        input_groups[name+'e'].rates = 0*Hz
    net.run(0*ms, namespace=brian_ns)

    # Run simulation
    print("\nRunning simulation...")
    all_activities = []
    single_example_time = float(cfg.single_example_time / second)

    for data_idx, label in tqdm(zip(selected_indices, selected_labels),
                                 total=len(selected_indices), desc="Processing"):
        current_data = images[data_idx]
        rates = current_data.reshape((cfg.n_input)) / 8. * cfg.input_intensity
        input_groups['Xe'].rates = rates * Hz

        start_time = float(net.t / second)
        net.run(cfg.single_example_time, namespace=brian_ns)

        spike_times_input = np.array(spike_monitor_input.t / second)
        spike_indices_input = np.array(spike_monitor_input.i)
        mask_input = spike_times_input >= start_time

        spike_times_e = np.array(spike_monitor_e.t / second)
        spike_indices_e = np.array(spike_monitor_e.i)
        mask_e = spike_times_e >= start_time

        example_times = np.concatenate([
            spike_times_input[mask_input] - start_time,
            spike_times_e[mask_e] - start_time
        ])
        example_indices = np.concatenate([
            spike_indices_input[mask_input],
            spike_indices_e[mask_e] + n_input
        ])

        activity = compute_activity_trajectory(
            example_times, example_indices, n_total, single_example_time, TIME_BINS
        )
        all_activities.append(activity)

        input_groups['Xe'].rates = 0*Hz
        net.run(cfg.resting_time, namespace=brian_ns)

    # PCA
    print("\nComputing PCA...")
    all_stacked = np.vstack(all_activities)
    all_stacked += np.random.randn(*all_stacked.shape) * 0.01

    pca = PCA(n_components=20)
    all_pca = pca.fit_transform(all_stacked)

    trajectories = []
    for i in range(len(selected_labels)):
        start = i * TIME_BINS
        end = (i + 1) * TIME_BINS
        trajectories.append(all_pca[start:end])

    # Compute separation metrics
    print("\nComputing separation dynamics...")
    centroids = compute_class_centroids(trajectories, selected_labels, TIME_BINS)
    pairwise_distances = compute_pairwise_distances(centroids, TIME_BINS)

    # Create visualizations
    print("\nCreating visualizations...")

    fig_mean = create_mean_separation_plot(pairwise_distances, TIME_BINS)
    fig_within_between = create_within_vs_between_class(trajectories, selected_labels, TIME_BINS)

    # Save as PDFs
    output_mean = save_figure(fig_mean, 'mean_separation')
    output_wb = save_figure(fig_within_between, 'within_vs_between_class')

    print(f"\nSaved: {output_mean}")
    print(f"Saved: {output_wb}")

    if args.show:
        fig_mean = create_mean_separation_plot(pairwise_distances, TIME_BINS)
        plt.show()


if __name__ == "__main__":
    main()
