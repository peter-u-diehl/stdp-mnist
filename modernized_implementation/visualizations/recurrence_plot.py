"""
Recurrence Plot Visualization

Shows when neural states revisit similar regions in state space,
revealing periodic patterns, chaotic dynamics, or stable attractors.

Outputs PDF figures with modern styling.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from config import Config

from viz_utils import (
    setup_network, load_mnist_data, select_balanced_examples,
    compute_activity_trajectory, get_output_path, get_digit_color, DIGIT_COLORS,
    setup_modern_style, save_figure, create_figure
)


def compute_recurrence_matrix(trajectory, threshold=None, threshold_percentile=10):
    """Compute recurrence matrix from trajectory.

    Args:
        trajectory: Array of shape (n_timepoints, n_dims)
        threshold: Distance threshold for recurrence (auto-computed if None)
        threshold_percentile: Percentile of distances to use as threshold

    Returns:
        recurrence_matrix: Binary matrix of shape (n_timepoints, n_timepoints)
        distances: Distance matrix
        threshold: The threshold used
    """
    distances = cdist(trajectory, trajectory, metric='euclidean')

    if threshold is None:
        threshold = np.percentile(distances, threshold_percentile)

    recurrence_matrix = (distances <= threshold).astype(float)
    return recurrence_matrix, distances, threshold


def compute_recurrence_quantification(recurrence_matrix):
    """Compute recurrence quantification analysis (RQA) measures."""
    n = len(recurrence_matrix)

    # Recurrence rate
    rr = np.sum(recurrence_matrix) / (n * n)

    # Determinism (fraction of recurrent points forming diagonal lines)
    det_points = 0
    total_recurrent = np.sum(recurrence_matrix) - n  # exclude main diagonal

    for offset in range(1, n):
        diag = np.diag(recurrence_matrix, offset)
        in_line = False
        line_length = 0
        for val in diag:
            if val > 0:
                line_length += 1
                in_line = True
            else:
                if in_line and line_length >= 2:
                    det_points += line_length
                line_length = 0
                in_line = False
        if in_line and line_length >= 2:
            det_points += line_length

    determinism = det_points / max(total_recurrent, 1)

    return {
        'recurrence_rate': rr,
        'determinism': determinism
    }


def create_recurrence_plot(recurrence_matrices, labels, rqa_results=None):
    """Create recurrence plot visualization for multiple examples.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    n_examples = len(recurrence_matrices)
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (rec_mat, label) in enumerate(zip(recurrence_matrices, labels)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        im = ax.imshow(rec_mat, cmap='Blues', aspect='equal', origin='lower')

        title = f'Digit {label}'
        if rqa_results and i < len(rqa_results):
            title += f'\nRR={rqa_results[i]["recurrence_rate"]:.2f}'

        ax.set_title(title, fontsize=10, color=get_digit_color(label))
        ax.set_xlabel('Time (bins)', fontsize=8)
        ax.set_ylabel('Time (bins)', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_examples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    fig.suptitle('Recurrence Plots: When Neural States Revisit Similar Regions',
                fontsize=14, fontweight='medium', y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Recurrence Plot Visualization')
    parser.add_argument('--examples-per-class', type=int, default=2,
                       help='Number of examples per class')
    parser.add_argument('--threshold-percentile', type=float, default=10,
                       help='Percentile of distances for recurrence threshold')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Recurrence Plot Visualization")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True
    cfg._compute_derived_params()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    TIME_BINS = 50

    print(f"\nConfiguration:")
    print(f"  Examples per class: {args.examples_per_class}")
    print(f"  Threshold percentile: {args.threshold_percentile}")

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

    pca = PCA(n_components=10)
    all_pca = pca.fit_transform(all_stacked)

    trajectories = []
    for i in range(len(selected_labels)):
        start = i * TIME_BINS
        end = (i + 1) * TIME_BINS
        trajectories.append(all_pca[start:end])

    # Compute recurrence matrices
    print("\nComputing recurrence matrices...")
    recurrence_matrices = []
    rqa_results = []

    for traj in trajectories:
        rec_mat, _, _ = compute_recurrence_matrix(traj, threshold_percentile=args.threshold_percentile)
        recurrence_matrices.append(rec_mat)
        rqa_results.append(compute_recurrence_quantification(rec_mat))

    # Print RQA results
    print("\nRecurrence Quantification Analysis:")
    for i, (label, rqa) in enumerate(zip(selected_labels, rqa_results)):
        print(f"  Example {i+1} (Digit {label}): RR={rqa['recurrence_rate']:.3f}, DET={rqa['determinism']:.3f}")

    # Create visualization
    print("\nCreating visualization...")

    fig_recurrence = create_recurrence_plot(recurrence_matrices, selected_labels, rqa_results)

    # Save as PDF
    output_rec = save_figure(fig_recurrence, 'recurrence_plots')

    print(f"\nSaved: {output_rec}")

    if args.show:
        fig_recurrence = create_recurrence_plot(recurrence_matrices, selected_labels, rqa_results)
        plt.show()


if __name__ == "__main__":
    main()
