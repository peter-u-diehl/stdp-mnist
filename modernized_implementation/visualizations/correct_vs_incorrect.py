"""
Correct vs Incorrect Trials Visualization

Overlay trajectories colored by prediction outcome to see where
the network "goes wrong" - comparing successful vs failed classifications.

Outputs PDF figures with modern styling.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from config import Config

from viz_utils import (
    setup_network, load_mnist_data, load_assignments, compute_prediction,
    compute_activity_trajectory, get_output_path, get_digit_color, DIGIT_COLORS,
    setup_modern_style, save_figure, create_figure
)


def create_trajectory_comparison_2d(trajectories, labels, predictions, correctness):
    """Create 2D trajectory plot comparing correct vs incorrect.

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Correct trajectories
    ax1 = axes[0]
    for traj, label, correct in zip(trajectories, labels, correctness):
        if correct:
            color = get_digit_color(label)
            ax1.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, alpha=0.6)
            ax1.scatter(traj[-1, 0], traj[-1, 1], c=color, s=50, marker='o',
                       edgecolors='white', linewidths=0.5, zorder=5)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(f'Correct Trials (n={sum(correctness)})', fontweight='medium')

    # Incorrect trajectories
    ax2 = axes[1]
    for traj, label, pred, correct in zip(trajectories, labels, predictions, correctness):
        if not correct:
            color = get_digit_color(label)
            ax2.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, alpha=0.6)
            ax2.scatter(traj[-1, 0], traj[-1, 1], c=color, s=50, marker='x',
                       linewidths=2, zorder=5)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'Incorrect Trials (n={sum(~np.array(correctness))})', fontweight='medium')

    # Add legend
    for cls in range(10):
        ax1.plot([], [], color=get_digit_color(cls), label=f'Digit {cls}')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    fig.suptitle('Neural Trajectories: Correct vs Incorrect', fontsize=14, fontweight='medium', y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Correct vs Incorrect Visualization')
    parser.add_argument('--examples-per-class', type=int, default=20,
                       help='Number of examples per class')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Correct vs Incorrect Trials Visualization")
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
    print("\nLoading data...")
    images, labels = load_mnist_data(cfg, use_test_set=True)
    assignments = load_assignments(weight_path, cfg.n_e, cfg)

    classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))

    # Select random examples
    selected_indices = []
    selected_labels = []
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        n_select = min(args.examples_per_class, len(cls_indices))
        chosen = np.random.choice(cls_indices, n_select, replace=False)
        selected_indices.extend(chosen)
        selected_labels.extend([cls] * n_select)

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
    predictions = []
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

        # Combined activity for trajectory
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

        # Compute prediction
        exc_spike_counts = np.zeros(n_e)
        for idx in spike_indices_e[mask_e]:
            if idx < n_e:
                exc_spike_counts[idx] += 1
        pred = compute_prediction(exc_spike_counts, assignments)
        predictions.append(pred)

        input_groups['Xe'].rates = 0*Hz
        net.run(cfg.resting_time, namespace=brian_ns)

    # PCA
    print("\nComputing PCA...")
    all_stacked = np.vstack(all_activities)
    all_stacked += np.random.randn(*all_stacked.shape) * 0.01

    pca = PCA(n_components=3)
    all_pca = pca.fit_transform(all_stacked)

    trajectories = []
    for i in range(len(selected_labels)):
        start = i * TIME_BINS
        end = (i + 1) * TIME_BINS
        trajectories.append(all_pca[start:end])

    # Determine correctness
    correctness = [p == l for p, l in zip(predictions, selected_labels)]
    accuracy = sum(correctness) / len(correctness)
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"Correct: {sum(correctness)}, Incorrect: {sum(~np.array(correctness))}")

    # Create visualization
    print("\nCreating visualization...")

    fig_trajectories = create_trajectory_comparison_2d(trajectories, selected_labels, predictions, correctness)

    # Save as PDF
    output_traj = save_figure(fig_trajectories, 'correct_vs_incorrect_trajectories')

    print(f"\nSaved: {output_traj}")

    if args.show:
        fig_trajectories = create_trajectory_comparison_2d(trajectories, selected_labels, predictions, correctness)
        plt.show()


if __name__ == "__main__":
    main()
