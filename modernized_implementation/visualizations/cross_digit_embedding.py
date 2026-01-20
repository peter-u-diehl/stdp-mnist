"""
Cross-Digit Similarity Visualization

Shows cross-digit neural similarity matrix.

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
    setup_network, load_mnist_data, select_balanced_examples,
    compute_activity_trajectory, get_output_path, get_digit_color, DIGIT_COLORS,
    setup_modern_style, save_figure, create_figure
)


def create_similarity_matrix(features, labels):
    """Create digit-digit similarity matrix based on neural representations.

    Returns:
        fig: matplotlib Figure object
        similarity: similarity matrix array
    """
    setup_modern_style()

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)

    # Compute mean representation per class
    class_means = {}
    for label in unique_labels:
        mask = np.array(labels) == label
        class_means[label] = np.mean(features[mask], axis=0)

    # Compute pairwise cosine similarity
    similarity = np.zeros((n_classes, n_classes))
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            v1, v2 = class_means[l1], class_means[l2]
            similarity[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels([str(l) for l in unique_labels])
    ax.set_yticklabels([str(l) for l in unique_labels])

    # Color the tick labels
    for i, label in enumerate(unique_labels):
        ax.get_xticklabels()[i].set_color(get_digit_color(label))
        ax.get_yticklabels()[i].set_color(get_digit_color(label))

    ax.set_xlabel('Digit')
    ax.set_ylabel('Digit')
    ax.set_title('Cross-Digit Neural Similarity Matrix', fontsize=14, fontweight='medium')

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if abs(similarity[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{similarity[i, j]:.2f}', ha='center', va='center',
                   fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=10)

    fig.tight_layout()
    return fig, similarity


def main():
    parser = argparse.ArgumentParser(description='Cross-Digit Similarity')
    parser.add_argument('--examples-per-class', type=int, default=30,
                       help='Number of examples per class')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Digit Similarity Visualization")
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

    print(f"Selected {len(selected_indices)} examples")

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

    # Extract features (trajectory flattened)
    print("\nExtracting features...")
    features = np.array([act.flatten() for act in all_activities])

    # PCA for dimensionality reduction
    print("Running PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.1%}")

    # Create visualization
    print("\nCreating visualization...")

    fig_similarity, similarity_matrix = create_similarity_matrix(features_pca, selected_labels)

    # Save as PDF
    output_similarity = save_figure(fig_similarity, 'digit_similarity_matrix')

    print(f"\nSaved: {output_similarity}")

    if args.show:
        fig_similarity, _ = create_similarity_matrix(features_pca, selected_labels)
        plt.show()


if __name__ == "__main__":
    main()
