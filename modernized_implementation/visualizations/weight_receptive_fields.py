"""
Weight / Receptive Field Visualization

Visualize the learned receptive fields (weight patterns) of excitatory neurons,
showing what input patterns each neuron has learned to detect.

Outputs PDF figures with modern styling.
"""

import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

from viz_utils import (
    load_weight_matrix, load_assignments, get_digit_color,
    setup_modern_style, save_figure
)


def create_receptive_field_grid(weight_matrix, assignments, n_cols=20, sort_by_class=True):
    """Create a grid visualization of receptive fields.

    Args:
        weight_matrix: (784, n_e) weight matrix
        assignments: Class assignments per neuron
        n_cols: Number of columns in grid
        sort_by_class: If True, sort neurons by assigned class

    Returns:
        matplotlib Figure object
    """
    setup_modern_style()

    n_input, n_e = weight_matrix.shape
    n_rows = (n_e + n_cols - 1) // n_cols

    # Sort neurons by class assignment
    if sort_by_class:
        sort_idx = np.argsort(assignments)
    else:
        sort_idx = np.arange(n_e)

    # Create composite image
    img_size = 28
    grid_img = np.zeros((n_rows * img_size, n_cols * img_size))

    for i, neuron_idx in enumerate(sort_idx):
        if i >= n_e:
            break

        row = i // n_cols
        col = i % n_cols

        # Get weights and reshape to 28x28
        weights = weight_matrix[:, neuron_idx].reshape(28, 28)

        # Normalize to [0, 1]
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)

        # Place in grid
        r_start, r_end = row * img_size, (row + 1) * img_size
        c_start, c_end = col * img_size, (col + 1) * img_size
        grid_img[r_start:r_end, c_start:c_end] = weights

    # Create figure
    fig_height = max(8, n_rows * 0.4)
    fig_width = max(10, n_cols * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(grid_img, cmap='viridis', aspect='equal')

    # Add class labels on left side
    for cls in range(10):
        neurons_in_class = np.where(assignments[sort_idx] == cls)[0]
        if len(neurons_in_class) > 0:
            first_idx = neurons_in_class[0]
            row = first_idx // n_cols
            ax.annotate(f'{int(cls)}', xy=(-0.02, (row * img_size + img_size // 2) / grid_img.shape[0]),
                       xycoords='axes fraction', fontsize=11, fontweight='bold',
                       color=get_digit_color(int(cls)), va='center', ha='right')

    ax.set_title(f'Learned Receptive Fields ({n_e} neurons, sorted by class)', fontsize=14, fontweight='medium')
    ax.axis('off')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Normalized Weight', fontsize=10)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Receptive Field Visualization')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Receptive Field Visualization")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    # Load weights
    print("\nLoading weight matrix...")
    weight_matrix = load_weight_matrix(weight_path, cfg.n_input, cfg.n_e)

    if weight_matrix is None:
        print("Error: Could not load weights")
        return

    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Weight range: [{weight_matrix.min():.4f}, {weight_matrix.max():.4f}]")

    # Load assignments
    print("\nLoading neuron assignments...")
    assignments = load_assignments(weight_path, cfg.n_e, cfg)

    # Class distribution
    print("\nNeurons per class:")
    for cls in range(10):
        count = np.sum(assignments == cls)
        print(f"  Digit {cls}: {count} neurons")

    # Create visualization
    print("\nCreating visualization...")
    fig_grid = create_receptive_field_grid(weight_matrix, assignments)

    # Save as PDF
    output_grid = save_figure(fig_grid, 'receptive_fields_grid')
    print(f"\nSaved: {output_grid}")

    if args.show:
        fig_grid = create_receptive_field_grid(weight_matrix, assignments)
        plt.show()


if __name__ == "__main__":
    main()
