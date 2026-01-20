"""
Spike Raster + Trajectory Synchronized Animation

Side-by-side animation of raw spike raster and PCA trajectory,
showing how population bursts map to trajectory movements.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from config import Config

from viz_utils import (
    setup_network, load_mnist_data, select_balanced_examples,
    compute_activity_trajectory, get_output_path, get_digit_color
)


def create_synchronized_animation(spike_data_list, trajectories, labels, n_e, time_bins=50):
    """Create synchronized spike raster and trajectory animation."""
    n_examples = len(labels)
    total_time = 0.35  # seconds
    time_per_bin = total_time / time_bins

    # Compute global bounds for trajectory
    all_trajs = np.vstack(trajectories)
    x_range = [all_trajs[:, 0].min() - 0.5, all_trajs[:, 0].max() + 0.5]
    y_range = [all_trajs[:, 1].min() - 0.5, all_trajs[:, 1].max() + 0.5]
    z_range = [all_trajs[:, 2].min() - 0.5, all_trajs[:, 2].max() + 0.5]

    # Create figure
    fig = go.Figure()

    # Add all traces for first example/first frame
    init_spike_times, init_spike_indices = spike_data_list[0]
    init_color = get_digit_color(labels[0])
    init_traj = trajectories[0]

    # Trace 0: Raster spikes (2D scatter, positioned on left)
    fig.add_trace(go.Scatter(
        x=init_spike_times * 1000,
        y=init_spike_indices,
        mode='markers',
        marker=dict(size=3, color=init_color, opacity=0.8),
        name='Spikes',
        xaxis='x',
        yaxis='y'
    ))

    # Trace 1: Current time line
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, n_e],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Current time',
        xaxis='x',
        yaxis='y'
    ))

    # Trace 2: Trajectory line (3D)
    fig.add_trace(go.Scatter3d(
        x=init_traj[:1, 0],
        y=init_traj[:1, 1],
        z=init_traj[:1, 2],
        mode='lines',
        line=dict(color=init_color, width=6),
        name='Trajectory'
    ))

    # Trace 3: Current position marker (3D)
    fig.add_trace(go.Scatter3d(
        x=[init_traj[0, 0]],
        y=[init_traj[0, 1]],
        z=[init_traj[0, 2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Current'
    ))

    # Build frames
    frames = []
    for ex_idx, (label, (spike_times, spike_indices), traj) in enumerate(
            zip(labels, spike_data_list, trajectories)):

        color = get_digit_color(label)

        for bin_idx in range(time_bins):
            t_end = (bin_idx + 1) * time_per_bin
            current_traj = traj[:bin_idx + 1]

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=spike_times * 1000,
                        y=spike_indices,
                        mode='markers',
                        marker=dict(size=3, color=color, opacity=0.8),
                    ),
                    go.Scatter(
                        x=[t_end * 1000, t_end * 1000],
                        y=[0, n_e],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                    ),
                    go.Scatter3d(
                        x=current_traj[:, 0],
                        y=current_traj[:, 1],
                        z=current_traj[:, 2],
                        mode='lines',
                        line=dict(color=color, width=6),
                    ),
                    go.Scatter3d(
                        x=[current_traj[-1, 0]],
                        y=[current_traj[-1, 1]],
                        z=[current_traj[-1, 2]],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='diamond'),
                    )
                ],
                name=f'ex{ex_idx}_bin{bin_idx}',
                traces=[0, 1, 2, 3]
            )
            frames.append(frame)

    fig.frames = frames

    # Slider steps
    steps = []
    for ex_idx in range(n_examples):
        label = labels[ex_idx]
        for bin_idx in range(time_bins):
            t_end = (bin_idx + 1) * time_per_bin
            steps.append(dict(
                args=[[f'ex{ex_idx}_bin{bin_idx}'],
                      {"frame": {"duration": 50, "redraw": True},
                       "mode": "immediate",
                       "transition": {"duration": 0}}],
                label=f'D{label}:{bin_idx}',
                method='animate'
            ))

    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix='',
            visible=True,
            xanchor='center',
            font=dict(size=14)
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=60),
        len=0.9,
        x=0.05,
        y=0,
        steps=steps
    )]

    # Play/pause buttons
    updatemenus = [dict(
        type='buttons',
        showactive=False,
        y=1.12,
        x=0.5,
        xanchor='center',
        buttons=[
            dict(label='&#9654; Play',
                 method='animate',
                 args=[None, {"frame": {"duration": 80, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 0}}]),
            dict(label='&#9724; Pause',
                 method='animate',
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}])
        ]
    )]

    # Layout with domain positioning for 2D axes
    fig.update_layout(
        title=dict(
            text='Spike Raster + Trajectory Animation',
            x=0.5,
            font=dict(size=22)
        ),
        height=650,
        width=1400,
        template='plotly_white',
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=False,
        # 2D axes on the left half
        xaxis=dict(
            domain=[0, 0.45],
            title='Time (ms)',
            range=[0, total_time * 1000],
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            domain=[0.1, 0.9],
            title='Neuron Index',
            range=[0, n_e],
            showgrid=True,
            gridcolor='lightgray'
        ),
        # 3D scene on the right half
        scene=dict(
            domain=dict(x=[0.5, 1], y=[0.1, 0.95]),
            xaxis=dict(title='PC1', range=x_range, backgroundcolor='rgb(245,245,245)'),
            yaxis=dict(title='PC2', range=y_range, backgroundcolor='rgb(245,245,245)'),
            zaxis=dict(title='PC3', range=z_range, backgroundcolor='rgb(245,245,245)'),
            bgcolor='rgb(250, 248, 245)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        annotations=[
            dict(text='Spike Raster (Excitatory)', x=0.22, y=1.02,
                 xref='paper', yref='paper', showarrow=False, font=dict(size=14)),
            dict(text='3D Trajectory (PCA)', x=0.75, y=1.02,
                 xref='paper', yref='paper', showarrow=False, font=dict(size=14)),
        ]
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Spike Raster + Trajectory Sync')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to show')
    args = parser.parse_args()

    print("=" * 60)
    print("Spike Raster + Trajectory Synchronized Animation")
    print("=" * 60)

    cfg = Config()
    cfg.test_mode = True
    cfg._compute_derived_params()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_path = os.path.join(base_path, 'mnist_data', 'weights') + '/'

    TIME_BINS = 50

    # Load data
    print("\nLoading MNIST data...")
    images, labels = load_mnist_data(cfg, use_test_set=True)

    classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))
    selected_indices, selected_labels = select_balanced_examples(
        labels, 1, classes[:args.num_examples], random_seed=42
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

    # Run simulation and collect spike data
    print("\nRunning simulation...")
    all_activities = []
    spike_data_list = []
    single_example_time = float(cfg.single_example_time / second)

    for data_idx, label in tqdm(zip(selected_indices, selected_labels),
                                 total=len(selected_indices), desc="Processing"):
        current_data = images[data_idx]
        rates = current_data.reshape((cfg.n_input)) / 8. * cfg.input_intensity
        input_groups['Xe'].rates = rates * Hz

        start_time = float(net.t / second)
        net.run(cfg.single_example_time, namespace=brian_ns)

        # Get input spikes
        spike_times_input = np.array(spike_monitor_input.t / second)
        spike_indices_input = np.array(spike_monitor_input.i)
        mask_input = spike_times_input >= start_time

        # Get excitatory spikes
        spike_times_e = np.array(spike_monitor_e.t / second)
        spike_indices_e = np.array(spike_monitor_e.i)
        mask_e = spike_times_e >= start_time

        # Store excitatory spikes for raster
        exc_times = spike_times_e[mask_e] - start_time
        exc_indices = spike_indices_e[mask_e]
        spike_data_list.append((exc_times, exc_indices))

        # Combined for PCA
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

    # Smooth
    for i in range(len(all_activities)):
        start = i * TIME_BINS
        end = (i + 1) * TIME_BINS
        for j in range(n_total):
            all_stacked[start:end, j] = gaussian_filter1d(all_stacked[start:end, j], sigma=2.0)

    all_stacked += np.random.randn(*all_stacked.shape) * 0.01

    pca = PCA(n_components=3)
    all_pca = pca.fit_transform(all_stacked)

    trajectories = []
    for i in range(len(selected_labels)):
        start = i * TIME_BINS
        end = (i + 1) * TIME_BINS
        trajectories.append(all_pca[start:end])

    # Create visualization
    print("\nCreating animation...")
    fig = create_synchronized_animation(
        spike_data_list, trajectories, selected_labels, n_e, TIME_BINS
    )

    output_path = get_output_path('spike_raster_trajectory.html')
    fig.write_html(output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
