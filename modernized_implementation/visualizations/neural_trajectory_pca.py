"""
Neural Trajectory PCA Visualization

This script runs the Diehl & Cook SNN on 10 different MNIST examples,
records the neural activity over time, performs PCA on the activity vectors,
and creates an animated 3D trajectory visualization using Plotly.

The idea is to visualize how the network's state evolves through a low-dimensional
manifold as it processes different digit examples.

Usage:
    python neural_trajectory_pca.py [--weight-path PATH] [--num-examples N]

    --weight-path: Path to trained weights folder (default: use migration weights)
    --num-examples: Number of MNIST examples to visualize (default: 10)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import time
import os
import sys
import argparse
from brian2 import *
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Default configuration
NUM_EXAMPLES = 10  # Number of different MNIST examples to visualize
TIME_BINS_PER_EXAMPLE = 50  # Number of time bins to record per example (temporal resolution)

# Default weight path (relative to parent directory)
DEFAULT_WEIGHT_PATH = '../mnist_data/weights/'


def setup_network(cfg, custom_weight_path=None):
    """Set up the Brian2 network (similar to main script but simplified for analysis)

    Args:
        cfg: Config object
        custom_weight_path: Optional path to trained weights (overrides cfg.weight_path)
    """

    # Extract parameters from config - these need to be global for Brian2
    global v_rest_e, v_rest_i, v_reset_e, v_reset_i, v_thresh_i
    global refrac_e, refrac_i, offset, v_thresh_e_const

    n_input = cfg.n_input
    n_e = cfg.n_e
    n_i = cfg.n_i
    v_rest_e = cfg.v_rest_e
    v_rest_i = cfg.v_rest_i
    v_reset_e = cfg.v_reset_e
    v_reset_i = cfg.v_reset_i
    v_thresh_i = cfg.v_thresh_i
    refrac_e = cfg.refrac_e
    refrac_i = cfg.refrac_i
    offset = cfg.offset
    v_thresh_e_const = cfg.v_thresh_e_const

    ending = cfg.ending
    data_path = cfg.data_path
    weight_path = custom_weight_path if custom_weight_path else cfg.weight_path

    population_names = cfg.population_names
    input_population_names = cfg.input_population_names
    recurrent_conn_names = cfg.recurrent_conn_names
    input_connection_names = cfg.input_connection_names
    input_conn_names = cfg.input_conn_names

    delay = {
        'ee_input': cfg.delay_ee_input,
        'ei_input': cfg.delay_ei_input
    }

    # Get neuron equations from config (ensures exact match with training)
    neuron_eqs_e = cfg.get_neuron_eqs_e()
    neuron_eqs_i = cfg.get_neuron_eqs_i()

    # Threshold and reset strings (from config/main script)
    v_thresh_e_str = '(v>(theta - offset + v_thresh_e_const)) and (timer>refrac_e)'
    scr_e = 'v = v_reset_e; timer = 0*ms'

    # Helper function to load weight matrices
    def get_matrix_from_file(fileName):
        fname_offset = len(ending) + 4
        if fileName[-4-fname_offset] == 'X':
            n_src = n_input
        else:
            if fileName[-3-fname_offset]=='e':
                n_src = n_e
            else:
                n_src = n_i
        if fileName[-1-fname_offset]=='e':
            n_tgt = n_e
        else:
            n_tgt = n_i
        readout = np.load(fileName, allow_pickle=True)
        value_arr = np.zeros((n_src, n_tgt))
        if not readout.shape == (0,):
            value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
        return value_arr

    # Create neuron groups
    neuron_groups = {}
    input_groups = {}
    connections = {}

    # Create neuron groups (matching main script exactly)
    neuron_groups['e'] = NeuronGroup(n_e*len(population_names), neuron_eqs_e,
                                      threshold=v_thresh_e_str, refractory=refrac_e,
                                      reset=scr_e, method='euler')
    neuron_groups['i'] = NeuronGroup(n_i*len(population_names), neuron_eqs_i,
                                      threshold='v > v_thresh_i', refractory=refrac_i,
                                      reset='v = v_reset_i', method='euler')

    # Set up populations
    for name in population_names:
        neuron_groups[name+'e'] = neuron_groups['e'][0:n_e]
        neuron_groups[name+'i'] = neuron_groups['i'][0:n_i]

        neuron_groups[name+'e'].v = v_rest_e - 40. * mV
        neuron_groups[name+'i'].v = v_rest_i - 40. * mV

        # Load theta values
        try:
            neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy') * volt
        except FileNotFoundError:
            print(f"Warning: Could not load theta, using default")
            neuron_groups['e'].theta = np.ones((n_e)) * 20.0*mV

        # Create recurrent connections
        for conn_type in recurrent_conn_names:
            connName = name+conn_type[0]+name+conn_type[1]
            try:
                weightMatrix = get_matrix_from_file(data_path + 'random/' + connName + ending + '.npy')
            except FileNotFoundError:
                print(f"Warning: Could not load {connName}")
                continue

            src_grp = neuron_groups[connName[0:2]]
            tgt_grp = neuron_groups[connName[2:4]]

            connections[connName] = Synapses(src_grp, tgt_grp, model='w : 1',
                                              on_pre='g'+conn_type[0]+'_post += w', method='euler')
            sources, targets = np.where(weightMatrix > 0)
            connections[connName].connect(i=sources, j=targets)
            connections[connName].w = weightMatrix[sources, targets]

    # Create input population
    for name in input_population_names:
        input_groups[name+'e'] = PoissonGroup(n_input, 0*Hz)

    # Create input connections
    for name in input_connection_names:
        for connType in input_conn_names:
            connName = name[0] + connType[0] + name[1] + connType[1]

            if connType == 'ei_input':
                load_path = data_path + 'random/'
            else:
                load_path = weight_path

            try:
                weightMatrix = get_matrix_from_file(load_path + connName + ending + '.npy')
            except FileNotFoundError:
                print(f"Warning: Could not load {connName}")
                continue

            src_grp = input_groups[connName[0:2]]
            tgt_grp = neuron_groups[connName[2:4]]

            connections[connName] = Synapses(src_grp, tgt_grp, model='w : 1',
                                              on_pre='g'+connType[0]+'_post += w',
                                              delay=delay[connType][1], method='euler')
            sources, targets = np.where(weightMatrix > 0)
            connections[connName].connect(i=sources, j=targets)
            connections[connName].w = weightMatrix[sources, targets]
            connections[connName].delay = 'rand() * ' + str(float(delay[connType][1]/ms)) + '*ms'

            # Verify weights loaded
            if connName == 'XeAe':
                print(f"  Loaded XeAe weights: {len(sources)} connections, "
                      f"avg={np.mean(weightMatrix[sources, targets]):.4f}, "
                      f"max={np.max(weightMatrix[sources, targets]):.4f}")

    # Create spike monitors for input and excitatory neurons
    spike_monitor_e = SpikeMonitor(neuron_groups['Ae'])
    spike_monitor_input = SpikeMonitor(input_groups['Xe'])

    # Build network
    net = Network()
    net.add(neuron_groups['e'], neuron_groups['i'])
    for name in input_population_names:
        net.add(input_groups[name+'e'])
    for conn in connections.values():
        net.add(conn)
    net.add(spike_monitor_e)
    net.add(spike_monitor_input)

    return net, input_groups, neuron_groups, spike_monitor_e, spike_monitor_input, n_input, n_e


def compute_activity_trajectory(spike_times, spike_indices, n_neurons, total_time, n_bins):
    """
    Convert spike train data into binned activity vectors.

    Returns an array of shape (n_bins, n_neurons) where each row is the
    spike count per neuron in that time bin.
    """
    bin_edges = np.linspace(0, total_time, n_bins + 1)
    activity = np.zeros((n_bins, n_neurons))

    for bin_idx in range(n_bins):
        t_start = bin_edges[bin_idx]
        t_end = bin_edges[bin_idx + 1]

        # Find spikes in this time bin
        mask = (spike_times >= t_start) & (spike_times < t_end)
        bin_spike_indices = spike_indices[mask]

        # Count spikes per neuron
        for neuron_idx in bin_spike_indices:
            if neuron_idx < n_neurons:
                activity[bin_idx, neuron_idx] += 1

    return activity


def create_trajectory_animation(trajectories_3d, labels, predictions, colors_per_class):
    """
    Create an animated Plotly figure showing neural trajectories through PCA space.
    Trajectories are shown one at a time, sequentially, with prediction display.
    """

    # Get all trajectories info
    n_examples = len(trajectories_3d)
    n_bins = trajectories_3d[0].shape[0]

    # Total frames = n_examples * n_bins (one trajectory at a time)
    total_frames = n_examples * n_bins

    # Create figure
    fig = go.Figure()

    # Color map for different digit classes
    colorscale = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Background colors
    bg_color = '#f0ebe0'  # Light warm background
    plot_bg = '#e8e0d0'   # Slightly darker for 3D scene

    # Initial state: empty traces for each trajectory line + one current position marker
    for i, (traj, label) in enumerate(zip(trajectories_3d, labels)):
        color = colorscale[label % len(colorscale)]

        # Trajectory line (initially empty)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color=color, width=5),
            name=f'Digit {label}',
            legendgroup=f'digit{label}',
            showlegend=(i == 0 or labels[i-1] != label)
        ))

    # Current position marker (small circle)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers',
        marker=dict(color='white', size=6, symbol='circle',
                    line=dict(color='black', width=1)),
        name='Current',
        showlegend=False
    ))

    # Create animation frames - one trajectory at a time
    frames = []
    for example_idx in range(n_examples):
        traj = trajectories_3d[example_idx]
        label = labels[example_idx]
        pred = predictions[example_idx]
        color = colorscale[label % len(colorscale)]
        pred_color = colorscale[pred % len(colorscale)]
        is_correct = pred == label

        for time_idx in range(1, n_bins + 1):
            frame_data = []

            # Add all completed trajectories
            for prev_idx in range(example_idx):
                prev_traj = trajectories_3d[prev_idx]
                prev_color = colorscale[labels[prev_idx] % len(colorscale)]
                frame_data.append(go.Scatter3d(
                    x=prev_traj[:, 0], y=prev_traj[:, 1], z=prev_traj[:, 2],
                    mode='lines',
                    line=dict(color=prev_color, width=5),
                ))

            # Add current trajectory (partial)
            frame_data.append(go.Scatter3d(
                x=traj[:time_idx, 0], y=traj[:time_idx, 1], z=traj[:time_idx, 2],
                mode='lines',
                line=dict(color=color, width=5),
            ))

            # Add empty traces for future trajectories
            for future_idx in range(example_idx + 1, n_examples):
                frame_data.append(go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='lines',
                ))

            # Current position marker (small circle)
            frame_data.append(go.Scatter3d(
                x=[traj[time_idx-1, 0]], y=[traj[time_idx-1, 1]], z=[traj[time_idx-1, 2]],
                mode='markers',
                marker=dict(color=color, size=6, symbol='circle',
                            line=dict(color='white', width=1)),
            ))

            # Simple, clean annotation - no emojis
            status_text = "CORRECT" if is_correct else "WRONG"
            status_color = '#006400' if is_correct else '#8B0000'

            frame_layout = dict(
                annotations=[
                    dict(
                        text=f"[{example_idx + 1}/{n_examples}]   True: {label}   Predicted: {pred}   ({status_text})",
                        x=0.5, y=0.98,
                        xref='paper', yref='paper',
                        showarrow=False,
                        font=dict(size=18, color=status_color, family='monospace'),
                    ),
                ]
            )

            frame_name = f'{example_idx * n_bins + time_idx}'
            frames.append(go.Frame(data=frame_data, layout=frame_layout, name=frame_name))

    fig.frames = frames

    # Calculate axis ranges
    all_points = np.vstack(trajectories_3d)
    margin = 0.15
    x_range = [all_points[:, 0].min() - margin, all_points[:, 0].max() + margin]
    y_range = [all_points[:, 1].min() - margin, all_points[:, 1].max() + margin]
    z_range = [all_points[:, 2].min() - margin, all_points[:, 2].max() + margin]

    # Clean layout - 3D plot on left, info on right
    fig.update_layout(
        paper_bgcolor=bg_color,
        margin=dict(l=0, r=200, t=50, b=60),
        scene=dict(
            domain=dict(x=[0, 0.85], y=[0.08, 0.95]),
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(range=x_range, backgroundcolor=plot_bg, gridcolor='#c8c0b0',
                       showbackground=True, title_font=dict(size=13)),
            yaxis=dict(range=y_range, backgroundcolor=plot_bg, gridcolor='#c8c0b0',
                       showbackground=True, title_font=dict(size=13)),
            zaxis=dict(range=z_range, backgroundcolor=plot_bg, gridcolor='#c8c0b0',
                       showbackground=True, title_font=dict(size=13)),
            bgcolor=plot_bg,
            camera=dict(
                eye=dict(x=0.8, y=0.8, z=0.5)
            ),
            aspectmode='data'
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                direction='down',
                y=0.5, x=1.02,
                xanchor='left', yanchor='middle',
                bgcolor='white',
                bordercolor='#999999',
                font=dict(size=12),
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=40, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top', xanchor='left',
            currentvalue=dict(visible=False),
            transition=dict(duration=0),
            pad=dict(b=0, t=0),
            len=0.85, x=0, y=0.05,
            bgcolor='#d8d0c0',
            bordercolor='#b0a890',
            ticklen=0,
            steps=[
                dict(
                    args=[[str(k)], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0)
                    )],
                    label='',
                    method='animate'
                )
                for k in range(1, total_frames + 1)
            ]
        )],
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=0.88,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#999999',
            borderwidth=1,
            font=dict(size=12)
        ),
        width=1400,
        height=800
    )

    return fig


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Trajectory PCA Visualization')
    parser.add_argument('--weight-path', type=str, default=DEFAULT_WEIGHT_PATH,
                        help='Path to trained weights folder')
    parser.add_argument('--examples-per-class', type=int, default=3,
                        help='Number of examples per digit class (default: 3)')
    parser.add_argument('--smooth', type=float, default=2.0,
                        help='Gaussian smoothing sigma for trajectories (0 to disable)')
    args = parser.parse_args()

    print("=" * 60)
    print("Neural Trajectory PCA Visualization")
    print("=" * 60)

    # Initialize configuration (force test mode)
    cfg = Config()
    cfg.set_test_mode(True)

    # Set data path relative to parent directory (migration folder)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.data_path = os.path.join(base_path, 'mnist_data') + '/'
    cfg.mnist_data_path = cfg.data_path

    # Resolve weight path
    weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.weight_path))
    if not weight_path.endswith('/'):
        weight_path += '/'

    available_classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))
    print(f"\nConfiguration:")
    print(f"  Examples per class: {args.examples_per_class}")
    print(f"  Classes: {available_classes}")
    print(f"  Time bins per example: {TIME_BINS_PER_EXAMPLE}")
    print(f"  Network: {cfg.n_e} excitatory neurons")
    print(f"  Weight path: {weight_path}")
    print(f"  Smoothing sigma: {args.smooth}")
    print()

    # Check if weights exist
    theta_file = weight_path + 'theta_A.npy'
    xe_ae_file = weight_path + 'XeAe.npy'
    if os.path.exists(theta_file) and os.path.exists(xe_ae_file):
        print("Found trained weights!")
    else:
        print("WARNING: Trained weights not found at specified path.")
        print(f"  Looking for: {theta_file}")
        print(f"  Looking for: {xe_ae_file}")
        print("  Will use default/random weights (results may not be meaningful)")
        print()

    # Load MNIST data
    print("Loading MNIST data...")
    data_loader = MNISTDataLoader(cfg)
    test_images, test_labels = data_loader.load_test_data()

    # Select diverse examples (one per class if possible)
    print("Selecting examples...")
    selected_indices = []
    selected_labels = []

    # Get examples per class - run N examples of digit 0, then N of digit 1, etc.
    available_classes = cfg.mnist_classes if cfg.mnist_classes else list(range(10))
    examples_per_class = args.examples_per_class

    print(f"Selecting {examples_per_class} examples per class, sequential by digit...")

    for cls in available_classes:
        cls_indices = np.where(test_labels == cls)[0]
        if len(cls_indices) > 0:
            # Randomly select N examples from this class
            n_select = min(examples_per_class, len(cls_indices))
            chosen_indices = np.random.choice(cls_indices, size=n_select, replace=False)
            for idx in chosen_indices:
                selected_indices.append(idx)
                selected_labels.append(cls)

    num_examples = len(selected_indices)  # Update num_examples to actual count
    print(f"Total examples: {num_examples}")
    print(f"Class sequence: {[f'{cls}x{examples_per_class}' for cls in available_classes]}")

    # Set up the network
    print("\nSetting up neural network...")
    defaultclock.dt = cfg.dt
    net, input_groups, neuron_groups, spike_monitor_e, spike_monitor_input, n_input, n_e = setup_network(cfg, custom_weight_path=weight_path)
    n_total = n_input + n_e  # Total neurons for combined activity vectors
    print(f"Recording from {n_input} input + {n_e} excitatory = {n_total} total neurons")

    # Initialize network
    for name in cfg.input_population_names:
        input_groups[name+'e'].rates = 0*Hz
    net.run(0*ms)

    # Load neuron assignments for predictions
    print("Loading neuron assignments...")
    assignments_path = weight_path.replace('weights/', 'activity/')
    try:
        # Try to load clean assignments first
        assignments_file = None
        for candidate in ['resultPopVecs2500_clean.npy', 'resultPopVecs5000_clean.npy', 'resultPopVecs10000_clean.npy']:
            try:
                result_monitor = np.load(assignments_path + candidate)
                input_numbers = np.load(assignments_path + candidate.replace('resultPopVecs', 'inputNumbers'))
                assignments_file = candidate
                break
            except FileNotFoundError:
                continue

        if assignments_file is None:
            raise FileNotFoundError("No assignment files found")

        # Compute assignments from loaded activity
        assignments = np.ones(n_e) * -1
        maximum_rate = [0] * n_e
        classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
        for j in classes_to_check:
            num_assignments = len(np.where(input_numbers == j)[0])
            if num_assignments > 0:
                rate = np.sum(result_monitor[input_numbers == j], axis=0) / num_assignments
                for i in range(n_e):
                    if rate[i] > maximum_rate[i]:
                        maximum_rate[i] = rate[i]
                        assignments[i] = j
        print(f"Loaded assignments from {assignments_file}")
    except Exception as e:
        print(f"Warning: Could not load assignments ({e}), predictions will be random")
        assignments = np.zeros(n_e)

    # Run network on selected examples and collect activity
    print("\nRunning simulation...")
    all_activities = []
    all_predictions = []
    single_example_time = float(cfg.single_example_time / second)
    resting_time = cfg.resting_time
    input_intensity = cfg.input_intensity

    for ex_idx, (data_idx, label) in enumerate(tqdm(zip(selected_indices, selected_labels),
                                                      total=len(selected_indices),
                                                      desc="Processing examples")):
        # Get current example
        current_data = test_images[data_idx]

        # Present stimulus
        rates = current_data.reshape((cfg.n_input)) / 8. * input_intensity
        input_groups['Xe'].rates = rates * Hz

        # Record start time
        start_time = float(net.t / second)

        # Run for single example time
        net.run(cfg.single_example_time)

        # Get input layer spikes from this example
        all_spike_times_input = np.array(spike_monitor_input.t / second)
        all_spike_indices_input = np.array(spike_monitor_input.i)
        mask_input = all_spike_times_input >= start_time
        example_spike_times_input = all_spike_times_input[mask_input] - start_time
        example_spike_indices_input = all_spike_indices_input[mask_input]

        # Get excitatory spikes from this example
        all_spike_times_e = np.array(spike_monitor_e.t / second)
        all_spike_indices_e = np.array(spike_monitor_e.i)
        mask_e = all_spike_times_e >= start_time
        example_spike_times_e = all_spike_times_e[mask_e] - start_time
        example_spike_indices_e = all_spike_indices_e[mask_e] + n_input  # Offset excitatory indices

        # Combine input and excitatory spikes
        example_spike_times = np.concatenate([example_spike_times_input, example_spike_times_e])
        example_spike_indices = np.concatenate([example_spike_indices_input, example_spike_indices_e])

        # Compute spike counts per excitatory neuron for prediction (only excitatory used for classification)
        # Use original indices before offset was applied
        spike_counts = np.zeros(n_e)
        original_exc_indices = all_spike_indices_e[mask_e]  # Before offset
        for idx in original_exc_indices:
            if idx < n_e:
                spike_counts[idx] += 1

        # Compute prediction based on assignments
        summed_rates = [0] * 10
        num_assignments_per_class = [0] * 10
        for i in range(10):
            num_assignments_per_class[i] = len(np.where(assignments == i)[0])
            if num_assignments_per_class[i] > 0:
                summed_rates[i] = np.sum(spike_counts[assignments == i]) / num_assignments_per_class[i]
        prediction = np.argmax(summed_rates)
        all_predictions.append(prediction)

        # Compute binned activity for combined E+I population
        activity = compute_activity_trajectory(
            example_spike_times,
            example_spike_indices,
            n_total,  # Combined excitatory + inhibitory
            single_example_time,
            TIME_BINS_PER_EXAMPLE
        )
        all_activities.append(activity)

        # Rest period
        input_groups['Xe'].rates = 0*Hz
        net.run(resting_time)

    print("\nComputing PCA...")

    # Stack all activity vectors for PCA fitting
    # Shape: (n_examples * n_bins, n_neurons)
    all_activity_stacked = np.vstack(all_activities)

    # Apply Gaussian smoothing to each neuron's time series if requested
    # This helps create smoother trajectories from sparse spike data
    if args.smooth > 0:
        print(f"Applying Gaussian smoothing (sigma={args.smooth})...")
        for ex_idx in range(num_examples):
            start_idx = ex_idx * TIME_BINS_PER_EXAMPLE
            end_idx = (ex_idx + 1) * TIME_BINS_PER_EXAMPLE
            for neuron_idx in range(n_total):  # Combined E+I neurons
                all_activity_stacked[start_idx:end_idx, neuron_idx] = gaussian_filter1d(
                    all_activity_stacked[start_idx:end_idx, neuron_idx],
                    sigma=args.smooth
                )

    # Add small noise to avoid issues with zero-variance features
    all_activity_stacked = all_activity_stacked + np.random.randn(*all_activity_stacked.shape) * 0.01

    # Fit PCA
    pca = PCA(n_components=3)
    all_activity_pca = pca.fit_transform(all_activity_stacked)

    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    # Reshape back to per-example trajectories
    trajectories_3d = []
    for i in range(num_examples):
        start_idx = i * TIME_BINS_PER_EXAMPLE
        end_idx = (i + 1) * TIME_BINS_PER_EXAMPLE
        trajectories_3d.append(all_activity_pca[start_idx:end_idx])

    # Create color mapping
    unique_labels = sorted(set(selected_labels))
    colors_per_class = {label: i for i, label in enumerate(unique_labels)}

    # Print prediction accuracy
    correct = sum(1 for p, l in zip(all_predictions, selected_labels) if p == l)
    print(f"\nPrediction accuracy: {correct}/{num_examples} ({100*correct/num_examples:.1f}%)")
    print(f"Predictions: {list(zip(selected_labels, all_predictions))}")

    # Create animated visualization
    print("\nCreating visualization...")
    fig = create_trajectory_animation(trajectories_3d, selected_labels, all_predictions, colors_per_class)

    # Save to HTML file
    output_path = os.path.join(os.path.dirname(__file__), 'neural_trajectory_pca.html')
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")

    # Also show in browser
    fig.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
