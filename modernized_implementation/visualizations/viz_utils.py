"""
Shared Utilities for SNN Visualizations

Common functions for setting up the network, loading data, running simulations,
and computing neural activity metrics used across all visualization scripts.
"""

import numpy as np
import os
import sys
from brian2 import *

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Default paths
DEFAULT_WEIGHT_PATH = '../mnist_data/weights/'
DEFAULT_RANDOM_PATH = '../mnist_data/random/'
DEFAULT_ACTIVITY_PATH = '../mnist_data/activity/'


def get_base_path():
    """Get the base path for the migration folder"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_network(cfg, custom_weight_path=None, monitor_inhibitory=False):
    """Set up the Brian2 network for visualization/analysis.

    Args:
        cfg: Config object
        custom_weight_path: Optional path to trained weights (overrides cfg.weight_path)
        monitor_inhibitory: If True, also return inhibitory spike monitor

    Returns:
        Tuple of (net, input_groups, neuron_groups, spike_monitor_e, spike_monitor_input, n_input, n_e)
        If monitor_inhibitory=True, also includes spike_monitor_i and n_i
    """
    # Extract parameters from config - need to be global for Brian2 namespace resolution
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

    # Get neuron equations from config
    neuron_eqs_e = cfg.get_neuron_eqs_e()
    neuron_eqs_i = cfg.get_neuron_eqs_i()

    # Threshold and reset strings
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

    # Create neuron groups
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

    # Create spike monitors
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

    # Create namespace dict for net.run() calls
    brian_namespace = {
        'v_rest_e': v_rest_e,
        'v_rest_i': v_rest_i,
        'v_reset_e': v_reset_e,
        'v_reset_i': v_reset_i,
        'v_thresh_i': v_thresh_i,
        'refrac_e': refrac_e,
        'refrac_i': refrac_i,
        'offset': offset,
        'v_thresh_e_const': v_thresh_e_const,
    }

    if monitor_inhibitory:
        spike_monitor_i = SpikeMonitor(neuron_groups['Ai'])
        net.add(spike_monitor_i)
        return net, input_groups, neuron_groups, connections, spike_monitor_e, spike_monitor_input, spike_monitor_i, n_input, n_e, n_i, brian_namespace

    return net, input_groups, neuron_groups, connections, spike_monitor_e, spike_monitor_input, n_input, n_e, brian_namespace


def load_assignments(weight_path, n_e, cfg):
    """Load neuron class assignments from activity files.

    Args:
        weight_path: Path to weights folder
        n_e: Number of excitatory neurons
        cfg: Config object

    Returns:
        assignments: Array of class assignments per neuron (-1 if unassigned)
    """
    assignments_path = weight_path.replace('weights/', 'activity/')
    try:
        assignments_file = None
        for candidate in ['resultPopVecs2500_clean.npy', 'resultPopVecs5000_clean.npy',
                         'resultPopVecs10000_clean.npy', 'resultPopVecs2500.npy']:
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
        return assignments
    except Exception as e:
        print(f"Warning: Could not load assignments ({e}), predictions will be random")
        return np.zeros(n_e)


def compute_prediction(spike_counts, assignments, n_classes=10):
    """Compute prediction based on spike counts and neuron assignments.

    Args:
        spike_counts: Array of spike counts per excitatory neuron
        assignments: Array of class assignments per neuron
        n_classes: Number of classes (default 10)

    Returns:
        prediction: Predicted class label
    """
    summed_rates = [0] * n_classes
    num_assignments_per_class = [0] * n_classes
    for i in range(n_classes):
        num_assignments_per_class[i] = len(np.where(assignments == i)[0])
        if num_assignments_per_class[i] > 0:
            summed_rates[i] = np.sum(spike_counts[assignments == i]) / num_assignments_per_class[i]
    return np.argmax(summed_rates)


def compute_activity_trajectory(spike_times, spike_indices, n_neurons, total_time, n_bins):
    """Convert spike train data into binned activity vectors.

    Args:
        spike_times: Array of spike times
        spike_indices: Array of neuron indices for each spike
        n_neurons: Total number of neurons
        total_time: Total time duration (seconds)
        n_bins: Number of time bins

    Returns:
        activity: Array of shape (n_bins, n_neurons) with spike counts
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


def load_weight_matrix(weight_path, n_input=784, n_e=400):
    """Load the XeAe weight matrix.

    Args:
        weight_path: Path to weights folder
        n_input: Number of input neurons
        n_e: Number of excitatory neurons

    Returns:
        weight_matrix: 2D array of shape (n_input, n_e)
    """
    try:
        readout = np.load(weight_path + 'XeAe.npy', allow_pickle=True)
        weight_matrix = np.zeros((n_input, n_e))
        if not readout.shape == (0,):
            weight_matrix[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
        return weight_matrix
    except FileNotFoundError:
        print(f"Warning: Could not load weights from {weight_path}")
        return None


def load_mnist_data(cfg, use_test_set=True):
    """Load and filter MNIST data.

    Args:
        cfg: Config object (must have data_path, mnist_data_path, and mnist_classes attributes)
        use_test_set: If True, load test set; otherwise load training set

    Returns:
        images: Array of images
        labels: Array of labels
    """
    # Ensure data paths are set correctly relative to migration folder
    base_path = get_base_path()
    if not os.path.isabs(cfg.data_path) or not os.path.exists(cfg.data_path):
        cfg.data_path = os.path.join(base_path, 'mnist_data') + '/'
    if not hasattr(cfg, 'mnist_data_path') or cfg.mnist_data_path is None or not os.path.exists(cfg.mnist_data_path):
        cfg.mnist_data_path = cfg.data_path

    data_loader = MNISTDataLoader(cfg)

    if use_test_set:
        images, labels = data_loader.load_test_data()
    else:
        images, labels = data_loader.load_training_data()

    # Filter by classes if specified
    if cfg.mnist_classes is not None:
        mask = np.isin(labels, cfg.mnist_classes)
        images = images[mask]
        labels = labels[mask]

    return images, labels


def select_balanced_examples(labels, examples_per_class, classes=None, random_seed=None):
    """Select balanced examples from each class.

    Args:
        labels: Array of labels
        examples_per_class: Number of examples per class
        classes: List of classes to include (default: unique labels)
        random_seed: Random seed for reproducibility

    Returns:
        selected_indices: List of selected indices
        selected_labels: List of corresponding labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if classes is None:
        classes = sorted(np.unique(labels))

    selected_indices = []
    selected_labels = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        n_available = len(cls_indices)
        n_select = min(examples_per_class, n_available)

        chosen = np.random.choice(cls_indices, n_select, replace=False)
        selected_indices.extend(chosen)
        selected_labels.extend([cls] * n_select)

    return selected_indices, selected_labels


def get_output_path(filename):
    """Get the output path for a visualization file.

    Args:
        filename: Name of the output file

    Returns:
        Full path to the output file in the visualizations directory
    """
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(viz_dir, filename)


# Color schemes for consistent visualization
DIGIT_COLORS = [
    '#1f77b4',  # 0 - blue
    '#ff7f0e',  # 1 - orange
    '#2ca02c',  # 2 - green
    '#d62728',  # 3 - red
    '#9467bd',  # 4 - purple
    '#8c564b',  # 5 - brown
    '#e377c2',  # 6 - pink
    '#7f7f7f',  # 7 - gray
    '#bcbd22',  # 8 - olive
    '#17becf',  # 9 - cyan
]

def get_digit_color(digit):
    """Get the color for a digit class."""
    return DIGIT_COLORS[digit % len(DIGIT_COLORS)]


# =============================================================================
# Modern Matplotlib Styling for PDF Export
# =============================================================================

def setup_modern_style():
    """Configure matplotlib with a modern, publication-ready style.

    Returns a dict of style parameters that can be used with plt.rcParams.update()
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    style_params = {
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 150,
        'figure.figsize': (10, 6),

        # Axes
        'axes.facecolor': '#fafafa',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.titlesize': 14,
        'axes.titleweight': 'medium',
        'axes.titlepad': 12,
        'axes.labelsize': 11,
        'axes.labelweight': 'medium',
        'axes.labelpad': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler(color=DIGIT_COLORS),

        # Grid
        'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,

        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#cccccc',
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,

        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,

        # Savefig
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # PDF
        'pdf.fonttype': 42,  # TrueType fonts for better compatibility
    }

    plt.rcParams.update(style_params)
    return style_params


def save_figure(fig, filename, output_dir=None):
    """Save figure as PDF with proper formatting.

    Args:
        fig: matplotlib Figure object
        filename: Base filename (without extension)
        output_dir: Output directory (defaults to visualizations/figures/)

    Returns:
        Path to saved PDF file
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

    os.makedirs(output_dir, exist_ok=True)

    if not filename.endswith('.pdf'):
        filename = filename + '.pdf'

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    return output_path


def create_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create a figure with modern styling applied.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments passed to plt.subplots

    Returns:
        fig, axes tuple
    """
    import matplotlib.pyplot as plt

    setup_modern_style()

    if figsize is None:
        # Calculate sensible default based on subplot grid
        width = 5 * ncols
        height = 4 * nrows
        figsize = (min(width, 16), min(height, 12))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.patch.set_facecolor('white')

    return fig, axes


def add_colorbar(fig, mappable, ax, label=None, **kwargs):
    """Add a nicely formatted colorbar.

    Args:
        fig: Figure object
        mappable: The image/contour to create colorbar for
        ax: Axes to attach colorbar to
        label: Colorbar label
        **kwargs: Additional colorbar arguments
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)

    if label:
        cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    return cbar
