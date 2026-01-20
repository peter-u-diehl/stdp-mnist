"""
Visualize Network Activity Over Time - Diehl & Cook 2015 Network

Creates animated GIF showing how the spiking network responds to MNIST digits.
Shows input layer (784 neurons in 28x28 grid) and excitatory layer (400 neurons in 20x20 grid).

Inspired by: github.com/ImagineOrange/Spiking-Neural-Network-Experiments/LIF_utils
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from brian2 import *
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Configuration
cfg = Config()
cfg.set_test_mode(True)  # Use test mode (load trained weights)

# Visualization parameters
NUM_EXAMPLES = 3  # How many MNIST digits to visualize
OUTPUT_PATH = "analysis_figures/network_activity.gif"
FPS = 100  # Frames per second in output GIF (higher = see individual spikes)
FRAME_DURATION_MS = 10.0  # Duration per frame in milliseconds (lower = less compression)
INPUT_DECAY_RATE = 0.85  # Input layer decay per frame (0-1, higher = longer trails)
EXC_DECAY_RATE = 0.85  # Excitatory layer decay (slower to accumulate sparse spikes)
EXC_SPIKE_BRIGHTNESS = 1.5  # How bright excitatory spikes appear (>1.0 for accumulation)
EXC_VMAX = 0.8  # Color scale max for excitatory layer (lower = more visible)

# Create output directory
os.makedirs("analysis_figures", exist_ok=True)

print("=" * 60)
print("NETWORK ACTIVITY VISUALIZATION")
print("=" * 60)

# Load network configuration
print("\nLoading configuration...")
n_input = cfg.n_input
n_e = cfg.n_e
single_example_time = cfg.single_example_time
resting_time = cfg.resting_time
dt = cfg.dt

# Calculate grid dimensions
input_grid_size = int(np.sqrt(n_input))  # 28x28 for MNIST
exc_grid_size = int(np.sqrt(n_e))  # 20x20 for 400 neurons

print(f"  Input layer: {n_input} neurons ({input_grid_size}x{input_grid_size} grid)")
print(f"  Excitatory layer: {n_e} neurons ({exc_grid_size}x{exc_grid_size} grid)")
print(f"  Time per example: {single_example_time}")

# Load data
print("\nLoading test data...")
data_loader = MNISTDataLoader(cfg)
test_images, test_labels = data_loader.load_test_data()

# Select specific examples to visualize
example_indices = np.random.choice(len(test_images), NUM_EXAMPLES, replace=False)
viz_images = test_images[example_indices]
viz_labels = test_labels[example_indices]

print(f"Selected {NUM_EXAMPLES} examples:")
for i, (idx, label) in enumerate(zip(example_indices, viz_labels)):
    print(f"  Example {i+1}: Digit {label} (index {idx})")

# Build network
print("\nBuilding network...")

# Import network parameters
data_path = cfg.data_path
weight_path = cfg.weight_path
v_rest_e = cfg.v_rest_e
v_rest_i = cfg.v_rest_i
v_reset_e = cfg.v_reset_e
v_reset_i = cfg.v_reset_i
v_thresh_e_const = cfg.v_thresh_e_const
v_thresh_i = cfg.v_thresh_i
refrac_e = cfg.refrac_e
refrac_i = cfg.refrac_i
offset = cfg.offset

# Set simulation parameters
defaultclock.dt = dt
np.random.seed(cfg.random_seed)

# Create neuron groups
print("  Creating neurons...")
neuron_groups = {}
population_names = ['A']
input_population_names = ['X']

# Excitatory neurons
neuron_groups['Ae'] = NeuronGroup(
    n_e,
    cfg.get_neuron_eqs_e(),
    threshold=cfg.get_v_thresh_e_str(),
    refractory=refrac_e,
    reset=cfg.get_scr_e(),
    method='euler',
    namespace={
        'v_rest_e': v_rest_e,
        'tc_theta': cfg.tc_theta,
        'theta_plus_e': cfg.theta_plus_e,
        'refrac_e': refrac_e,
        'v_thresh_e_const': v_thresh_e_const,
        'offset': offset
    }
)

# Inhibitory neurons
neuron_groups['Ai'] = NeuronGroup(
    n_e,
    cfg.get_neuron_eqs_i(),
    threshold='v>v_thresh_i',
    refractory=refrac_i,
    reset='v=v_reset_i',
    method='euler',
    namespace={
        'v_rest_i': v_rest_i,
        'v_thresh_i': v_thresh_i,
        'v_reset_i': v_reset_i,
        'refrac_i': refrac_i
    }
)

# Input neurons
input_groups = {}
input_groups['Xe'] = PoissonGroup(n_input, rates=0*Hz)

# Spike monitors
spike_monitors = {
    'Ae': SpikeMonitor(neuron_groups['Ae']),
    'Ai': SpikeMonitor(neuron_groups['Ai']),
    'Xe': SpikeMonitor(input_groups['Xe'])
}

# Load connections
print("  Loading trained weights...")
connections = {}

# Input to excitatory
XeAe_weights = np.load(weight_path + 'XeAe.npy', allow_pickle=True)
conn_matrix = np.zeros((n_input, n_e))
for i, j, w in XeAe_weights:
    conn_matrix[int(i), int(j)] = w

connections['XeAe'] = Synapses(
    input_groups['Xe'],
    neuron_groups['Ae'],
    model='w : 1',
    on_pre='ge_post += w',
    method='euler'
)
# Connect only where weights exist (sparse connectivity - critical!)
sources, targets = np.where(conn_matrix > 0)
connections['XeAe'].connect(i=sources, j=targets)
connections['XeAe'].w = conn_matrix[sources, targets]

# Excitatory to inhibitory
AeAi_matrix = np.load(data_path + 'random/AeAi.npy', allow_pickle=True)
connections['AeAi'] = Synapses(
    neuron_groups['Ae'],
    neuron_groups['Ai'],
    model='w : 1',
    on_pre='ge_post += w',
    method='euler'
)
connections['AeAi'].connect(condition='i==j')  # One-to-one connection
conn_weights = np.zeros(n_e)
for i, j, w in AeAi_matrix:
    if int(i) == int(j):
        conn_weights[int(i)] = w
connections['AeAi'].w = conn_weights

# Inhibitory to excitatory (all-to-all except self)
AiAe_matrix = np.load(data_path + 'random/AiAe.npy', allow_pickle=True)
connections['AiAe'] = Synapses(
    neuron_groups['Ai'],
    neuron_groups['Ae'],
    model='w : 1',
    on_pre='gi_post += w',
    method='euler'
)
connections['AiAe'].connect(condition='i!=j')
# Create weight matrix
aiae_weights = np.zeros((n_e, n_e))
for i, j, w in AiAe_matrix:
    aiae_weights[int(i), int(j)] = w
connections['AiAe'].w = aiae_weights[connections['AiAe'].i[:], connections['AiAe'].j[:]]

# Load theta values (stored as volts, not millivolts!)
theta_values = np.load(weight_path + 'theta_A.npy', allow_pickle=True)
neuron_groups['Ae'].theta = theta_values * volt

# Initialize network state
neuron_groups['Ae'].v = v_rest_e
neuron_groups['Ai'].v = v_rest_i

# Build network
net = Network()
net.add(neuron_groups.values())
net.add(input_groups.values())
net.add(connections.values())
net.add(spike_monitors.values())

print("Network built and weights loaded")

# Run simulation and record activity
print(f"\nRunning simulation for {NUM_EXAMPLES} examples...")

# Storage for spike times
all_input_spikes = []
all_exc_spikes = []
example_boundaries = []  # Time points where examples change

current_time = 0
for example_idx in range(NUM_EXAMPLES):
    print(f"  Processing example {example_idx + 1}/{NUM_EXAMPLES} (digit {viz_labels[example_idx]})...")

    # Set input rates
    rates = viz_images[example_idx].reshape((n_input)) / 8. * cfg.input_intensity
    input_groups['Xe'].rates = rates * Hz

    # Record start time
    start_t = current_time

    # Present stimulus
    net.run(single_example_time, report=None)
    current_time += single_example_time / ms

    # Rest period
    input_groups['Xe'].rates = 0 * Hz
    net.run(resting_time, report=None)
    current_time += resting_time / ms

    # Record boundary
    example_boundaries.append(current_time)

print("Simulation complete")

# Prepare data for visualization
print("\nPreparing visualization data...")

# Get spike times and neurons
input_spike_times = np.array(spike_monitors['Xe'].t / ms)
input_spike_neurons = np.array(spike_monitors['Xe'].i)
exc_spike_times = np.array(spike_monitors['Ae'].t / ms)
exc_spike_neurons = np.array(spike_monitors['Ae'].i)
inh_spike_times = np.array(spike_monitors['Ai'].t / ms)
inh_spike_neurons = np.array(spike_monitors['Ai'].i)

total_time = current_time
time_step = FRAME_DURATION_MS  # ms per frame
num_frames = int(total_time / time_step)

print(f"  Total time: {total_time:.1f} ms")
print(f"  Frames: {num_frames} ({time_step:.1f} ms per frame)")
print(f"  Playback speed: {FPS} FPS (GIF will play at {total_time/(num_frames/FPS)/1000:.2f}x realtime)")

# Create animation
print("\nGenerating animation...")

fig = plt.figure(figsize=(20, 7), facecolor='#1a1a1a')
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

# Input layer visualization (28x28)
ax_input = fig.add_subplot(gs[:, 0])
ax_input.set_facecolor('#1a1a1a')
ax_input.set_title('Input Layer (784 neurons)', color='white', fontsize=14, fontweight='bold')
ax_input.set_xticks([])
ax_input.set_yticks([])

# Excitatory layer visualization (20x20)
ax_exc = fig.add_subplot(gs[:, 1])
ax_exc.set_facecolor('#1a1a1a')
ax_exc.set_title('Excitatory Layer (400 neurons)', color='white', fontsize=14, fontweight='bold')
ax_exc.set_xticks([])
ax_exc.set_yticks([])

# Inhibitory layer visualization (20x20)
ax_inh = fig.add_subplot(gs[:, 2])
ax_inh.set_facecolor('#1a1a1a')
ax_inh.set_title('Inhibitory Layer (400 neurons)', color='white', fontsize=14, fontweight='bold')
ax_inh.set_xticks([])
ax_inh.set_yticks([])

# Activity plot (spike counts over time)
ax_activity = fig.add_subplot(gs[0, 3])
ax_activity.set_facecolor('#1a1a1a')
ax_activity.set_title('Network Activity', color='white', fontsize=12, fontweight='bold')
ax_activity.set_xlabel('Time (ms)', color='white')
ax_activity.set_ylabel('Spikes per frame', color='white')
ax_activity.tick_params(colors='white')
ax_activity.spines['bottom'].set_color('white')
ax_activity.spines['left'].set_color('white')
ax_activity.spines['top'].set_visible(False)
ax_activity.spines['right'].set_visible(False)

# Digit label display
ax_label = fig.add_subplot(gs[1, 3])
ax_label.set_facecolor('#1a1a1a')
ax_label.axis('off')

# Initialize activity grids
input_activity = np.zeros((input_grid_size, input_grid_size))
exc_activity = np.zeros((exc_grid_size, exc_grid_size))
inh_activity = np.zeros((exc_grid_size, exc_grid_size))  # Same size as excitatory

# Initialize images (use hot colormap: black -> red -> yellow -> white)
im_input = ax_input.imshow(input_activity, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
im_exc = ax_exc.imshow(exc_activity, cmap='hot', vmin=0, vmax=EXC_VMAX, interpolation='nearest')
im_inh = ax_inh.imshow(inh_activity, cmap='hot', vmin=0, vmax=EXC_VMAX, interpolation='nearest')

# Initialize activity tracking
activity_times = []
input_counts = []
exc_counts = []
inh_counts = []

def update_frame(frame_num):
    global input_activity, exc_activity, inh_activity

    current_time_ms = frame_num * time_step
    window_start = current_time_ms
    window_end = current_time_ms + time_step

    # Decay existing activity (separate rates for each layer)
    input_activity *= INPUT_DECAY_RATE
    exc_activity *= EXC_DECAY_RATE
    inh_activity *= EXC_DECAY_RATE

    # Add new spikes in this time window
    input_mask = (input_spike_times >= window_start) & (input_spike_times < window_end)
    input_spikes_now = input_spike_neurons[input_mask]

    exc_mask = (exc_spike_times >= window_start) & (exc_spike_times < window_end)
    exc_spikes_now = exc_spike_neurons[exc_mask]

    inh_mask = (inh_spike_times >= window_start) & (inh_spike_times < window_end)
    inh_spikes_now = inh_spike_neurons[inh_mask]

    # Update input grid
    for neuron_id in input_spikes_now:
        row = neuron_id // input_grid_size
        col = neuron_id % input_grid_size
        input_activity[row, col] = 1.0

    # Update excitatory grid (with higher brightness for sparse spikes)
    for neuron_id in exc_spikes_now:
        row = neuron_id // exc_grid_size
        col = neuron_id % exc_grid_size
        # Add to existing activity (accumulation effect for sparse spikes)
        exc_activity[row, col] = min(exc_activity[row, col] + EXC_SPIKE_BRIGHTNESS, EXC_VMAX * 2)

    # Update inhibitory grid (same parameters as excitatory)
    for neuron_id in inh_spikes_now:
        row = neuron_id // exc_grid_size
        col = neuron_id % exc_grid_size
        # Add to existing activity (accumulation effect for sparse spikes)
        inh_activity[row, col] = min(inh_activity[row, col] + EXC_SPIKE_BRIGHTNESS, EXC_VMAX * 2)

    # Update images
    im_input.set_array(input_activity)
    im_exc.set_array(exc_activity)
    im_inh.set_array(inh_activity)

    # Update activity plot
    activity_times.append(current_time_ms)
    input_counts.append(len(input_spikes_now))
    exc_counts.append(len(exc_spikes_now))
    inh_counts.append(len(inh_spikes_now))

    ax_activity.clear()
    ax_activity.set_facecolor('#1a1a1a')
    ax_activity.plot(activity_times, input_counts, color='#ff6666', label='Input', linewidth=2, alpha=0.8)
    ax_activity.plot(activity_times, exc_counts, color='#6666ff', label='Excitatory', linewidth=2, alpha=0.8)
    ax_activity.plot(activity_times, inh_counts, color='#66ff66', label='Inhibitory', linewidth=2, alpha=0.8)

    # Mark example boundaries
    for boundary_time in example_boundaries:
        if boundary_time <= current_time_ms:
            ax_activity.axvline(boundary_time, color='yellow', linestyle='--', alpha=0.5, linewidth=1)

    ax_activity.set_xlim(0, total_time)
    ax_activity.set_ylim(0, max(max(input_counts + [1]), max(exc_counts + [1]), max(inh_counts + [1])) * 1.1)
    ax_activity.set_xlabel('Time (ms)', color='white')
    ax_activity.set_ylabel('Spikes per frame', color='white')
    ax_activity.set_title('Network Activity', color='white', fontsize=12, fontweight='bold')
    ax_activity.tick_params(colors='white')
    ax_activity.spines['bottom'].set_color('white')
    ax_activity.spines['left'].set_color('white')
    ax_activity.spines['top'].set_visible(False)
    ax_activity.spines['right'].set_visible(False)
    ax_activity.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax_activity.grid(alpha=0.2, color='white')

    # Update digit label
    current_example = 0
    for i, boundary in enumerate(example_boundaries):
        if current_time_ms < boundary:
            current_example = i
            break

    ax_label.clear()
    ax_label.set_facecolor('#1a1a1a')
    ax_label.axis('off')
    ax_label.text(0.5, 0.6, f'Digit: {viz_labels[current_example]}',
                  ha='center', va='center', color='white', fontsize=48, fontweight='bold')
    ax_label.text(0.5, 0.3, f'Example {current_example + 1}/{NUM_EXAMPLES}',
                  ha='center', va='center', color='white', fontsize=16)
    ax_label.text(0.5, 0.1, f'Time: {current_time_ms:.0f} ms',
                  ha='center', va='center', color='white', fontsize=14)

    return im_input, im_exc, im_inh

# Create animation
print("  Creating frames...")
anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=tqdm(range(num_frames), desc='Generating frames'),
    interval=1000/FPS,
    blit=False,
    repeat=True
)

# Save as GIF
print(f"\nSaving animation to {OUTPUT_PATH}...")
writer = animation.PillowWriter(fps=FPS)
anim.save(OUTPUT_PATH, writer=writer, dpi=80)

print(f"\n{'='*60}")
print("VISUALIZATION COMPLETE!")
print(f"{'='*60}")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Duration: {total_time/1000:.2f} seconds")
print(f"Frames: {num_frames}")
print(f"Frame rate: {FPS} FPS")
print(f"\nVisualized digits: {viz_labels}")
