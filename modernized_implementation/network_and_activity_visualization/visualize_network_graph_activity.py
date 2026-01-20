"""
Network Graph Activity Visualization - Animated GIF

Shows network topology as a graph with nodes lighting up when neurons spike.
Combines structural view (graph layout) with temporal dynamics (spiking activity).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import networkx as nx
from tqdm import tqdm
from brian2 import *
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Configuration
cfg = Config()
cfg.set_test_mode(True)

# Visualization parameters
NUM_EXAMPLES = 5
OUTPUT_PATH = "analysis_figures/network_graph_activity.gif"
FPS = 30  # Frames per second for playback
FRAME_DURATION_MS = 0.5  # ms per frame (matches simulation dt timestep)
NODE_DECAY_RATE = 0.80  # How fast node brightness fades

# Graph visualization parameters - SHOW ALL NEURONS
SHOW_ALL_NEURONS = True  # Display the entire network
# For edge visualization (to avoid clutter, we sample edges)
EDGE_SAMPLE_RATE = 0.02  # Show 2% of inhibitory connections for clarity
SHOW_TOP_INPUT_CONNECTIONS_PER_EXC = 3  # Top N input connections per exc neuron

os.makedirs("analysis_figures", exist_ok=True)

print("="*70)
print("NETWORK GRAPH ACTIVITY VISUALIZATION")
print("="*70)

# Load configuration and data
n_input = cfg.n_input
n_e = cfg.n_e
single_example_time = cfg.single_example_time
resting_time = cfg.resting_time
dt = cfg.dt
data_path = cfg.data_path
weight_path = cfg.weight_path

print("\nLoading test data...")
data_loader = MNISTDataLoader(cfg)
test_images, test_labels = data_loader.load_test_data()

# Select examples
example_indices = np.random.choice(len(test_images), NUM_EXAMPLES, replace=False)
viz_images = test_images[example_indices]
viz_labels = test_labels[example_indices]

print(f"Selected {NUM_EXAMPLES} examples: digits {viz_labels}")

# Build network (same as main visualization)
print("\nBuilding network...")

defaultclock.dt = dt
np.random.seed(cfg.random_seed)

# Create neuron groups
neuron_groups = {}

neuron_groups['Ae'] = NeuronGroup(
    n_e, cfg.get_neuron_eqs_e(),
    threshold=cfg.get_v_thresh_e_str(), refractory=cfg.refrac_e,
    reset=cfg.get_scr_e(), method='euler',
    namespace={'v_rest_e': cfg.v_rest_e, 'v_reset_e': cfg.v_reset_e,
               'tc_theta': cfg.tc_theta, 'theta_plus_e': cfg.theta_plus_e,
               'refrac_e': cfg.refrac_e, 'v_thresh_e_const': cfg.v_thresh_e_const,
               'offset': cfg.offset}
)

neuron_groups['Ai'] = NeuronGroup(
    n_e, cfg.get_neuron_eqs_i(),
    threshold='v>v_thresh_i', refractory=cfg.refrac_i,
    reset='v=v_reset_i', method='euler',
    namespace={'v_rest_i': cfg.v_rest_i, 'v_thresh_i': cfg.v_thresh_i,
               'v_reset_i': cfg.v_reset_i, 'refrac_i': cfg.refrac_i}
)

input_groups = {'Xe': PoissonGroup(n_input, rates=0*Hz)}

spike_monitors = {
    'Ae': SpikeMonitor(neuron_groups['Ae']),
    'Ai': SpikeMonitor(neuron_groups['Ai']),
    'Xe': SpikeMonitor(input_groups['Xe'])
}

# Load connections
connections = {}

XeAe_weights = np.load(weight_path + 'XeAe.npy', allow_pickle=True)
conn_matrix = np.zeros((n_input, n_e))
for i, j, w in XeAe_weights:
    conn_matrix[int(i), int(j)] = w

connections['XeAe'] = Synapses(input_groups['Xe'], neuron_groups['Ae'],
                               model='w : 1', on_pre='ge_post += w', method='euler')
sources, targets = np.where(conn_matrix > 0)
connections['XeAe'].connect(i=sources, j=targets)
connections['XeAe'].w = conn_matrix[sources, targets]

AeAi_matrix = np.load(data_path + 'random/AeAi.npy', allow_pickle=True)
connections['AeAi'] = Synapses(neuron_groups['Ae'], neuron_groups['Ai'],
                               model='w : 1', on_pre='ge_post += w', method='euler')
connections['AeAi'].connect(condition='i==j')
conn_weights = np.zeros(n_e)
for i, j, w in AeAi_matrix:
    if int(i) == int(j):
        conn_weights[int(i)] = w
connections['AeAi'].w = conn_weights

AiAe_matrix = np.load(data_path + 'random/AiAe.npy', allow_pickle=True)
connections['AiAe'] = Synapses(neuron_groups['Ai'], neuron_groups['Ae'],
                               model='w : 1', on_pre='gi_post += w', method='euler')
connections['AiAe'].connect(condition='i!=j')
aiae_weights = np.zeros((n_e, n_e))
for i, j, w in AiAe_matrix:
    aiae_weights[int(i), int(j)] = w
connections['AiAe'].w = aiae_weights[connections['AiAe'].i[:], connections['AiAe'].j[:]]

theta_values = np.load(weight_path + 'theta_A.npy', allow_pickle=True)
neuron_groups['Ae'].theta = theta_values * volt

neuron_groups['Ae'].v = cfg.v_rest_e
neuron_groups['Ai'].v = cfg.v_rest_i

net = Network()
net.add(neuron_groups.values())
net.add(input_groups.values())
net.add(connections.values())
net.add(spike_monitors.values())

print("Network built")

# Run simulation
print(f"\nRunning simulation for {NUM_EXAMPLES} examples...")

current_time = 0
example_boundaries = []

for example_idx in range(NUM_EXAMPLES):
    rates = viz_images[example_idx].reshape((n_input)) / 8. * cfg.input_intensity
    input_groups['Xe'].rates = rates * Hz
    net.run(single_example_time, report=None)
    current_time += single_example_time / ms
    input_groups['Xe'].rates = 0 * Hz
    net.run(resting_time, report=None)
    current_time += resting_time / ms
    example_boundaries.append(current_time)

print("Simulation complete")

# Get spike data
input_spike_times = np.array(spike_monitors['Xe'].t / ms)
input_spike_neurons = np.array(spike_monitors['Xe'].i)
exc_spike_times = np.array(spike_monitors['Ae'].t / ms)
exc_spike_neurons = np.array(spike_monitors['Ae'].i)
inh_spike_times = np.array(spike_monitors['Ai'].t / ms)
inh_spike_neurons = np.array(spike_monitors['Ai'].i)

# Load assignments and compute predictions
print("\nLoading neuron assignments and computing predictions...")

# Load activity data for assignments (use CLEAN files for better accuracy)
activity_path = data_path + 'activity/'
input_numbers_file = np.load(activity_path + 'inputNumbers2500_clean.npy', allow_pickle=True)
result_pop_vecs = np.load(activity_path + 'resultPopVecs2500_clean.npy', allow_pickle=True)

# Compute neuron assignments
def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    for j in classes_to_check:
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments

assignments = get_new_assignments(result_pop_vecs, input_numbers_file)

# Function to get prediction from spike counts
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

# Compute predictions for each example
predictions = []
for ex_idx in range(NUM_EXAMPLES):
    start_time = ex_idx * (single_example_time + resting_time) / ms
    end_time = start_time + (single_example_time / ms)
    mask = (exc_spike_times >= start_time) & (exc_spike_times < end_time)
    exc_spikes_in_example = exc_spike_neurons[mask]

    if len(exc_spikes_in_example) > 0:
        spike_counts = np.bincount(exc_spikes_in_example, minlength=n_e)
        ranking = get_recognized_number_ranking(assignments, spike_counts)
        pred = ranking[0]
    else:
        pred = -1

    predictions.append(pred)
    print(f"  Example {ex_idx+1}: True={viz_labels[ex_idx]}, Predicted={pred}")

print(f"Predictions computed: {predictions}")

# Build NetworkX graph for visualization - ALL NEURONS
print("\nBuilding network graph with ALL neurons...")

G = nx.DiGraph()

# Add ALL INPUT nodes (784 neurons)
for i in range(n_input):
    G.add_node(f'X{i}', neuron_type='input', layer=0, idx=i)

# Add ALL EXCITATORY nodes (400 neurons)
for i in range(n_e):
    G.add_node(f'E{i}', neuron_type='excitatory', layer=1, idx=i)

# Add ALL INHIBITORY nodes (400 neurons)
for i in range(n_e):
    G.add_node(f'I{i}', neuron_type='inhibitory', layer=2, idx=i)

# Add edges (sample for visualization clarity)
edge_count = 0

# Input → Excitatory connections (sample top connections per excitatory)
print("  Adding Input→Excitatory edges...")
for exc_idx in range(n_e):
    input_weights = conn_matrix[:, exc_idx]
    top_inputs = np.argsort(input_weights)[-SHOW_TOP_INPUT_CONNECTIONS_PER_EXC:]
    for input_idx in top_inputs:
        if input_weights[input_idx] > 0:
            G.add_edge(f'X{input_idx}', f'E{exc_idx}',
                      weight=float(input_weights[input_idx]),
                      conn_type='input_to_exc')
            edge_count += 1

# Excitatory → Inhibitory (ALL one-to-one connections)
print("  Adding Excitatory→Inhibitory edges...")
for i in range(n_e):
    G.add_edge(f'E{i}', f'I{i}', weight=1.0, conn_type='exc_to_inh')
    edge_count += 1

# Inhibitory → Excitatory (sample lateral inhibition to reduce clutter)
print("  Adding Inhibitory→Excitatory edges (sampled)...")
np.random.seed(42)  # For reproducible sampling
for i in range(n_e):
    for j in range(n_e):
        if i != j and np.random.random() < EDGE_SAMPLE_RATE:
            G.add_edge(f'I{i}', f'E{j}', weight=1.0, conn_type='lateral_inh')
            edge_count += 1

# Position nodes in three distinct columns/layers
print("  Positioning nodes...")
pos = {}

# Layer 0: Input neurons (28x28 grid on left)
input_grid_size = 28  # MNIST is 28x28
for i in range(n_input):
    row = i // input_grid_size
    col = i % input_grid_size
    pos[f'X{i}'] = (col * 0.4, -row * 0.4)

# Layer 1: Excitatory neurons (20x20 grid in middle)
exc_grid_size = 20  # 400 = 20x20
for i in range(n_e):
    row = i // exc_grid_size
    col = i % exc_grid_size
    pos[f'E{i}'] = (col * 0.5 + 14, -row * 0.5)

# Layer 2: Inhibitory neurons (20x20 grid on right)
for i in range(n_e):
    row = i // exc_grid_size
    col = i % exc_grid_size
    pos[f'I{i}'] = (col * 0.5 + 26, -row * 0.5)

print(f"Graph built:")
print(f"  Input nodes: {n_input}")
print(f"  Excitatory nodes: {n_e}")
print(f"  Inhibitory nodes: {n_e}")
print(f"  Total nodes: {len(G.nodes())}")
print(f"  Total edges: {edge_count}")

# Create animation
print("\nGenerating animation...")

total_time = current_time
time_step = FRAME_DURATION_MS
num_frames = int(total_time / time_step)

fig = plt.figure(figsize=(18, 10), facecolor='#1a1a1a')
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, height_ratios=[3, 1])

# Main graph view
ax_graph = fig.add_subplot(gs[0, :])
ax_graph.set_facecolor('#1a1a1a')
ax_graph.set_title('Network Graph (Nodes light up when neurons spike)',
                   color='white', fontsize=16, fontweight='bold')
ax_graph.axis('off')

# Activity plot
ax_activity = fig.add_subplot(gs[1, 0])
ax_activity.set_facecolor('#1a1a1a')
ax_activity.set_title('Population Activity', color='white', fontsize=12)
ax_activity.set_xlabel('Time (ms)', color='white')
ax_activity.set_ylabel('Spikes per frame', color='white')
ax_activity.tick_params(colors='white')
for spine in ax_activity.spines.values():
    spine.set_color('white')

# Label info
ax_label = fig.add_subplot(gs[1, 1])
ax_label.set_facecolor('#1a1a1a')
ax_label.axis('off')

# Initialize node activity state
node_activity = {node: 0.0 for node in G.nodes()}

# Track activity over time
activity_times = []
exc_counts = []
inh_counts = []

def update_frame(frame_num):
    global node_activity

    current_time_ms = frame_num * time_step
    window_start = current_time_ms
    window_end = current_time_ms + time_step

    # Decay all node activities
    for node in node_activity:
        node_activity[node] *= NODE_DECAY_RATE

    # Update activities for spiking neurons
    input_mask = (input_spike_times >= window_start) & (input_spike_times < window_end)
    input_spikes_now = input_spike_neurons[input_mask]

    exc_mask = (exc_spike_times >= window_start) & (exc_spike_times < window_end)
    exc_spikes_now = exc_spike_neurons[exc_mask]

    inh_mask = (inh_spike_times >= window_start) & (inh_spike_times < window_end)
    inh_spikes_now = inh_spike_neurons[inh_mask]

    # Mark all spiking input neurons
    for neuron_id in input_spikes_now:
        node_activity[f'X{neuron_id}'] = 1.0

    # Mark all spiking excitatory neurons
    for neuron_id in exc_spikes_now:
        node_activity[f'E{neuron_id}'] = 1.0

    # Mark all spiking inhibitory neurons
    for neuron_id in inh_spikes_now:
        node_activity[f'I{neuron_id}'] = 1.0

    # Clear and redraw graph
    ax_graph.clear()
    ax_graph.set_facecolor('#1a1a1a')
    ax_graph.axis('off')
    ax_graph.set_title('Network Graph (Nodes light up when neurons spike)',
                      color='white', fontsize=16, fontweight='bold')

    # Prepare node colors based on activity
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        activity = node_activity[node]
        if node.startswith('X'):
            # Input: red when inactive, bright orange when active
            color = (0.6 + activity * 0.4, activity * 0.4, activity * 0.2)
            node_sizes.append(20 + activity * 80)
        elif node.startswith('E'):
            # Excitatory: blue when inactive, bright cyan when active
            color = (activity * 0.5, activity * 0.8, 0.4 + activity * 0.6)
            node_sizes.append(40 + activity * 160)
        else:  # Inhibitory (I)
            # Inhibitory: green when inactive, bright lime when active
            color = (activity * 0.5, 0.4 + activity * 0.6, activity * 0.5)
            node_sizes.append(40 + activity * 160)
        node_colors.append(color)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          ax=ax_graph, alpha=0.8, edgecolors='white', linewidths=0.5)

    # Draw edges with transparency
    input_to_exc_edges = [(u, v) for u, v, d in G.edges(data=True)
                          if d.get('conn_type') == 'input_to_exc']
    exc_to_inh_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('conn_type') == 'exc_to_inh']
    lat_inh_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get('conn_type') == 'lateral_inh']

    if input_to_exc_edges:
        nx.draw_networkx_edges(G, pos, edgelist=input_to_exc_edges, edge_color='#ff9999',
                              width=0.3, ax=ax_graph, alpha=0.2, arrows=True,
                              arrowsize=3, arrowstyle='->')

    if exc_to_inh_edges:
        nx.draw_networkx_edges(G, pos, edgelist=exc_to_inh_edges, edge_color='#6666ff',
                              width=0.5, ax=ax_graph, alpha=0.3, arrows=True,
                              arrowsize=5, arrowstyle='->')

    if lat_inh_edges:
        nx.draw_networkx_edges(G, pos, edgelist=lat_inh_edges, edge_color='#66ff66',
                              width=0.2, ax=ax_graph, alpha=0.15, arrows=True,
                              arrowsize=3, arrowstyle='->', connectionstyle='arc3,rad=0.1')

    # Add layer labels (positioned based on actual grid layouts)
    ax_graph.text(5, 1, 'Input Layer (784)', color='#ff6666',
                 fontsize=12, fontweight='bold', ha='center')
    ax_graph.text(19, 1, 'Excitatory (400)', color='#6666ff',
                 fontsize=12, fontweight='bold', ha='center')
    ax_graph.text(31, 1, 'Inhibitory (400)', color='#66ff66',
                 fontsize=12, fontweight='bold', ha='center')

    # Update activity plot
    activity_times.append(current_time_ms)
    exc_counts.append(len(exc_spikes_now))
    inh_counts.append(len(inh_spikes_now))

    ax_activity.clear()
    ax_activity.set_facecolor('#1a1a1a')
    ax_activity.plot(activity_times, exc_counts, color='#6666ff',
                    label='Excitatory', linewidth=2, alpha=0.8)
    ax_activity.plot(activity_times, inh_counts, color='#66ff66',
                    label='Inhibitory', linewidth=2, alpha=0.8)

    for boundary_time in example_boundaries:
        if boundary_time <= current_time_ms:
            ax_activity.axvline(boundary_time, color='yellow', linestyle='--',
                               alpha=0.5, linewidth=1)

    ax_activity.set_xlim(0, total_time)
    ax_activity.set_ylim(0, max(max(exc_counts + [1]), max(inh_counts + [1])) * 1.1)
    ax_activity.set_xlabel('Time (ms)', color='white')
    ax_activity.set_ylabel('Spikes per frame', color='white')
    ax_activity.set_title('Population Activity', color='white', fontsize=12)
    ax_activity.tick_params(colors='white')
    for spine in ax_activity.spines.values():
        spine.set_color('white')
    ax_activity.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax_activity.grid(alpha=0.2, color='white')

    # Update label
    current_example = 0
    for i, boundary in enumerate(example_boundaries):
        if current_time_ms < boundary:
            current_example = i
            break

    ax_label.clear()
    ax_label.set_facecolor('#1a1a1a')
    ax_label.axis('off')

    # Show true digit and prediction
    true_digit = viz_labels[current_example]
    pred_digit = predictions[current_example]
    correct = (true_digit == pred_digit)

    ax_label.text(0.5, 0.7, f'True: {true_digit}',
                 ha='center', va='center', color='white', fontsize=32, fontweight='bold')

    pred_color = '#00ff00' if correct else '#ff4444'  # Green if correct, red if wrong
    ax_label.text(0.5, 0.4, f'Pred: {pred_digit}',
                 ha='center', va='center', color=pred_color, fontsize=32, fontweight='bold')

    ax_label.text(0.5, 0.15, f'Example {current_example + 1}/{NUM_EXAMPLES}',
                 ha='center', va='center', color='white', fontsize=12)
    ax_label.text(0.5, 0.0, f'Time: {current_time_ms:.0f} ms',
                 ha='center', va='center', color='white', fontsize=10)

print("  Creating frames...")
anim = animation.FuncAnimation(
    fig, update_frame,
    frames=tqdm(range(num_frames), desc='Generating frames'),
    interval=1000/FPS, blit=False, repeat=True
)

print(f"\nSaving animation to {OUTPUT_PATH}...")
writer = animation.PillowWriter(fps=FPS)
anim.save(OUTPUT_PATH, writer=writer, dpi=100)

print("\n" + "="*70)
print("GRAPH ACTIVITY VISUALIZATION COMPLETE!")
print("="*70)
print(f"Saved to: {OUTPUT_PATH}")
print(f"Showing ALL {len(G.nodes())} neurons (784 input + 400 excitatory + 400 inhibitory)")
print(f"Duration: {total_time/1000:.2f} seconds")
print(f"Frames: {num_frames} at {FPS} FPS")
