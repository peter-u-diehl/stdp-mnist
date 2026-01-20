"""
Static Network Structure Visualization - 3D Version

Shows the network topology as a 3D graph with:
- Three distinct depth layers (Input, Excitatory, Inhibitory)
- Connection paths flowing through 3D space
- Interactive rotation capability
- Node colors representing neuron assignments
- Node sizes representing learned properties
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import networkx as nx
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Configuration
cfg = Config()
cfg.set_test_mode(True)

# Visualization parameters
OUTPUT_PATH = "analysis_figures/static_network_structure_3d.png"
OUTPUT_HTML = "analysis_figures/interactive_network_3d.html"
SHOW_TOP_INPUT_CONNECTIONS_PER_EXC = 5  # Top N input connections per exc neuron
EDGE_SAMPLE_RATE = 0.02  # Show 2% of inhibitory connections
RANDOM_SEED = 42

# 3D view angles
ELEVATION = 20  # Viewing angle from above
AZIMUTH = 45    # Rotation angle

# Animation parameters
NUM_EXAMPLES_TO_ANIMATE = 10  # Number of examples to show in the animation
ANIMATION_MS_PER_FRAME = 20  # Milliseconds per animation frame (faster: 50→20)
FRAMES_PER_MS = 2  # How many animation frames per simulation ms

os.makedirs("analysis_figures", exist_ok=True)

print("="*70)
print("STATIC NETWORK STRUCTURE VISUALIZATION - 3D")
print("="*70)

# Load configuration and network data
n_input = cfg.n_input
n_e = cfg.n_e
data_path = cfg.data_path
weight_path = cfg.weight_path

print("\nLoading network weights...")

# Load Input → Excitatory weights
XeAe_weights = np.load(weight_path + 'XeAe.npy', allow_pickle=True)
conn_matrix = np.zeros((n_input, n_e))
for i, j, w in XeAe_weights:
    conn_matrix[int(i), int(j)] = w

# Load Excitatory → Inhibitory weights (one-to-one)
AeAi_matrix = np.load(data_path + 'random/AeAi.npy', allow_pickle=True)
aeai_weights = np.zeros(n_e)
for i, j, w in AeAi_matrix:
    if int(i) == int(j):
        aeai_weights[int(i)] = w

# Load Inhibitory → Excitatory weights (lateral inhibition)
AiAe_matrix = np.load(data_path + 'random/AiAe.npy', allow_pickle=True)
aiae_weights = np.zeros((n_e, n_e))
for i, j, w in AiAe_matrix:
    aiae_weights[int(i), int(j)] = w

# Load theta values (adaptive thresholds)
theta_values = np.load(weight_path + 'theta_A.npy', allow_pickle=True)

print("Weights loaded")

# Load neuron assignments
print("\nLoading neuron assignments...")

activity_path = data_path + 'activity/'
input_numbers_file = np.load(activity_path + 'inputNumbers2500_clean.npy', allow_pickle=True)
result_pop_vecs = np.load(activity_path + 'resultPopVecs2500_clean.npy', allow_pickle=True)

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

# Get the configured classes
configured_classes = cfg.mnist_classes if cfg.mnist_classes is not None else list(range(10))
num_classes = len(configured_classes)

# Count assignments only for configured classes
assignment_counts = np.bincount(assignments.astype(int), minlength=max(configured_classes)+1)

print("Assignments loaded")
print(f"  Configured classes: {configured_classes}")
print(f"  Neuron assignments per digit: {[assignment_counts[i] for i in configured_classes]}")

# Build NetworkX graph
print("\nBuilding network graph...")

G = nx.DiGraph()

# Add input nodes
for i in range(n_input):
    G.add_node(f'X{i}', neuron_type='input', layer=0, idx=i)

# Add excitatory nodes with assignment info
for i in range(n_e):
    G.add_node(f'E{i}', neuron_type='excitatory', layer=1, idx=i,
               assignment=int(assignments[i]), theta=float(theta_values[i]))

# Add inhibitory nodes
for i in range(n_e):
    G.add_node(f'I{i}', neuron_type='inhibitory', layer=2, idx=i)

# Add edges with weight information
edge_count = 0
input_edge_weights = []

# Input → Excitatory connections (top connections per excitatory)
print("  Adding Input→Excitatory edges...")
for exc_idx in range(n_e):
    input_weights = conn_matrix[:, exc_idx]
    top_inputs = np.argsort(input_weights)[-SHOW_TOP_INPUT_CONNECTIONS_PER_EXC:]
    for input_idx in top_inputs:
        if input_weights[input_idx] > 0:
            weight = float(input_weights[input_idx])
            G.add_edge(f'X{input_idx}', f'E{exc_idx}',
                      weight=weight, conn_type='input_to_exc')
            input_edge_weights.append(weight)
            edge_count += 1

# Excitatory → Inhibitory (all one-to-one connections)
print("  Adding Excitatory→Inhibitory edges...")
for i in range(n_e):
    G.add_edge(f'E{i}', f'I{i}', weight=float(aeai_weights[i]), conn_type='exc_to_inh')
    edge_count += 1

# Inhibitory → Excitatory (sample lateral inhibition)
print("  Adding Inhibitory→Excitatory edges (sampled)...")
np.random.seed(RANDOM_SEED)
lateral_inh_weights = []
for i in range(n_e):
    for j in range(n_e):
        if i != j and np.random.random() < EDGE_SAMPLE_RATE:
            weight = float(aiae_weights[i, j])
            G.add_edge(f'I{i}', f'E{j}', weight=weight, conn_type='lateral_inh')
            lateral_inh_weights.append(weight)
            edge_count += 1

print(f"Graph built: {len(G.nodes())} nodes, {edge_count} edges")

# Position nodes in 3D space
print("  Positioning nodes in 3D...")
pos_3d = {}

# Layer 0: Input neurons (28x28 grid) - z=0 plane
# Wider layout for better visibility
input_grid_size = 28
for i in range(n_input):
    row = i // input_grid_size
    col = i % input_grid_size
    # Center the grid around origin with wider spacing
    x = (col - 13.5) * 0.8  # Increased from 0.5 to 0.8
    y = (row - 13.5) * 0.8  # Increased from 0.5 to 0.8
    z = 0.0
    pos_3d[f'X{i}'] = (x, y, z)

# Layer 1: Excitatory neurons (20x20 grid) - z=10 plane
exc_grid_size = 20
for i in range(n_e):
    row = i // exc_grid_size
    col = i % exc_grid_size
    # Center the grid around origin with wider spacing
    x = (col - 9.5) * 1.0  # Increased from 0.6 to 1.0
    y = (row - 9.5) * 1.0  # Increased from 0.6 to 1.0
    z = 10.0
    pos_3d[f'E{i}'] = (x, y, z)

# Layer 2: Inhibitory neurons (20x20 grid) - z=20 plane
for i in range(n_e):
    row = i // exc_grid_size
    col = i % exc_grid_size
    # Center the grid around origin with wider spacing
    x = (col - 9.5) * 1.0  # Increased from 0.6 to 1.0
    y = (row - 9.5) * 1.0  # Increased from 0.6 to 1.0
    z = 20.0
    pos_3d[f'I{i}'] = (x, y, z)

# Create 3D visualization
print("\nCreating 3D visualization...")

fig = plt.figure(figsize=(20, 12), facecolor='white')  # Wider: 16→20
ax_3d = fig.add_subplot(111, projection='3d')
ax_3d.set_title('3D Network Structure: Input → Excitatory → Inhibitory',
                fontsize=18, fontweight='bold', pad=20)

# Color map for digit assignments
digit_color_map = {}
if num_classes <= 10:
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in configured_classes:
        digit_color_map[digit] = base_colors[digit]
else:
    base_colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i, digit in enumerate(configured_classes):
        digit_color_map[digit] = base_colors[i]

# Prepare node data for 3D plotting
from matplotlib.colors import to_rgba

node_xs, node_ys, node_zs = [], [], []
node_colors = []
node_sizes = []

for node in G.nodes():
    x, y, z = pos_3d[node]
    node_xs.append(x)
    node_ys.append(y)
    node_zs.append(z)

    if node.startswith('X'):
        # Input: gray with small size
        node_colors.append(to_rgba('#888888'))
        node_sizes.append(10)
    elif node.startswith('E'):
        # Excitatory: colored by digit assignment
        assignment = int(G.nodes[node]['assignment'])
        if assignment in digit_color_map:
            node_colors.append(to_rgba(digit_color_map[assignment]))
        else:
            node_colors.append(to_rgba('#888888'))
        # Size based on theta value
        theta = G.nodes[node]['theta']
        node_sizes.append(30 + theta * 150)
    else:  # Inhibitory
        # Inhibitory: light gray
        node_colors.append(to_rgba('#cccccc'))
        node_sizes.append(30)

# Draw edges in 3D
input_to_exc_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('conn_type') == 'input_to_exc']
exc_to_inh_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('conn_type') == 'exc_to_inh']
lat_inh_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('conn_type') == 'lateral_inh']

# Draw Input → Exc edges
if input_to_exc_edges:
    input_weights = [G.edges[e]['weight'] for e in input_to_exc_edges]
    max_input_weight = max(input_weights)
    min_input_weight = min(input_weights)
    weight_range = max_input_weight - min_input_weight

    for edge in input_to_exc_edges:
        src, dst = edge
        x_src, y_src, z_src = pos_3d[src]
        x_dst, y_dst, z_dst = pos_3d[dst]

        weight = G.edges[edge]['weight']
        alpha = 0.1 + 0.4 * ((weight - min_input_weight) / weight_range if weight_range > 0 else 0.5)

        ax_3d.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst],
                  color='#ff6666', alpha=alpha, linewidth=0.3)

# Draw Exc → Inh edges
for edge in exc_to_inh_edges:
    src, dst = edge
    x_src, y_src, z_src = pos_3d[src]
    x_dst, y_dst, z_dst = pos_3d[dst]

    ax_3d.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst],
              color='#4477ff', alpha=0.4, linewidth=0.5)

# Draw lateral inhibition edges (sampled)
for edge in lat_inh_edges:
    src, dst = edge
    x_src, y_src, z_src = pos_3d[src]
    x_dst, y_dst, z_dst = pos_3d[dst]

    ax_3d.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst],
              color='#66cc66', alpha=0.15, linewidth=0.2)

# Draw nodes (convert sizes to numpy array)
ax_3d.scatter(node_xs, node_ys, node_zs, c=node_colors, s=np.array(node_sizes),
             alpha=0.8, edgecolors='black', linewidths=0.3, depthshade=True)

# Add layer planes (semi-transparent)
grid_size = 12  # Wider grid to match wider layout
xx, yy = np.meshgrid(np.linspace(-grid_size, grid_size, 10),
                     np.linspace(-grid_size, grid_size, 10))

# Input layer plane
zz_input = np.zeros_like(xx)
ax_3d.plot_surface(xx, yy, zz_input, alpha=0.1, color='#ff6666')

# Excitatory layer plane
zz_exc = np.ones_like(xx) * 10
ax_3d.plot_surface(xx, yy, zz_exc, alpha=0.1, color='#4477ff')

# Inhibitory layer plane
zz_inh = np.ones_like(xx) * 20
ax_3d.plot_surface(xx, yy, zz_inh, alpha=0.1, color='#66cc66')

# Add layer labels in 3D space
ax_3d.text(0, -12, 0, 'Input Layer\n(784 neurons)', fontsize=12, fontweight='bold',
          ha='center', color='#ff3333',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ff6666'))

ax_3d.text(0, -10, 10, 'Excitatory Layer\n(400 neurons)', fontsize=12, fontweight='bold',
          ha='center', color='#3366ff',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#3366ff'))

ax_3d.text(0, -10, 20, 'Inhibitory Layer\n(400 neurons)', fontsize=12, fontweight='bold',
          ha='center', color='#339933',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#66cc66'))

# Set viewing angle
ax_3d.view_init(elev=ELEVATION, azim=AZIMUTH)

# Axis labels and styling
ax_3d.set_xlabel('X Position', fontsize=10, labelpad=10)
ax_3d.set_ylabel('Y Position', fontsize=10, labelpad=10)
ax_3d.set_zlabel('Network Depth (Layer)', fontsize=10, labelpad=10)

# Set z-axis ticks to show layers
ax_3d.set_zticks([0, 10, 20])
ax_3d.set_zticklabels(['Input', 'Excitatory', 'Inhibitory'])

# Grid styling
ax_3d.grid(True, alpha=0.3)
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False

# Save figure
print(f"\nSaving figure to {OUTPUT_PATH}...")
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')

print("\n" + "="*70)
print("3D NETWORK STRUCTURE VISUALIZATION COMPLETE!")
print("="*70)
print(f"Saved to: {OUTPUT_PATH}")
print(f"\nKey features:")
print(f"  • Configured digit classes: {configured_classes}")
print(f"  • 3D layered architecture with depth")
print(f"  • Node colors show digit assignments (excitatory neurons)")
print(f"  • Node sizes reflect adaptive thresholds")
print(f"  • Semi-transparent layer planes")
print(f"  • Viewing angle: elevation={ELEVATION}°, azimuth={AZIMUTH}°")
print(f"\nNote: The interactive display window (plt.show()) will open after saving")

# Create Interactive Plotly Animation
print("\n" + "="*70)
print("CREATING INTERACTIVE PLOTLY ANIMATION")
print("="*70)

try:
    import plotly.graph_objects as go
    from brian2 import *
    from data_loader import MNISTDataLoader

    print("\nRunning network simulation for animation...")

    # Load test data
    data_loader = MNISTDataLoader(cfg)
    test_images, test_labels = data_loader.load_test_data()

    # Select examples for animation
    np.random.seed(RANDOM_SEED)
    example_indices = np.random.choice(len(test_images), NUM_EXAMPLES_TO_ANIMATE, replace=False)
    anim_images = test_images[example_indices]
    anim_labels = test_labels[example_indices]

    print(f"Selected {NUM_EXAMPLES_TO_ANIMATE} examples: digits {anim_labels}")

    # Build network for simulation
    defaultclock.dt = cfg.dt

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

    # Recreate connections with delays
    connections = {}

    # Input → Excitatory: 0-10ms random delays
    connections['XeAe'] = Synapses(input_groups['Xe'], neuron_groups['Ae'],
                                   model='w : 1', on_pre='ge_post += w',
                                   delay=cfg.delay_ee_input[1], method='euler')
    sources, targets = np.where(conn_matrix > 0)
    connections['XeAe'].connect(i=sources, j=targets)
    connections['XeAe'].w = conn_matrix[sources, targets]
    connections['XeAe'].delay = 'rand() * 10*ms'  # Random delays 0-10ms

    # Excitatory → Inhibitory: 0-5ms random delays
    AeAi_matrix = np.load(data_path + 'random/AeAi.npy', allow_pickle=True)
    connections['AeAi'] = Synapses(neuron_groups['Ae'], neuron_groups['Ai'],
                                   model='w : 1', on_pre='ge_post += w',
                                   delay=cfg.delay_ei_input[1], method='euler')
    connections['AeAi'].connect(condition='i==j')
    conn_weights = np.zeros(n_e)
    for i, j, w in AeAi_matrix:
        if int(i) == int(j):
            conn_weights[int(i)] = w
    connections['AeAi'].w = conn_weights
    connections['AeAi'].delay = 'rand() * 5*ms'  # Random delays 0-5ms

    # Inhibitory → Excitatory: No delays (instantaneous lateral inhibition)
    AiAe_matrix = np.load(data_path + 'random/AiAe.npy', allow_pickle=True)
    connections['AiAe'] = Synapses(neuron_groups['Ai'], neuron_groups['Ae'],
                                   model='w : 1', on_pre='gi_post += w', method='euler')
    connections['AiAe'].connect(condition='i!=j')
    aiae_weights_full = np.zeros((n_e, n_e))
    for i, j, w in AiAe_matrix:
        aiae_weights_full[int(i), int(j)] = w
    connections['AiAe'].w = aiae_weights_full[connections['AiAe'].i[:], connections['AiAe'].j[:]]

    neuron_groups['Ae'].theta = theta_values * volt
    neuron_groups['Ae'].v = cfg.v_rest_e
    neuron_groups['Ai'].v = cfg.v_rest_i

    net = Network()
    net.add(neuron_groups.values())
    net.add(input_groups.values())
    net.add(connections.values())
    net.add(spike_monitors.values())

    # Run simulation
    print("  Running simulation...")
    single_example_time = cfg.single_example_time
    resting_time = cfg.resting_time

    for example_idx in range(NUM_EXAMPLES_TO_ANIMATE):
        rates = anim_images[example_idx].reshape((n_input)) / 8. * cfg.input_intensity
        input_groups['Xe'].rates = rates * Hz
        net.run(single_example_time, report=None)
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time, report=None)

    print("  Simulation complete")

    # Extract spike data
    input_spike_times = np.array(spike_monitors['Xe'].t / ms)
    input_spike_neurons = np.array(spike_monitors['Xe'].i)
    exc_spike_times = np.array(spike_monitors['Ae'].t / ms)
    exc_spike_neurons = np.array(spike_monitors['Ae'].i)
    inh_spike_times = np.array(spike_monitors['Ai'].t / ms)
    inh_spike_neurons = np.array(spike_monitors['Ai'].i)

    # Compute predictions for each example
    def get_recognized_number_ranking(assignments, spike_rates):
        summed_rates = [0] * 10
        num_assignments = [0] * 10
        for i in range(10):
            num_assignments[i] = len(np.where(assignments == i)[0])
            if num_assignments[i] > 0:
                summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
        return np.argsort(summed_rates)[::-1]

    predictions = []
    for ex_idx in range(NUM_EXAMPLES_TO_ANIMATE):
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

    print(f"  Predictions: {predictions} (True: {anim_labels})")

    # Create animation frames
    print("\n  Creating Plotly animation frames...")

    total_time = NUM_EXAMPLES_TO_ANIMATE * (single_example_time + resting_time) / ms
    time_step = 1.0  # 1ms per frame
    num_frames = int(total_time / time_step)

    # Node decay rate for visual effect
    NODE_DECAY = 0.7

    # Track activity over time for the activity chart
    activity_history = {
        'time': [],
        'input': [],
        'excitatory': [],
        'inhibitory': []
    }

    # No edge traces - edges removed per user request
    edge_traces = []

    # Create frames
    frames = []
    node_activity = {f'X{i}': 0.0 for i in range(n_input)}
    node_activity.update({f'E{i}': 0.0 for i in range(n_e)})
    node_activity.update({f'I{i}': 0.0 for i in range(n_e)})

    print(f"  Generating {num_frames} frames...")

    for frame_idx in range(0, num_frames, 5):  # Every 5ms for performance
        current_time_ms = frame_idx * time_step
        window_start = current_time_ms
        window_end = current_time_ms + time_step * 5

        # Decay all activities
        for node in node_activity:
            node_activity[node] *= NODE_DECAY

        # Mark spiking neurons
        input_mask = (input_spike_times >= window_start) & (input_spike_times < window_end)
        input_spike_count = len(input_spike_neurons[input_mask])
        for neuron_id in input_spike_neurons[input_mask]:
            node_activity[f'X{neuron_id}'] = 1.0

        exc_mask = (exc_spike_times >= window_start) & (exc_spike_times < window_end)
        exc_spike_count = len(exc_spike_neurons[exc_mask])
        for neuron_id in exc_spike_neurons[exc_mask]:
            node_activity[f'E{neuron_id}'] = 1.0

        inh_mask = (inh_spike_times >= window_start) & (inh_spike_times < window_end)
        inh_spike_count = len(inh_spike_neurons[inh_mask])
        for neuron_id in inh_spike_neurons[inh_mask]:
            node_activity[f'I{neuron_id}'] = 1.0

        # Record activity for chart
        activity_history['time'].append(current_time_ms)
        activity_history['input'].append(input_spike_count)
        activity_history['excitatory'].append(exc_spike_count)
        activity_history['inhibitory'].append(inh_spike_count)

        # Determine current example
        example_idx = int(current_time_ms / ((single_example_time + resting_time) / ms))
        if example_idx >= NUM_EXAMPLES_TO_ANIMATE:
            example_idx = NUM_EXAMPLES_TO_ANIMATE - 1

        # Prepare node data
        node_colors_frame = []
        node_sizes_frame = []

        for node in [f'X{i}' for i in range(n_input)] + [f'E{i}' for i in range(n_e)] + [f'I{i}' for i in range(n_e)]:
            activity = node_activity[node]

            if node.startswith('X'):
                # Input nodes - larger and more visible
                brightness = 0.6 + activity * 0.4
                node_colors_frame.append(f'rgba(130,130,130,{brightness})')
                node_sizes_frame.append(4 + activity * 6)  # Increased from 2+4
            elif node.startswith('E'):
                # Excitatory nodes - larger and more visible
                idx = int(node[1:])
                assignment = int(assignments[idx])
                base_color = digit_color_map.get(assignment, (0.5, 0.5, 0.5, 1.0))

                if hasattr(base_color, '__iter__') and len(base_color) >= 3:
                    r, g, b = base_color[0], base_color[1], base_color[2]
                    brightness = 0.6 + activity * 0.4
                    node_colors_frame.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{brightness})')
                else:
                    node_colors_frame.append(f'rgba(100,100,255,{0.6 + activity * 0.4})')

                node_sizes_frame.append(8 + activity * 12)  # Increased from 5+10
            else:  # Inhibitory
                # Inhibitory nodes - larger and more visible
                brightness = 0.5 + activity * 0.5
                node_colors_frame.append(f'rgba(180,180,180,{brightness})')
                node_sizes_frame.append(6 + activity * 10)  # Increased from 4+8

        # Create frame - now includes both 3D network and 2D activity chart
        # Get activity history up to current time
        current_frame_idx = len(activity_history['time']) - 1

        # 3D Network scatter
        network_trace = go.Scatter3d(
            x=[pos_3d[node][0] for node in [f'X{i}' for i in range(n_input)] + [f'E{i}' for i in range(n_e)] + [f'I{i}' for i in range(n_e)]],
            y=[pos_3d[node][1] for node in [f'X{i}' for i in range(n_input)] + [f'E{i}' for i in range(n_e)] + [f'I{i}' for i in range(n_e)]],
            z=[pos_3d[node][2] for node in [f'X{i}' for i in range(n_input)] + [f'E{i}' for i in range(n_e)] + [f'I{i}' for i in range(n_e)]],
            mode='markers',
            marker=dict(
                size=node_sizes_frame,
                color=node_colors_frame,
                line=dict(color='black', width=0.5)
            ),
            hoverinfo='none',
            showlegend=False
        )

        # Only include 3D network data (no activity chart)
        frame_data = list(edge_traces) + [network_trace]

        frame_name = f"Frame {frame_idx}"

        # Determine if correct
        is_correct = predictions[example_idx] == anim_labels[example_idx]
        pred_color = '#00ff00' if is_correct else '#ff4444'

        layout_update = dict(
            title=dict(
                text=f'<b>Interactive 3D Network Animation</b><br>' +
                     f'<span style="font-size:14px">Time: {current_time_ms:.0f}ms | ' +
                     f'Example {example_idx+1}/{NUM_EXAMPLES_TO_ANIMATE} | ' +
                     f'True Label: <b>{anim_labels[example_idx]}</b> | ' +
                     f'Prediction: <b style="color:{pred_color}">{predictions[example_idx]}</b></span>',
                x=0.5,
                xanchor='center'
            )
        )

        frames.append(go.Frame(data=frame_data, name=frame_name, layout=layout_update))

    print("  Frames generated")

    # Create initial figure (single 3D view, no subplots)
    print("  Creating interactive figure...")

    fig = go.Figure(data=frames[0].data)

    # Update layout with better formatting
    fig.update_layout(
        title=dict(
            text='<b>Interactive 3D Spiking Neural Network Animation</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        # 3D scene configuration - full screen
        scene=dict(
            xaxis=dict(title='', showbackground=False, showgrid=True,
                      gridcolor='lightgray', showticklabels=False),
            yaxis=dict(title='', showbackground=False, showgrid=True,
                      gridcolor='lightgray', showticklabels=False),
            zaxis=dict(title='Layer Depth', tickvals=[0, 10, 20],
                      ticktext=['Input', 'Excitatory', 'Inhibitory'],
                      showbackground=False, showgrid=True, gridcolor='lightgray',
                      titlefont=dict(size=12)),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.0)),  # Zoomed in closer (was 1.5, 1.5, 1.2)
            bgcolor='rgba(250, 240, 230, 0.5)'  # Eggshell white with some transparency
        ),
        # Control buttons
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=ANIMATION_MS_PER_FRAME, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='⏮ Reset',
                        method='animate',
                        args=[[frames[0].name], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate'
                        )]
                    )
                ],
                direction='left',
                pad=dict(r=10, t=10),
                x=0.02,
                xanchor='left',
                y=0.02,
                yanchor='bottom',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#666',
                borderwidth=1,
                font=dict(size=12)
            )
        ],
        # Timeline slider
        sliders=[dict(
            active=0,
            steps=[dict(
                args=[[f.name], dict(
                    frame=dict(duration=0, redraw=True),
                    mode='immediate',
                    transition=dict(duration=0)
                )],
                method='animate',
                label=f'{i*5:.0f}ms'
            ) for i, f in enumerate(frames)],
            x=0.02,
            len=0.96,
            xanchor='left',
            y=0.01,
            yanchor='top',
            pad=dict(b=10, t=10),
            currentvalue=dict(
                visible=True,
                prefix='Time: ',
                xanchor='center',
                font=dict(size=14)
            ),
            transition=dict(duration=0),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#666',
            borderwidth=1
        )],
        # Overall layout styling
        width=1600,
        height=1000,
        paper_bgcolor='#FAF0E6',  # Eggshell white
        font=dict(family='Arial, sans-serif', size=11),
        showlegend=False,  # No legend needed
        margin=dict(l=50, r=50, t=100, b=100),
        hovermode='closest'
    )

    # Add frames to figure
    fig.frames = frames

    print(f"\n  Saving interactive animation to {OUTPUT_HTML}...")

    # Write HTML with custom JavaScript to auto-start animation
    html_string = fig.to_html(include_plotlyjs='cdn')

    # Inject JavaScript to auto-start the animation after page load
    auto_start_script = """
    <script>
    // Auto-start animation when page loads
    window.addEventListener('load', function() {
        setTimeout(function() {
            var playButton = document.querySelector('[data-title="Play"]');
            if (playButton) {
                playButton.click();
            }
        }, 500);  // Small delay to ensure Plotly is fully initialized
    });
    </script>
    """

    # Insert the script before closing body tag
    html_string = html_string.replace('</body>', auto_start_script + '</body>')

    with open(OUTPUT_HTML, 'w') as f:
        f.write(html_string)

    print("\n" + "="*70)
    print("INTERACTIVE PLOTLY ANIMATION COMPLETE!")
    print("="*70)
    print(f"Saved to: {OUTPUT_HTML}")
    print(f"\nOpen this file in your browser to see the animated network!")
    print(f"\nNew Features:")
    print(f"  • {NUM_EXAMPLES_TO_ANIMATE} examples with live predictions")
    print(f"  • Real-time activity chart showing spike counts over time")
    print(f"  • Nodes light up when neurons spike with decay effect")
    print(f"  • Fully interactive 3D rotation (works during playback!)")
    print(f"  • Play/Pause/Reset controls")
    print(f"  • Timeline slider for scrubbing")
    print(f"  • Color-coded predictions (green=correct, red=incorrect)")
    print(f"  • Larger view (1600x1000) with better formatting")

except ImportError as e:
    print(f"\nCould not create Plotly animation: {e}")
    print("Install plotly with: pip install plotly")
except Exception as e:
    print(f"\nError creating animation: {e}")
    import traceback
    traceback.print_exc()
