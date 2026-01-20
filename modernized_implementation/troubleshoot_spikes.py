"""
Troubleshooting Script: Analyze Network Spike Patterns

This script runs a detailed analysis of what's happening during digit presentation
to diagnose why all neurons appear to be firing indiscriminately.
"""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

print("="*70)
print("SPIKE PATTERN TROUBLESHOOTING")
print("="*70)

# Configuration
cfg = Config()
cfg.set_test_mode(True)

# Load data
data_loader = MNISTDataLoader(cfg)
test_images, test_labels = data_loader.load_test_data()

# Select one example to analyze in detail
example_idx = np.random.choice(len(test_images))
example_image = test_images[example_idx]
example_label = test_labels[example_idx]

print(f"\nAnalyzing digit {example_label} (index {example_idx})")

# Build network
print("\nBuilding network...")

n_input = cfg.n_input
n_e = cfg.n_e
single_example_time = cfg.single_example_time
dt = cfg.dt

# Network parameters
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

defaultclock.dt = dt
np.random.seed(cfg.random_seed)

# Create neuron groups
neuron_groups = {}

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

# Monitors
spike_monitors = {
    'Ae': SpikeMonitor(neuron_groups['Ae']),
    'Ai': SpikeMonitor(neuron_groups['Ai']),
    'Xe': SpikeMonitor(input_groups['Xe'])
}

# State monitors for detailed voltage traces
state_monitors = {
    'Ae_v': StateMonitor(neuron_groups['Ae'], 'v', record=[0, 50, 100, 150, 200]),
    'Ae_theta': StateMonitor(neuron_groups['Ae'], 'theta', record=[0, 50, 100, 150, 200])
}

# Load connections
print("  Loading connections...")
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
# Connect only where weights exist (sparse connectivity)
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
connections['AeAi'].connect(condition='i==j')
conn_weights = np.zeros(n_e)
for i, j, w in AeAi_matrix:
    if int(i) == int(j):
        conn_weights[int(i)] = w
connections['AeAi'].w = conn_weights

# Inhibitory to excitatory
AiAe_matrix = np.load(data_path + 'random/AiAe.npy', allow_pickle=True)
connections['AiAe'] = Synapses(
    neuron_groups['Ai'],
    neuron_groups['Ae'],
    model='w : 1',
    on_pre='gi_post += w',
    method='euler'
)
connections['AiAe'].connect(condition='i!=j')
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
net.add(state_monitors.values())

print("Network built")

# Run simulation
print("\nRunning simulation...")

# Set input rates
rates = example_image.reshape((n_input)) / 8. * cfg.input_intensity
input_groups['Xe'].rates = rates * Hz

# Check input statistics
print(f"\nInput statistics:")
print(f"  Non-zero pixels: {np.sum(example_image > 0)}/{n_input}")
print(f"  Mean rate (non-zero): {np.mean(rates[rates > 0]):.1f} Hz")
print(f"  Max rate: {np.max(rates):.1f} Hz")
print(f"  Total expected input spikes: ~{np.sum(rates) * (single_example_time/second):.0f}")

# Run
net.run(single_example_time, report='text')

print("\nSimulation complete")

# Analyze spike patterns
print("\n" + "="*70)
print("SPIKE ANALYSIS")
print("="*70)

# Get spikes
input_spike_times = np.array(spike_monitors['Xe'].t / ms)
input_spike_neurons = np.array(spike_monitors['Xe'].i)
exc_spike_times = np.array(spike_monitors['Ae'].t / ms)
exc_spike_neurons = np.array(spike_monitors['Ae'].i)
inh_spike_times = np.array(spike_monitors['Ai'].t / ms)
inh_spike_neurons = np.array(spike_monitors['Ai'].i)

print(f"\nTotal spikes:")
print(f"  Input layer: {len(input_spike_times)} spikes")
print(f"  Excitatory layer: {len(exc_spike_times)} spikes")
print(f"  Inhibitory layer: {len(inh_spike_times)} spikes")

# Analyze excitatory spike distribution
print(f"\nExcitatory neuron participation:")
unique_exc_neurons = np.unique(exc_spike_neurons)
print(f"  Neurons that spiked: {len(unique_exc_neurons)}/{n_e} ({100*len(unique_exc_neurons)/n_e:.1f}%)")

# Spike count per neuron
spike_counts = np.bincount(exc_spike_neurons.astype(int), minlength=n_e)
print(f"\nSpikes per neuron statistics:")
print(f"  Mean: {np.mean(spike_counts):.2f}")
print(f"  Median: {np.median(spike_counts):.1f}")
print(f"  Max: {np.max(spike_counts)}")
print(f"  Neurons with >5 spikes: {np.sum(spike_counts > 5)}")
print(f"  Neurons with >10 spikes: {np.sum(spike_counts > 10)}")

# Temporal distribution
print(f"\nTemporal spike distribution (excitatory):")
time_windows = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350)]
for start, end in time_windows:
    mask = (exc_spike_times >= start) & (exc_spike_times < end)
    n_spikes = np.sum(mask)
    n_neurons = len(np.unique(exc_spike_neurons[mask]))
    print(f"  {start:3d}-{end:3d} ms: {n_spikes:3d} spikes from {n_neurons:3d} neurons")

# Check for synchrony (multiple neurons spiking at same time)
print(f"\nSynchrony analysis:")
time_bins = np.arange(0, 350, 1)  # 1ms bins
spike_counts_per_bin = np.histogram(exc_spike_times, bins=time_bins)[0]
print(f"  Max spikes in any 1ms bin: {np.max(spike_counts_per_bin)}")
print(f"  Bins with >10 simultaneous spikes: {np.sum(spike_counts_per_bin > 10)}")
print(f"  Bins with >5 simultaneous spikes: {np.sum(spike_counts_per_bin > 5)}")

# Check inhibitory activity
print(f"\nInhibitory activity:")
if len(inh_spike_times) > 0:
    print(f"  First inhibitory spike at: {np.min(inh_spike_times):.1f} ms")
    print(f"  Inhibitory neurons that spiked: {len(np.unique(inh_spike_neurons))}/{n_e}")
else:
    print(f"  WARNING: No inhibitory spikes!")

# Analyze input-to-excitatory relationship
print(f"\nInput-Excitatory relationship:")
# Calculate when input spikes occur
input_spike_count_per_neuron = np.bincount(input_spike_neurons.astype(int), minlength=n_input)
print(f"  Input neurons that spiked: {np.sum(input_spike_count_per_neuron > 0)}/{n_input}")
print(f"  Mean input spikes per active neuron: {np.mean(input_spike_count_per_neuron[input_spike_count_per_neuron > 0]):.2f}")

# Visualize
print("\nGenerating diagnostic plots...")

fig = plt.figure(figsize=(16, 12))

# 1. Input image
ax1 = plt.subplot(4, 3, 1)
ax1.imshow(example_image, cmap='gray')
ax1.set_title(f'Input: Digit {example_label}')
ax1.axis('off')

# 2. Input rates heatmap
ax2 = plt.subplot(4, 3, 2)
ax2.imshow(rates.reshape(28, 28), cmap='hot')
ax2.set_title('Input Rates (Hz)')
ax2.axis('off')
plt.colorbar(ax2.images[0], ax=ax2)

# 3. Excitatory spike counts heatmap
ax3 = plt.subplot(4, 3, 3)
spike_counts_grid = spike_counts.reshape(20, 20)
im = ax3.imshow(spike_counts_grid, cmap='hot')
ax3.set_title('Excitatory Spike Counts')
ax3.axis('off')
plt.colorbar(im, ax=ax3)

# 4. Input raster plot
ax4 = plt.subplot(4, 3, 4)
if len(input_spike_times) > 0:
    ax4.scatter(input_spike_times, input_spike_neurons, s=1, c='red', alpha=0.5)
ax4.set_xlim(0, 350)
ax4.set_ylim(0, n_input)
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Input Neuron')
ax4.set_title('Input Layer Raster')

# 5. Excitatory raster plot
ax5 = plt.subplot(4, 3, 5)
if len(exc_spike_times) > 0:
    ax5.scatter(exc_spike_times, exc_spike_neurons, s=2, c='blue', alpha=0.8)
ax5.set_xlim(0, 350)
ax5.set_ylim(0, n_e)
ax5.set_xlabel('Time (ms)')
ax5.set_ylabel('Excitatory Neuron')
ax5.set_title('Excitatory Layer Raster')

# 6. Inhibitory raster plot
ax6 = plt.subplot(4, 3, 6)
if len(inh_spike_times) > 0:
    ax6.scatter(inh_spike_times, inh_spike_neurons, s=2, c='green', alpha=0.8)
ax6.set_xlim(0, 350)
ax6.set_ylim(0, n_e)
ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('Inhibitory Neuron')
ax6.set_title('Inhibitory Layer Raster')

# 7. Spike count histogram
ax7 = plt.subplot(4, 3, 7)
ax7.hist(spike_counts[spike_counts > 0], bins=30, edgecolor='black')
ax7.set_xlabel('Spikes per neuron')
ax7.set_ylabel('Count')
ax7.set_title('Excitatory Spike Distribution')
ax7.axvline(np.mean(spike_counts[spike_counts > 0]), color='red', linestyle='--', label='Mean')
ax7.legend()

# 8. Temporal spike rate
ax8 = plt.subplot(4, 3, 8)
time_bins_fine = np.arange(0, 350, 5)
input_counts = np.histogram(input_spike_times, bins=time_bins_fine)[0]
exc_counts = np.histogram(exc_spike_times, bins=time_bins_fine)[0]
inh_counts = np.histogram(inh_spike_times, bins=time_bins_fine)[0]
bin_centers = (time_bins_fine[:-1] + time_bins_fine[1:]) / 2
ax8.plot(bin_centers, input_counts, 'r-', label='Input', alpha=0.7)
ax8.plot(bin_centers, exc_counts, 'b-', label='Excitatory', alpha=0.7)
ax8.plot(bin_centers, inh_counts, 'g-', label='Inhibitory', alpha=0.7)
ax8.set_xlabel('Time (ms)')
ax8.set_ylabel('Spikes per 5ms bin')
ax8.set_title('Population Activity Over Time')
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Sample voltage traces
ax9 = plt.subplot(4, 3, 9)
for i, idx in enumerate([0, 1, 2, 3, 4]):
    ax9.plot(state_monitors['Ae_v'].t / ms, state_monitors['Ae_v'].v[i] / mV,
             alpha=0.7, label=f'Neuron {idx}')
ax9.set_xlabel('Time (ms)')
ax9.set_ylabel('Membrane Potential (mV)')
ax9.set_title('Sample Excitatory Neuron Voltages')
ax9.axhline(v_thresh_e_const/mV, color='red', linestyle='--', alpha=0.3, label='Base Threshold')
ax9.legend(fontsize=8)
ax9.grid(alpha=0.3)

# 10. Theta values
ax10 = plt.subplot(4, 3, 10)
for i, idx in enumerate([0, 1, 2, 3, 4]):
    ax10.plot(state_monitors['Ae_theta'].t / ms, state_monitors['Ae_theta'].theta[i] / mV,
              alpha=0.7, label=f'Neuron {idx}')
ax10.set_xlabel('Time (ms)')
ax10.set_ylabel('Theta (mV)')
ax10.set_title('Sample Adaptive Thresholds')
ax10.legend(fontsize=8)
ax10.grid(alpha=0.3)

# 11. Synchrony over time
ax11 = plt.subplot(4, 3, 11)
time_bins_sync = np.arange(0, 350, 1)
spike_counts_per_bin = np.histogram(exc_spike_times, bins=time_bins_sync)[0]
ax11.plot(time_bins_sync[:-1], spike_counts_per_bin)
ax11.set_xlabel('Time (ms)')
ax11.set_ylabel('Simultaneous Spikes (1ms bins)')
ax11.set_title('Excitatory Synchrony')
ax11.axhline(10, color='red', linestyle='--', alpha=0.5, label='High synchrony')
ax11.legend()
ax11.grid(alpha=0.3)

# 12. Connection weights summary
ax12 = plt.subplot(4, 3, 12)
ax12.text(0.1, 0.9, 'Connection Statistics:', fontsize=12, fontweight='bold', transform=ax12.transAxes)
ax12.text(0.1, 0.75, f'XeAe weights:', transform=ax12.transAxes)
ax12.text(0.1, 0.65, f'  Mean: {np.mean(conn_matrix[conn_matrix > 0]):.4f}', transform=ax12.transAxes)
ax12.text(0.1, 0.55, f'  Max: {np.max(conn_matrix):.4f}', transform=ax12.transAxes)
ax12.text(0.1, 0.40, f'Theta values:', transform=ax12.transAxes)
ax12.text(0.1, 0.30, f'  Mean: {np.mean(theta_values):.2f} mV', transform=ax12.transAxes)
ax12.text(0.1, 0.20, f'  Max: {np.max(theta_values):.2f} mV', transform=ax12.transAxes)
ax12.axis('off')

plt.tight_layout()
plt.savefig('analysis_figures/spike_troubleshooting.png', dpi=150, bbox_inches='tight')
print(f"Saved diagnostic plot to analysis_figures/spike_troubleshooting.png")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Check for common issues
issues = []

if len(exc_spike_times) == 0:
    issues.append("CRITICAL: No excitatory spikes! Network may be too suppressed.")

if len(exc_spike_times) > n_e * 20:
    issues.append(f"WARNING: Very high spike count ({len(exc_spike_times)} spikes). Network may be over-excited.")

if len(np.unique(exc_spike_neurons)) > n_e * 0.8:
    issues.append(f"WARNING: {100*len(np.unique(exc_spike_neurons))/n_e:.1f}% of neurons spiked. Expected ~20-40% for selective response.")

if np.max(spike_counts_per_bin) > 50:
    issues.append(f"WARNING: High synchrony detected ({np.max(spike_counts_per_bin)} spikes in 1ms). May indicate network instability.")

if len(inh_spike_times) == 0:
    issues.append("WARNING: No inhibitory spikes. Lateral inhibition not working!")

if len(issues) == 0:
    print("\nNetwork behavior appears normal")
else:
    print("\nIssues detected:")
    for issue in issues:
        print(f"  • {issue}")

print("\n" + "="*70)
