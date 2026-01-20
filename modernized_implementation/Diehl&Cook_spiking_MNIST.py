'''
Created on 15.12.2014

@author: Peter U. Diehl
'''


import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import time
import os.path
import scipy
import pickle
from brian2 import *
from struct import unpack
from tqdm import tqdm
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Initialize configuration
cfg = Config()

# Initialize data loader
data_loader = MNISTDataLoader(cfg)

# functions

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName, allow_pickle=True)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print('save connections')
    os.makedirs(data_path + 'weights', exist_ok=True)
    for connName in save_conns:
        conn = connections[connName]
        connMatrix = np.zeros((n_input, n_e))
        connMatrix[conn.i[:], conn.j[:]] = conn.w[:]
        connListSparse = ([(i,j,connMatrix[i,j]) for i in range(connMatrix.shape[0]) for j in range(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print('save theta')
    os.makedirs(data_path + 'weights', exist_ok=True)
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            conn = connections[connName]
            temp_conn = np.zeros((n_input, n_e))
            temp_conn[conn.i[:], conn.j[:]] = conn.w[:]
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in range(n_e):
                temp_conn[:,j] *= colFactors[j]
            conn.w[:] = temp_conn[conn.i[:], conn.j[:]]

def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    conn = connections[name]
    weight_matrix[conn.i[:], conn.j[:]] = conn.w[:]
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = plt.figure(fig_num, figsize = (18, 18))
    im2 = plt.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    plt.colorbar(im2)
    plt.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(np.ceil(num_examples/update_interval))
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = plt.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    plt.ylim(ymax = 100)
    plt.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

def get_recognized_number_ranking(assignments, spike_rates):
    # Create arrays for all 10 digits (even if only subset used for training)
    # This ensures the ranking indices correspond to actual digit labels
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    # Determine which classes to iterate over based on config
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    for j in classes_to_check:
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments


# load MNIST
print(f'\n{"="*60}')
print(f'Configuration Summary')
print(f'{"="*60}')
print(cfg)
if cfg.mnist_classes is not None:
    print(f'Class filtering enabled: Only using examples from {cfg.mnist_classes}')
    print(f'   Balanced sampling: Equal examples per class each epoch')
print(f'{"="*60}\n')

# Load and filter data
print('Loading MNIST data...')
start = time.time()
if cfg.use_testing_set:
    # Test mode: load test set
    dataset_images, dataset_labels = data_loader.load_test_data()
else:
    # Training mode: load training set
    dataset_images, dataset_labels = data_loader.load_training_data()
end = time.time()
print(f'Data loaded and filtered in {end - start:.2f}s')

# Verify we have enough data for balanced sampling
if cfg.mnist_classes is not None:
    num_classes = len(cfg.mnist_classes)
    # Use different logic for test vs train mode
    if cfg.test_mode:
        # In test mode, use all available examples (limited by smallest class)
        min_class_size = min([np.sum(dataset_labels == cls) for cls in cfg.mnist_classes])
        examples_per_class_per_epoch = min_class_size
        print(f'\nBalanced sampling configuration (TEST MODE):')
        print(f'  Examples per class: {examples_per_class_per_epoch} (using all available)')
        print(f'  Total examples per epoch: {examples_per_class_per_epoch * num_classes}')
    else:
        # In training mode, use configured amount
        examples_per_class_per_epoch = cfg.num_train_examples // num_classes
        print(f'\nBalanced sampling configuration:')
        print(f'  Examples per epoch: {cfg.num_train_examples}')

    print(f'  Classes: {num_classes}')
    print(f'  Examples per class per epoch: {examples_per_class_per_epoch}')
    print()

    for cls in cfg.mnist_classes:
        available = np.sum(dataset_labels == cls)
        if available < examples_per_class_per_epoch:
            raise ValueError(
                f'Insufficient data for class {cls}: '
                f'need {examples_per_class_per_epoch} per epoch, '
                f'but only {available} available'
            )
        print(f'  Class {cls}: {available} available (need {examples_per_class_per_epoch} per epoch) ')

print(f'\nDataset ready: {len(dataset_labels)} examples cached in memory')
print(f'{"="*60}\n')


# set parameters and equations
# All parameters now loaded from config
test_mode = cfg.test_mode
defaultclock.dt = cfg.dt

np.random.seed(cfg.random_seed)
data_path = cfg.data_path
weight_path = cfg.weight_path
num_examples = cfg.num_examples
use_testing_set = cfg.use_testing_set
do_plot_performance = cfg.do_plot_performance
record_spikes = cfg.record_spikes
ee_STDP_on = cfg.ee_STDP_on
update_interval = cfg.update_interval
weight_update_interval = cfg.weight_update_interval
save_connections_interval = cfg.save_connections_interval
enable_live_plots = cfg.enable_live_plots

ending = cfg.ending
n_input = cfg.n_input
n_e = cfg.n_e
n_i = cfg.n_i
single_example_time = cfg.single_example_time
resting_time = cfg.resting_time
runtime = cfg.runtime

v_rest_e = cfg.v_rest_e
v_rest_i = cfg.v_rest_i
v_reset_e = cfg.v_reset_e
v_reset_i = cfg.v_reset_i
v_thresh_e_const = cfg.v_thresh_e_const
v_thresh_i = cfg.v_thresh_i
refrac_e = cfg.refrac_e
refrac_i = cfg.refrac_i

weight = {}
delay = {}
input_population_names = cfg.input_population_names
population_names = cfg.population_names
input_connection_names = cfg.input_connection_names
save_conns = cfg.save_conns
input_conn_names = cfg.input_conn_names
recurrent_conn_names = cfg.recurrent_conn_names
weight['ee_input'] = cfg.weight_ee_input
delay['ee_input'] = cfg.delay_ee_input
delay['ei_input'] = cfg.delay_ei_input
input_intensity = cfg.input_intensity
start_input_intensity = cfg.start_input_intensity

tc_pre_ee = cfg.tc_pre_ee
tc_post_1_ee = cfg.tc_post_1_ee
tc_post_2_ee = cfg.tc_post_2_ee
nu_ee_pre = cfg.nu_ee_pre
nu_ee_post = cfg.nu_ee_post
wmax_ee = cfg.wmax_ee
exp_ee_pre = cfg.exp_ee_pre
exp_ee_post = cfg.exp_ee_post
STDP_offset = cfg.STDP_offset

tc_theta = cfg.tc_theta
theta_plus_e = cfg.theta_plus_e
offset = cfg.offset

scr_e = cfg.get_scr_e()
v_thresh_e_str = cfg.get_v_thresh_e_str()

neuron_eqs_e = cfg.get_neuron_eqs_e()
neuron_eqs_i = cfg.get_neuron_eqs_i()
eqs_stdp_ee = cfg.get_stdp_eqs()
eqs_stdp_pre_ee = cfg.get_stdp_pre_eq()
eqs_stdp_post_ee = cfg.get_stdp_post_eq()

if enable_live_plots:
    plt.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['i'] = NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= 'v > v_thresh_i', refractory= refrac_i, reset= 'v = v_reset_i', method='euler')


# create network population and recurrent connections
for name in population_names:
    print('create neuron group', name)

    neuron_groups[name+'e'] = neuron_groups['e'][0:n_e]
    neuron_groups[name+'i'] = neuron_groups['i'][0:n_i]

    neuron_groups[name+'e'].v = v_rest_e - 40. * mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * mV
    if test_mode or weight_path[-8:] == 'weights/':
        try:
            neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy') * volt
        except FileNotFoundError:
            print(f"\nERROR: Could not find pretrained weights!")
            print(f"Missing file: {weight_path}theta_{name}{ending}.npy")
            print(f"\nFor test mode, you need pretrained weights in '{weight_path}'")
            print(f"Either:")
            print(f"  1. Train a network first by setting test_mode = False")
            print(f"  2. Use the provided pretrained weights")
            exit(1)
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*mV

    print('create recurrent connections')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        try:
            weightMatrix = get_matrix_from_file(data_path + 'random/' + connName + ending + '.npy')
        except FileNotFoundError:
            print(f"\nERROR: Could not find recurrent connection file!")
            print(f"Missing file: {data_path}random/{connName}{ending}.npy")
            print(f"\nPlease run 'Diehl&Cook_MNIST_random_conn_generator.py' first to generate initial weights.")
            exit(1)

        src_grp = neuron_groups[connName[0:2]]
        tgt_grp = neuron_groups[connName[2:4]]

        connections[connName] = Synapses(src_grp, tgt_grp, model='w : 1', on_pre='g'+conn_type[0]+'_post += w', method='euler')

        sources, targets = np.where(weightMatrix > 0)
        connections[connName].connect(i=sources, j=targets)
        connections[connName].w = weightMatrix[sources, targets]

    if ee_STDP_on:
        if 'ee' in recurrent_conn_names:
            connName = name+'e'+name+'e'
            src_grp = neuron_groups[connName[0:2]]
            tgt_grp = neuron_groups[connName[2:4]]
            stdp_methods[connName] = Synapses(src_grp, tgt_grp, model=eqs_stdp_ee,
                                             on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee, method='euler')

    print('create monitors for', name)
    rate_monitors[name+'e'] = PopulationRateMonitor(neuron_groups[name+'e'])
    rate_monitors[name+'i'] = PopulationRateMonitor(neuron_groups[name+'i'])
    spike_counters[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])

    if record_spikes:
        spike_monitors[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = SpikeMonitor(neuron_groups[name+'i'])

# Live progress plot
if record_spikes and enable_live_plots:
    plt.ion()
    fig_live = plt.figure(fig_num, figsize=(10, 4))
    fig_num += 1
    ax_progress = fig_live.add_subplot(111)
    progress_line, = ax_progress.plot([], [], 'b-', linewidth=2)
    ax_progress.set_xlim(0, num_examples)
    ax_progress.set_ylim(0, 100)
    ax_progress.set_xlabel('Examples processed')
    ax_progress.set_ylabel('Spike count (last example)')
    ax_progress.set_title('Simulation Progress')
    ax_progress.grid(True, alpha=0.3)
    progress_data = {'examples': [], 'spikes': []}
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


# create input population and connections from input populations
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = PoissonGroup(n_input, 0*Hz)
    rate_monitors[name+'e'] = PopulationRateMonitor(input_groups[name+'e'])

for name in input_connection_names:
    print('create connections between', name[0], 'and', name[1])
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        try:
            weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        except FileNotFoundError:
            print(f"\nERROR: Could not find weight file!")
            print(f"Missing file: {weight_path}{connName}{ending}.npy")
            print(f"\nPlease run 'Diehl&Cook_MNIST_random_conn_generator.py' first to generate initial weights.")
            exit(1)

        src_grp = input_groups[connName[0:2]]
        tgt_grp = neuron_groups[connName[2:4]]

        if not ee_STDP_on:
            connections[connName] = Synapses(src_grp, tgt_grp, model='w : 1', on_pre='g'+connType[0]+'_post += w', delay=delay[connType][1], method='euler')

            sources, targets = np.where(weightMatrix > 0)
            connections[connName].connect(i=sources, j=targets)
            connections[connName].w = weightMatrix[sources, targets]
            connections[connName].delay = 'rand() * ' + str(float(delay[connType][1]/ms)) + '*ms'
        else:
            stdp_methods[name[0]+'e'+name[1]+'e'] = Synapses(src_grp, tgt_grp, model=eqs_stdp_ee,
                                                             on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee,
                                                             delay=delay[connType][1], method='euler')
            sources, targets = np.where(weightMatrix > 0)
            stdp_methods[name[0]+'e'+name[1]+'e'].connect(i=sources, j=targets)
            stdp_methods[name[0]+'e'+name[1]+'e'].w = weightMatrix[sources, targets]
            stdp_methods[name[0]+'e'+name[1]+'e'].delay = 'rand() * ' + str(float(delay[connType][1]/ms)) + '*ms'
            connections[connName] = stdp_methods[name[0]+'e'+name[1]+'e']


# run the simulation and set inputs
net = Network()
net.add(neuron_groups['e'], neuron_groups['i'])
for name in input_population_names:
    net.add(input_groups[name+'e'])
for name in list(connections.values()):
    net.add(name)
for name in list(rate_monitors.values()):
    net.add(name)
if record_spikes:
    for name in list(spike_monitors.values()):
        net.add(name)
for name in list(spike_counters.values()):
    net.add(name)

previous_spike_count = np.zeros(n_e)

# Initialize assignments based on mode
if test_mode:
    # Test mode: Load training activity to compute assignments
    print('\nComputing assignments from training data...')
    training_activity_path = data_path + 'activity/'

    # Try to find the most recent training activity file
    # Priority: Clean post-training activity (STDP off, final weights) > regular training activity
    num_classes = len(cfg.mnist_classes) if cfg.mnist_classes is not None else 10
    assignment_examples = cfg.assignment_examples_per_class * num_classes

    training_file_candidates = [
        f'resultPopVecs{assignment_examples}_clean.npy',  # PRIORITY: Post-training evaluation pass
        f'resultPopVecs{cfg.num_train_examples * 3}_clean.npy',  # Legacy: old format (full training set)
        f'resultPopVecs{cfg.num_train_examples * 3}.npy',  # Regular training activity (may be polluted)
        'resultPopVecs10000_clean.npy',  # Old clean checkpoint
        'resultPopVecs10000.npy',  # Old checkpoint (may be from different class set)
    ]

    training_result_monitor = None
    training_input_numbers = None
    used_filename = None

    for filename in training_file_candidates:
        try:
            result_path = training_activity_path + filename
            labels_path = training_activity_path + filename.replace('resultPopVecs', 'inputNumbers')
            training_result_monitor = np.load(result_path)
            training_input_numbers = np.load(labels_path)
            used_filename = filename
            break
        except FileNotFoundError:
            continue

    if training_result_monitor is None:
        print(f'\n{"="*60}')
        print(f'ERROR: Cannot compute assignments - training activity not found!')
        print(f'{"="*60}')
        print(f'Searched for:')
        for filename in training_file_candidates:
            print(f'  - {training_activity_path}{filename}')
        print(f'\nTo fix this, run training mode first:')
        print(f'  1. Set test_mode = False in config.py')
        print(f'  2. Run: python Diehl&Cook_spiking_MNIST.py')
        print(f'  3. Training will save activity files to {training_activity_path}')
        print(f'  4. Then you can run test mode')
        print(f'{"="*60}\n')
        exit(1)

    # Compute assignments from training data
    # Note: result_monitor is a circular buffer (size = update_interval)
    # input_numbers contains all examples, so we need to match the sizes
    num_activity_examples = training_result_monitor.shape[0]
    training_labels_subset = training_input_numbers[-num_activity_examples:]  # Last N examples

    assignments = get_new_assignments(training_result_monitor, training_labels_subset)

    # Indicate what type of activity was used
    is_clean = '_clean' in used_filename
    activity_type = "CLEAN post-training" if is_clean else "training (may be polluted)"

    print(f'Assignments computed from {num_activity_examples} training examples')
    print(f'  Source: {activity_type}')
    print(f'  File: {used_filename}')

    if not is_clean:
        print(f'\n  WARNING: Using training activity (STDP was ON)')
        print(f'  For best accuracy, use clean post-training activity.')
        print(f'  Run a full training session to generate *_clean.npy files.')

    # Count neurons per class
    print('\nNeuron assignments by class:')
    for cls in range(10):
        count = np.sum(assignments == cls)
        if count > 0:
            print(f'  Class {cls}: {count} neurons assigned')

    # Warn if assignments include classes not in current test set
    if cfg.mnist_classes is not None:
        assigned_classes = np.unique(assignments).astype(int)
        extra_classes = [c for c in assigned_classes if c not in cfg.mnist_classes]
        if extra_classes:
            print(f'\nWarning: Some neurons assigned to classes not in test set: {extra_classes}')
            print(f'   This suggests training data included different classes.')
            print(f'   Current test set classes: {cfg.mnist_classes}')
    print()

else:
    # Training mode: Start with zeros, will be updated during training
    assignments = np.zeros(n_e)

input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
if not test_mode and enable_live_plots:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1
if do_plot_performance and enable_live_plots:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0*Hz
net.run(0*ms)

# Normalize weights after loading (critical for proper initialization)
if not test_mode:
    print('Normalizing initial weights to target sum per neuron...')
    # Check weights before normalization
    if 'XeAe' in connections:
        temp_conn = np.zeros((n_input, n_e))
        temp_conn[connections['XeAe'].i[:], connections['XeAe'].j[:]] = connections['XeAe'].w[:]
        print(f'  Before: column sums range [{np.min(np.sum(temp_conn, axis=0)):.2f}, {np.max(np.sum(temp_conn, axis=0)):.2f}]')

    normalize_weights()

    # Check weights after normalization
    if 'XeAe' in connections:
        temp_conn = np.zeros((n_input, n_e))
        temp_conn[connections['XeAe'].i[:], connections['XeAe'].j[:]] = connections['XeAe'].w[:]
        print(f'  After:  column sums range [{np.min(np.sum(temp_conn, axis=0)):.2f}, {np.max(np.sum(temp_conn, axis=0)):.2f}]')
    print('Initial normalization complete\n')

# Create balanced epoch generator
# Use the examples_per_class calculated earlier based on available data
if cfg.mnist_classes is not None:
    total_examples_per_epoch = examples_per_class_per_epoch * len(cfg.mnist_classes)
else:
    total_examples_per_epoch = cfg.num_train_examples if not cfg.test_mode else cfg.num_test_examples

num_epochs = cfg.num_examples // total_examples_per_epoch if not cfg.test_mode else 1
epoch_generator = data_loader.create_epoch_generator(
    dataset_images,
    dataset_labels,
    total_examples_per_epoch,
    num_epochs,
    seed=cfg.random_seed
)

print(f'Starting simulation with {num_epochs} balanced epochs')
print(f'Each epoch: {total_examples_per_epoch} examples')
if cfg.mnist_classes is not None:
    print(f'Balanced: {examples_per_class_per_epoch} examples per class\n')

# Initial weight sanity check
if 'XeAe' in connections:
    initial_weights = connections['XeAe'].w[:]
    print('Initial synaptic weight statistics (Input→Excitatory):')
    print(f'  Count: {len(initial_weights)} connections')
    print(f'  Average: {np.mean(initial_weights):.6f}')
    print(f'  Std dev: {np.std(initial_weights):.6f}')
    print(f'  Range: [{np.min(initial_weights):.6f}, {np.max(initial_weights):.6f}]')
    print(f'  Target normalized sum per neuron: {cfg.weight_ee_input}')
    print()

# Initialize progress tracking
start_sim_time = time.time()
pbar = tqdm(total=num_examples, desc='Processing MNIST', unit='img',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

j = 0  # Global example counter

# Process all epochs
for epoch_images, epoch_labels, epoch_num in epoch_generator:
    pbar.write(f'\n--- Epoch {epoch_num + 1}/{num_epochs} ---')

    # Process each example in this epoch
    for example_idx in range(len(epoch_labels)):
        current_data = epoch_images[example_idx]
        current_label = epoch_labels[example_idx]

        # Normalize weights BEFORE presenting example (maintains constant total input weight)
        if not test_mode:
            normalize_weights()

        # Retry loop for low spike counts
        # Early in training, be more lenient as network is still learning
        min_spikes_required = 5 if j > 100 else max(1, j // 20)  # Gradually increase requirement
        max_retries = 10 if j > 100 else 5  # Fewer retries early on
        retry_count = 0

        while retry_count < max_retries:
            # Process this example
            rates = current_data.reshape((n_input)) / 8. *  input_intensity
            input_groups['Xe'].rates = rates * Hz
            net.run(single_example_time, report=None)

            current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
            previous_spike_count = np.copy(spike_counters['Ae'].count[:])
            total_spikes = np.sum(current_spike_count)

            if total_spikes < min_spikes_required:
                input_intensity += 2  # Increase faster early on
                retry_count += 1
                if retry_count < max_retries:
                    for i,name in enumerate(input_population_names):
                        input_groups[name+'e'].rates = 0*Hz
                    net.run(resting_time)
                    continue  # Try again
                else:
                    # Max retries reached, accept low spike count
                    if j < 20:  # Only warn for first 20 examples
                        pbar.write(f'Info: Example {j} produced {total_spikes} spikes (intensity reached {input_intensity:.1f})')
                    break
            else:
                # Sufficient spikes, exit retry loop
                break

        # Process results (always execute, even with low spikes)
        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
        if j % weight_update_interval == 0 and not test_mode and enable_live_plots:
            update_2d_input_weights(input_weight_monitor, fig_weights)
        if j % save_connections_interval == 0 and j > 0 and not test_mode:
            save_connections(str(j))
            save_theta(str(j))

        # Store results (always, even with low spikes)
        result_monitor[j%update_interval,:] = current_spike_count
        input_numbers[j] = current_label
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])

        # Update progress bar with statistics
        total_spikes = np.sum(current_spike_count)

        # Calculate average synaptic weight for monitoring
        avg_weight = np.mean(connections['XeAe'].w[:]) if 'XeAe' in connections else 0.0

        pbar.set_postfix({
            'spikes': f'{total_spikes:.0f}',
            'intensity': f'{input_intensity:.1f}',
            'retries': retry_count,
            'avg_w': f'{avg_weight:.4f}'
        })
        pbar.update(1)

        # Update live progress plot every 10 examples
        if record_spikes and enable_live_plots and j % 10 == 0:
            progress_data['examples'].append(j)
            progress_data['spikes'].append(np.sum(current_spike_count))
            progress_line.set_data(progress_data['examples'], progress_data['spikes'])
            ax_progress.set_ylim(0, max(10, max(progress_data['spikes']) * 1.1))
            fig_live.canvas.draw()
            fig_live.canvas.flush_events()

        # Periodic weight statistics
        if j % 100 == 0 and j > 0:
            if 'XeAe' in connections:
                weights = connections['XeAe'].w[:]
                pbar.write(f'[{j:5d}] Weight stats: avg={np.mean(weights):.4f}, '
                          f'std={np.std(weights):.4f}, min={np.min(weights):.4f}, '
                          f'max={np.max(weights):.4f}')

        if j % update_interval == 0 and j > 0:
            if do_plot_performance and enable_live_plots:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                pbar.write(f'Classification performance: {performance[:(j//update_interval)+1]}')

        # Reset for next example
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0*Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1

# Close progress bar and print summary
pbar.close()
elapsed_time = time.time() - start_sim_time
print(f'\n{"="*60}')
print(f'Simulation Complete!')
print(f'{"="*60}')
print(f'Total examples processed: {num_examples}')
print(f'Total simulation time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)')
print(f'Average time per example: {elapsed_time/num_examples:.2f}s')
print(f'{"="*60}\n')

# save results
print('Saving results...')
if not test_mode:
    save_theta()
    save_connections()

    # POST-TRAINING EVALUATION PASS
    # Run training set through network with STDP OFF to get clean assignments
    print('\n' + '='*60)
    print('POST-TRAINING EVALUATION PASS')
    print('='*60)
    print('Running training set through network with STDP OFF')
    print('to compute clean neuron assignments...')
    print()

    # Determine how many examples to use for assignment computation
    # Use balanced sampling from config
    num_classes = len(cfg.mnist_classes) if cfg.mnist_classes is not None else 10
    examples_per_class_for_assignment = cfg.assignment_examples_per_class
    assignment_examples = examples_per_class_for_assignment * num_classes
    print(f'Using {assignment_examples} training examples for assignments ({examples_per_class_for_assignment} per class)')

    # Create temporary result monitor for evaluation pass
    eval_result_monitor = np.zeros((assignment_examples, n_e))
    eval_input_numbers = np.zeros(assignment_examples, dtype=int)

    # Reset network state completely before evaluation pass
    print('Resetting network state...')
    for name in input_population_names:
        input_groups[name+'e'].rates = 0*Hz

    # Reset membrane potentials and conductances to resting state
    for name in population_names:
        neuron_groups[name + 'e'].v = v_rest_e
        neuron_groups[name + 'e'].ge = 0
        neuron_groups[name + 'e'].gi = 0
        neuron_groups[name + 'i'].v = v_rest_i
        neuron_groups[name + 'i'].ge = 0
        neuron_groups[name + 'i'].gi = 0

    # Note: Can't reset spike counters directly (read-only in Brian2)
    # Instead, we'll record the current count as baseline and subtract it
    eval_spike_baseline = np.copy(spike_counters['Ae'].count[:])

    print('Network state reset complete.')

    # Create balanced sample for assignment computation
    print('Creating balanced sample for assignment computation...')
    eval_images, eval_labels, _ = data_loader.create_balanced_epoch(
        dataset_images, dataset_labels,
        assignment_examples,
        random_state=np.random.RandomState(cfg.random_seed)
    )
    print(f'Balanced sample created: {len(eval_labels)} examples')

    # Run evaluation pass (STDP is already off - we're not in test_mode but not updating weights)
    print('Processing examples...')

    # Use tqdm for progress bar
    from tqdm import tqdm
    eval_intensity = start_input_intensity

    with tqdm(total=assignment_examples, desc='Eval pass', unit='img') as pbar:
        for eval_idx in range(assignment_examples):
            # Get example from balanced sample
            current_data = eval_images[eval_idx]
            current_label = eval_labels[eval_idx]

            # Initialize previous spike count (relative to baseline)
            if eval_idx == 0:
                eval_previous_spike_count = np.copy(eval_spike_baseline)

            # Present stimulus
            rates = current_data.reshape((n_input)) / 8. * eval_intensity
            input_groups['Xe'].rates = rates * Hz
            net.run(single_example_time, report=None)

            # Record spikes (subtract previous count to get spikes for this example only)
            current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - eval_previous_spike_count
            eval_previous_spike_count = np.copy(spike_counters['Ae'].count[:])

            eval_result_monitor[eval_idx, :] = current_spike_count
            eval_input_numbers[eval_idx] = current_label

            # Rest period
            input_groups['Xe'].rates = 0*Hz
            net.run(resting_time)

            pbar.update(1)

            if (eval_idx + 1) % 100 == 0:
                avg_spikes = eval_result_monitor[:eval_idx+1].sum(axis=1).mean()
                pbar.set_postfix({'avg_spikes': f'{avg_spikes:.1f}'})

    print('\nEvaluation pass complete!')
    print(f'Average spikes per example: {eval_result_monitor.sum(axis=1).mean():.1f}')

    # Save clean evaluation activity for assignment computation
    print('\nSaving clean assignment activity...')
    os.makedirs(data_path + 'activity', exist_ok=True)
    activity_filename = 'resultPopVecs' + str(assignment_examples) + '_clean'
    labels_filename = 'inputNumbers' + str(assignment_examples) + '_clean'
    np.save(data_path + 'activity/' + activity_filename, eval_result_monitor)
    np.save(data_path + 'activity/' + labels_filename, eval_input_numbers)
    print(f'  Saved: {activity_filename}.npy ({eval_result_monitor.shape[0]} examples)')
    print(f'  Saved: {labels_filename}.npy')
    print()
    print('These files should be used for computing assignments in test mode.')
    print('='*60)
else:
    # Create activity directory if it doesn't exist
    os.makedirs(data_path + 'activity', exist_ok=True)
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)


# plot results
if enable_live_plots:
    if rate_monitors:
        fig = plt.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(rate_monitors):
            plt.subplot(len(rate_monitors), 1, i+1)
            plt.plot(rate_monitors[name].t/second, rate_monitors[name].smooth_rate(window='flat', width=0.5*second)/Hz, '.')
            plt.title('Rates of population ' + name)

    if spike_monitors:
        fig = plt.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(spike_monitors):
            plt.subplot(len(spike_monitors), 1, i+1)
            plt.plot(spike_monitors[name].t/ms, spike_monitors[name].i, '.k')
            plt.title('Spikes of population ' + name)

    if spike_counters:
        fig = plt.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(spike_counters):
            plt.subplot(len(spike_counters), 1, i+1)
            plt.plot(spike_counters['Ae'].count[:])
            plt.title('Spike count of population ' + name)

    plot_2d_input_weights()
    plt.ioff()
    plt.show()



