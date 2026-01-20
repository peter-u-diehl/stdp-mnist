# Spiking Neural Network for MNIST Classification

## Migration Notice

This codebase has been migrated from **Python 2 to Python 3** and from **Brian v1 to Brian2**. The original implementation was published in 2015, and after 11 years, the migration modernizes the code for current Python environments and neuromorphic simulation frameworks. All functional behavior has been preserved to replicate the original experiment results.

For detailed migration information, technical fixes, and compatibility notes, see [MIGRATION_NOTES.md](notes/MIGRATION_NOTES.md).

---

<img width="939" height="483" alt="Screenshot 2026-01-12 at 2 00 23 AM" src="https://github.com/user-attachments/assets/e45609ac-f20b-4cbb-8cfe-c52efc4e6774" />





Implementation of the paper **"Unsupervised learning of digit recognition using spike-timing-dependent plasticity"** by Diehl & Cook (2015).

[Paper Link](http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/abstract)
[Original Repo](https://github.com/peter-u-diehl/stdp-mnist)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd modernized_implementation

# Install dependencies
pip install -r requirements.txt
```

## Pretrained Weights Available

This repository includes **pretrained weights** for a network trained on digits 0-4 (5 classes). You can immediately run evaluation and visualization without training:

- **Trained weights**: `mnist_data/weights/` (XeAe.npy, theta_A.npy)
- **Recorded activity**: `mnist_data/activity/` (spike counts from training and testing)
- **Analysis figures**: `analysis_figures/` (evaluation plots, specificity analysis, comprehensive metrics)
- **Visualization outputs**: `visualizations/figures/` (receptive fields, tuning curves, PCA trajectories, etc.)

To run evaluation with pretrained weights:
```bash
python Diehl\&Cook_MNIST_evaluation.py
```

### Run Your Own Experiment

You can train your own network and override all pre-saved weights, activity data, and figures. Modify experimental parameters in [config.py](config.py):

- Change `mnist_classes` to train on different digit subsets (e.g., `[0, 1]` for binary, `None` for all 10 digits)
- Set `test_mode = False` to train from scratch
- Adjust `num_train_examples`, `num_epochs`, learning rates, and network size

Training will overwrite the existing weights in `mnist_data/weights/` and activity in `mnist_data/activity/`. Running evaluation will regenerate analysis figures in `analysis_figures/`.

## Process Flow

**Step 1: Generate Initial Weights**
```bash
python sim_and_eval_utils/Diehl\&Cook_MNIST_random_conn_generator.py
```
Creates random initial connection matrices in `mnist_data/random/`

**Step 2: Train the Network**
```bash
# Edit config.py: Set test_mode = False
python Diehl\&Cook_spiking_MNIST.py
```
Trains network with STDP learning, saves weights to `mnist_data/weights/` and activity to `mnist_data/activity/`

**Step 3: Test the Network**
```bash
# Edit config.py: Set test_mode = True
python Diehl\&Cook_spiking_MNIST.py
```
Runs test set through trained network with STDP disabled, records spike activity

**Step 4: Evaluate Performance**
```bash
python Diehl\&Cook_MNIST_evaluation.py
```
Computes neuron assignments from training activity and calculates test accuracy

**Step 5 (Optional): Visualize Network Activity**
```bash
cd network_and_activity_visualization/
python visualize_network_activity.py
python visualize_network_graph_activity.py
python visualize_static_network_structure_3d.py
```
Generate animated visualizations of network structure and spiking activity

## Overview

This project implements the unsupervised learning spiking neural network (SNN) originally proposed by Diehl & Cook (2015). The network learns to classify handwritten digits from the MNIST dataset using biologically-plausible learning mechanisms including spike-timing-dependent plasticity (STDP) and lateral inhibition. Performance results and analysis visualizations are available in the `analysis_figures/` directory after running the evaluation script.

### Two-Phase Learning Process

This implementation follows a **two-phase unsupervised learning** approach:

1. **Phase 1 - Training (Unsupervised)**:
   - Network learns via STDP without any label information
   - Neurons develop receptive fields through competitive dynamics
   - Spike activity is recorded for each presented example
   - Runs in [Diehl&Cook_spiking_MNIST.py](Diehl&Cook_spiking_MNIST.py)

2. **Phase 2 - Evaluation (Post-hoc labeling)**:
   - After training completes, neurons are assigned to digit classes
   - Assignment based on which class each neuron fired most for during training
   - Test set accuracy is computed using these assignments
   - Runs in [Diehl&Cook_MNIST_evaluation.py](Diehl&Cook_MNIST_evaluation.py)

**Key insight**: Labels are never used during training - they only appear during the evaluation phase to assign neurons to classes and measure accuracy.

## Network Architecture

### Neuron Populations

The network consists of three main populations:

1. **Input Layer (X)**: 784 Poisson neurons (28×28 pixels)
   - Encodes MNIST images as spike rates
   - Firing rates proportional to pixel intensity

2. **Excitatory Layer (Ae)**: 400 adaptive LIF neurons
   - Primary learning layer
   - Implements STDP plasticity
   - Develops receptive fields for digit features

3. **Inhibitory Layer (Ai)**: 400 LIF neurons
   - Implements lateral inhibition
   - Enforces winner-take-all dynamics

### Network Connectivity

| Connection | Type | Pattern | Initial Weight | Plasticity |
|------------|------|---------|----------------|------------|
| X → Ae | Excitatory | Dense (100%) | Random [0.01, 0.31] | STDP enabled (training) |
| X → Ai | Excitatory | Sparse (10%) | Random [0.2] | Static |
| Ae → Ai | Excitatory | One-to-one | 10.4 | Static |
| Ai → Ae | Inhibitory | All-to-all (except self) | 17.0 | Static |

## Leaky Integrate-and-Fire (LIF) Neurons

### Excitatory Neuron Model

The excitatory neurons implement an adaptive LIF model with the following dynamics:

```python
dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100*ms)
I_synE = ge * nS * (-v)                    # Excitatory current
I_synI = gi * nS * (-100*mV - v)           # Inhibitory current
dge/dt = -ge / (1.0*ms)                    # Excitatory conductance decay
dgi/dt = -gi / (2.0*ms)                    # Inhibitory conductance decay
dtheta/dt = -theta / (tc_theta)            # Adaptive threshold decay
```

#### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `v_rest_e` | -65 mV | Resting potential |
| `v_reset_e` | -65 mV | Reset potential after spike |
| `v_thresh_e` | -52 mV | Base threshold voltage |
| `theta` | Adaptive | Additional threshold (increases with spiking) |
| `theta_plus_e` | 0.05 mV | Threshold increase per spike |
| `tc_theta` | 1e7 ms | Threshold decay time constant |
| `refrac_e` | 5 ms | Refractory period |
| `tau_mem` | 100 ms | Membrane time constant |

**Adaptive Threshold**: The threshold dynamically adjusts to implement homeostatic regulation:
```
v_threshold = v_thresh_e_const + theta - 20mV
```
Each time a neuron spikes, `theta` increases, making it temporarily harder to fire again. This prevents single neurons from dominating and encourages diverse feature learning.

### Inhibitory Neuron Model

The inhibitory neurons use a simpler, faster LIF model:

```python
dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10*ms)
I_synE = ge * nS * (-v)                    # Excitatory current
I_synI = gi * nS * (-85*mV - v)            # Inhibitory current
dge/dt = -ge / (1.0*ms)                    # Excitatory conductance decay
dgi/dt = -gi / (2.0*ms)                    # Inhibitory conductance decay
```

#### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `v_rest_i` | -60 mV | Resting potential |
| `v_reset_i` | -45 mV | Reset potential |
| `v_thresh_i` | -40 mV | Threshold voltage |
| `refrac_i` | 2 ms | Refractory period |
| `tau_mem` | 10 ms | Membrane time constant (10× faster) |

## Learning Rule: Spike-Timing-Dependent Plasticity (STDP)

### STDP Implementation

The network implements a triplet-based STDP rule on X → Ae synapses:

```python
# Synaptic state variables
dpre/dt = -pre / (tc_pre_ee)           # Pre-synaptic trace
dpost1/dt = -post1 / (tc_post_1_ee)    # Post-synaptic trace 1
dpost2/dt = -post2 / (tc_post_2_ee)    # Post-synaptic trace 2

# On pre-synaptic spike:
ge_post += w                                               # Synaptic transmission
pre = 1.0
w = clip(w - nu_ee_pre * post1, 0, wmax_ee)              # Depression (LTD)

# On post-synaptic spike:
post2before = post2
w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)  # Potentiation (LTP)
post1 = 1.0
post2 = 1.0
```

### STDP Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tc_pre_ee` | 20 ms | Pre-synaptic trace time constant |
| `tc_post_1_ee` | 20 ms | Post-synaptic trace 1 time constant |
| `tc_post_2_ee` | 40 ms | Post-synaptic trace 2 time constant |
| `nu_ee_pre` | 0.0001 | Learning rate for depression |
| `nu_ee_post` | 0.01 | Learning rate for potentiation (100× larger) |
| `wmax_ee` | 1.0 | Maximum synaptic weight |
| `exp_ee_pre` | 0.2 | Pre-synaptic exponent |
| `exp_ee_post` | 0.2 | Post-synaptic exponent |
| `STDP_offset` | 0.4 | STDP offset parameter |

### Weight Normalization

Synaptic weights are normalized **before presenting each example** to maintain constant total input:

```python
# For each excitatory neuron j:
sum_j = sum(w[i,j] for all i)
w[:,j] = w[:,j] * (weight['ee_input'] / sum_j)
```

Target sum: `weight['ee_input'] = 78.0`

**Important timing**: Normalization happens BEFORE each example is presented, not after. This ensures the network always sees properly scaled inputs from the start of each presentation.

This normalization is important for:
- Preventing runaway potentiation during STDP learning
- Ensuring fair competition between neurons
- Maintaining stable network dynamics
- Avoiding the need for retry loops when neurons fail to spike

## Simulation Algorithm

### Brian2 Integration

The simulation uses the **Brian2** neuromorphic simulator with explicit numerical integration:

- **Integration method**: Euler method
- **Time step**: `dt = 0.5 ms`
- **Clock**: `defaultclock.dt = 0.5*ms`

### Input Encoding

MNIST images (28×28 grayscale) are encoded as Poisson spike trains:

```python
# Normalize pixel values to spike rates
pixel_values = image.flatten()  # 784 pixels
rates = (pixel_values / 8.0) * input_intensity  # Scale to Hz
input_group.rates = rates * Hz
```

- `input_intensity`: Adjustable scaling factor (default: 2.0)
- Higher pixel values → higher firing rates
- Dynamic intensity adjustment if network receives too few spikes

### Simulation Loop

Each training/testing example follows this sequence:

1. **Weight normalization** (training only)
   - Normalize X → Ae weights BEFORE presenting example
   - Ensures network sees properly scaled inputs from the start

2. **Stimulus presentation** (350 ms)
   - Present MNIST image as Poisson input
   - Network processes input and generates spikes
   - STDP updates weights (training mode only)
   - Record excitatory neuron spike counts

3. **Resting period** (150 ms)
   - Input rates set to 0 Hz
   - Network activity decays to baseline
   - Prevents interference between consecutive examples

4. **Checkpointing** (training only)
   - Save weights every 10,000 examples

Total time per example: **500 ms** (350ms presentation + 150ms rest)

### Adaptive Input Intensity

If a presented example generates fewer than 5 total spikes:
```python
if sum(spike_counts) < 5:
    input_intensity += 1.0  # Increase intensity
    # Re-present the same example
```

This ensures sufficient network activity for learning.

## Classification Mechanism

### Neuron Assignment (Post-hoc)

After unsupervised training completes, neurons are assigned to digit classes using a **label assignment** procedure:

1. Load the recorded spike activity from training phase
2. Assign each neuron to the class for which it fired most frequently:
   ```python
   for each neuron i:
       for each class j in [0-9]:
           average_rate[i, j] = mean(spike_count[i] for examples labeled j)
       assignment[i] = argmax(average_rate[i, :])
   ```

**Important**: This assignment happens AFTER training in the evaluation script, NOT during training. The training phase is completely unsupervised.

### Prediction

To classify a test example:

1. Present the image and record spike counts: `spike_counts[i]` for neuron i
2. For each digit class j, compute summed response:
   ```python
   class_response[j] = sum(spike_counts[i] for neurons assigned to j)
   ```
3. Predicted class: `argmax(class_response)`

### Evaluation Timing

During a typical training run with 18,621 examples (3 epochs × 6,207 examples):
- Evaluations happen at fixed intervals (every 10,000 examples)
- **Only ONE evaluation** occurs during training at example 10,000
- Evaluations do NOT happen at epoch boundaries
- Final evaluation happens after all training completes

## Configurable Parameters

### Configuration System

The implementation uses a modular configuration system:

- **[config.py](config.py)**: Contains the `Config` class with all network parameters
- **[data_loader.py](data_loader.py)**: Handles MNIST data loading with class filtering and balanced sampling
- **Main script**: Uses config object for all parameter access

Key configuration options in [config.py](config.py):

```python
class Config:
    def __init__(self):
        # Mode selection
        self.test_mode = False              # True: test with pretrained weights
                                           # False: train from random initialization

        # Dataset filtering
        self.mnist_classes = [0, 1, 2]     # None = all classes [0-9]
                                           # Or list like [0, 1] for subset

        # Training examples
        self.num_train_examples = 6207     # Examples per epoch (balanced)
        self.num_test_examples = 2940      # Test examples to use

        # Simulation parameters
        self.single_example_time = 0.35 * second  # Stimulus duration
        self.resting_time = 0.15 * second         # Inter-stimulus interval
        self.dt = 0.5 * ms                        # Simulation time step

        # Network size
        self.n_input = 784                 # Input neurons (28×28)
        self.n_e = 400                     # Excitatory neurons (20×20)
        self.n_i = 400                     # Inhibitory neurons

        # Input encoding
        self.input_intensity = 2.0         # Base scaling factor for spike rates
```

### Learning Parameters

```python
# STDP time constants
tc_pre_ee = 20*ms            # Pre-synaptic trace decay
tc_post_1_ee = 20*ms         # Post-synaptic trace 1 decay
tc_post_2_ee = 40*ms         # Post-synaptic trace 2 decay

# Learning rates
nu_ee_pre = 0.0001          # Depression learning rate
nu_ee_post = 0.01           # Potentiation learning rate

# Weight bounds
wmax_ee = 1.0               # Maximum synaptic weight
weight['ee_input'] = 78.0   # Target sum for normalization
```

### Neuron Parameters

```python
# Excitatory neurons
v_rest_e = -65. * mV
v_reset_e = -65. * mV
v_thresh_e_const = -52. * mV
refrac_e = 5. * ms
theta_plus_e = 0.05 * mV    # Threshold adaptation
tc_theta = 1e7 * ms         # Threshold decay

# Inhibitory neurons
v_rest_i = -60. * mV
v_reset_i = -45. * mV
v_thresh_i = -40. * mV
refrac_i = 2. * ms
```

### Connection Weights and Delays

```python
# Synaptic weights
weight['ee_input'] = 78.    # Input → Excitatory (normalized sum)
weight['ei'] = 10.4         # Excitatory → Inhibitory
weight['ie'] = 17.0         # Inhibitory → Excitatory

# Synaptic delays
delay['ee_input'] = (0*ms, 10*ms)  # Uniform random [0, 10] ms
delay['ei_input'] = (0*ms, 5*ms)   # Uniform random [0, 5] ms
```

### Update Intervals

```python
update_interval = 10000              # Neuron assignment update
weight_update_interval = 20          # Weight normalization
save_connections_interval = 10000    # Weight checkpoint saving
```

## Usage

### Current Configuration

The default configuration in [config.py](config.py) is set up for **5-class testing** (digits 0, 1, 2, 3, 4):

```python
# config.py default settings:
self.test_mode = True               # Test mode (use pretrained weights)
self.mnist_classes = [0, 1, 2, 3, 4]  # Use only digits 0-4
self.num_train_examples = 10345     # 2,069 per class × 5 classes
self.num_test_examples = 4900       # 980 per class × 5 classes
```

This configuration:
- By default runs in TEST mode with pretrained weights
- Uses 5 classes (digits 0-4) instead of full 10-class MNIST
- For training: set `test_mode = False` in config.py
- Uses balanced sampling (exactly 2,069 examples per class per epoch for training)
- Takes significantly less time than full 10-class MNIST

To switch to **full 10-class MNIST**, modify [config.py:50](config.py#L50):
```python
self.mnist_classes = None           # Use all classes [0-9]
self.num_train_examples = 60000     # Full training set
```

### 1. Generate Initial Random Weights

```bash
python Diehl\&Cook_MNIST_random_conn_generator.py
```

This creates the `mnist_data/random/` directory with initial connection matrices:
- `XeAe.npy`: Input → Excitatory (dense, random)
- `XeAi.npy`: Input → Inhibitory (sparse, 10%)
- `AeAi.npy`: Excitatory → Inhibitory (one-to-one)
- `AiAe.npy`: Inhibitory → Excitatory (all-to-all except self)

### 2. Train from Scratch

```bash
python Diehl\&Cook_spiking_MNIST.py
```

After setting `test_mode = False` in [config.py:19](config.py#L19):
- Trains on 5 classes (digits 0-4) by default
- Runs for 3 epochs × 10,345 examples = 31,035 total examples
- Uses balanced sampling (2,069 examples per class per epoch)
- STDP updates weights continuously
- Saves weights to `mnist_data/weights/` periodically
- Saves spike activity to `mnist_data/activity/` directory
- Takes 1-2 hours on modern hardware

To test with pretrained weights, set `test_mode = True` in [config.py:19](config.py#L19) (this is the default)

### 3. Evaluate Performance

After training completes, evaluate the network:

```bash
python Diehl\&Cook_MNIST_evaluation.py
```

This script:
1. Loads the spike activity recorded during training
2. Assigns each neuron to the digit class it fired most for
3. Computes test accuracy using these assignments
4. Results are displayed in the console output

Performance varies based on training configuration, number of epochs, and class subset. See results in `analysis_figures/` for visualizations and detailed performance metrics.

## File Structure

```
diehl_2015_migration/
├── Diehl&Cook_spiking_MNIST.py              # Main simulation script
├── Diehl&Cook_MNIST_evaluation.py           # Performance evaluation
├── config.py                                # Configuration class
├── troubleshoot_spikes.py                   # Network debugging utility
├── README.md                                # This file
│
├── sim_and_eval_utils/                      # Simulation utilities
│   ├── __init__.py
│   ├── data_loader.py                       # MNIST data loading with balanced sampling
│   └── Diehl&Cook_MNIST_random_conn_generator.py  # Weight initialization
│
├── network_and_activity_visualization/      # Visualization scripts
│   ├── visualize_network_activity.py        # Animated network activity
│   ├── visualize_network_graph_activity.py  # Graph-based activity visualization
│   └── visualize_static_network_structure_3d.py  # 3D network structure
│
├── notes/                                   # Documentation
│   └── MIGRATION_NOTES.md                  # Python2→3, Brian1→2 migration notes
│
├── mnist_data/                              # MNIST data directory
│   ├── train-images.idx3-ubyte             # Training images
│   ├── train-labels.idx1-ubyte             # Training labels
│   ├── t10k-images.idx3-ubyte              # Test images
│   ├── t10k-labels.idx1-ubyte              # Test labels
│   │
│   ├── weights/                            # Trained network weights
│   │   ├── XeAe.npy                        # Input → Excitatory weights
│   │   └── theta_A.npy                     # Adaptive thresholds
│   │
│   ├── random/                             # Initial random connections
│   │   ├── XeAe.npy                        # Input → Excitatory (random)
│   │   ├── XeAi.npy                        # Input → Inhibitory (sparse)
│   │   ├── AeAi.npy                        # Excitatory → Inhibitory
│   │   └── AiAe.npy                        # Inhibitory → Excitatory
│   │
│   └── activity/                           # Simulation outputs
│       ├── resultPopVecs*.npy              # Spike count matrices
│       └── inputNumbers*.npy               # Ground truth labels
│
└── analysis_figures/                        # Generated visualizations (created at runtime)
    └── *.png, *.gif                        # Network activity animations and plots
```

### New Organization in This Migration

The repository has been reorganized for better modularity:

- **sim_and_eval_utils/**: Utility modules for simulation and evaluation
  - **[data_loader.py](sim_and_eval_utils/data_loader.py)**: MNIST data loading with balanced sampling
  - **[Diehl&Cook_MNIST_random_conn_generator.py](sim_and_eval_utils/Diehl&Cook_MNIST_random_conn_generator.py)**: Weight initialization

- **network_and_activity_visualization/**: All visualization scripts
  - Animated network activity visualizations
  - Graph-based activity plots
  - 3D network structure visualization

- **notes/**: Documentation and migration notes
  - **[MIGRATION_NOTES.md](notes/MIGRATION_NOTES.md)**: Python2→3, Brian1→2 migration details

## Data Loading and Balanced Sampling

The implementation includes a new `MNISTDataLoader` class that provides canonical ML-style data loading with important features for unsupervised learning:

### Class Filtering

Configure which MNIST classes to use in [config.py](config.py):

```python
self.mnist_classes = [0, 1, 2]  # Use only digits 0-2
# OR
self.mnist_classes = None       # Use all digits 0-9
```

This is useful for:
- Binary classification experiments (e.g., `[0, 1]`)
- Subset experiments to reduce training time
- Studying class-specific learning dynamics

### Balanced Sampling

For unsupervised learning, **balanced class representation** is important:

```python
# Example: Training with 3 classes [0, 1, 2]
# Total training examples: 6,207 = 2,069 per class × 3 classes
# Each epoch: exactly 2,069 examples from each class

data_loader = MNISTDataLoader(cfg)
images, labels = data_loader.load_training_data()

# Create balanced epoch
epoch_images, epoch_labels, indices = data_loader.create_balanced_epoch(
    images, labels,
    examples_per_epoch=6207,
    random_state=np.random.RandomState(seed)
)
```

**Why balanced sampling matters**:
- Prevents network bias toward overrepresented classes
- Ensures fair competition during STDP learning
- More stable convergence
- Important for unsupervised learning where labels aren't used during training

### Epoch Generation

The data loader provides a generator for multi-epoch training:

```python
# Generate 3 epochs of balanced, shuffled data
for epoch_images, epoch_labels, epoch_num in data_loader.create_epoch_generator(
    images, labels,
    examples_per_epoch=6207,
    num_epochs=3,
    seed=0
):
    # Each epoch has exactly 6,207 examples (balanced across classes)
    # Each epoch is shuffled differently
    # Examples are sampled without replacement until pool exhausted
```

## Key Features

### Biological Plausibility

- **Spiking neurons**: Event-based communication using action potentials
- **STDP learning**: Hebbian learning rule based on spike timing
- **Lateral inhibition**: Winner-take-all competition
- **Adaptive threshold**: Homeostatic regulation
- **Conductance-based synapses**: Biologically realistic current computation

### Unsupervised Learning

- No backpropagation or error signals
- Network self-organizes to recognize digit features
- Neurons specialize through competitive dynamics
- Label assignment performed post-hoc (after learning)

### Computational Efficiency

- Sparse spike-based computation
- Event-driven STDP (only updates on spikes)
- Efficient Brian2 code generation
- Progress tracking with tqdm

## Performance

Performance depends on training configuration, number of epochs, class subset, and random seed. After running the evaluation script, results and visualizations are saved to the `analysis_figures/` directory.

### Performance Notes

1. **Test set assignment**: The default evaluation uses test set to assign neuron labels. For proper evaluation following ML best practices, use training set assignments (modify line 83 in evaluation script).

2. **Variance**: Training from random initialization shows variance depending on random seed and initialization.

3. **Training time**: Full training takes several hours on modern CPUs depending on the number of examples and epochs.

## Monitoring and Visualization

The simulation provides real-time feedback:

### Progress Bar
```
Processing MNIST: |████████████| 10000/10000 [29:45<00:00, 5.60img/s, spikes=42, intensity=2.0]
```

### Live Plotting (Training Mode)

- **Weight evolution**: 2D visualization of learned receptive fields
- **Spike raster plots**: Network activity over time
- **Population rates**: Firing rates of each population
- **Performance plot**: Classification accuracy over training

### Saved Outputs

- `resultPopVecs*.npy`: Spike count matrix (examples × neurons)
- `inputNumbers*.npy`: Ground truth labels
- `XeAe*.npy`: Learned weight matrices
- `theta_*.npy`: Adaptive threshold values

## Theoretical Background

### Why STDP?

Spike-Timing-Dependent Plasticity strengthens synapses when pre-synaptic spikes consistently precede post-synaptic spikes (causality). This allows neurons to learn predictive features in the input data.

### Winner-Take-All Dynamics

The lateral inhibition (Ai → Ae) implements competition:
1. When an excitatory neuron spikes, it activates its inhibitory counterpart
2. The inhibitory neuron suppresses all other excitatory neurons
3. This ensures sparse, distributed representations
4. Different neurons learn different digit features

### Adaptive Threshold

The threshold adaptation (`theta`) implements homeostatic plasticity:
- Prevents single neurons from dominating
- Encourages all neurons to participate
- Maintains stable firing rates
- Important for learning diverse features

### Unsupervised Feature Learning

Without labels, neurons learn to respond to recurring patterns:
1. Initially, neurons have random connectivity
2. STDP strengthens connections from frequently co-active inputs
3. Neurons develop receptive fields for stroke patterns
4. Competition ensures different neurons learn different features
5. After training, neurons can be assigned to digit classes

## Troubleshooting

### Installation Issues

**Brian2 import error**:
```bash
pip install --upgrade brian2
```

**Matplotlib backend error**:
- The code sets backend to 'TkAgg' (line 10)
- If unavailable, try 'Qt5Agg' or 'Agg'

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{diehl2015unsupervised,
  title={Unsupervised learning of digit recognition using spike-timing-dependent plasticity},
  author={Diehl, Peter U and Cook, Matthew},
  journal={Frontiers in computational neuroscience},
  volume={9},
  pages={99},
  year={2015},
  publisher={Frontiers}
}
```

## References

- Original paper: [Frontiers in Computational Neuroscience](http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/abstract)
- Brian2 documentation: [briansimulator.org](https://briansimulator.org/)
- MNIST dataset: [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)

## Author

Code modifications by **Ethan Crouse** (2026)
Contact: ethancrouse98@gmail.com

**Peter U. Diehl**
Original implementation (2015)
Contact: peter.u.diehl@gmail.com

## License

Please refer to the original paper and contact the author for licensing information.

---

## Original README (from Peter U. Diehl)

This is the code for the paper "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" available at http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/abstract#

To run the simulations you also need to download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and install the brian simulator (the easiest way is to run the following command from a shell "easy_install brian", otherwise see http://briansimulator.org/docs/installation.html).

Testing with pretrained weights:
First run the main file "Diehl&Cook_spiking_MNIST.py" (which by default uses the pretrained weights) and wait until the simulation finished (on recent hardware this should take about 30min). After the simulation finished, you can run "Diehl&Cook_MNIST_evaluation.py" to evaluate the result, which should show a performance of 91.56%.

Training a new network:
At first you have to modify the main file "Diehl&Cook_spiking_MNIST.py" by changing line 210 to "test_mode = False" to train the network. The resulting weights will be stored in the subfolder "weights". Those weights can then be used to test the performance of the network by running a simulation using "Diehl&Cook_spiking_MNIST.py" with line 210 changed back to "test_mode = True". After the simulation finished, the performance can be evaluated using "Diehl&Cook_MNIST_evaluation.py" and should be around 89% using the given parameters.


If you have any questions don't hesitate to contact me via peter.u.diehl@gmail.com.



Note:
In this simple demo the performance is evaluated using neuron assignments of the test set. This leads to a slight increase in performance but violates good practice in machine learning. The results presented in the paper did NOT use the test set to determine neuron assignments. Instead the assignments were generated by running the same script in testing mode but using the 60000 examples of the training set to determine neuron assignments (this can be done by changing line 85 in "Diehl&Cook_MNIST_evaluation.py" to training_ending = '60000').
