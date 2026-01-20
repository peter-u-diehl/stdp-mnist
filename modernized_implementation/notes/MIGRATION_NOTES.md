# Migration from Brian to Brian2 and Python3

This codebase has been migrated from Brian v1 (Python 2) to Brian2 (Python 3). All functional behavior has been preserved to replicate the original experiment.

## Files Migrated

1. **Diehl&Cook_spiking_MNIST.py** - Main simulation script
2. **Diehl&Cook_MNIST_random_conn_generator.py** - Weight initialization
3. **Diehl&Cook_MNIST_evaluation.py** - Performance evaluation
4. **weights/WeightReadout.py** - Weight visualization

## Key Changes

### Python 2 to Python 3
- `import cPickle as pickle` → `import pickle`
- `xrange()` → `range()`
- `print` statements → `print()` functions
- Added `encoding='latin1'` to pickle.load() for compatibility
- Added `allow_pickle=True` to np.load() for .npy files

### Brian to Brian2

#### Import Changes
- `import brian as b` → `from brian2 import *`
- `from brian import *` → removed
- `import brian_no_units` → removed (units are mandatory in Brian2)

#### Clock and Preferences
- `b.set_global_preferences(...)` → `defaultclock.dt = 0.5*ms`
- Old preference system removed (codegen is always on in Brian2)

#### Neuron Groups
- `b.NeuronGroup(...)` → `NeuronGroup(...)`
- Added `method='euler'` parameter to all NeuronGroups
- String thresholds now use `and` instead of `*`
- Added `(unless refractory)` flag to differential equations
- Subgroup syntax: `.subgroup(n)` → `[0:n]`

#### Synapses (formerly Connections)
- `b.Connection(...)` → `Synapses(...)`
- Complete rewrite of connection/synapse creation:
  - Model defined with `model='w : 1'`
  - Spike effects: `on_pre='ge_post += w'`
  - Connection setup: `.connect(i=sources, j=targets)`
  - Weight assignment: `.w = values`
  - Delay assignment: `.delay = ...`

#### STDP Implementation
- STDP variables marked as `(event-driven)`
- Weight updates now use `clip()` function for bounds
- STDP is now implemented as Synapses with `on_pre` and `on_post`

#### Monitors
- `b.PopulationRateMonitor(...)` → `PopulationRateMonitor(...)`
- `b.SpikeMonitor(...)` → `SpikeMonitor(...)`
- `b.SpikeCounter(...)` → `SpikeMonitor(...)` (use .count attribute)
- Rate monitor access: `.rate` → `.smooth_rate(window='flat', width=0.5*second)/Hz`
- Spike monitor: now returns `.t` and `.i` instead of nested structure

#### Network Management
- Explicit `Network()` object required in Brian2
- All objects must be added to network: `net.add(...)`
- Run command: `b.run(...)` → `net.run(...)`

#### Plotting
- `b.raster_plot(...)` → manual plotting with `plt.plot(spike_monitor.t/ms, spike_monitor.i, '.k')`
- `b.figure(...)` → `plt.figure(...)`
- `b.subplot(...)` → `plt.subplot(...)`
- `b.ion()` → `plt.ion()`
- `b.show()` → `plt.show()`

#### Poisson Groups
- `b.PoissonGroup(n, 0)` → `PoissonGroup(n, 0*Hz)`
- `.rate` attribute → `.rates` attribute
- Must include units: `0*Hz` or `rates * Hz`

#### Units
- All values must have units in Brian2
- `b.second` → `second`
- `b.ms` → `ms`
- `b.mV` → `mV`
- Units come from `from brian2 import *`

## Installation Requirements

To run the migrated code, you need:

```bash
pip install brian2 numpy matplotlib scipy
```

## Functional Equivalence

All changes maintain functional equivalence with the original code:
- Same neuron equations and parameters
- Same network architecture
- Same STDP learning rules
- Same weight normalization
- Same input processing
- Same output evaluation

The simulation should produce identical results to the original Brian v1 implementation.

## Testing

To verify the migration:
1. Run `Diehl&Cook_MNIST_random_conn_generator.py` to generate initial weights
2. Run `Diehl&Cook_spiking_MNIST.py` in test mode (default)
3. Run `Diehl&Cook_MNIST_evaluation.py` to evaluate performance
4. Expected accuracy: ~91.56% (same as original)

## Notes

- The migration preserves all original comments and structure
- No functional changes or improvements were made
- The code replicates the experiment from the paper: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity"
- Matplotlib backend set to 'TkAgg' for compatibility
