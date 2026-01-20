"""
Configuration for Diehl & Cook 2015 Spiking MNIST Experiment

This module contains all configurable parameters for the spiking neural network
MNIST classification experiment. Default values match the original paper.
"""

from brian2 import ms, mV, second, Hz
import numpy as np


class Config:
    """Configuration class for spiking MNIST experiment"""

    def __init__(self):
        # ======================================================================
        # Mode Settings
        # ======================================================================
        self.test_mode = True  # True: test with pretrained weights, False: train from scratch

        # ======================================================================
        # Path Settings
        # ======================================================================
        self.mnist_data_path = 'mnist_data/'  # Path to MNIST data files
        self.data_path = 'mnist_data/'  # Base path for saving/loading data

        # ======================================================================
        # Simulation Settings
        # ======================================================================
        self.dt = 0.5 * ms  # Simulation timestep
        self.random_seed = 0  # Random seed for reproducibility

        # ======================================================================
        # Network Architecture
        # ======================================================================
        self.n_input = 784  # Number of input neurons (28x28 MNIST images)
        self.n_e = 400  # Number of excitatory neurons
        self.n_i = 400  # Number of inhibitory neurons (typically same as n_e)

        # ======================================================================
        # Timing Parameters
        # ======================================================================
        self.single_example_time = 0.35 * second  # Presentation time per example
        self.resting_time = 0.15 * second  # Rest period between examples

        # ======================================================================
        # Training/Testing Parameters
        # ======================================================================
        # Dataset filtering
        self.mnist_classes = [0,1,2,3,4]  # None = all classes [0-9], or list like [0, 1] for binary classification
        self.num_train_examples = 10345  # Number of training examples per epoch (2069 per class × 5 classes)
        self.num_test_examples = 4900  # Number of test examples to use (980 per class × 5 classes, max available)
        self.test_examples_per_class = 980  # Max examples per class in test set (for balanced sampling)

        # Post-training assignment computation
        self.assignment_examples_per_class = 500  # Examples per class for computing neuron assignments (post-training eval pass)

        # These are set based on test_mode in _compute_derived_params()
        self.num_examples = None  # Will be set based on test_mode
        self.use_testing_set = None  # Will be set based on test_mode
        self.do_plot_performance = None  # Will be set based on test_mode
        self.record_spikes = True  # Record spike trains
        self.ee_STDP_on = None  # Will be set based on test_mode

        # Live plotting settings (can be disabled to improve performance)
        self.enable_live_plots = False  # Enable real-time plots during training (weight plots, progress plots, etc.)

        # Update intervals
        self.update_interval = None  # Will be computed based on num_examples
        self.weight_update_interval = 20  # How often to update weight plots
        self.save_connections_interval = 10000  # How often to save weights

        # ======================================================================
        # Neuron Parameters - Excitatory
        # ======================================================================
        self.v_rest_e = -65. * mV  # Resting potential
        self.v_reset_e = -65. * mV  # Reset potential after spike
        self.v_thresh_e_const = -52. * mV  # Base threshold voltage
        self.refrac_e = 5. * ms  # Refractory period
        self.offset = 20.0 * mV  # Threshold offset for adaptive threshold

        # ======================================================================
        # Neuron Parameters - Inhibitory
        # ======================================================================
        self.v_rest_i = -60. * mV  # Resting potential
        self.v_reset_i = -45. * mV  # Reset potential after spike
        self.v_thresh_i = -40. * mV  # Threshold voltage
        self.refrac_i = 2. * ms  # Refractory period

        # ======================================================================
        # Synaptic Weight Parameters
        # ======================================================================
        self.weight_ee_input = 78.  # Total synaptic weight for input->excitatory connections
        self.wmax_ee = 1.0  # Maximum weight for STDP

        # TEMPORARY DEBUG: Try higher weight sum to test if normalization is too aggressive
        # self.weight_ee_input = 150.  # Testing if original 78 is too low

        # ======================================================================
        # Synaptic Delay Parameters
        # ======================================================================
        self.delay_ee_input = (0*ms, 10*ms)  # Min and max delay for excitatory input
        self.delay_ei_input = (0*ms, 5*ms)  # Min and max delay for exc->inh connections

        # ======================================================================
        # Input Parameters
        # ======================================================================
        self.input_intensity = 2.0  # Scaling factor for input rates
        self.start_input_intensity = self.input_intensity  # Store initial value

        # ======================================================================
        # STDP (Spike-Timing Dependent Plasticity) Parameters
        # ======================================================================
        self.tc_pre_ee = 20 * ms  # Time constant for pre-synaptic trace
        self.tc_post_1_ee = 20 * ms  # Time constant for post-synaptic trace 1
        self.tc_post_2_ee = 40 * ms  # Time constant for post-synaptic trace 2
        self.nu_ee_pre = 0.0001  # Learning rate for LTD (long-term depression)
        self.nu_ee_post = 0.01  # Learning rate for LTP (long-term potentiation)
        self.exp_ee_pre = 0.2  # Exponent for pre-synaptic component
        self.exp_ee_post = 0.2  # Exponent for post-synaptic component
        self.STDP_offset = 0.4  # STDP offset parameter

        # ======================================================================
        # Adaptive Threshold Parameters
        # ======================================================================
        self.tc_theta = 1e7 * ms  # Time constant for threshold decay
        self.theta_plus_e = 0.05 * mV  # Threshold increase after each spike

        # ======================================================================
        # Population and Connection Names
        # ======================================================================
        self.input_population_names = ['X']
        self.population_names = ['A']
        self.input_connection_names = ['XA']
        self.save_conns = ['XeAe']
        self.input_conn_names = ['ee_input']
        self.recurrent_conn_names = ['ei', 'ie']

        self.ending = ''  # Suffix for saved weight files

        # Compute derived parameters based on mode
        self._compute_derived_params()

    def _compute_derived_params(self):
        """Compute parameters that depend on test_mode and other settings"""

        if self.test_mode:
            # Test mode settings
            self.weight_path = self.data_path + 'weights/'
            self.num_examples = self.num_test_examples
            self.use_testing_set = True
            self.do_plot_performance = False
            self.ee_STDP_on = False
            self.update_interval = self.num_examples
            self.weight_update_interval = 20
            self.save_connections_interval = 10000
        else:
            # Training mode settings
            self.weight_path = self.data_path + 'random/'
            # Default: 3 epochs over the training set
            self.num_examples = self.num_train_examples * 3
            self.use_testing_set = False
            self.do_plot_performance = True
            self.ee_STDP_on = True
            self.record_spikes = True

            # Set update intervals based on number of examples (training mode only)
            if self.num_examples <= 10000:
                self.update_interval = self.num_examples
                self.weight_update_interval = 20
            else:
                self.update_interval = 10000
                self.weight_update_interval = 100

            if self.num_examples <= 60000:
                self.save_connections_interval = 10000
            else:
                self.save_connections_interval = 10000
                self.update_interval = 10000

        # Compute runtime
        self.runtime = self.num_examples * (self.single_example_time + self.resting_time)

    def get_neuron_eqs_e(self):
        """Get excitatory neuron equations"""
        eqs = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
        if self.test_mode:
            eqs += '\n  theta      :volt'
        else:
            eqs += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
        eqs += '\n  dtimer/dt = 100.0*msecond/second  : second'
        return eqs

    def get_neuron_eqs_i(self):
        """Get inhibitory neuron equations"""
        return '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

    def get_stdp_eqs(self):
        """Get STDP synapse equations"""
        eqs_stdp_ee = '''
                w : 1
                post2before : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
        return eqs_stdp_ee

    def get_stdp_pre_eq(self):
        """Get STDP pre-synaptic update equation"""
        return 'ge_post += w; pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'

    def get_stdp_post_eq(self):
        """Get STDP post-synaptic update equation"""
        return 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

    def get_v_thresh_e_str(self):
        """Get excitatory threshold condition string"""
        return '(v>(theta - offset + v_thresh_e_const)) and (timer>refrac_e)'

    def get_scr_e(self):
        """Get excitatory reset equation"""
        if self.test_mode:
            return 'v = v_reset_e; timer = 0*ms'
        else:
            return 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

    def set_test_mode(self, test_mode):
        """Change test mode and recompute derived parameters"""
        self.test_mode = test_mode
        self._compute_derived_params()

    def get_class_filter_str(self):
        """Get a string describing the MNIST class filter"""
        if self.mnist_classes is None:
            return "all classes [0-9]"
        else:
            return f"classes {self.mnist_classes}"

    def should_use_example(self, label):
        """Check if an example with given label should be used based on mnist_classes filter"""
        if self.mnist_classes is None:
            return True
        return label in self.mnist_classes

    def __repr__(self):
        """String representation of configuration"""
        mode = "TEST" if self.test_mode else "TRAIN"
        classes_str = self.get_class_filter_str()
        return f"""Config({mode} mode):
  Network: {self.n_input} input -> {self.n_e} excitatory, {self.n_i} inhibitory
  Dataset: {classes_str}
  Examples: {self.num_examples} ({'test' if self.use_testing_set else 'train'} set)
  Timing: {self.single_example_time} presentation + {self.resting_time} rest
  STDP: {'ON' if self.ee_STDP_on else 'OFF'}
  Learning rates: nu_pre={self.nu_ee_pre}, nu_post={self.nu_ee_post}
"""


# Create a default config instance that can be imported
default_config = Config()
