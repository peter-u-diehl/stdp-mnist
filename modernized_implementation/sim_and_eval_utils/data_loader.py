"""
Data loading and preparation for Diehl & Cook 2015 Spiking MNIST

This module handles:
- Loading MNIST data with class filtering
- Creating balanced datasets for unsupervised learning
- Generating proper epochs with shuffling
- Caching filtered data to avoid redundant filtering

For unsupervised learning, we:
1. Filter to specified classes only
2. Balance classes within each epoch (equal representation)
3. Shuffle data each epoch to avoid order effects
4. Cache filtered data for efficiency
"""

import numpy as np
import pickle
import os.path
from struct import unpack


class MNISTDataLoader:
    """
    Handles MNIST data loading with filtering and balanced sampling.

    For unsupervised learning, balanced class representation ensures:
    - Network doesn't develop bias toward overrepresented classes
    - Fair competition during STDP learning
    - More stable convergence
    """

    def __init__(self, config):
        """
        Initialize data loader with configuration.

        Args:
            config: Config object with mnist_data_path and mnist_classes
        """
        self.config = config
        self.mnist_data_path = config.mnist_data_path
        self.mnist_classes = config.mnist_classes

        # Cached filtered data
        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None

    def _load_raw_mnist(self, pickle_name, is_train=True):
        """
        Load raw MNIST data from pickle or IDX files.

        Args:
            pickle_name: Path to pickle file (without .pickle extension)
            is_train: Whether this is training (True) or test (False) data

        Returns:
            dict with keys: 'x' (images), 'y' (labels), 'rows', 'cols'
        """
        pickle_path = f'{pickle_name}.pickle'

        if os.path.isfile(pickle_path):
            print(f'Loading cached MNIST data from {pickle_path}')
            data = pickle.load(open(pickle_path, 'rb'), encoding='latin1')
        else:
            print(f'Loading MNIST data from IDX files...')
            if is_train:
                images_file = self.mnist_data_path + 'train-images.idx3-ubyte'
                labels_file = self.mnist_data_path + 'train-labels.idx1-ubyte'
            else:
                images_file = self.mnist_data_path + 't10k-images.idx3-ubyte'
                labels_file = self.mnist_data_path + 't10k-labels.idx1-ubyte'

            with open(images_file, 'rb') as images, open(labels_file, 'rb') as labels:
                # Read image metadata
                images.read(4)  # skip magic number
                number_of_images = unpack('>I', images.read(4))[0]
                rows = unpack('>I', images.read(4))[0]
                cols = unpack('>I', images.read(4))[0]

                # Read label metadata
                labels.read(4)  # skip magic number
                N = unpack('>I', labels.read(4))[0]

                if number_of_images != N:
                    raise Exception('Number of labels did not match number of images')

                # Load all data
                x = np.zeros((N, rows, cols), dtype=np.uint8)
                y = np.zeros((N, 1), dtype=np.uint8)

                for i in range(N):
                    if i % 1000 == 0:
                        print(f"  Loading image {i}/{N}")
                    x[i] = [[unpack('>B', images.read(1))[0] for _ in range(cols)]
                            for _ in range(rows)]
                    y[i] = unpack('>B', labels.read(1))[0]

                data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}

                # Cache for future use
                pickle.dump(data, open(pickle_path, "wb"))
                print(f'Cached MNIST data to {pickle_path}')

        return data

    def _filter_by_classes(self, data):
        """
        Filter dataset to only include specified classes.

        Args:
            data: dict with 'x' (images) and 'y' (labels)

        Returns:
            tuple: (filtered_images, filtered_labels)
        """
        if self.mnist_classes is None:
            # No filtering - use all classes
            return data['x'], data['y'].flatten()

        # Create mask for specified classes
        mask = np.isin(data['y'].flatten(), self.mnist_classes)
        filtered_images = data['x'][mask]
        filtered_labels = data['y'].flatten()[mask]

        print(f'\nFiltered to classes {self.mnist_classes}:')
        for cls in self.mnist_classes:
            count = np.sum(filtered_labels == cls)
            print(f'  Class {cls}: {count:5d} examples')
        print(f'  Total:   {len(filtered_labels):5d} examples\n')

        return filtered_images, filtered_labels

    def load_training_data(self):
        """
        Load and filter training data. Results are cached.

        Returns:
            tuple: (images, labels) as numpy arrays
        """
        if self._train_data is None:
            raw_data = self._load_raw_mnist(self.mnist_data_path + 'training', is_train=True)
            self._train_data, self._train_labels = self._filter_by_classes(raw_data)

        return self._train_data, self._train_labels

    def load_test_data(self):
        """
        Load and filter test data. Results are cached.

        Returns:
            tuple: (images, labels) as numpy arrays
        """
        if self._test_data is None:
            raw_data = self._load_raw_mnist(self.mnist_data_path + 'testing', is_train=False)
            self._test_data, self._test_labels = self._filter_by_classes(raw_data)

        return self._test_data, self._test_labels

    def create_balanced_epoch(self, images, labels, examples_per_epoch, random_state=None):
        """
        Create a balanced epoch with equal examples per class.

        For unsupervised learning, balanced sampling is crucial:
        - Prevents network bias toward frequent classes
        - Ensures fair STDP competition
        - More stable learning dynamics

        Args:
            images: Full dataset images
            labels: Full dataset labels
            examples_per_epoch: Total examples in epoch
            random_state: numpy RandomState for reproducibility

        Returns:
            tuple: (epoch_images, epoch_labels, indices_used)
        """
        if random_state is None:
            random_state = np.random.RandomState()

        # Determine which classes to use
        if self.mnist_classes is None:
            classes = np.unique(labels)
        else:
            classes = self.mnist_classes

        num_classes = len(classes)
        examples_per_class = examples_per_epoch // num_classes
        remainder = examples_per_epoch % num_classes

        # Check if we have enough data
        for cls in classes:
            available = np.sum(labels == cls)
            # Worst case: this class gets one extra sample due to remainder
            needed = examples_per_class + (1 if remainder > 0 else 0)
            if available < needed:
                raise ValueError(
                    f'Class {cls} has only {available} examples, '
                    f'but {needed} requested per class (with remainder)'
                )

        # Sample balanced examples from each class
        epoch_indices = []
        for i, cls in enumerate(classes):
            # Get all indices for this class
            cls_indices = np.where(labels == cls)[0]
            # Distribute remainder: first 'remainder' classes get one extra example
            count = examples_per_class + (1 if i < remainder else 0)
            # Randomly sample without replacement
            sampled = random_state.choice(cls_indices, size=count, replace=False)
            epoch_indices.extend(sampled)

        # Shuffle the epoch to mix classes
        epoch_indices = np.array(epoch_indices)
        random_state.shuffle(epoch_indices)

        return images[epoch_indices], labels[epoch_indices], epoch_indices

    def create_epoch_generator(self, images, labels, examples_per_epoch, num_epochs, seed=None):
        """
        Generator that yields balanced epochs.

        Each epoch:
        - Has exactly examples_per_epoch samples
        - Is balanced across classes
        - Is shuffled differently
        - Uses different samples from the pool (until exhausted)

        Args:
            images: Full dataset images
            labels: Full dataset labels
            examples_per_epoch: Samples per epoch
            num_epochs: Number of epochs to generate
            seed: Random seed for reproducibility

        Yields:
            tuple: (epoch_images, epoch_labels, epoch_number)
        """
        rng = np.random.RandomState(seed)

        # Determine classes
        if self.mnist_classes is None:
            classes = np.unique(labels)
        else:
            classes = self.mnist_classes

        num_classes = len(classes)
        examples_per_class = examples_per_epoch // num_classes
        remainder = examples_per_epoch % num_classes

        # Create per-class pools
        class_pools = {}
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            # Shuffle each class pool
            rng.shuffle(cls_indices)
            class_pools[cls] = cls_indices.tolist()

        for epoch in range(num_epochs):
            epoch_indices = []

            for i, cls in enumerate(classes):
                # Get indices for this class
                pool = class_pools[cls]

                # Distribute remainder: first 'remainder' classes get one extra
                count = examples_per_class + (1 if i < remainder else 0)

                # If pool is exhausted, refill and reshuffle
                if len(pool) < count:
                    cls_indices = np.where(labels == cls)[0]
                    rng.shuffle(cls_indices)
                    pool = cls_indices.tolist()
                    class_pools[cls] = pool

                # Take examples from pool
                sampled = pool[:count]
                epoch_indices.extend(sampled)
                class_pools[cls] = pool[count:]

            # Shuffle epoch to mix classes
            epoch_indices = np.array(epoch_indices)
            rng.shuffle(epoch_indices)

            yield images[epoch_indices], labels[epoch_indices], epoch

    def get_summary(self):
        """
        Get summary statistics of loaded data.

        Returns:
            dict with training and test statistics
        """
        summary = {}

        if self._train_data is not None:
            summary['train_total'] = len(self._train_labels)
            summary['train_per_class'] = {
                cls: np.sum(self._train_labels == cls)
                for cls in (self.mnist_classes if self.mnist_classes else range(10))
            }

        if self._test_data is not None:
            summary['test_total'] = len(self._test_labels)
            summary['test_per_class'] = {
                cls: np.sum(self._test_labels == cls)
                for cls in (self.mnist_classes if self.mnist_classes else range(10))
            }

        return summary
