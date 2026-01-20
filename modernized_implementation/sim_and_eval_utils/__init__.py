"""
Simulation and Evaluation Utilities

This module contains utilities for:
- Loading and preprocessing MNIST data (data_loader.py)
- Generating initial random connection matrices (Diehl&Cook_MNIST_random_conn_generator.py)
"""

from .data_loader import MNISTDataLoader

__all__ = ['MNISTDataLoader']
