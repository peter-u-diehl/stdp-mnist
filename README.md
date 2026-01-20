# Spiking Neural Network for MNIST Classification

## Migration Notice

This codebase has been migrated from **Python 2 to Python 3** and from **Brian v1 to Brian2**. The original implementation was published in 2015, and after 11 years, the migration modernizes the code for current Python environments and neuromorphic simulation frameworks. All functional behavior has been preserved to replicate the original experiment results.

For detailed migration information, technical fixes, and compatibility notes, see [MIGRATION_NOTES.md](modernized_implementation/notes/MIGRATION_NOTES.md).

---

<img width="939" height="483" alt="Screenshot 2026-01-12 at 2 00 23 AM" src="https://github.com/user-attachments/assets/e45609ac-f20b-4cbb-8cfe-c52efc4e6774" />

Implementation of the paper **"Unsupervised learning of digit recognition using spike-timing-dependent plasticity"** by Diehl & Cook (2015).

[Paper Link](http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/abstract)
[Original Repo](https://github.com/peter-u-diehl/stdp-mnist)

## Repository Structure

```
stdp-mnist-diehl-2015/
├── modernized_implementation/    # Python 3 + Brian2 (use this!)
│   ├── README.md                 # Full documentation
│   ├── Diehl&Cook_spiking_MNIST.py
│   ├── Diehl&Cook_MNIST_evaluation.py
│   ├── config.py
│   └── ...
└── legacy_implementation/        # Original Python 2 + Brian1 code (archived)
```

## Quick Start

```bash
cd modernized_implementation

# Install dependencies
pip install -r requirements.txt

# Run evaluation with pretrained weights
python Diehl\&Cook_MNIST_evaluation.py
```

See the full documentation in [modernized_implementation/README.md](modernized_implementation/README.md).

## Pretrained Weights Available

This repository includes **pretrained weights** for a network trained on digits 0-4 (5 classes). You can immediately run evaluation and visualization without training:

- **Trained weights**: `modernized_implementation/mnist_data/weights/`
- **Recorded activity**: `modernized_implementation/mnist_data/activity/`
- **Analysis figures**: `modernized_implementation/analysis_figures/`
- **Visualization outputs**: `modernized_implementation/visualizations/figures/`

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

## Author

Code modifications by **Ethan Crouse** (2026)
Contact: ethancrouse98@gmail.com

**Peter U. Diehl**
Original implementation (2015)
Contact: peter.u.diehl@gmail.com

## License

Please refer to the original paper and contact the author for licensing information.
