#!/usr/bin/env python
"""
Run All Visualizations

Master script to generate all SNN visualizations.
Each visualization is run as a subprocess to avoid Brian2 state conflicts.
"""

import subprocess
import sys
import os
from datetime import datetime

# List of visualization scripts to run
VISUALIZATIONS = [
    {
        'script': 'neural_trajectory_pca.py',
        'name': 'Neural Trajectory PCA',
        'description': '3D animated trajectory through PCA space'
    },
    {
        'script': 'recurrence_plot.py',
        'name': 'Recurrence Plots',
        'description': 'When neural states revisit similar regions'
    },
    {
        'script': 'spike_raster_trajectory.py',
        'name': 'Spike Raster + Trajectory',
        'description': 'Synchronized spike raster and trajectory animation'
    },
    {
        'script': 'weight_receptive_fields.py',
        'name': 'Receptive Fields',
        'description': 'Learned weight patterns of excitatory neurons'
    },
    {
        'script': 'neuron_tuning_curves.py',
        'name': 'Neuron Tuning Curves',
        'description': 'Response profiles across digit classes'
    },
    {
        'script': 'separation_dynamics.py',
        'name': 'Separation Dynamics',
        'description': 'Inter-class distance over time'
    },
    {
        'script': 'correct_vs_incorrect.py',
        'name': 'Correct vs Incorrect',
        'description': 'Compare successful and failed classifications'
    },
    {
        'script': 'cross_digit_embedding.py',
        'name': 'Cross-Digit Embedding',
        'description': 't-SNE/UMAP embedding of neural representations'
    },
    {
        'script': 'spike_timing_analysis.py',
        'name': 'Spike Timing Analysis',
        'description': 'Temporal precision and ISI distributions'
    },
    {
        'script': 'effective_connectivity.py',
        'name': 'Effective Connectivity',
        'description': 'Co-activation graph between neurons'
    },
]


def run_visualization(script, name, show=False):
    """Run a single visualization script."""
    script_path = os.path.join(os.path.dirname(__file__), script)

    if not os.path.exists(script_path):
        print(f"  [SKIP] Script not found: {script}")
        return False

    cmd = [sys.executable, script_path]
    if not show:
        # Add flag to suppress display if the script supports it
        pass

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per visualization
        )

        if result.returncode == 0:
            print(f"  [OK] {name}")
            return True
        else:
            print(f"  [FAIL] {name}")
            if result.stderr:
                # Print last few lines of error
                error_lines = result.stderr.strip().split('\n')[-5:]
                for line in error_lines:
                    print(f"        {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all SNN visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Show each visualization as it completes')
    parser.add_argument('--only', type=str,
                       help='Only run visualization matching this name (partial match)')
    parser.add_argument('--list', action='store_true',
                       help='List available visualizations and exit')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable visualizations:")
        print("=" * 60)
        for viz in VISUALIZATIONS:
            print(f"  {viz['name']}")
            print(f"    Script: {viz['script']}")
            print(f"    {viz['description']}")
            print()
        return

    print("=" * 60)
    print("SNN Visualization Suite")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Filter if --only specified
    visualizations = VISUALIZATIONS
    if args.only:
        visualizations = [v for v in VISUALIZATIONS
                         if args.only.lower() in v['name'].lower() or
                            args.only.lower() in v['script'].lower()]
        if not visualizations:
            print(f"No visualizations matching '{args.only}'")
            return

    print(f"Running {len(visualizations)} visualizations...")
    print()

    successes = 0
    failures = 0

    for viz in visualizations:
        print(f"[{successes + failures + 1}/{len(visualizations)}] {viz['name']}")
        print(f"    {viz['description']}")

        if run_visualization(viz['script'], viz['name'], show=args.show):
            successes += 1
        else:
            failures += 1
        print()

    print("=" * 60)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {successes} succeeded, {failures} failed")
    print()

    # List generated files
    viz_dir = os.path.dirname(__file__)
    html_files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]

    if html_files:
        print("Generated HTML files:")
        for f in sorted(html_files):
            print(f"  {f}")


if __name__ == "__main__":
    main()
