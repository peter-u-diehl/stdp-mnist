'''
Created on 15.12.2014

@author: Peter U. Diehl

Modified for 4:1 E:I Ratio Experiment with Specificity Analysis

Added metrics to diagnose Winner-Take-All (WTA) strength:
- Assignment confidence (how strongly each neuron prefers its assigned class)
- Neurons per class distribution
- Spike count statistics per presentation
- Selectivity index
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import pickle
import matplotlib.pyplot as plt
import os
from config import Config
from sim_and_eval_utils.data_loader import MNISTDataLoader

# Initialize configuration
cfg = Config()

# Initialize data loader
data_loader = MNISTDataLoader(cfg)


# functions

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    # Determine which classes to iterate over based on config
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    for j in classes_to_check:
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments


def compute_confusion_matrix(assignments, result_monitor, input_numbers, n_classes):
    """
    Compute confusion matrix from population voting.

    Returns:
    - confusion_matrix: n_classes x n_classes matrix
    - per_class_accuracy: accuracy for each class
    - most_confused_pairs: list of (true_class, predicted_class, count) tuples
    """
    input_nums = np.asarray(input_numbers)
    n_examples = result_monitor.shape[0]
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    class_list = list(classes_to_check)
    n_classes = len(class_list)

    # Create class index mapping
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(n_examples):
        true_label = int(input_nums[i])
        if true_label not in class_to_idx:
            continue

        # Get prediction via population voting
        spike_rates = result_monitor[i, :]
        summed_rates = np.zeros(n_classes)
        for idx, c in enumerate(class_list):
            neurons_for_class = assignments == c
            if np.sum(neurons_for_class) > 0:
                summed_rates[idx] = np.mean(spike_rates[neurons_for_class])

        predicted_idx = np.argmax(summed_rates)
        true_idx = class_to_idx[true_label]
        confusion[true_idx, predicted_idx] += 1

    # Per-class accuracy
    per_class_accuracy = {}
    for idx, c in enumerate(class_list):
        total = np.sum(confusion[idx, :])
        correct = confusion[idx, idx]
        per_class_accuracy[c] = (correct / total * 100) if total > 0 else 0

    # Find most confused pairs
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion[i, j] > 0:
                confused_pairs.append((class_list[i], class_list[j], confusion[i, j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        'confusion_matrix': confusion,
        'class_labels': class_list,
        'per_class_accuracy': per_class_accuracy,
        'most_confused_pairs': confused_pairs[:10]  # Top 10
    }


def compute_receptive_field_quality(result_monitor, input_numbers, assignments, training_images):
    """
    Analyze the quality of learned receptive fields.

    Computes:
    - Weight concentration: How localized are the learned features?
    - Feature diversity: How different are receptive fields within same class?
    - Dead neuron analysis: Neurons that rarely fire
    """
    input_nums = np.asarray(input_numbers)
    n_neurons = result_monitor.shape[1]
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)

    metrics = {}

    # === 1. Activity Distribution Analysis ===
    total_spikes_per_neuron = np.sum(result_monitor, axis=0)

    # Identify dead/low-activity neurons
    mean_spikes = np.mean(total_spikes_per_neuron)
    std_spikes = np.std(total_spikes_per_neuron)

    dead_threshold = 0.01 * mean_spikes  # Less than 1% of average
    low_threshold = 0.1 * mean_spikes    # Less than 10% of average
    high_threshold = mean_spikes + 2 * std_spikes  # Dominating neurons

    metrics['dead_neurons'] = np.sum(total_spikes_per_neuron < dead_threshold)
    metrics['low_activity_neurons'] = np.sum(total_spikes_per_neuron < low_threshold)
    metrics['high_activity_neurons'] = np.sum(total_spikes_per_neuron > high_threshold)
    metrics['activity_gini'] = compute_gini(total_spikes_per_neuron)

    # === 2. Spike Rate Statistics per Neuron ===
    metrics['spike_distribution'] = {
        'mean': np.mean(total_spikes_per_neuron),
        'std': np.std(total_spikes_per_neuron),
        'min': np.min(total_spikes_per_neuron),
        'max': np.max(total_spikes_per_neuron),
        'median': np.median(total_spikes_per_neuron)
    }

    # === 3. Class-wise Activity Analysis ===
    class_activity = {}
    for c in classes_to_check:
        class_mask = input_nums == c
        if np.sum(class_mask) > 0:
            class_spikes = np.sum(result_monitor[class_mask, :], axis=0)
            class_activity[c] = {
                'mean': np.mean(class_spikes),
                'std': np.std(class_spikes),
                'active_neurons': np.sum(class_spikes > 0)
            }
    metrics['class_activity'] = class_activity

    # === 4. Neuron Utilization ===
    # What fraction of neurons contribute meaningfully?
    cumulative_spikes = np.cumsum(np.sort(total_spikes_per_neuron)[::-1])
    total_spikes = cumulative_spikes[-1] if len(cumulative_spikes) > 0 else 1

    # Find how many neurons account for 50%, 80%, 90% of activity
    pct_50 = np.searchsorted(cumulative_spikes, 0.5 * total_spikes) + 1
    pct_80 = np.searchsorted(cumulative_spikes, 0.8 * total_spikes) + 1
    pct_90 = np.searchsorted(cumulative_spikes, 0.9 * total_spikes) + 1

    metrics['neurons_for_50pct_activity'] = pct_50
    metrics['neurons_for_80pct_activity'] = pct_80
    metrics['neurons_for_90pct_activity'] = pct_90

    return metrics


def compute_gini(values):
    """Compute Gini coefficient (0=perfect equality, 1=perfect inequality)"""
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_population_coding_metrics(result_monitor, input_numbers, assignments):
    """
    Analyze population coding quality.

    Measures how well the population of neurons encodes digit identity:
    - Population vector consistency: Do same digits produce similar patterns?
    - Inter-class separation: Are different digit patterns distinguishable?
    - Information theoretic measures
    """
    input_nums = np.asarray(input_numbers)
    n_neurons = result_monitor.shape[1]
    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    class_list = list(classes_to_check)
    n_classes = len(class_list)

    metrics = {}

    # === 1. Compute mean population vector per class ===
    class_centroids = np.zeros((n_classes, n_neurons))
    class_stds = np.zeros((n_classes, n_neurons))

    for idx, c in enumerate(class_list):
        class_mask = input_nums == c
        if np.sum(class_mask) > 0:
            class_centroids[idx, :] = np.mean(result_monitor[class_mask, :], axis=0)
            class_stds[idx, :] = np.std(result_monitor[class_mask, :], axis=0)

    # === 2. Intra-class consistency (average correlation within class) ===
    intra_class_corr = {}
    for idx, c in enumerate(class_list):
        class_mask = input_nums == c
        class_examples = result_monitor[class_mask, :]
        if len(class_examples) > 1:
            # Sample to avoid memory issues
            n_sample = min(100, len(class_examples))
            sample_idx = np.random.choice(len(class_examples), n_sample, replace=False)
            sample = class_examples[sample_idx, :]

            # Compute pairwise correlations
            corrs = []
            for i in range(n_sample):
                for j in range(i+1, n_sample):
                    if np.std(sample[i]) > 0 and np.std(sample[j]) > 0:
                        corrs.append(np.corrcoef(sample[i], sample[j])[0, 1])
            intra_class_corr[c] = np.mean(corrs) if corrs else 0
        else:
            intra_class_corr[c] = 0

    metrics['intra_class_correlation'] = intra_class_corr
    metrics['mean_intra_class_corr'] = np.mean(list(intra_class_corr.values()))

    # === 3. Inter-class separation (distance between class centroids) ===
    inter_class_distances = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            dist = np.linalg.norm(class_centroids[i] - class_centroids[j])
            inter_class_distances[i, j] = dist
            inter_class_distances[j, i] = dist

    # Average distance
    metrics['mean_inter_class_distance'] = np.mean(inter_class_distances[np.triu_indices(n_classes, k=1)])

    # Compute Fisher's criterion: inter-class variance / intra-class variance
    inter_class_var = np.var(class_centroids, axis=0)
    intra_class_var = np.mean(class_stds**2, axis=0)

    # Avoid division by zero
    fisher_ratio = np.zeros(n_neurons)
    valid_mask = intra_class_var > 0
    fisher_ratio[valid_mask] = inter_class_var[valid_mask] / intra_class_var[valid_mask]

    metrics['mean_fisher_ratio'] = np.mean(fisher_ratio)
    metrics['median_fisher_ratio'] = np.median(fisher_ratio)

    # === 4. Silhouette-like score ===
    # Simplified: for each example, compare distance to own centroid vs nearest other centroid
    silhouette_scores = []
    sample_size = min(500, result_monitor.shape[0])
    sample_indices = np.random.choice(result_monitor.shape[0], sample_size, replace=False)

    for i in sample_indices:
        true_class_idx = None
        for idx, c in enumerate(class_list):
            if input_nums[i] == c:
                true_class_idx = idx
                break

        if true_class_idx is None:
            continue

        # Distance to own centroid
        own_dist = np.linalg.norm(result_monitor[i] - class_centroids[true_class_idx])

        # Distance to nearest other centroid
        other_dists = []
        for idx in range(n_classes):
            if idx != true_class_idx:
                other_dists.append(np.linalg.norm(result_monitor[i] - class_centroids[idx]))

        if other_dists:
            nearest_other = min(other_dists)
            max_dist = max(own_dist, nearest_other)
            if max_dist > 0:
                silhouette_scores.append((nearest_other - own_dist) / max_dist)

    metrics['mean_silhouette'] = np.mean(silhouette_scores) if silhouette_scores else 0

    return metrics


def compute_temporal_dynamics(result_monitor, input_numbers):
    """
    Analyze temporal learning dynamics across training.

    If we have access to multiple checkpoints, track how metrics evolve.
    Otherwise, analyze trends within the available data.
    """
    input_nums = np.asarray(input_numbers)
    n_examples = result_monitor.shape[0]

    metrics = {}

    # Divide training into epochs/chunks to see progression
    n_chunks = min(10, n_examples // 100)  # At least 100 examples per chunk
    if n_chunks < 2:
        n_chunks = 2
    chunk_size = n_examples // n_chunks

    # Track metrics per chunk
    spikes_per_chunk = []
    active_neurons_per_chunk = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else n_examples

        chunk_data = result_monitor[start:end, :]

        spikes_per_chunk.append(np.mean(np.sum(chunk_data, axis=1)))
        active_neurons_per_chunk.append(np.mean(np.sum(chunk_data > 0, axis=1)))

    metrics['chunks'] = n_chunks
    metrics['spikes_per_chunk'] = spikes_per_chunk
    metrics['active_neurons_per_chunk'] = active_neurons_per_chunk

    # Compute trend (is activity increasing/decreasing over training?)
    if len(spikes_per_chunk) > 1:
        spike_trend = np.polyfit(range(len(spikes_per_chunk)), spikes_per_chunk, 1)[0]
        active_trend = np.polyfit(range(len(active_neurons_per_chunk)), active_neurons_per_chunk, 1)[0]
        metrics['spike_trend'] = spike_trend
        metrics['active_neuron_trend'] = active_trend
    else:
        metrics['spike_trend'] = 0
        metrics['active_neuron_trend'] = 0

    return metrics


def compute_specificity_analysis(result_monitor, input_numbers, assignments):
    """
    Compute detailed specificity metrics to diagnose WTA strength.

    Returns a dictionary with:
    - assignment_confidence: For each neuron, % of activity for assigned class
    - neurons_per_class: Count of neurons assigned to each class
    - avg_spikes_per_presentation: Mean spikes per input across all neurons
    - active_neurons_per_presentation: How many neurons fire per input
    - selectivity_index: Standard selectivity metric (0=no selectivity, 1=perfect)
    """
    input_nums = np.asarray(input_numbers)
    n_neurons = result_monitor.shape[1]
    n_examples = result_monitor.shape[0]

    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    n_classes = len(classes_to_check)

    metrics = {}

    # === 1. Assignment Confidence ===
    # For each neuron, what fraction of its total activity is for its assigned class?
    assignment_confidence = np.zeros(n_neurons)
    for i in range(n_neurons):
        assigned_class = int(assignments[i])
        if assigned_class < 0:
            assignment_confidence[i] = 0
            continue

        total_spikes = np.sum(result_monitor[:, i])
        if total_spikes == 0:
            assignment_confidence[i] = 0
            continue

        assigned_class_mask = input_nums == assigned_class
        spikes_for_assigned = np.sum(result_monitor[assigned_class_mask, i])
        assignment_confidence[i] = spikes_for_assigned / total_spikes

    metrics['assignment_confidence'] = assignment_confidence
    metrics['mean_confidence'] = np.mean(assignment_confidence[assignments >= 0])
    metrics['median_confidence'] = np.median(assignment_confidence[assignments >= 0])

    # === 2. Neurons per Class ===
    neurons_per_class = {}
    for c in classes_to_check:
        neurons_per_class[c] = np.sum(assignments == c)
    metrics['neurons_per_class'] = neurons_per_class
    metrics['unassigned_neurons'] = np.sum(assignments < 0)

    # === 3. Spike Statistics per Presentation ===
    spikes_per_presentation = np.sum(result_monitor, axis=1)  # Total spikes per example
    active_per_presentation = np.sum(result_monitor > 0, axis=1)  # Neurons that fired per example

    metrics['avg_spikes_per_presentation'] = np.mean(spikes_per_presentation)
    metrics['std_spikes_per_presentation'] = np.std(spikes_per_presentation)
    metrics['avg_active_neurons'] = np.mean(active_per_presentation)
    metrics['std_active_neurons'] = np.std(active_per_presentation)

    # === 4. Selectivity Index ===
    # For each neuron, compute selectivity as: (max_response - mean_response) / (max_response + mean_response)
    # Values near 1 = highly selective, near 0 = responds equally to all classes
    selectivity_indices = np.zeros(n_neurons)

    # Compute mean response per class for each neuron
    class_responses = np.zeros((n_neurons, n_classes))
    for idx, c in enumerate(classes_to_check):
        class_mask = input_nums == c
        if np.sum(class_mask) > 0:
            class_responses[:, idx] = np.mean(result_monitor[class_mask, :], axis=0)

    for i in range(n_neurons):
        max_resp = np.max(class_responses[i, :])
        mean_resp = np.mean(class_responses[i, :])
        if max_resp + mean_resp > 0:
            selectivity_indices[i] = (max_resp - mean_resp) / (max_resp + mean_resp)
        else:
            selectivity_indices[i] = 0

    metrics['selectivity_indices'] = selectivity_indices
    metrics['mean_selectivity'] = np.mean(selectivity_indices)
    metrics['median_selectivity'] = np.median(selectivity_indices)

    # === 5. Response Overlap ===
    # Count neurons that respond significantly (>5 spikes avg) to multiple classes
    significant_threshold = 5
    neurons_responding_to_multiple = 0
    classes_per_neuron = []

    for i in range(n_neurons):
        significant_classes = np.sum(class_responses[i, :] > significant_threshold)
        classes_per_neuron.append(significant_classes)
        if significant_classes > 1:
            neurons_responding_to_multiple += 1

    metrics['neurons_responding_to_multiple_classes'] = neurons_responding_to_multiple
    metrics['avg_classes_per_neuron'] = np.mean(classes_per_neuron)
    metrics['class_responses'] = class_responses

    return metrics


def print_specificity_report(metrics):
    """Print a detailed specificity analysis report."""

    print(f'\n{"="*70}')
    print('SPECIFICITY ANALYSIS - WTA STRENGTH DIAGNOSIS')
    print(f'{"="*70}')

    # Assignment confidence
    print(f'\n1. ASSIGNMENT CONFIDENCE')
    print(f'   (What % of each neuron\'s activity is for its assigned class?)')
    print(f'   {"─"*50}')
    print(f'   Mean confidence:   {metrics["mean_confidence"]*100:.1f}%')
    print(f'   Median confidence: {metrics["median_confidence"]*100:.1f}%')

    conf = metrics['assignment_confidence']
    conf_valid = conf[conf > 0]
    print(f'   Distribution:')
    print(f'     >90% confident: {np.sum(conf_valid > 0.9)} neurons')
    print(f'     70-90%:         {np.sum((conf_valid > 0.7) & (conf_valid <= 0.9))} neurons')
    print(f'     50-70%:         {np.sum((conf_valid > 0.5) & (conf_valid <= 0.7))} neurons')
    print(f'     <50%:           {np.sum(conf_valid <= 0.5)} neurons')

    if metrics['mean_confidence'] < 0.5:
        print(f'\n   WARNING: LOW CONFIDENCE - Neurons not developing class preferences!')
        print(f'       This suggests weak WTA - too many neurons firing per input.')
    elif metrics['mean_confidence'] > 0.8:
        print(f'\n   GOOD: HIGH CONFIDENCE - Neurons strongly prefer their assigned class.')

    # Neurons per class
    print(f'\n2. NEURONS PER CLASS')
    print(f'   (Should be roughly equal for balanced learning)')
    print(f'   {"─"*50}')

    npc = metrics['neurons_per_class']
    classes = sorted(npc.keys())
    total_assigned = sum(npc.values())
    expected = total_assigned / len(classes) if classes else 0

    for c in classes:
        bar_len = int(npc[c] / max(npc.values()) * 30) if max(npc.values()) > 0 else 0
        bar = '#' * bar_len
        deviation = ((npc[c] - expected) / expected * 100) if expected > 0 else 0
        print(f'   Class {c}: {npc[c]:3d} neurons {bar} ({deviation:+.0f}%)')

    print(f'   Unassigned: {metrics["unassigned_neurons"]} neurons')

    # Check for imbalance
    if max(npc.values()) > 3 * min(npc.values()) and min(npc.values()) > 0:
        print(f'\n   WARNING: IMBALANCED - Some classes have {max(npc.values())/min(npc.values()):.1f}x more neurons!')

    # Spike statistics
    print(f'\n3. ACTIVITY STATISTICS (per presentation)')
    print(f'   {"─"*50}')
    print(f'   Avg total spikes:    {metrics["avg_spikes_per_presentation"]:.1f} +/- {metrics["std_spikes_per_presentation"]:.1f}')
    print(f'   Avg active neurons:  {metrics["avg_active_neurons"]:.1f} +/- {metrics["std_active_neurons"]:.1f}')

    # Interpret
    n_e = len(metrics['assignment_confidence'])
    active_pct = metrics['avg_active_neurons'] / n_e * 100
    print(f'   Active percentage:   {active_pct:.1f}% of {n_e} neurons')

    if active_pct > 30:
        print(f'\n   WARNING: TOO MANY ACTIVE - {active_pct:.0f}% neurons firing suggests weak WTA!')
        print(f'       Expected: 5-15% for strong winner-take-all.')
        print(f'       Consider increasing I->E weight to strengthen inhibition.')
    elif active_pct < 2:
        print(f'\n   WARNING: TOO FEW ACTIVE - Only {active_pct:.1f}% neurons firing!')
        print(f'       Network may be over-inhibited or dead.')
    else:
        print(f'\n   GOOD: HEALTHY SPARSITY - {active_pct:.1f}% neurons active.')

    # Selectivity index
    print(f'\n4. SELECTIVITY INDEX')
    print(f'   (0 = responds equally to all classes, 1 = responds to one class only)')
    print(f'   {"─"*50}')
    print(f'   Mean selectivity:   {metrics["mean_selectivity"]:.3f}')
    print(f'   Median selectivity: {metrics["median_selectivity"]:.3f}')

    sel = metrics['selectivity_indices']
    print(f'   Distribution:')
    print(f'     High (>0.7):      {np.sum(sel > 0.7)} neurons')
    print(f'     Medium (0.3-0.7): {np.sum((sel > 0.3) & (sel <= 0.7))} neurons')
    print(f'     Low (<0.3):       {np.sum(sel <= 0.3)} neurons')

    if metrics['mean_selectivity'] < 0.3:
        print(f'\n   WARNING: LOW SELECTIVITY - Neurons responding to multiple classes!')
    elif metrics['mean_selectivity'] > 0.6:
        print(f'\n   GOOD: HIGH SELECTIVITY - Neurons strongly class-specific.')

    # Response overlap
    print(f'\n5. RESPONSE OVERLAP')
    print(f'   {"─"*50}')
    print(f'   Neurons responding to >1 class: {metrics["neurons_responding_to_multiple_classes"]}')
    print(f'   Avg classes per neuron:         {metrics["avg_classes_per_neuron"]:.2f}')

    overlap_pct = metrics['neurons_responding_to_multiple_classes'] / n_e * 100
    if overlap_pct > 50:
        print(f'\n   WARNING: HIGH OVERLAP - {overlap_pct:.0f}% neurons respond to multiple classes!')

    # Overall diagnosis
    print(f'\n{"="*70}')
    print('OVERALL WTA DIAGNOSIS')
    print(f'{"="*70}')

    issues = []
    if metrics['mean_confidence'] < 0.5:
        issues.append('Low assignment confidence')
    if active_pct > 30:
        issues.append('Too many neurons active per input')
    if metrics['mean_selectivity'] < 0.3:
        issues.append('Low selectivity')
    if overlap_pct > 50:
        issues.append('High response overlap')

    if not issues:
        print('\n   WTA appears HEALTHY')
        print('   Neurons are developing class-specific receptive fields.')
    else:
        print('\n   WTA appears WEAK')
        print('   Issues detected:')
        for issue in issues:
            print(f'     - {issue}')
        print('\n   Recommendation: Increase I->E weight from 1.5 to 2.5-3.0')
        print('   This will strengthen lateral inhibition and sharpen competition.')


def print_confusion_analysis(conf_metrics):
    """Print confusion matrix analysis."""

    print(f'\n{"="*70}')
    print('CONFUSION MATRIX ANALYSIS')
    print(f'{"="*70}')

    # Per-class accuracy
    print(f'\n1. PER-CLASS ACCURACY')
    print(f'   {"─"*50}')

    pca = conf_metrics['per_class_accuracy']
    classes = conf_metrics['class_labels']

    for c in classes:
        acc = pca[c]
        bar_len = int(acc / 100 * 30)
        bar = '#' * bar_len
        status = 'GOOD' if acc > 50 else 'POOR' if acc > 20 else 'FAILING'
        print(f'   Class {c}: {acc:5.1f}% {bar} [{status}]')

    # Accuracy stats
    accs = list(pca.values())
    print(f'\n   Mean accuracy:  {np.mean(accs):.1f}%')
    print(f'   Std deviation:  {np.std(accs):.1f}%')
    print(f'   Best class:     {classes[np.argmax(accs)]} ({max(accs):.1f}%)')
    print(f'   Worst class:    {classes[np.argmin(accs)]} ({min(accs):.1f}%)')

    # Most confused pairs
    print(f'\n2. MOST CONFUSED PAIRS')
    print(f'   {"─"*50}')

    for true_c, pred_c, count in conf_metrics['most_confused_pairs'][:5]:
        print(f'   {true_c} → {pred_c}: {count} misclassifications')

    # Confusion matrix
    print(f'\n3. CONFUSION MATRIX')
    print(f'   {"─"*50}')

    cm = conf_metrics['confusion_matrix']
    print('      Predicted →')
    print('   T  ', end='')
    for c in classes:
        print(f'{c:5d}', end='')
    print()
    print('   ↓  ' + '-' * (5 * len(classes)))

    for i, c in enumerate(classes):
        print(f'   {c}  ', end='')
        for j in range(len(classes)):
            if i == j:
                print(f'[{cm[i,j]:3d}]', end='')
            else:
                print(f' {cm[i,j]:3d} ', end='')
        print()


def print_receptive_field_analysis(rf_metrics):
    """Print receptive field quality analysis."""

    print(f'\n{"="*70}')
    print('RECEPTIVE FIELD QUALITY ANALYSIS')
    print(f'{"="*70}')

    print(f'\n1. NEURON ACTIVITY DISTRIBUTION')
    print(f'   {"─"*50}')

    sd = rf_metrics['spike_distribution']
    print(f'   Total spikes per neuron:')
    print(f'     Mean:   {sd["mean"]:.1f}')
    print(f'     Std:    {sd["std"]:.1f}')
    print(f'     Median: {sd["median"]:.1f}')
    print(f'     Range:  {sd["min"]:.0f} - {sd["max"]:.0f}')

    print(f'\n2. NEURON HEALTH')
    print(f'   {"─"*50}')
    print(f'   Dead neurons (<1% avg activity):    {rf_metrics["dead_neurons"]}')
    print(f'   Low activity (<10% avg):            {rf_metrics["low_activity_neurons"]}')
    print(f'   High activity (>2σ above mean):     {rf_metrics["high_activity_neurons"]}')
    print(f'   Activity Gini coefficient:          {rf_metrics["activity_gini"]:.3f}')

    if rf_metrics['activity_gini'] > 0.6:
        print(f'\n   WARNING: High Gini ({rf_metrics["activity_gini"]:.2f}) indicates unequal activity!')
        print(f'       A few neurons may be dominating.')
    elif rf_metrics['activity_gini'] < 0.3:
        print(f'\n   GOOD: Low Gini indicates well-distributed activity.')

    print(f'\n3. NEURON UTILIZATION')
    print(f'   {"─"*50}')
    print(f'   Neurons for 50% of activity: {rf_metrics["neurons_for_50pct_activity"]}')
    print(f'   Neurons for 80% of activity: {rf_metrics["neurons_for_80pct_activity"]}')
    print(f'   Neurons for 90% of activity: {rf_metrics["neurons_for_90pct_activity"]}')

    # Ideal: 50% of neurons for 50% of activity
    n_total = 400  # cfg.n_e
    utilization = rf_metrics['neurons_for_50pct_activity'] / n_total * 100
    print(f'   50% activity from {utilization:.0f}% of neurons')

    if utilization < 10:
        print(f'\n   WARNING: Only {utilization:.0f}% neurons produce 50% activity!')
        print(f'       Network is dominated by few neurons.')


def print_population_coding_analysis(pop_metrics):
    """Print population coding analysis."""

    print(f'\n{"="*70}')
    print('POPULATION CODING ANALYSIS')
    print(f'{"="*70}')

    print(f'\n1. INTRA-CLASS CONSISTENCY')
    print(f'   (Do same-class inputs produce similar population patterns?)')
    print(f'   {"─"*50}')

    icc = pop_metrics['intra_class_correlation']
    for c, corr in icc.items():
        status = 'HIGH' if corr > 0.5 else 'MODERATE' if corr > 0.2 else 'LOW'
        bar_len = int(max(0, corr) * 20)
        bar = '#' * bar_len
        print(f'   Class {c}: {corr:.3f} {bar} [{status}]')

    print(f'\n   Mean intra-class correlation: {pop_metrics["mean_intra_class_corr"]:.3f}')

    print(f'\n2. INTER-CLASS SEPARATION')
    print(f'   (Are different classes well-separated in neural space?)')
    print(f'   {"─"*50}')
    print(f'   Mean inter-class distance: {pop_metrics["mean_inter_class_distance"]:.2f}')
    print(f'   Mean Fisher ratio:         {pop_metrics["mean_fisher_ratio"]:.3f}')
    print(f'   Median Fisher ratio:       {pop_metrics["median_fisher_ratio"]:.3f}')

    print(f'\n3. CLUSTERING QUALITY (Silhouette Score)')
    print(f'   {"─"*50}')
    print(f'   Mean silhouette: {pop_metrics["mean_silhouette"]:.3f}')
    print(f'   (Range: -1 to 1, higher = better separation)')

    if pop_metrics['mean_silhouette'] > 0.3:
        print(f'\n   GOOD: Positive silhouette indicates reasonable class separation.')
    elif pop_metrics['mean_silhouette'] > 0:
        print(f'\n   MODERATE: Weak but positive class separation.')
    else:
        print(f'\n   WARNING: Negative silhouette indicates overlapping class representations!')


def print_temporal_analysis(temp_metrics):
    """Print temporal dynamics analysis."""

    print(f'\n{"="*70}')
    print('TEMPORAL DYNAMICS ANALYSIS')
    print(f'{"="*70}')

    print(f'\n1. ACTIVITY OVER TRAINING')
    print(f'   (Divided into {temp_metrics["chunks"]} chunks)')
    print(f'   {"─"*50}')

    spikes = temp_metrics['spikes_per_chunk']
    active = temp_metrics['active_neurons_per_chunk']

    print(f'   Chunk | Avg Spikes | Active Neurons')
    print(f'   ' + '-' * 40)
    for i in range(len(spikes)):
        print(f'     {i+1:2d}  |   {spikes[i]:6.1f}   |    {active[i]:5.1f}')

    print(f'\n2. TRENDS')
    print(f'   {"─"*50}')

    spike_trend = temp_metrics['spike_trend']
    active_trend = temp_metrics['active_neuron_trend']

    spike_dir = '↑ increasing' if spike_trend > 0.5 else '↓ decreasing' if spike_trend < -0.5 else '→ stable'
    active_dir = '↑ increasing' if active_trend > 0.1 else '↓ decreasing' if active_trend < -0.1 else '→ stable'

    print(f'   Spike count trend:      {spike_trend:+.2f}/chunk ({spike_dir})')
    print(f'   Active neurons trend:   {active_trend:+.2f}/chunk ({active_dir})')

    if spike_trend < -1:
        print(f'\n   NOTE: Decreasing spikes may indicate theta adaptation is working.')
    if active_trend < -0.5:
        print(f'\n   NOTE: Fewer active neurons over time suggests WTA is strengthening.')


def plot_specificity_analysis(metrics, save_path=None):
    """Generate visualization of specificity metrics."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Specificity Analysis - WTA Strength Diagnosis', fontsize=14, fontweight='bold')

    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)

    # 1. Assignment confidence histogram
    ax = axes[0, 0]
    conf = metrics['assignment_confidence']
    conf_valid = conf[conf > 0]
    ax.hist(conf_valid, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['mean_confidence'], color='red', linestyle='--', label=f'Mean: {metrics["mean_confidence"]:.2f}')
    ax.axvline(0.5, color='orange', linestyle=':', label='50% threshold')
    ax.set_xlabel('Assignment Confidence')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Assignment Confidence Distribution')
    ax.legend()

    # 2. Neurons per class
    ax = axes[0, 1]
    npc = metrics['neurons_per_class']
    classes = sorted(npc.keys())
    counts = [npc[c] for c in classes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    ax.bar([str(c) for c in classes], counts, color=colors, edgecolor='black')
    ax.axhline(np.mean(counts), color='red', linestyle='--', label=f'Expected: {np.mean(counts):.0f}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Neurons per Class')
    ax.legend()

    # 3. Selectivity index histogram
    ax = axes[0, 2]
    sel = metrics['selectivity_indices']
    ax.hist(sel, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['mean_selectivity'], color='red', linestyle='--', label=f'Mean: {metrics["mean_selectivity"]:.2f}')
    ax.axvline(0.5, color='orange', linestyle=':', label='0.5 threshold')
    ax.set_xlabel('Selectivity Index')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Selectivity Index Distribution')
    ax.legend()

    # 4. Class response heatmap (sample neurons)
    ax = axes[1, 0]
    class_resp = metrics['class_responses']
    # Sort neurons by their assigned class for visualization
    n_sample = min(100, class_resp.shape[0])
    sample_idx = np.random.choice(class_resp.shape[0], n_sample, replace=False)
    sample_resp = class_resp[sample_idx, :]

    im = ax.imshow(sample_resp, aspect='auto', cmap='viridis')
    ax.set_xlabel('Class')
    ax.set_ylabel('Neuron (sample)')
    ax.set_title(f'Class Responses ({n_sample} sample neurons)')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes])
    plt.colorbar(im, ax=ax, label='Avg Spikes')

    # 5. Confidence vs Selectivity scatter
    ax = axes[1, 1]
    valid_mask = conf > 0
    ax.scatter(conf[valid_mask], sel[valid_mask], alpha=0.5, s=20)
    ax.set_xlabel('Assignment Confidence')
    ax.set_ylabel('Selectivity Index')
    ax.set_title('Confidence vs Selectivity')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    # 6. Summary metrics
    ax = axes[1, 2]
    ax.axis('off')

    n_e = len(conf)
    active_pct = metrics['avg_active_neurons'] / n_e * 100

    summary_text = f"""
    SUMMARY METRICS
    ----------------------------------------

    Assignment Confidence
      Mean:   {metrics['mean_confidence']*100:.1f}%
      Median: {metrics['median_confidence']*100:.1f}%

    Activity (per presentation)
      Avg spikes:  {metrics['avg_spikes_per_presentation']:.1f}
      Active neurons: {metrics['avg_active_neurons']:.1f} ({active_pct:.1f}%)

    Selectivity
      Mean:   {metrics['mean_selectivity']:.3f}
      Median: {metrics['median_selectivity']:.3f}

    Response Overlap
      Multi-class neurons: {metrics['neurons_responding_to_multiple_classes']}
      Avg classes/neuron:  {metrics['avg_classes_per_neuron']:.2f}

    Unassigned neurons: {metrics['unassigned_neurons']}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nSpecificity analysis plot saved to: {save_path}')

    return fig


def plot_comprehensive_analysis(specificity_metrics, conf_metrics, rf_metrics, pop_metrics, temp_metrics, save_path=None):
    """Generate a comprehensive multi-page visualization of all metrics."""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 4, hspace=0.35, wspace=0.3)

    fig.suptitle('Comprehensive Evaluation Report - 4:1 E:I Architecture', fontsize=16, fontweight='bold', y=0.995)

    classes_to_check = cfg.mnist_classes if cfg.mnist_classes is not None else range(10)
    class_list = list(classes_to_check)

    # === Row 1: Confusion Matrix and Per-Class Accuracy ===

    # 1.1 Confusion Matrix Heatmap
    ax = fig.add_subplot(gs[0, 0:2])
    cm = conf_metrics['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_list)))
    ax.set_yticks(range(len(class_list)))
    ax.set_xticklabels(class_list)
    ax.set_yticklabels(class_list)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Training Data)')

    # Add text annotations
    for i in range(len(class_list)):
        for j in range(len(class_list)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8)

    # 1.2 Per-Class Accuracy Bar Chart
    ax = fig.add_subplot(gs[0, 2:4])
    pca = conf_metrics['per_class_accuracy']
    colors = ['#2ecc71' if pca[c] > 50 else '#f39c12' if pca[c] > 20 else '#e74c3c' for c in class_list]
    bars = ax.bar([str(c) for c in class_list], [pca[c] for c in class_list], color=colors, edgecolor='black')
    ax.axhline(np.mean(list(pca.values())), color='blue', linestyle='--', label=f'Mean: {np.mean(list(pca.values())):.1f}%')
    ax.axhline(20, color='red', linestyle=':', alpha=0.5, label='Random (20%)')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy (Training Data)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # === Row 2: WTA Specificity Metrics ===

    # 2.1 Assignment Confidence
    ax = fig.add_subplot(gs[1, 0])
    conf = specificity_metrics['assignment_confidence']
    conf_valid = conf[conf > 0]
    ax.hist(conf_valid, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(specificity_metrics['mean_confidence'], color='red', linestyle='--', label=f'Mean: {specificity_metrics["mean_confidence"]:.2f}')
    ax.axvline(0.5, color='orange', linestyle=':', label='50% threshold')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Neurons')
    ax.set_title('Assignment Confidence')
    ax.legend(fontsize=7)

    # 2.2 Neurons per Class
    ax = fig.add_subplot(gs[1, 1])
    npc = specificity_metrics['neurons_per_class']
    counts = [npc[c] for c in class_list]
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_list)))
    ax.bar([str(c) for c in class_list], counts, color=colors, edgecolor='black')
    ax.axhline(np.mean(counts), color='red', linestyle='--', label=f'Expected: {np.mean(counts):.0f}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Neurons')
    ax.set_title('Neurons per Class')
    ax.legend(fontsize=7)

    # 2.3 Selectivity Index
    ax = fig.add_subplot(gs[1, 2])
    sel = specificity_metrics['selectivity_indices']
    ax.hist(sel, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(specificity_metrics['mean_selectivity'], color='red', linestyle='--', label=f'Mean: {specificity_metrics["mean_selectivity"]:.2f}')
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Neurons')
    ax.set_title('Selectivity Index')
    ax.legend(fontsize=7)

    # 2.4 Confidence vs Selectivity
    ax = fig.add_subplot(gs[1, 3])
    valid_mask = conf > 0
    ax.scatter(conf[valid_mask], sel[valid_mask], alpha=0.4, s=15, c='purple')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Selectivity')
    ax.set_title('Confidence vs Selectivity')

    # === Row 3: Population Coding ===

    # 3.1 Intra-class Correlation
    ax = fig.add_subplot(gs[2, 0])
    icc = pop_metrics['intra_class_correlation']
    colors = ['#2ecc71' if icc[c] > 0.5 else '#f39c12' if icc[c] > 0.2 else '#e74c3c' for c in class_list]
    ax.bar([str(c) for c in class_list], [icc[c] for c in class_list], color=colors, edgecolor='black')
    ax.axhline(pop_metrics['mean_intra_class_corr'], color='blue', linestyle='--', label=f'Mean: {pop_metrics["mean_intra_class_corr"]:.2f}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Correlation')
    ax.set_title('Intra-Class Consistency')
    ax.legend(fontsize=7)
    ax.set_ylim(-0.1, 1.0)

    # 3.2 Population Coding Summary
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    summary = f"""
    POPULATION CODING
    ────────────────────────
    Intra-class corr:   {pop_metrics['mean_intra_class_corr']:.3f}
    Inter-class dist:   {pop_metrics['mean_inter_class_distance']:.1f}
    Fisher ratio:       {pop_metrics['mean_fisher_ratio']:.3f}
    Silhouette score:   {pop_metrics['mean_silhouette']:.3f}

    INTERPRETATION
    ────────────────────────
    Silhouette > 0.3: Good
    Silhouette > 0: Moderate
    Silhouette < 0: Poor
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 3.3 Class Response Heatmap
    ax = fig.add_subplot(gs[2, 2:4])
    class_resp = specificity_metrics['class_responses']
    # Sort by assigned class for better visualization
    im = ax.imshow(class_resp.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Class')
    ax.set_yticks(range(len(class_list)))
    ax.set_yticklabels(class_list)
    ax.set_title('Class Response Profile (all neurons)')
    plt.colorbar(im, ax=ax, label='Avg Spikes')

    # === Row 4: Neuron Health ===

    # 4.1 Activity Distribution
    ax = fig.add_subplot(gs[3, 0:2])
    sd = rf_metrics['spike_distribution']

    # Create synthetic data for visualization based on stats
    x = np.linspace(0, sd['max'], 100)
    # Show key percentiles
    ax.axvline(sd['mean'], color='red', linestyle='-', linewidth=2, label=f'Mean: {sd["mean"]:.0f}')
    ax.axvline(sd['median'], color='blue', linestyle='--', linewidth=2, label=f'Median: {sd["median"]:.0f}')
    ax.axvline(sd['mean'] - sd['std'], color='gray', linestyle=':', alpha=0.5)
    ax.axvline(sd['mean'] + sd['std'], color='gray', linestyle=':', alpha=0.5, label=f'±1σ ({sd["std"]:.0f})')

    # Text annotation
    ax.text(0.95, 0.95, f'Range: {sd["min"]:.0f} - {sd["max"]:.0f}\nGini: {rf_metrics["activity_gini"]:.3f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Total Spikes per Neuron')
    ax.set_title('Neuron Activity Distribution')
    ax.legend(fontsize=8)

    # 4.2 Neuron Utilization
    ax = fig.add_subplot(gs[3, 2])
    categories = ['50%', '80%', '90%']
    neurons = [rf_metrics['neurons_for_50pct_activity'],
               rf_metrics['neurons_for_80pct_activity'],
               rf_metrics['neurons_for_90pct_activity']]
    ax.bar(categories, neurons, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    ax.axhline(200, color='gray', linestyle='--', alpha=0.5, label='Ideal (50%)')
    ax.set_xlabel('% of Total Activity')
    ax.set_ylabel('Neurons Required')
    ax.set_title('Neuron Utilization')
    ax.legend(fontsize=7)

    # 4.3 Neuron Health Summary
    ax = fig.add_subplot(gs[3, 3])
    health_data = {
        'Dead': rf_metrics['dead_neurons'],
        'Low': rf_metrics['low_activity_neurons'],
        'Normal': 400 - rf_metrics['low_activity_neurons'] - rf_metrics['high_activity_neurons'],
        'High': rf_metrics['high_activity_neurons']
    }
    colors = ['#95a5a6', '#f39c12', '#2ecc71', '#e74c3c']
    ax.pie(list(health_data.values()), labels=list(health_data.keys()), colors=colors,
           autopct='%1.0f%%', startangle=90)
    ax.set_title('Neuron Health Distribution')

    # === Row 5: Temporal Dynamics ===

    # 5.1 Spikes over Training
    ax = fig.add_subplot(gs[4, 0:2])
    chunks = range(1, temp_metrics['chunks'] + 1)
    ax.plot(chunks, temp_metrics['spikes_per_chunk'], 'b-o', linewidth=2, markersize=6, label='Avg Spikes')
    ax.set_xlabel('Training Chunk')
    ax.set_ylabel('Average Spikes per Presentation', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    ax2 = ax.twinx()
    ax2.plot(chunks, temp_metrics['active_neurons_per_chunk'], 'r-s', linewidth=2, markersize=6, label='Active Neurons')
    ax2.set_ylabel('Active Neurons', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_title(f'Activity Over Training (trend: spikes {temp_metrics["spike_trend"]:+.2f}, active {temp_metrics["active_neuron_trend"]:+.2f})')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # 5.2 Trend Analysis
    ax = fig.add_subplot(gs[4, 2:4])
    ax.axis('off')

    # Summary panel
    overall_acc = np.mean(list(conf_metrics['per_class_accuracy'].values()))
    active_pct = specificity_metrics['avg_active_neurons'] / 400 * 100

    status_color = '#2ecc71' if overall_acc > 50 else '#f39c12' if overall_acc > 25 else '#e74c3c'

    summary = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║              OVERALL EVALUATION SUMMARY                   ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  TRAINING ACCURACY:  {overall_acc:5.1f}%  (on data used for STDP)   ║
    ║  Random baseline:    20.0%                               ║
    ║  Improvement:        {overall_acc - 20:+5.1f}%                          ║
    ║                                                          ║
    ║  WTA HEALTH:                                             ║
    ║    Active neurons:   {active_pct:4.1f}% (target: 2-15%)           ║
    ║    Selectivity:      {specificity_metrics['mean_selectivity']:.3f} (target: >0.5)            ║
    ║    Confidence:       {specificity_metrics['mean_confidence']*100:4.1f}% (target: >60%)           ║
    ║                                                          ║
    ║  POPULATION CODING:                                      ║
    ║    Silhouette:       {pop_metrics['mean_silhouette']:+.3f} (target: >0.3)            ║
    ║    Fisher ratio:     {pop_metrics['mean_fisher_ratio']:.3f} (higher = better)        ║
    ║                                                          ║
    ║  NEURON HEALTH:                                          ║
    ║    Dead neurons:     {rf_metrics['dead_neurons']:3d}/400                            ║
    ║    Gini coefficient: {rf_metrics['activity_gini']:.3f} (target: <0.5)            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # === Row 6: Recommendations ===
    ax = fig.add_subplot(gs[5, :])
    ax.axis('off')

    # Generate recommendations based on metrics
    recommendations = []

    if overall_acc < 30:
        recommendations.append("• TRAINING NEEDED: Accuracy near random. Run more epochs (3+ recommended).")

    if active_pct > 15:
        recommendations.append(f"• WEAK WTA: {active_pct:.0f}% neurons active. Consider increasing I→E weight from 1.5 to 2.0-2.5.")
    elif active_pct < 2:
        recommendations.append(f"• OVER-INHIBITED: Only {active_pct:.1f}% neurons active. Consider decreasing I→E weight.")

    if specificity_metrics['mean_confidence'] < 0.5:
        recommendations.append("• LOW CONFIDENCE: Neurons not developing preferences. May need more training or stronger WTA.")

    if rf_metrics['activity_gini'] > 0.6:
        recommendations.append(f"• UNEQUAL UTILIZATION: Gini={rf_metrics['activity_gini']:.2f}. A few neurons dominate.")

    npc = specificity_metrics['neurons_per_class']
    max_npc, min_npc = max(npc.values()), min(npc.values())
    if min_npc > 0 and max_npc / min_npc > 3:
        recommendations.append(f"• CLASS IMBALANCE: {max_npc/min_npc:.1f}x difference. May resolve with more training.")

    if pop_metrics['mean_silhouette'] < 0:
        recommendations.append("• POOR SEPARATION: Negative silhouette indicates overlapping class representations.")

    if not recommendations:
        recommendations.append("• Network appears healthy! Consider running full evaluation with test set.")

    rec_text = "RECOMMENDATIONS\n" + "─" * 70 + "\n\n" + "\n\n".join(recommendations)

    ax.text(0.02, 0.95, rec_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nComprehensive analysis plot saved to: {save_path}')

    return fig


# Load parameters from config
MNIST_data_path = cfg.mnist_data_path
data_path = cfg.data_path + 'activity/'
n_e = cfg.n_e
n_input = cfg.n_input
ending = cfg.ending

# Evaluation settings - auto-detect from available files
# Look for training activity files (prioritize clean post-training activity)
import glob
training_files = glob.glob(data_path + 'resultPopVecs*.npy')
if not training_files:
    print(f"\nERROR: No training activity files found in {data_path}")
    print("Please run training mode first.")
    exit(1)

# Extract numbers from filenames and find the most recent
# Track both clean and regular files separately
clean_sizes = []
regular_sizes = []
for f in training_files:
    fname = f.split('/')[-1]
    if fname.startswith('resultPopVecs'):
        size_str = fname.replace('resultPopVecs', '').replace('.npy', '').replace('_clean', '')
        try:
            size = int(size_str)
            if '_clean' in fname:
                clean_sizes.append(size)
            else:
                regular_sizes.append(size)
        except ValueError:
            continue

# Prioritize clean files for training (assignments)
if clean_sizes:
    training_ending = str(max(clean_sizes)) + '_clean'
    is_clean = True
    print(f"\nUsing CLEAN post-training activity for assignments (STDP off, final weights)")
elif regular_sizes:
    training_ending = str(max(regular_sizes))
    is_clean = False
    print(f"\nUsing training activity for assignments (STDP was ON - may reduce accuracy)")
    print(f"   For best results, run a full training session to generate clean activity.")
else:
    print(f"\nERROR: Could not parse training file sizes")
    exit(1)

# For testing, check if we have a MATCHING test file (same or newer than training)
# Old test files from previous runs should not be used with new training data
if regular_sizes:
    max_regular = max(regular_sizes)
    training_size = max(clean_sizes) if clean_sizes else 0

    # Only use regular file if it's from the same run (similar size) or we don't have clean
    if not clean_sizes:
        # No clean files, use regular for both
        testing_ending = str(max_regular)
        print(f"Using training activity for evaluation (no clean files)")
    elif max_regular >= training_size * 0.9:  # Within 10% of training size = same run
        testing_ending = str(max_regular)
        print(f"Using test activity for evaluation")
    else:
        # Regular file is old/stale, use clean file for both
        print(f"\nWARNING: Test activity file ({max_regular} examples) is from a different run")
        print(f"   than training ({training_size} examples). Using training activity for both.")
        testing_ending = training_ending
else:
    print(f"\nWARNING: No test activity files found!")
    print(f"   Please run test mode (set test_mode=True) to generate test activity.")
    print(f"   Using training activity for both...")
    testing_ending = training_ending

print(f"   Training file (for assignments): resultPopVecs{training_ending}.npy")
print(f"   Testing file (for evaluation): resultPopVecs{testing_ending}.npy")

start_time_training = 0
# Extract numeric part for end time (remove '_clean' suffix if present)
training_numeric_ending = int(training_ending.replace('_clean', ''))
end_time_training = training_numeric_ending
start_time_testing = 0
testing_numeric_ending = int(testing_ending.replace('_clean', ''))
end_time_testing = testing_numeric_ending

print('load MNIST')
print('Loading training data...')
training_images, training_labels = data_loader.load_training_data()
print('Loading test data...')
testing_images, testing_labels = data_loader.load_test_data()
print(f'Training set: {len(training_labels)} examples')
print(f'Test set: {len(testing_labels)} examples')

print('load results')
try:
    training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
    training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
    testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
    testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
    print(training_result_monitor.shape)
except FileNotFoundError as e:
    print(f"\nERROR: Could not find simulation results!")
    print(f"Missing file: {e.filename}")
    print(f"\nPlease run 'Diehl&Cook_spiking_MNIST.py' first to generate the activity files.")
    print(f"Expected files in '{data_path}':")
    print(f"  - resultPopVecs{training_ending}.npy")
    print(f"  - inputNumbers{training_ending}.npy")
    print(f"  - resultPopVecs{testing_ending}.npy")
    print(f"  - inputNumbers{testing_ending}.npy")
    exit(1)

print('get assignments')
test_results = np.zeros((10, end_time_testing-start_time_testing))
test_results_max = np.zeros((10, end_time_testing-start_time_testing))
test_results_top = np.zeros((10, end_time_testing-start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing-start_time_testing))

# Match sizes: activity buffer is circular, use last N labels
num_activity_examples = training_result_monitor.shape[0]
training_labels_subset = training_input_numbers[-num_activity_examples:]

assignments = get_new_assignments(training_result_monitor[start_time_training:end_time_training],
                                  training_labels_subset[start_time_training:end_time_training])
print(assignments)

# === COMPREHENSIVE ANALYSIS ===
print('\n' + '='*70)
print('COMPUTING COMPREHENSIVE EVALUATION METRICS')
print('='*70)

# Slice data for analysis
train_data = training_result_monitor[start_time_training:end_time_training]
train_labels = training_labels_subset[start_time_training:end_time_training]

# 1. Specificity Analysis (WTA strength)
print('\n[1/5] Computing specificity analysis...')
specificity_metrics = compute_specificity_analysis(train_data, train_labels, assignments)
print_specificity_report(specificity_metrics)

# 2. Confusion Matrix Analysis
print('\n[2/5] Computing confusion matrix...')
conf_metrics = compute_confusion_matrix(assignments, train_data, train_labels, n_classes=5)
print_confusion_analysis(conf_metrics)

# 3. Receptive Field Quality
print('\n[3/5] Computing receptive field quality metrics...')
rf_metrics = compute_receptive_field_quality(train_data, train_labels, assignments, training_images)
print_receptive_field_analysis(rf_metrics)

# 4. Population Coding Analysis
print('\n[4/5] Computing population coding metrics...')
pop_metrics = compute_population_coding_metrics(train_data, train_labels, assignments)
print_population_coding_analysis(pop_metrics)

# 5. Temporal Dynamics
print('\n[5/5] Computing temporal dynamics...')
temp_metrics = compute_temporal_dynamics(train_data, train_labels)
print_temporal_analysis(temp_metrics)

# Generate visualizations
print('\n' + '='*70)
print('GENERATING VISUALIZATIONS')
print('='*70)

# Save specificity plot (legacy)
specificity_plot_path = data_path + '../specificity_analysis.png'
plot_specificity_analysis(specificity_metrics, save_path=specificity_plot_path)

# Save comprehensive analysis plot
comprehensive_plot_path = data_path + '../comprehensive_evaluation.png'
plot_comprehensive_analysis(specificity_metrics, conf_metrics, rf_metrics, pop_metrics, temp_metrics,
                           save_path=comprehensive_plot_path)

# === ACCURACY CALCULATION ===
# Calculate accuracy in chunks (max 10000 at a time to avoid memory issues)
chunk_size = 10000
num_tests = max(1, (end_time_testing + chunk_size - 1) // chunk_size)  # Ceiling division, at least 1
sum_accurracy = [0] * num_tests

print(f'\nCalculating accuracy on {end_time_testing} test examples in {num_tests} chunk(s)...')

counter = 0
while (counter < num_tests):
    start_time = min(chunk_size * counter, end_time_testing)
    end_time = min(chunk_size * (counter + 1), end_time_testing)

    if start_time >= end_time:
        break

    chunk_examples = end_time - start_time
    test_results = np.zeros((10, chunk_examples))

    print(f'\nChunk {counter+1}/{num_tests}: Processing examples {start_time} to {end_time}')

    for i in range(chunk_examples):
        test_results[:,i] = get_recognized_number_ranking(assignments,
                                                          testing_result_monitor[i+start_time,:])

    # testing_input_numbers contains the actual labels used during evaluation
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(chunk_examples) * 100
    print(f'  Accuracy: {sum_accurracy[counter]:.2f}% ({correct}/{chunk_examples} correct, {len(incorrect)} incorrect)')
    counter += 1

print(f'\n{"="*60}')
print(f'FINAL RESULTS')
print(f'{"="*60}')
print(f'Overall accuracy: {np.mean(sum_accurracy):.2f}% ± {np.std(sum_accurracy):.2f}%')
print(f'Total test examples: {end_time_testing}')
print(f'{"="*60}')


plt.show()
