#!/usr/bin/env python3
"""
Compute pass@k and mean@k metrics with bootstrap confidence intervals.

For each JSONL file:
- Groups examples by dataset
- For each dataset, computes:
  - pass@16: from 16 responses, bootstrap sample (with replacement) k responses,
    check if at least one is correct. Repeat N times, report mean and std.
  - mean@16: from 16 responses, bootstrap sample (with replacement) k responses,
    compute the mean score. Repeat N times, report mean and std.

Usage:
    python eval_metrics.py [--k 16] [--n_bootstrap 100] [--seed 42] [file1.jsonl file2.jsonl ...]

If no files specified, will use all .jsonl files in the current directory.
"""

import json
import argparse
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_jsonl(filepath):
    """Load a JSONL file and return list of records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_pass_at_k_bootstrap(scores_per_example, k, n_bootstrap, rng):
    """
    Compute pass@k with bootstrap.

    For each bootstrap iteration:
      - For each example, sample k responses with replacement from its 16 responses
      - If at least one sampled response is correct, count as pass
      - pass@k = number of passed examples / total examples

    Returns:
        mean, std of pass@k across bootstrap iterations
    """
    n_examples = len(scores_per_example)
    pass_rates = []

    for _ in range(n_bootstrap):
        n_pass = 0
        for scores in scores_per_example:
            # Sample k responses with replacement from the available responses
            sampled_indices = rng.integers(0, len(scores), size=k)
            sampled_scores = [scores[i] for i in sampled_indices]
            # pass@k: at least one correct
            if any(s > 0 for s in sampled_scores):
                n_pass += 1
        pass_rates.append(n_pass / n_examples)

    return np.mean(pass_rates), np.std(pass_rates)


def compute_mean_at_k_bootstrap(scores_per_example, k, n_bootstrap, rng):
    """
    Compute mean@k with bootstrap.

    For each bootstrap iteration:
      - For each example, sample k responses with replacement from its 16 responses
      - Compute mean score of the k sampled responses
      - mean@k = average of per-example mean scores

    Returns:
        mean, std of mean@k across bootstrap iterations
    """
    n_examples = len(scores_per_example)
    mean_scores = []

    for _ in range(n_bootstrap):
        total_score = 0.0
        for scores in scores_per_example:
            # Sample k responses with replacement
            sampled_indices = rng.integers(0, len(scores), size=k)
            sampled_scores = [scores[i] for i in sampled_indices]
            # mean@k: average score of sampled responses
            total_score += np.mean(sampled_scores)
        mean_scores.append(total_score / n_examples)

    return np.mean(mean_scores), np.std(mean_scores)


def process_file(filepath, k, n_bootstrap, seed):
    """Process a single JSONL file and compute metrics."""
    records = load_jsonl(filepath)

    # Group by dataset
    dataset_scores = defaultdict(list)
    for record in records:
        dataset = record.get('dataset', 'unknown')
        scores = [r['score'] for r in record['rollouts']]
        dataset_scores[dataset].append(scores)

    # Also compute overall (all datasets combined)
    all_scores = []
    for record in records:
        scores = [r['score'] for r in record['rollouts']]
        all_scores.append(scores)

    rng = np.random.default_rng(seed)

    results = {}

    # Per-dataset metrics
    for dataset in sorted(dataset_scores.keys()):
        scores_per_example = dataset_scores[dataset]
        pass_mean, pass_std = compute_pass_at_k_bootstrap(scores_per_example, k, n_bootstrap, rng)
        mean_mean, mean_std = compute_mean_at_k_bootstrap(scores_per_example, k, n_bootstrap, rng)
        results[dataset] = {
            'n_examples': len(scores_per_example),
            'pass@k_mean': pass_mean,
            'pass@k_std': pass_std,
            'mean@k_mean': mean_mean,
            'mean@k_std': mean_std,
        }

    # Overall
    pass_mean, pass_std = compute_pass_at_k_bootstrap(all_scores, k, n_bootstrap, rng)
    mean_mean, mean_std = compute_mean_at_k_bootstrap(all_scores, k, n_bootstrap, rng)
    results['OVERALL'] = {
        'n_examples': len(all_scores),
        'pass@k_mean': pass_mean,
        'pass@k_std': pass_std,
        'mean@k_mean': mean_mean,
        'mean@k_std': mean_std,
    }

    return results


def print_results(filepath, results, k):
    """Pretty print results for a file."""
    filename = Path(filepath).stem
    print(f"\n{'='*100}")
    print(f"File: {filename}")
    print(f"{'='*100}")
    print(f"{'Dataset':<20} {'N':>6} {'pass@'+str(k)+' mean':>14} {'pass@'+str(k)+' std':>14} {'mean@'+str(k)+' mean':>14} {'mean@'+str(k)+' std':>14}")
    print(f"{'-'*20} {'-'*6} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    # Print per-dataset first, then overall
    datasets = [d for d in sorted(results.keys()) if d != 'OVERALL']
    for dataset in datasets:
        r = results[dataset]
        print(f"{dataset:<20} {r['n_examples']:>6} {r['pass@k_mean']:>14.4f} {r['pass@k_std']:>14.4f} {r['mean@k_mean']:>14.4f} {r['mean@k_std']:>14.4f}")

    print(f"{'-'*20} {'-'*6} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    r = results['OVERALL']
    print(f"{'OVERALL':<20} {r['n_examples']:>6} {r['pass@k_mean']:>14.4f} {r['pass@k_std']:>14.4f} {r['mean@k_mean']:>14.4f} {r['mean@k_std']:>14.4f}")


def main():
    parser = argparse.ArgumentParser(description='Compute pass@k and mean@k with bootstrap confidence intervals')
    parser.add_argument('files', nargs='*', help='JSONL files to process (default: all .jsonl in current dir)')
    parser.add_argument('--file', '-f', type=str, default=None, help='Specify a single JSONL file to process')
    parser.add_argument('--k', type=int, default=16, help='Number of responses to sample (default: 16)')
    parser.add_argument('--n_bootstrap', type=int, default=100, help='Number of bootstrap iterations (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    # Get files: --file takes priority, then positional args, then all .jsonl in current dir
    if args.file:
        files = [args.file]
    elif args.files:
        files = args.files
    else:
        files = sorted(glob.glob('*.jsonl'))

    if not files:
        print("No JSONL files found!")
        return

    print(f"Configuration: k={args.k}, n_bootstrap={args.n_bootstrap}, seed={args.seed}")
    print(f"Found {len(files)} file(s) to process")

    for filepath in files:
        try:
            results = process_file(filepath, args.k, args.n_bootstrap, args.seed)
            print_results(filepath, results, args.k)
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
