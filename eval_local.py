#!/usr/bin/env python3
"""
Local evaluation script.

Loads the public eval JSON, sends it to the local endpoint,
and compares predictions to ground truth.

Usage:
    python3 eval_local.py [--url http://localhost:8000/predict] [--batch-size 100]
"""

import argparse
import json
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Number of cases per request batch")
    parser.add_argument("--data", default="relevant_priors_public.json")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    truth_by_key = {}
    for t in data["truth"]:
        truth_by_key[(t["case_id"], t["study_id"])] = t["is_relevant_to_current"]

    cases = data["cases"]
    all_predictions = {}

    total_start = time.monotonic()

    # Send in batches
    for batch_start in range(0, len(cases), args.batch_size):
        batch = cases[batch_start:batch_start + args.batch_size]
        batch_priors = sum(len(c["prior_studies"]) for c in batch)

        payload = {
            "challenge_id": "relevant-priors-v1",
            "schema_version": 1,
            "cases": batch,
        }

        print(f"Sending batch {batch_start//args.batch_size + 1}: "
              f"{len(batch)} cases, {batch_priors} priors...", end=" ", flush=True)

        batch_start_time = time.monotonic()
        resp = requests.post(args.url, json=payload, timeout=360)
        batch_elapsed = time.monotonic() - batch_start_time

        if resp.status_code != 200:
            print(f"ERROR {resp.status_code}: {resp.text[:200]}")
            sys.exit(1)

        result = resp.json()
        for pred in result["predictions"]:
            all_predictions[(pred["case_id"], pred["study_id"])] = pred["predicted_is_relevant"]

        print(f"got {len(result['predictions'])} predictions in {batch_elapsed:.1f}s")

    total_elapsed = time.monotonic() - total_start

    # Score
    correct = 0
    incorrect = 0
    missing = 0
    tp = fp = tn = fn = 0

    for key, actual in truth_by_key.items():
        if key not in all_predictions:
            missing += 1
            incorrect += 1
            continue

        pred = all_predictions[key]
        if pred == actual:
            correct += 1
            if actual:
                tp += 1
            else:
                tn += 1
        else:
            incorrect += 1
            if pred:
                fp += 1
            else:
                fn += 1

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Total truth labels: {len(truth_by_key)}")
    print(f"Missing predictions: {missing}")
    print(f"Correct: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if tp + fp > 0:
        print(f"  Precision: {tp/(tp+fp):.4f}")
    if tp + fn > 0:
        print(f"  Recall: {tp/(tp+fn):.4f}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
