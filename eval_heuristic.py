#!/usr/bin/env python3
"""Evaluate the heuristic classifier on the public eval data."""

import json
from collections import Counter
from classifier import heuristic_predict, extract_regions, extract_modality

with open("relevant_priors_public.json") as f:
    data = json.load(f)

truth_by_key = {}
for t in data["truth"]:
    truth_by_key[(t["case_id"], t["study_id"])] = t["is_relevant_to_current"]

correct = 0
incorrect = 0
conf_buckets = {"high_correct": 0, "high_wrong": 0, "low_correct": 0, "low_wrong": 0}
false_negatives = []
false_positives = []

for case in data["cases"]:
    current = case["current_study"]
    for prior in case["prior_studies"]:
        key = (case["case_id"], prior["study_id"])
        if key not in truth_by_key:
            continue
        actual = truth_by_key[key]

        pred, conf = heuristic_predict(
            current["study_description"],
            prior["study_description"],
            current.get("study_date", ""),
            prior.get("study_date", ""),
        )

        is_correct = pred == actual
        if is_correct:
            correct += 1
        else:
            incorrect += 1

        bucket = ("high" if conf >= 0.7 else "low") + "_" + ("correct" if is_correct else "wrong")
        conf_buckets[bucket] += 1

        if not is_correct and actual and not pred:
            false_negatives.append({
                "current": current["study_description"],
                "prior": prior["study_description"],
                "confidence": conf,
                "current_regions": sorted(extract_regions(current["study_description"])),
                "prior_regions": sorted(extract_regions(prior["study_description"])),
            })
        elif not is_correct and not actual and pred:
            false_positives.append({
                "current": current["study_description"],
                "prior": prior["study_description"],
                "confidence": conf,
                "current_regions": sorted(extract_regions(current["study_description"])),
                "prior_regions": sorted(extract_regions(prior["study_description"])),
            })

total = correct + incorrect
print(f"=== HEURISTIC RESULTS ===")
print(f"Correct: {correct}/{total} = {correct/total:.4f}")
print(f"Incorrect: {incorrect}")
print()
print(f"=== CONFIDENCE BUCKETS ===")
for k, v in conf_buckets.items():
    print(f"  {k}: {v}")

low_total = conf_buckets["low_correct"] + conf_buckets["low_wrong"]
if low_total > 0:
    print(f"  Low-confidence accuracy: {conf_buckets['low_correct']/low_total:.3f} ({low_total} items)")
high_total = conf_buckets["high_correct"] + conf_buckets["high_wrong"]
if high_total > 0:
    print(f"  High-confidence accuracy: {conf_buckets['high_correct']/high_total:.3f} ({high_total} items)")

print(f"\n=== TOP FALSE NEGATIVES (relevant but predicted not) ===")
fn_descs = Counter()
for fn in false_negatives:
    fn_descs[(fn["current"], fn["prior"])] += 1
for (cur, pri), count in fn_descs.most_common(25):
    cr = sorted(extract_regions(cur))
    pr = sorted(extract_regions(pri))
    print(f"  [{count}] '{cur}' {cr} vs '{pri}' {pr}")

print(f"\n=== TOP FALSE POSITIVES (not relevant but predicted relevant) ===")
fp_descs = Counter()
for fp in false_positives:
    fp_descs[(fp["current"], fp["prior"])] += 1
for (cur, pri), count in fp_descs.most_common(25):
    cr = sorted(extract_regions(cur))
    pr = sorted(extract_regions(pri))
    print(f"  [{count}] '{cur}' {cr} vs '{pri}' {pr}")
