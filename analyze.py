#!/usr/bin/env python3
"""Analyze the public eval data to understand what predicts relevance."""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime

with open("relevant_priors_public.json") as f:
    data = json.load(f)

truth_by_key = {}
for t in data["truth"]:
    truth_by_key[(t["case_id"], t["study_id"])] = t["is_relevant_to_current"]

total = len(truth_by_key)
relevant = sum(1 for v in truth_by_key.values() if v)
print(f"=== BASIC STATS ===")
print(f"Total priors: {total}")
print(f"Relevant: {relevant} ({relevant/total*100:.1f}%)")
print(f"Not relevant: {total - relevant} ({(total-relevant)/total*100:.1f}%)")
print(f"Cases: {len(data['cases'])}")
print(f"Avg priors/case: {total / len(data['cases']):.1f}")
print()

# --- Extract body region and modality from study descriptions ---

BODY_REGIONS = {
    "chest": r"\b(chest|lung|thorax|thoracic|pulmon)\b",
    "brain": r"\b(brain|head|cranial|neuro|skull)\b",
    "abdomen": r"\b(abd|abdom|liver|hepat|pancrea|renal|kidney|gallbladder|spleen)\b",
    "pelvis": r"\b(pelvi|bladder|uterus|ovari|prostat|rectal|rectum)\b",
    "spine_cervical": r"\b(cervical\s*spine|c[\-\s]?spine)\b",
    "spine_thoracic": r"\b(thoracic\s*spine|t[\-\s]?spine)\b",
    "spine_lumbar": r"\b(lumbar|l[\-\s]?spine)\b",
    "spine": r"\b(spine|spinal)\b",
    "knee": r"\b(knee)\b",
    "shoulder": r"\b(shoulder)\b",
    "hip": r"\b(hip|femur|femoral)\b",
    "ankle": r"\b(ankle|foot|feet)\b",
    "wrist": r"\b(wrist|hand|finger|carpal)\b",
    "elbow": r"\b(elbow)\b",
    "cardiac": r"\b(heart|cardiac|coronar|myocard|myo\s*perf|aort)\b",
    "breast": r"\b(breast|mamm|mam\b)\b",
    "neck": r"\b(neck|thyroid|carotid)\b",
    "face": r"\b(face|facial|sinus|orbit|mandib|maxfac|maxil|nasal)\b",
    "extremity": r"\b(extremit|arm|leg|tibia|fibula|humerus|forearm|calf)\b",
    "whole_body": r"\b(whole\s*body|skull.*thigh|vertex.*toe|total\s*body)\b",
    "vascular": r"\b(angiogr|angio|venous|arter|vein|vascu|dvt)\b",
}

MODALITIES = {
    "CT": r"\bCT\b",
    "MRI": r"\bMR[I]?\b",
    "XR": r"\b(XR|x[\-\s]?ray|radiograph|CR)\b",
    "US": r"\b(US\b|ultra|sono|doppler)",
    "NM": r"\b(NM|nuclear|SPECT|scintig|bone\s*scan)\b",
    "PET": r"\bPET\b",
    "MAM": r"\b(MAM|mammo|tomo)\b",
    "FLUORO": r"\b(fluoro|barium|swallow)\b",
    "DEXA": r"\b(DEXA|DXA|bone\s*dens)\b",
}

def extract_regions(desc):
    desc_lower = desc.lower()
    regions = set()
    for region, pattern in BODY_REGIONS.items():
        if re.search(pattern, desc_lower, re.IGNORECASE):
            regions.add(region)
    if not regions:
        if re.search(r"\bpet/ct\b.*skull|whole", desc_lower):
            regions.add("whole_body")
    return regions

def extract_modality(desc):
    desc_upper = desc.upper()
    for mod, pattern in MODALITIES.items():
        if re.search(pattern, desc_upper):
            return mod
    return "UNKNOWN"

# --- Analyze patterns ---

region_match_stats = {"match_relevant": 0, "match_not": 0, "no_match_relevant": 0, "no_match_not": 0}
modality_match_stats = {"match_relevant": 0, "match_not": 0, "no_match_relevant": 0, "no_match_not": 0}
unmatched_descriptions = Counter()
region_pair_stats = defaultdict(lambda: {"relevant": 0, "not_relevant": 0})

for case in data["cases"]:
    current = case["current_study"]
    current_regions = extract_regions(current["study_description"])
    current_mod = extract_modality(current["study_description"])
    current_date = datetime.strptime(current["study_date"], "%Y-%m-%d")

    for prior in case["prior_studies"]:
        key = (case["case_id"], prior["study_id"])
        if key not in truth_by_key:
            continue
        is_relevant = truth_by_key[key]

        prior_regions = extract_regions(prior["study_description"])
        prior_mod = extract_modality(prior["study_description"])

        if not current_regions:
            unmatched_descriptions[current["study_description"]] += 1
        if not prior_regions:
            unmatched_descriptions[prior["study_description"]] += 1

        regions_overlap = bool(current_regions & prior_regions)

        pair_key = (frozenset(current_regions), frozenset(prior_regions))
        if is_relevant:
            region_pair_stats[pair_key]["relevant"] += 1
        else:
            region_pair_stats[pair_key]["not_relevant"] += 1

        if regions_overlap:
            if is_relevant:
                region_match_stats["match_relevant"] += 1
            else:
                region_match_stats["match_not"] += 1
        else:
            if is_relevant:
                region_match_stats["no_match_relevant"] += 1
            else:
                region_match_stats["no_match_not"] += 1

        mods_match = current_mod == prior_mod
        if mods_match:
            if is_relevant:
                modality_match_stats["match_relevant"] += 1
            else:
                modality_match_stats["match_not"] += 1
        else:
            if is_relevant:
                modality_match_stats["no_match_relevant"] += 1
            else:
                modality_match_stats["no_match_not"] += 1

print("=== BODY REGION MATCH AS PREDICTOR ===")
for k, v in region_match_stats.items():
    print(f"  {k}: {v}")
tp = region_match_stats["match_relevant"]
fp = region_match_stats["match_not"]
fn = region_match_stats["no_match_relevant"]
tn = region_match_stats["no_match_not"]
if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"  Precision (if regions overlap, predict relevant): {precision:.3f}")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"  Recall: {recall:.3f}")
accuracy = (tp + tn) / (tp + fp + fn + tn)
print(f"  Accuracy: {accuracy:.3f}")
print()

print("=== MODALITY MATCH AS PREDICTOR ===")
for k, v in modality_match_stats.items():
    print(f"  {k}: {v}")
tp = modality_match_stats["match_relevant"]
fp = modality_match_stats["match_not"]
fn = modality_match_stats["no_match_relevant"]
tn = modality_match_stats["no_match_not"]
if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"  Precision (if modality matches, predict relevant): {precision:.3f}")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"  Recall: {recall:.3f}")
accuracy = (tp + tn) / (tp + fp + fn + tn)
print(f"  Accuracy: {accuracy:.3f}")
print()

print("=== TOP UNMATCHED DESCRIPTIONS (no body region extracted) ===")
for desc, count in unmatched_descriptions.most_common(30):
    print(f"  [{count:4d}] {desc}")
print()

print("=== ALWAYS-FALSE BASELINE ===")
print(f"  Accuracy: {(total - relevant) / total:.3f}")
print()

print("=== ALWAYS-TRUE BASELINE ===")
print(f"  Accuracy: {relevant / total:.3f}")
