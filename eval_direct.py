#!/usr/bin/env python3
"""
Direct evaluation — calls the classifier functions directly without HTTP.
Two-pass approach: heuristic all cases first, then batch LLM for uncertain pairs.

Usage:
    python3 eval_direct.py                          # heuristic only
    ANTHROPIC_API_KEY=sk-... python3 eval_direct.py  # with LLM
    ANTHROPIC_MODEL=claude-sonnet-4-6 ANTHROPIC_API_KEY=sk-... python3 eval_direct.py
"""

import asyncio
import json
import os
import time

import anthropic

from classifier import heuristic_predict
from llm_classifier import classify_pairs_batch, get_cache_stats

HEURISTIC_CONFIDENCE_THRESHOLD = 0.75


async def main():
    with open("relevant_priors_public.json") as f:
        data = json.load(f)

    truth_by_key = {}
    for t in data["truth"]:
        truth_by_key[(t["case_id"], t["study_id"])] = t["is_relevant_to_current"]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_llm = bool(api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

    if use_llm:
        print(f"Mode: Hybrid (heuristic + LLM {model})")
    else:
        print("Mode: Heuristic only (no ANTHROPIC_API_KEY)")

    cases = data["cases"]
    predictions = {}
    heuristic_used = 0
    start = time.monotonic()

    # Phase 1: Heuristic pass on all cases
    uncertain: list[tuple[str, str, str, str]] = []

    for case in cases:
        for prior in case["prior_studies"]:
            key = (case["case_id"], prior["study_id"])
            pred, conf = heuristic_predict(
                case["current_study"]["study_description"],
                prior["study_description"],
            )
            if not use_llm or conf >= HEURISTIC_CONFIDENCE_THRESHOLD:
                predictions[key] = pred
                heuristic_used += 1
            else:
                uncertain.append((
                    case["case_id"],
                    prior["study_id"],
                    case["current_study"]["study_description"],
                    prior["study_description"],
                ))

    phase1_elapsed = time.monotonic() - start
    print(
        f"Phase 1 (heuristic): {heuristic_used} resolved, "
        f"{len(uncertain)} uncertain ({phase1_elapsed:.1f}s)"
    )

    # Phase 2: Batch LLM for uncertain pairs
    if uncertain and use_llm:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        unique_pairs = list({(cdesc, pdesc) for _, _, cdesc, pdesc in uncertain})
        print(f"Unique uncertain pairs: {len(unique_pairs)} (from {len(uncertain)} total)")

        pair_results = await classify_pairs_batch(unique_pairs, client)

        llm_resolved = 0
        for case_id, study_id, cdesc, pdesc in uncertain:
            key = (case_id, study_id)
            pair_key = (cdesc.strip().lower(), pdesc.strip().lower())
            if pair_key in pair_results:
                predictions[key] = pair_results[pair_key]
                llm_resolved += 1
            else:
                pred, _ = heuristic_predict(cdesc, pdesc)
                predictions[key] = pred
                heuristic_used += 1

        print(f"Phase 2 (LLM): {llm_resolved} resolved, {heuristic_used} heuristic total")
    elif uncertain:
        for case_id, study_id, cdesc, pdesc in uncertain:
            key = (case_id, study_id)
            pred, _ = heuristic_predict(cdesc, pdesc)
            predictions[key] = pred

    elapsed = time.monotonic() - start

    # Score
    correct = incorrect = tp = fp = tn = fn = 0
    for key, actual in truth_by_key.items():
        if key not in predictions:
            incorrect += 1
            continue
        pred = predictions[key]
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
    print(f"RESULTS ({model if use_llm else 'heuristic only'})")
    print(f"{'='*50}")
    print(f"Correct: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if tp + fp > 0:
        print(f"  Precision: {tp/(tp+fp):.4f}")
    if tp + fn > 0:
        print(f"  Recall: {tp/(tp+fn):.4f}")
    if use_llm:
        print(f"  Heuristic-only: {heuristic_used}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Cache: {get_cache_stats()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
