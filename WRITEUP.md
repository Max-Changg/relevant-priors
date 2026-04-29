# Relevant Priors — Writeup

## Problem

Given a current radiology examination and a list of prior examinations for the same patient, predict which priors a radiologist would want to see for comparison. The challenge: handle 996 cases with 27,614 priors under a 360-second timeout, while respecting Anthropic API rate limits (50 req/min, 30K tokens/min).

## Approach

A two-pass hybrid system: a fast heuristic classifier resolves the easy cases with high confidence, and a batched LLM call handles only the truly unknowable remainder.

### Pass 1 — Heuristic Classifier (`classifier.py`)

Regex-based extraction of body regions and imaging modalities from study descriptions, followed by rule-based relevance prediction with confidence scoring.

**Key design choices:**

- **Same-description shortcut**: Strips contrast keywords (`WITH CONTRAST`, `WITHOUT CNTRST`, etc.) and compares normalized descriptions. Identical-modulo-contrast pairs return `(True, 0.95)` immediately.

- **Body region extraction**: 25 region categories (brain, chest, abdomen, pelvis, cardiac, breast, spine variants, extremity joints, vascular, whole-body, bone scan, bone density, face/sinuses, neck).

- **Related-region graph**: Pairs like abdomen/pelvis, abdomen/renal, and neck/spine_cervical are treated as relevant with moderate confidence (0.76).

- **Ambiguous-region detection**: Pairs like cardiac/chest, neck/brain, thoracic-spine/chest, and bone-scan/regional are marked as low confidence (0.5) and deferred to the LLM. These relationships are directional — e.g., an echocardiogram reader wants prior chest CTs, but a chest XR reader doesn't need prior echocardiograms.

- **Modality-based exclusions**: DEXA bone density scans are never comparable to structural imaging (returns False, 0.85). Same-modality pairs in the same region get boosted confidence (0.95).

- **Spine segment specificity**: Cervical spine vs lumbar spine share the generic "spine" region but are clinically distinct — these are marked as low confidence for LLM deferral.

- **Confidence thresholds**: Each heuristic path returns a confidence score. Pairs above 0.45 are accepted as-is; those below (currently only "both descriptions unrecognized") are sent to the LLM.

### Pass 2 — Batched LLM (`llm_classifier.py`)

Uncertain pairs are deduplicated across all cases, grouped into batches of 25, and sent to Claude with a clinically-informed system prompt covering edge cases like cardiac/chest directionality, bone scan coverage, and DEXA exclusion.

**Rate limiting & reliability:**

- **Sliding-window rate limiter** (40 req/min, under the 50 req/min hard limit) prevents 429 errors proactively.
- **60-second backoff** on 429 responses (matching Anthropic's window reset).
- **Description-pair cache**: Results are keyed by `(current_desc.lower(), prior_desc.lower())`. Duplicate pairs across cases and retried evaluator requests hit the cache instead of the API.
- **4 retries** with exponential backoff for transient failures and JSON parse errors.

### Server (`server.py`)

FastAPI endpoint at `POST /predict`. Runs both passes in sequence. Falls back to heuristic-only mode when `ANTHROPIC_API_KEY` is not set.

## Experiments & Results

### Experiment 1 — Baseline heuristic (region overlap only)

Initial implementation with basic body-region matching. No same-description shortcut, no related-region handling, limited regex coverage.

| Metric | Value |
|--------|-------|
| Accuracy | ~85% |
| Uncertain pairs | 519 |
| False negatives | High (missed related regions like abd/pelvis) |

### Experiment 2 — Improved heuristic with modality rules

Added same-description shortcut, abdomen patterns (barium, esophagram, swallow, UGI, small bowel), breast screen patterns, related-region pairs, modality-based exclusions (DEXA), spine segment handling, and ambiguous-pair detection for LLM deferral.

| Metric | Value |
|--------|-------|
| Accuracy | 93.79% (25,898 / 27,614) |
| Precision | 0.8319 |
| Recall | 0.8954 |
| High-confidence accuracy | 94.9% (on 25,898 resolved pairs) |
| Uncertain pairs deferred | 1,716 (666 unique) |
| Runtime | 3.0s for all 996 cases |

### Experiment 3 — Measured hybrid LLM evaluation

Tested batched LLM on the 666 unique uncertain pairs with both Haiku and Sonnet:

| Model | Accuracy | Precision | Recall | Runtime |
|-------|----------|-----------|--------|---------|
| Heuristic only | **93.79%** | 0.8319 | 0.8954 | 3.0s |
| + Claude Haiku 4.5 | 92.57% | 0.7912 | 0.9342 | 36.5s |
| + Claude Sonnet 4.6 | 93.08% | 0.8036 | 0.9383 | 92.4s |

**Key finding**: The LLM improves recall (+4 pp) but hurts precision (−4 pp), netting lower overall accuracy. The ambiguous pairs (cardiac/chest, neck/brain) are **directionally ambiguous** — the same description pair can be relevant or not depending on which is current vs prior — and the LLM over-predicts True without sufficient clinical context.

**Deployed configuration**: Confidence threshold set to 0.45 so only truly unknowable pairs (both descriptions unrecognized; ~11 pairs) are sent to the LLM. This gives 93.79% accuracy with 3-second runtime and minimal API usage.

## Clinical Error Analysis

Categorized the remaining 1,716 errors by radiologist-workflow patterns:

### False Negatives (relevant but predicted not relevant): 1,317 pairs

| Category | Count | Example | Root Cause |
|----------|-------|---------|------------|
| Cardiac ↔ Chest (directional) | ~59 | ECHO TTE vs CT CHEST | Echocardiogram reader needs chest CT for heart anatomy |
| Bone Scan ↔ Regional | ~67 | Bone Scan vs CT abdomen | Bone scans are whole-body, relevant for staging |
| Thoracic Spine ↔ Chest | ~28 | XR thoracic spine vs CHEST 2V | Anatomically adjacent, overlapping coverage |
| Neck/Carotid ↔ Brain | ~22 | CT angio carotid vs CT HEAD | Stroke workup: carotid feeds brain vasculature |
| Lumbar Spine ↔ Abdomen | ~12 | MRI lumbar vs CT ABD/PELVIS | Overlapping imaging field of view |
| Unrecognized descriptions | ~12 | CT guided FNA vs CT CHEST | Study types we can't parse |

These are all deferred to the LLM (conf 0.5), but the LLM currently doesn't resolve them accurately because the directionality is ambiguous from descriptions alone.

### False Positives (not relevant but predicted relevant): 399 pairs

| Category | Count | Example | Root Cause |
|----------|-------|---------|------------|
| Brain Imaging ↔ EEG/TCD | ~28 | CT HEAD vs NE EEG Request | EEG shares "brain" region but isn't imaging |
| Spine Segment Confusion | ~20 | T-spine vs C-spine | Share generic "spine" region |
| Abdomen/Pelvis overmatching | ~40 | CT abd/pelvis vs US endovaginal | Region overlap but different clinical purposes |
| Whole-body NM ↔ Regional | ~9 | PET/CT vs Modified Barium | Whole-body expansion too broad |

### Clinical Insights

1. **Relevance is directional**: When a cardiologist reads an echocardiogram, they want prior chest CTs (shows heart anatomy in context). But when a general radiologist reads a chest XR, they don't need prior echocardiograms. The same (cardiac, chest) pair is relevant in one direction and not the other.

2. **EEG is not imaging**: The "brain" region parser matches EEG (electroencephalography) because it's a neurological test, but radiologists don't compare EEGs with brain imaging. This is a modality-level distinction the heuristic could handle.

3. **Bone scans are contextual**: A nuclear medicine bone scan covers the whole body, but its relevance depends on the clinical question (oncology staging vs. fracture workup). The same bone scan might be relevant to a chest CT for lung cancer follow-up but not for a routine chest CT.

4. **Abdomen descriptions are overloaded**: "Abdomen" as a plain XR is a different clinical context than CT/MRI abdomen, but they share the same body region.

## Architecture Decisions

1. **Heuristic-first, LLM-second**: Minimizes API calls and latency. The heuristic resolves 99.96% of pairs in <3 seconds.

2. **Cross-case deduplication**: Many patients share identical study descriptions. Deduplicating `(current_desc, prior_desc)` pairs before LLM batching cuts API calls by 60%.

3. **Confidence-gated deferral with ambiguous-pair detection**: Known-ambiguous region pairs (cardiac/chest, neck/brain) get low confidence even though the heuristic picks a default answer. This enables future improvements (better LLM prompts, fine-tuned models) without changing the heuristic.

4. **Graceful degradation**: Without an API key, the server runs in heuristic-only mode (93.79% accuracy). Rate limit errors trigger backoff rather than failure.

## Test Suite

45 tests in `test_plan.py` covering:

- Regex pattern extraction for all body regions
- Same-description shortcut with contrast normalization
- Batch deduplication, sizing, and caching
- Rate limiter behavior (under/at/over limit, window sliding)
- 429 backoff timing (verifies ≥60s sleep)
- `/predict` endpoint schema contract
- No missing predictions (every prior gets a result)
- Heuristic-only mode (no API key)
- **Clinical edge cases**: echo/chest directionality, c-spine/l-spine distinction, DEXA exclusion, bone scan deferral, PET/CT coverage, mammogram/breast US, carotid/brain ambiguity
- Public-set accuracy ≥ 93% and ambiguous-pair deferral bounds
- Runtime < 300s

## Reproducibility

```bash
# Production
pip install -r requirements.txt

# Development (adds pytest, httpx, requests)
pip install -r requirements-dev.txt

# Run tests
pytest test_plan.py -v

# Run heuristic eval
python3 eval_heuristic.py

# Run hybrid eval (requires API key)
ANTHROPIC_API_KEY=sk-... python3 eval_direct.py
```

## Next Improvements

1. **Direction-aware classification**: Encode which study is current vs prior into the decision logic. A cardiac reader seeing prior chest CT is different from a chest reader seeing prior echo. This likely requires case-level features (study type of current) rather than just pair descriptions.

2. **EEG/TCD exclusion**: Add modality-based rule to exclude non-imaging neurological tests (EEG, transcranial Doppler) from brain imaging comparisons. These share the "brain" region tag but serve entirely different diagnostic purposes.

3. **Bone scan conditional expansion**: Bone scans should be relevant to regional studies only in oncology/staging contexts. Date proximity and patient history could help — a bone scan done the same week as a chest CT is likely part of the same workup.

4. **Fine-tuned classifier**: Train a small model (logistic regression or BERT) on the labeled pairs with features: body region overlap, modality match, description similarity, date gap, and study type. This could handle the directional ambiguity that both the heuristic and zero-shot LLM miss.

5. **Persistent cache**: Store LLM results in a lightweight database (SQLite or Redis) so they survive server restarts and accumulate across evaluator retries.

6. **Abdomen plain-XR disambiguation**: Distinguish "Abdomen" as a plain XR (limited utility for comparison) from advanced abdominal imaging. This would reduce false positives where CT/MRI abdomen is matched to a plain abdominal film.
