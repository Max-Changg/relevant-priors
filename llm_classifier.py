"""
LLM-based classifier using Anthropic Claude for radiology prior relevance.

Supports per-case classification and cross-case batch pair classification.
Caches results by (current_description, prior_description) pair.
Includes token-bucket rate limiter and rate-limit-aware backoff.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

_cache: dict[tuple[str, str], bool] = {}

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")


class RateLimiter:
    """Sliding-window rate limiter for API calls."""

    def __init__(self, max_per_minute: int = 40):
        self.timestamps: list[float] = []
        self.max = max_per_minute

    async def acquire(self):
        while True:
            now = time.monotonic()
            self.timestamps = [t for t in self.timestamps if now - t < 60]
            if len(self.timestamps) < self.max:
                self.timestamps.append(now)
                return
            sleep_for = 60 - (now - self.timestamps[0]) + 0.1
            await asyncio.sleep(sleep_for)


_rate_limiter = RateLimiter(max_per_minute=40)


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


SYSTEM_PROMPT = """You are an expert radiologist assistant. Your task is to determine whether each prior imaging study is relevant to the current study a radiologist is about to read.

A prior study is RELEVANT if a radiologist would want to see it for comparison while interpreting the current study. This typically means:
- Same or overlapping body region (e.g., prior chest CT is relevant to current chest CT)
- Related anatomy that would inform interpretation (e.g., prior mammogram relevant to current breast ultrasound)
- Cross-modality studies of the same region (e.g., prior chest X-ray relevant to current chest CT)
- Whole-body studies (like PET/CT skull-to-thigh, bone scan) are relevant to studies of regions they cover

A prior study is NOT RELEVANT if:
- It examines a completely different body region (e.g., knee MRI is not relevant to brain CT)
- It's a different organ system with no diagnostic overlap (e.g., echocardiogram is usually not relevant to a chest X-ray)
- The anatomy is adjacent but clinically distinct (e.g., cervical spine MRI is not relevant to brain MRI unless there's clinical overlap)

Respond ONLY with a JSON array of booleans, one per prior study, in the same order as listed. No other text."""

BATCH_SYSTEM_PROMPT = """You are an expert radiologist assistant. For each numbered pair, determine whether the prior study is relevant when a radiologist reads the current study.

RELEVANT means the radiologist would benefit from seeing the prior for comparison. Key rules:
- Same body region and modality: RELEVANT (e.g., prior CT chest for current CT chest)
- Same body region, different modality: usually RELEVANT (e.g., prior MRI brain for current CT head)
- Overlapping anatomy: RELEVANT (e.g., prior abdomen/pelvis CT for current pelvis MRI)
- Cardiac studies (echo/TTE) vs chest CT: RELEVANT only if current is CT/MRI chest (shows heart)
- Cardiac studies vs plain chest XR: NOT RELEVANT (chest XR doesn't show cardiac detail)
- Carotid/neck vascular vs brain imaging: RELEVANT when current is brain CT/MRI angiography
- Bone scan (whole body) vs regional studies: RELEVANT for musculoskeletal/oncology regions
- DEXA bone density vs structural imaging: NOT RELEVANT (different clinical purpose)
- EEG vs brain imaging: NOT RELEVANT (electrophysiology, not imaging)
- Different spine segments (cervical vs lumbar): usually NOT RELEVANT
- Adjacent spine segments (thoracic vs lumbar): often RELEVANT

Respond ONLY with a JSON array of booleans, one per pair, in order. No other text."""


def _parse_llm_json(text: str) -> list:
    """Extract and parse a JSON array from LLM response text."""
    if not text:
        return []
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        text = json_match.group()
    return json.loads(text)


async def classify_pairs_batch(
    pairs: list[tuple[str, str]],
    client: anthropic.AsyncAnthropic,
    rate_limiter: Optional[RateLimiter] = None,
    batch_size: int = 25,
) -> dict[tuple[str, str], bool]:
    """Classify (current_desc, prior_desc) pairs in batches via LLM.

    Returns dict mapping normalized (current, prior) -> bool.
    """
    if rate_limiter is None:
        rate_limiter = _rate_limiter

    results: dict[tuple[str, str], bool] = {}
    uncached: list[tuple[str, str]] = []

    for current, prior in pairs:
        key = (current.strip().lower(), prior.strip().lower())
        if key in _cache:
            results[key] = _cache[key]
        else:
            uncached.append((current, prior))

    if not uncached:
        return results

    num_batches = -(-len(uncached) // batch_size)
    logger.info(f"Batch LLM: {len(uncached)} uncached pairs in {num_batches} batches")

    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        pair_lines = []
        for idx, (current, prior) in enumerate(batch):
            pair_lines.append(f'{idx + 1}. Current: "{current}" | Prior: "{prior}"')

        user_msg = f"Evaluate these {len(batch)} study pairs:\n" + "\n".join(pair_lines)

        max_retries = 4
        for attempt in range(max_retries + 1):
            await rate_limiter.acquire()
            try:
                start = time.monotonic()
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=BATCH_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                elapsed = time.monotonic() - start
                logger.info(f"Batch LLM call: {len(batch)} pairs, {elapsed:.2f}s")

                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text = block.text.strip()
                        break

                if not text:
                    if attempt < max_retries:
                        await asyncio.sleep(1)
                        continue
                    break

                predictions = _parse_llm_json(text)

                if len(predictions) != len(batch):
                    logger.warning(
                        f"LLM returned {len(predictions)} for {len(batch)} pairs"
                    )

                for idx, (current, prior) in enumerate(batch):
                    if idx < len(predictions):
                        pred = bool(predictions[idx])
                        key = (current.strip().lower(), prior.strip().lower())
                        _cache[key] = pred
                        results[key] = pred

                break

            except anthropic.RateLimitError:
                logger.warning(f"Rate limited (attempt {attempt+1}), waiting 60s")
                await asyncio.sleep(60)
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Batch parse error (attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Batch LLM failed (attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 * (attempt + 1))

    return results


async def classify_case_llm(
    current_desc: str,
    current_date: str,
    priors: list[dict],
    client: anthropic.AsyncAnthropic,
    semaphore: Optional[asyncio.Semaphore] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> list[Optional[bool]]:
    """
    Classify all priors for a single case using one LLM call.
    Returns list of bool predictions aligned with the priors list.
    Uses cache for previously seen description pairs.
    """
    if rate_limiter is None:
        rate_limiter = _rate_limiter

    results: list[Optional[bool]] = [None] * len(priors)
    uncached_indices = []

    for i, prior in enumerate(priors):
        cache_key = (current_desc.strip().lower(), prior["study_description"].strip().lower())
        if cache_key in _cache:
            results[i] = _cache[cache_key]
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return results

    prior_lines = []
    for idx, i in enumerate(uncached_indices):
        p = priors[i]
        prior_lines.append(f'{idx + 1}. "{p["study_description"]}" (date: {p.get("study_date", "unknown")})')

    user_msg = (
        f'Current study: "{current_desc}" (date: {current_date})\n\n'
        f"Prior studies to evaluate ({len(uncached_indices)} total):\n"
        + "\n".join(prior_lines)
    )

    max_retries = 4
    for attempt in range(max_retries + 1):
        acquired_sem = False
        try:
            if semaphore:
                await semaphore.acquire()
                acquired_sem = True
            await rate_limiter.acquire()

            start = time.monotonic()
            response = await client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            elapsed = time.monotonic() - start
            logger.info(f"LLM call: {len(uncached_indices)} priors, {elapsed:.2f}s")

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text.strip()
                    break

            if not text:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                break

            predictions = _parse_llm_json(text)

            if len(predictions) != len(uncached_indices):
                logger.warning(
                    f"LLM returned {len(predictions)} for {len(uncached_indices)} priors"
                )

            for idx, i in enumerate(uncached_indices):
                if idx < len(predictions):
                    pred = bool(predictions[idx])
                    results[i] = pred
                    cache_key = (
                        current_desc.strip().lower(),
                        priors[i]["study_description"].strip().lower(),
                    )
                    _cache[cache_key] = pred

            return results

        except anthropic.RateLimitError:
            logger.warning(f"Rate limited (attempt {attempt+1}), waiting 60s")
            await asyncio.sleep(60)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"LLM parse error (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                break
        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                break
        finally:
            if acquired_sem and semaphore:
                semaphore.release()

    return results


def get_cache_stats() -> dict:
    return {"cache_size": len(_cache)}


def clear_cache():
    _cache.clear()
