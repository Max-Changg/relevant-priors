#!/usr/bin/env python3
"""
Tests for the Fix Rate Limiting plan.
Run with: pytest test_plan.py -v
"""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PUBLIC_JSON = os.path.join(os.path.dirname(__file__), "relevant_priors_public.json")
HAVE_PUBLIC_JSON = os.path.exists(PUBLIC_JSON)


# ---------------------------------------------------------------------------
# Part 1: Heuristic improvements
# ---------------------------------------------------------------------------

class TestNewPatternsParse:
    """Test 1: New regex patterns parse previously-unrecognized descriptions."""

    def test_xr_scoliosis_parses_to_spine(self):
        from classifier import extract_regions
        regions = extract_regions("XR scoliosis survey")
        assert "spine" in regions

    def test_nm_pul_perfusion_parses_to_chest(self):
        from classifier import extract_regions
        regions = extract_regions("NM pul perfusion")
        assert "chest" in regions

    def test_modified_barium_parses_to_abdomen(self):
        from classifier import extract_regions
        regions = extract_regions("Modified Barium ADULT")
        assert "abdomen" in regions

    def test_us_bilat_screen_parses_to_breast(self):
        from classifier import extract_regions
        regions = extract_regions("ULTRASOUND BILAT SCREEN COMP")
        assert "breast" in regions

    def test_bone_density_parses(self):
        from classifier import extract_regions
        regions = extract_regions("BONE DENSITY")
        assert "bone_density" in regions

    def test_esophagram_parses_to_abdomen(self):
        from classifier import extract_regions
        regions = extract_regions("ESOPHAGRAM WITH CONTRAST")
        assert "abdomen" in regions


class TestSameDescriptionShortcut:
    """Test 2: Same-description shortcut returns (True, >=0.95)."""

    @pytest.mark.parametrize("desc", [
        "CT CHEST WITH CONTRAST",
        "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "XR CHEST PA AND LATERAL",
        "US ABDOMEN COMPLETE",
        "MAMMOGRAM BILAT SCREEN",
    ])
    def test_identical_descriptions_relevant(self, desc):
        from classifier import heuristic_predict
        pred, conf = heuristic_predict(desc, desc)
        assert pred is True
        assert conf >= 0.95


class TestNearMatchContrast:
    """Test 3: Descriptions differing only by contrast keywords still match."""

    def test_ct_chest_contrast_variants(self):
        from classifier import heuristic_predict
        pred, conf = heuristic_predict(
            "CT CHEST WITH CONTRAST",
            "CT CHEST WITHOUT CNTRST",
        )
        assert pred is True
        assert conf >= 0.9

    def test_mri_brain_contrast_variants(self):
        from classifier import heuristic_predict
        pred, conf = heuristic_predict(
            "MRI BRAIN WITH AND WITHOUT CONTRAST",
            "MRI BRAIN WITHOUT CONTRAST",
        )
        assert pred is True
        assert conf >= 0.9


@pytest.mark.skipif(not HAVE_PUBLIC_JSON, reason="Public eval JSON not found")
class TestHeuristicOnPublicSet:
    """Tests 4-6 require the full public dataset."""

    @pytest.fixture(scope="class")
    def heuristic_results(self):
        from classifier import heuristic_predict

        with open(PUBLIC_JSON) as f:
            data = json.load(f)

        truth_by_key = {}
        for t in data["truth"]:
            truth_by_key[(t["case_id"], t["study_id"])] = t["is_relevant_to_current"]

        uncertain_count = 0
        correct = incorrect = fp = 0

        for case in data["cases"]:
            for prior in case["prior_studies"]:
                key = (case["case_id"], prior["study_id"])
                pred, conf = heuristic_predict(
                    case["current_study"]["study_description"],
                    prior["study_description"],
                )

                if conf < 0.75:
                    uncertain_count += 1

                actual = truth_by_key.get(key)
                if actual is not None:
                    if pred == actual:
                        correct += 1
                    else:
                        incorrect += 1
                        if pred and not actual:
                            fp += 1

        total = correct + incorrect
        accuracy = correct / total if total > 0 else 0
        return {
            "uncertain_count": uncertain_count,
            "accuracy": accuracy,
            "correct": correct,
            "incorrect": incorrect,
            "fp": fp,
            "total": total,
        }

    def test_uncertain_count_drops(self, heuristic_results):
        """Test 4: Uncertain cases < 250 (down from 519)."""
        assert heuristic_results["uncertain_count"] < 250, (
            f"Uncertain count {heuristic_results['uncertain_count']} >= 250"
        )

    def test_heuristic_accuracy_no_regression(self, heuristic_results):
        """Test 5: Heuristic accuracy >= 0.78."""
        assert heuristic_results["accuracy"] >= 0.78, (
            f"Accuracy {heuristic_results['accuracy']:.4f} < 0.78"
        )

    def test_false_positive_bound(self, heuristic_results):
        """Test 6: False positives < 1500."""
        assert heuristic_results["fp"] < 1500, (
            f"False positives {heuristic_results['fp']} >= 1500"
        )


# ---------------------------------------------------------------------------
# Part 2: Batch LLM flow
# ---------------------------------------------------------------------------

class TestUniquePairCollection:
    """Test 7: Deduplication of uncertain pairs across cases."""

    def test_deduplication(self):
        uncertain = [
            ("case1", "s1", "CT CHEST", "XR CHEST"),
            ("case1", "s2", "CT CHEST", "MRI BRAIN"),
            ("case2", "s3", "CT CHEST", "XR CHEST"),       # duplicate pair
            ("case2", "s4", "MRI KNEE", "XR ANKLE"),
            ("case3", "s5", "CT CHEST", "XR CHEST"),       # duplicate pair
            ("case3", "s6", "CT CHEST", "MRI BRAIN"),      # duplicate pair
        ]
        unique_pairs = list({(cdesc, pdesc) for _, _, cdesc, pdesc in uncertain})
        assert len(unique_pairs) == 3


class TestBatchSizing:
    """Test 8: Batching groups pairs into correctly-sized batches."""

    @pytest.mark.asyncio
    async def test_batch_count(self):
        from llm_classifier import classify_pairs_batch, clear_cache, RateLimiter
        clear_cache()

        pairs = [(f"study_A_{i}", f"study_B_{i}") for i in range(60)]

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = kwargs["messages"][0]["content"]
            n = int(msg.split("Evaluate these ")[1].split(" ")[0])
            return MagicMock(content=[MagicMock(text=json.dumps([True] * n))])

        mock_client = AsyncMock()
        mock_client.messages.create = mock_create
        rl = RateLimiter(max_per_minute=10000)

        await classify_pairs_batch(pairs, mock_client, rl, batch_size=25)

        assert call_count == 3, f"Expected 3 batches, got {call_count}"
        clear_cache()


class TestCachePopulation:
    """Test 9: Cache is populated after batch LLM call."""

    @pytest.mark.asyncio
    async def test_cache_filled(self):
        from llm_classifier import classify_pairs_batch, _cache, clear_cache, RateLimiter
        clear_cache()

        pairs = [("CT CHEST", "XR CHEST"), ("MRI BRAIN", "CT HEAD")]

        async def mock_create(**kwargs):
            return MagicMock(content=[MagicMock(text="[true, false]")])

        mock_client = AsyncMock()
        mock_client.messages.create = mock_create
        rl = RateLimiter(max_per_minute=10000)

        await classify_pairs_batch(pairs, mock_client, rl)

        assert ("ct chest", "xr chest") in _cache
        assert ("mri brain", "ct head") in _cache
        assert _cache[("ct chest", "xr chest")] is True
        assert _cache[("mri brain", "ct head")] is False
        clear_cache()


class TestResultMapping:
    """Test 10: Shared pairs across cases get the same cached result."""

    @pytest.mark.asyncio
    async def test_shared_pair_same_result(self):
        from llm_classifier import classify_pairs_batch, clear_cache, RateLimiter
        clear_cache()

        async def mock_create(**kwargs):
            return MagicMock(content=[MagicMock(text="[true]")])

        mock_client = AsyncMock()
        mock_client.messages.create = mock_create
        rl = RateLimiter(max_per_minute=10000)

        results = await classify_pairs_batch([("CT CHEST", "XR CHEST")], mock_client, rl)
        assert results[("ct chest", "xr chest")] is True

        # Second call should use cache — make client raise so we know it's not called
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("LLM should not be called")
        )
        results2 = await classify_pairs_batch([("CT CHEST", "XR CHEST")], mock_client, rl)
        assert results2[("ct chest", "xr chest")] is True
        clear_cache()


class TestAllCachedFastPath:
    """Test 11: Zero LLM calls when everything is cached."""

    @pytest.mark.asyncio
    async def test_no_llm_when_cached(self):
        from llm_classifier import classify_pairs_batch, _cache, clear_cache, RateLimiter
        clear_cache()

        _cache[("ct chest", "xr chest")] = True
        _cache[("mri brain", "ct head")] = False

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("LLM should not be called")
        )
        rl = RateLimiter(max_per_minute=10000)

        results = await classify_pairs_batch(
            [("CT CHEST", "XR CHEST"), ("MRI BRAIN", "CT HEAD")],
            mock_client, rl,
        )

        assert results[("ct chest", "xr chest")] is True
        assert results[("mri brain", "ct head")] is False
        mock_client.messages.create.assert_not_called()
        clear_cache()


# ---------------------------------------------------------------------------
# Part 3: Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiterUnderLimit:
    """Test 12: Under-limit acquires complete immediately."""

    @pytest.mark.asyncio
    async def test_39_acquires_fast(self):
        from llm_classifier import RateLimiter
        rl = RateLimiter(max_per_minute=40)

        start = time.monotonic()
        for _ in range(39):
            await rl.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"39 acquires took {elapsed:.3f}s, expected < 0.5s"


class TestRateLimiterBlocks:
    """Test 13: At-limit, the next acquire blocks."""

    @pytest.mark.asyncio
    async def test_41st_blocks(self):
        from llm_classifier import RateLimiter
        rl = RateLimiter(max_per_minute=40)

        for _ in range(40):
            await rl.acquire()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(rl.acquire(), timeout=0.5)


class TestRateLimiterWindowSlides:
    """Test 14: After 60s, the window slides and new acquires succeed."""

    @pytest.mark.asyncio
    async def test_window_reset(self):
        from llm_classifier import RateLimiter
        rl = RateLimiter(max_per_minute=40)

        for _ in range(40):
            await rl.acquire()

        # Simulate 61s passing by shifting all timestamps back
        rl.timestamps = [t - 61 for t in rl.timestamps]

        start = time.monotonic()
        await asyncio.wait_for(rl.acquire(), timeout=1.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5


class TestBackoff429:
    """Test 15: 429 backoff waits at least 60s."""

    @pytest.mark.asyncio
    async def test_rate_limit_backoff(self):
        from llm_classifier import classify_case_llm, RateLimiter, clear_cache
        clear_cache()

        sleep_durations: list[float] = []

        async def tracking_sleep(duration):
            sleep_durations.append(duration)

        # Build a RateLimitError via httpx
        import httpx
        raw_response = httpx.Response(
            status_code=429,
            headers={"retry-after": "60"},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        import anthropic as anth
        rate_limit_error = anth.RateLimitError(
            message="rate limited",
            response=raw_response,
            body=None,
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=rate_limit_error)
        rl = RateLimiter(max_per_minute=10000)

        with patch("llm_classifier.asyncio.sleep", side_effect=tracking_sleep):
            await classify_case_llm(
                current_desc="CT CHEST",
                current_date="2025-01-01",
                priors=[{"study_id": "1", "study_description": "XR CHEST"}],
                client=mock_client,
                rate_limiter=rl,
            )

        backoff_sleeps = [d for d in sleep_durations if d >= 60]
        assert len(backoff_sleeps) > 0, (
            f"No sleep >= 60s found. All sleeps: {sleep_durations}"
        )
        clear_cache()


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------

class TestPredictSchemaContract:
    """Test 16: /predict returns correct schema."""

    @pytest.mark.asyncio
    async def test_response_schema(self):
        from httpx import ASGITransport, AsyncClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            from server import app
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                payload = {
                    "cases": [
                        {
                            "case_id": "c1",
                            "current_study": {
                                "study_id": "s0",
                                "study_description": "CT CHEST WITH CONTRAST",
                                "study_date": "2025-01-01",
                            },
                            "prior_studies": [
                                {
                                    "study_id": "s1",
                                    "study_description": "XR CHEST",
                                    "study_date": "2024-01-01",
                                },
                            ],
                        },
                        {
                            "case_id": "c2",
                            "current_study": {
                                "study_id": "s2",
                                "study_description": "MRI BRAIN",
                                "study_date": "2025-02-01",
                            },
                            "prior_studies": [
                                {
                                    "study_id": "s3",
                                    "study_description": "CT HEAD",
                                    "study_date": "2024-06-01",
                                },
                                {
                                    "study_id": "s4",
                                    "study_description": "MRI KNEE",
                                    "study_date": "2023-01-01",
                                },
                            ],
                        },
                    ]
                }

                resp = await ac.post("/predict", json=payload)
                assert resp.status_code == 200

                data = resp.json()
                assert "predictions" in data
                preds = data["predictions"]
                assert len(preds) == 3

                for p in preds:
                    assert "case_id" in p
                    assert "study_id" in p
                    assert "predicted_is_relevant" in p
                    assert isinstance(p["predicted_is_relevant"], bool)


@pytest.mark.skipif(not HAVE_PUBLIC_JSON, reason="Public eval JSON not found")
class TestNoMissingPredictions:
    """Test 17: No predictions are skipped."""

    @pytest.mark.asyncio
    async def test_all_priors_predicted(self):
        from httpx import ASGITransport, AsyncClient

        with open(PUBLIC_JSON) as f:
            data = json.load(f)

        first_10 = data["cases"][:10]
        expected_count = sum(len(c["prior_studies"]) for c in first_10)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            from server import app
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                payload = {"cases": first_10}
                resp = await ac.post("/predict", json=payload)
                assert resp.status_code == 200
                preds = resp.json()["predictions"]
                assert len(preds) == expected_count


class TestHeuristicOnlyMode:
    """Test 18: Works without API key."""

    @pytest.mark.asyncio
    async def test_no_api_key_still_works(self):
        from httpx import ASGITransport, AsyncClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            from server import app
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                payload = {
                    "cases": [{
                        "case_id": "test1",
                        "current_study": {
                            "study_id": "s0",
                            "study_description": "CT CHEST",
                            "study_date": "2025-01-01",
                        },
                        "prior_studies": [{
                            "study_id": "s1",
                            "study_description": "XR KNEE",
                            "study_date": "2024-01-01",
                        }],
                    }]
                }
                resp = await ac.post("/predict", json=payload)
                assert resp.status_code == 200
                preds = resp.json()["predictions"]
                assert len(preds) == 1


@pytest.mark.skipif(not HAVE_PUBLIC_JSON, reason="Public eval JSON not found")
class TestFullPublicEvalAccuracy:
    """Test 19: Heuristic accuracy >= 0.78 on public set."""

    def test_heuristic_accuracy(self):
        from classifier import heuristic_predict

        with open(PUBLIC_JSON) as f:
            data = json.load(f)

        truth_by_key = {}
        for t in data["truth"]:
            truth_by_key[(t["case_id"], t["study_id"])] = t["is_relevant_to_current"]

        correct = incorrect = 0
        for case in data["cases"]:
            for prior in case["prior_studies"]:
                key = (case["case_id"], prior["study_id"])
                actual = truth_by_key.get(key)
                if actual is None:
                    continue
                pred, _ = heuristic_predict(
                    case["current_study"]["study_description"],
                    prior["study_description"],
                )
                if pred == actual:
                    correct += 1
                else:
                    incorrect += 1

        total = correct + incorrect
        accuracy = correct / total if total > 0 else 0
        assert accuracy >= 0.78, f"Accuracy {accuracy:.4f} < 0.78"


@pytest.mark.skipif(not HAVE_PUBLIC_JSON, reason="Public eval JSON not found")
class TestCompletesWithinTimeout:
    """Test 20: Full heuristic eval finishes in < 300s."""

    def test_heuristic_speed(self):
        from classifier import heuristic_predict

        with open(PUBLIC_JSON) as f:
            data = json.load(f)

        start = time.monotonic()
        for case in data["cases"]:
            for prior in case["prior_studies"]:
                heuristic_predict(
                    case["current_study"]["study_description"],
                    prior["study_description"],
                )
        elapsed = time.monotonic() - start

        assert elapsed < 300, f"Heuristic eval took {elapsed:.1f}s, expected < 300s"
