"""
FastAPI server for the Relevant Priors challenge.

POST /predict — accepts cases, returns relevance predictions.
Two-pass approach: heuristic on all cases first, then batch LLM for uncertain pairs.
"""

import logging
import os
import time
import uuid
from typing import Optional

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

from classifier import heuristic_predict
from llm_classifier import classify_pairs_batch, get_cache_stats, get_rate_limiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Relevant Priors API")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))


# --- Request/Response Models ---

class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class Case(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    current_study: Study
    prior_studies: list[Study]

class PredictRequest(BaseModel):
    challenge_id: Optional[str] = None
    schema_version: Optional[int] = None
    generated_at: Optional[str] = None
    cases: list[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class PredictResponse(BaseModel):
    predictions: list[Prediction]


# --- Endpoint ---

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())[:8]
    total_priors = sum(len(c.prior_studies) for c in request.cases)
    logger.info(
        f"[{request_id}] Received request: {len(request.cases)} cases, "
        f"{total_priors} total priors"
    )
    start = time.monotonic()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_llm = bool(api_key)

    if not use_llm:
        logger.warning(f"[{request_id}] No ANTHROPIC_API_KEY set, using heuristic only")

    predictions: list[Prediction] = []
    uncertain: list[tuple[str, str, str, str]] = []

    # Phase 1: Heuristic pass on all priors
    for case in request.cases:
        for prior in case.prior_studies:
            pred, conf = heuristic_predict(
                case.current_study.study_description,
                prior.study_description,
                case.current_study.study_date,
                prior.study_date,
            )
            if not use_llm or conf >= CONFIDENCE_THRESHOLD:
                predictions.append(Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=pred,
                ))
            else:
                uncertain.append((
                    case.case_id,
                    prior.study_id,
                    case.current_study.study_description,
                    prior.study_description,
                ))

    # Phase 2: Batch LLM for uncertain pairs
    if uncertain and use_llm:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        unique_pairs = list({(cdesc, pdesc) for _, _, cdesc, pdesc in uncertain})

        logger.info(
            f"[{request_id}] {len(uncertain)} uncertain priors, "
            f"{len(unique_pairs)} unique pairs for LLM"
        )

        pair_results = await classify_pairs_batch(
            unique_pairs, client, get_rate_limiter(),
        )

        for case_id, study_id, cdesc, pdesc in uncertain:
            key = (cdesc.strip().lower(), pdesc.strip().lower())
            if key in pair_results:
                pred_val = pair_results[key]
            else:
                pred_val, _ = heuristic_predict(cdesc, pdesc)
            predictions.append(Prediction(
                case_id=case_id,
                study_id=study_id,
                predicted_is_relevant=pred_val,
            ))

    elapsed = time.monotonic() - start
    cache_stats = get_cache_stats()
    logger.info(
        f"[{request_id}] Done in {elapsed:.2f}s: {len(predictions)} predictions, "
        f"uncertain={len(uncertain)}, cache={cache_stats}"
    )

    return PredictResponse(predictions=predictions)


@app.get("/health")
async def health():
    return {"status": "ok", "cache": get_cache_stats()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
