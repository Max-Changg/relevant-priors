# Challenge Brief

Build and host an HTTP API that receives one current patient examination and a list of previous examinations for the same patient. For each previous examination, return whether it should be shown to the radiologist while reading the current examination. Final submission includes your endpoint URL, a zip of your codebase, and a markdown write-up of experiments plus next-step improvements.

## 1. Rules

- Expose a POST endpoint that accepts the JSON request format shown below.
- Each request contains one or more patient examination cases. Each case has one current examination and a list of previous examinations for the same patient.
- For every previous examination in the request, return exactly one prediction with the same `case_id` and `study_id`.
- If you skip an examination, that missing prediction counts as incorrect.
- The browser quick check only uses 10 public cases. It is a fast contract check, not your final score.
- Download the public eval JSON below and use it for local testing. Final scoring uses a separate private JSON split.
- Your final submission is accepted immediately and graded in the background. Come back in 2–3 minutes for score and feedback.
- Your final submission must include three things: endpoint URL, code zip, and markdown write-up.
- Your final result is based on both private-split accuracy and review of the code zip and write-up.

## 2. Submission Checklist

- HTTP endpoint URL that our evaluator can reach
- Zip file containing the full codebase used by that endpoint
- Markdown write-up describing experiments, results, and next improvements

## 3. Hints

- Log request IDs, case counts, and prior counts so you can debug evaluator calls quickly.
- The evaluator will stop waiting after 360 seconds.
- If you use an LLM or external service, send all previous examinations for a case in one batched call instead of one call per examination.
- One LLM call per previous examination will usually time out on the private evaluation.
- Cache repeated work so retries and repeated study pairs do not trigger the same expensive call again.
- If the hidden evaluation times out or fails, fix the endpoint and submit again.
- Return `true` or `false` for `predicted_is_relevant`.

## Accuracy Definition

```
accuracy = correct_predictions / (correct_predictions + incorrect_predictions)
```

Where skipped previous examinations count as incorrect.

## Public Local Eval
@relevant_priors_public.json
Use this labeled public split for local iteration and debugging. The final leaderboard is based on the separate hidden private split.

- **996 cases**
- **27,614 labeled previous exams**

## API Request Schema

```json
{
  "challenge_id": "relevant-priors-v1",
  "schema_version": 1,
  "generated_at": "2026-04-16T12:00:00.000Z",
  "cases": [
    {
      "case_id": "1001016",
      "patient_id": "606707",
      "patient_name": "Andrews, Micheal",
      "current_study": {
        "study_id": "3100042",
        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "study_date": "2026-03-08"
      },
      "prior_studies": [
        {
          "study_id": "2453245",
          "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
          "study_date": "2020-03-08"
        },
        {
          "study_id": "992654",
          "study_description": "CT HEAD WITHOUT CNTRST",
          "study_date": "2021-03-08"
        }
      ]
    }
  ]
}
```

## API Response Schema

```json
{
  "predictions": [
    {
      "case_id": "1001016",
      "study_id": "2453245",
      "predicted_is_relevant": true
    },
    {
      "case_id": "1001016",
      "study_id": "992654",
      "predicted_is_relevant": false
    }
  ]
}
```

- Each case includes one current patient examination and a list of previous examinations.
- Return one prediction for every previous examination in the request.
- Use bulk inference: a single evaluator request may contain many cases and many previous examinations.
