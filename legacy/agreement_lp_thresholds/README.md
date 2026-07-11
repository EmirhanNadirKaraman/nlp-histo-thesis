# Quarantined: agreement LP-threshold branch

Superseded, unreferenced scoring code moved here on the repository-reorganization
branch (post thesis-submission tag). All of it had **zero constructor calls**
anywhere in the live tree, tests, eval, or scripts:

- `composite.py` — `CascadedCompositeScorer` (LP-thresholded cascade)
- `hybrid_scorer.py` — `HybridScorer` (fed only the composite scorer)
- `llm_judge.py` — `LLMJudgeScorer` (never instantiated)
- `calibration/` — `dataset.py` / `gold_labeler.py` / `threshold_optimizer.py`
  (offline LP-threshold fitting; reachable only from `composite.py`)

The live agreement path is `SemanticAgreementScorer` + `AgreementChecker` with
config-selected `EmbeddingSimilarityStrategy` / `HybridStructuredSimilarity`.
Retained for provenance; not imported at runtime.
