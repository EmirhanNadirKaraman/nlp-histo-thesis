### MAP stage

For the MAP stage of the summarization pipeline, we have a lot of thresholds that we should fine-tune our model. There is a randomly selected dataset consisting of 15 papers, and the fine-tuning is done on that dataset. 

In the SummarizationConfig, we already have:

map.theta: Accept threshold
map.reject_theta: Reject threshold
map.chunk_size: Chunk window (sentences per LLM call)
map.chunk_overlap: Sliding-window overlap

routing.enable_router: False = legacy L1→L2→L3; True = router L1→L3
routing.router_single_voter_policy: Router N=1 handling. 'escalate' (default) → L3; 'keep' → accept lone vetted voter.
routing.legacy_single_voter_policy: Legacy AgreementChecker N=1 handling. 'keep' (default; preserves prior implicit
                                    behaviour — synthetic confidence=1.0) vs 'escalate' (treat N=1 as low-evidence
                                    and route up the cascade). Done 2026-05-26.

> **Config layout v2 (2026-05-26).** The three routing-policy fields above moved from `MapConfig` into a dedicated
> `RoutingConfig` because they are agreement-decision parameters, not MAP extraction parameters (answers the inline
> question: "is it possible to have a router config with the single_voter_policy separately so it looks cleaner?"
> — yes, done as a pure refactor with no behaviour change). Defaults preserved verbatim; strict YAML loader rejects
> the v1 paths (`summarization.map.enable_router` etc.) with a clear error. See `STRUCTURE.md` pipeline changelog
> 2026-05-26 + `THESIS.md` Decisions log.
--------------------------------------------------------------------------------

#### AgreementConfig

These parameters control how agreement between MAP voters is scored.

agreement.tau:
Pairwise similarity values below this threshold are treated as zero before computing coverage. Higher tau makes agreement stricter.

agreement.count_alpha:
Penalty strength for voter outputs with very different finding counts. This helps prevent one voter emitting many findings while another emits one broad finding from looking artificially aligned.

agreement.reuse_weight:
Penalty for concentrated reuse, where one finding is reused to match many findings from another voter. Higher values punish many-to-one matching more strongly.

agreement.contradiction_weight:
Penalty applied when comparable findings appear to disagree in polarity/numeric direction.

> **Wiring (verified 2026-05-26):** all four fields flow `config.py` → `configs/run.yaml` (lines 128–132) → `SummarizationRunner.__init__` / `BatchSummarizationRunner.__init__` → `EmbeddingSimilarityStrategy.from_config(cfg.agreement)`. No hardcoded overrides on the path. Tests pin defaults + YAML round-trip + override + `from_config` propagation (`tests/test_config_loader.py::test_agreement_*`). The calibration sweep (`map_theta_sweep.py`, `run_summarization_sweeps.py`) builds `AgreementConfig()` with dataclass defaults explicitly so the sweep grids are independent of the production YAML — this is by design (sweeps iterate over weight variants).


--------------------------------------------------------------------------------

### TODO: these are not in the config, promote to a config field

~~Scorer kind: EmbeddingSimilarityStrategy vs HybridStructuredSimilarity~~ **Done 2026-05-26.**
  → Now at `summarization.agreement.scorer_kind` (`embedding` | `hybrid`, default `embedding`).
  Dispatched by `SemanticAgreementScorer.from_agreement_config` across both runners.
  Stage 1 of the MAP sweep (`run_summarization_sweeps --stage map_scorer`) now pins
  the winner via this YAML field; the harness's `BEST_SCORER` constant is the
  pre-pin staging area.
Voter profile: real / cheap. we can continue with real.

Embedder: OpenAI text-embedding-3-small / Gemini. The embeddings are currently being generated using Gemini. 

max_tokens: Currently 16384. Shorter might cause problems about not generating the full correct output, longer might mean the models keep generating if there is a bug on the LLM side, so no early exit. 

--------------------------------------------------------------------------------

### The following are not in the SummarizationConfig either, promote these to a config field

grounding_floor: Default = 0.50. Post-embedding multiply if a finding's grounding-pass fraction is low. Only fires when an AgreementContext is supplied (router path)

#### TODO: check 

Hybrid blend weights: 

w_category / w_embedding / w_entity / w_evidence
Default = 0.25 / 0.40 / 0.25 / 0.10

#### TODO: check and think about how to find the optimal weights 

~~Polarity Conflict: Escalate iff {positive, negative} comparable pair. These trigger forced escalation.~~ **Done 2026-05-26.**
  → Now at `summarization.agreement.force_escalate_on_polarity_conflict` (`true` | `false`, default `true`).
  Implemented as `AgreementChecker.force_escalate_on_polarity_conflict`. Marker is **always** recorded
  in `score_details["hard_fail_reason"]` regardless of flag value (so sweep harnesses can count chunks
  even when the override doesn't fire); the decision override to `ChunkDecision.ESCALATE` is the gated
  part. Stage 3b `--stage map_polarity_flag` in `run_summarization_sweeps` runs the 2-cell ablation;
  CSV gains `n_polarity_conflict_chunks` + `polarity_conflict_rate`. Default-flip is a *safety* decision,
  not a tuning one — polarity overrides exist for semantic safety (B-051), not cost control.

--------------------------------------------------------------------------------

## In what order do we do the experiments now?

The questions we have for this part are as follows: 

### EXP 1 — Tuned scorer comparison under the default embedder

Default embedder: Gemini

Question:
Which agreement scoring strategy gives the best MAP-stage performance under the current Gemini embedding setup?

Compared strategies:
1. EmbeddingSimilarityStrategy
2. HybridStructuredSimilarity with multiple blend-weight variants:
   - hybrid_default
   - hybrid_balanced
   - hybrid_embedding_heavy
   - hybrid_category_heavy
   - hybrid_entity_heavy
   - hybrid_evidence_heavy

Method:
For each candidate scorer configuration:
1. Run the same θ / reject_θ grid.
2. Select that candidate's best θ / reject_θ pair by strict_f1.
3. Compare the best tuned row from each scorer configuration.

Output:
- BEST_GEMINI_SCORER
- BEST_GEMINI_HYBRID_WEIGHTS, if hybrid wins
- BEST_GEMINI_THETA
- BEST_GEMINI_REJECT_THETA


### EXP 2 — Agreement soft-weight sweep under the selected Gemini config

Question:
Can tau / count_alpha / reuse_weight / contradiction_weight improve the selected Gemini-based scorer configuration?

Method:
Run a local sweep around the selected scorer configuration from EXP 1.

Output:
- BEST_GEMINI_AGREEMENT_WEIGHTS


### EXP 3 — Polarity conflict escalation ablation under the selected Gemini config

Question:
Does forced escalation on polarity conflict improve results?

Compare:
- Best Gemini config with force_escalate_on_polarity_conflict = true
- Best Gemini config with force_escalate_on_polarity_conflict = false

Output:
- BEST_GEMINI_POLARITY_FLAG
- FINAL_GEMINI_MAP_CONFIG

### EXP 4-6 - Same experiments with OpenAI 

### EXP 7 — Embedder branch comparison: Gemini vs OpenAI

Question:
Does OpenAI or Gemini give better MAP-stage performance after each embedder is allowed to use its own calibrated configuration?

Important:
Do not compare OpenAI using Gemini-tuned θ / reject_θ. Switching embedder changes similarity geometry, so each embedder needs its own calibration branch.

Branch A — Gemini:
Use FINAL_GEMINI_MAP_CONFIG from EXP 1–3.

Branch B — OpenAI:
Repeat the same calibration procedure with OpenAI embeddings:
1. Tuned scorer comparison
2. Agreement soft-weight sweep, if needed
3. Polarity conflict ablation, if needed

Final comparison:
- FINAL_GEMINI_MAP_CONFIG
- FINAL_OPENAI_MAP_CONFIG

Decision rule:
Choose the embedder whose best calibrated configuration gives the best strict_f1.
If statistically tied, choose the cheaper / faster / more stable embedder.

Output:
- BEST_EMBEDDER
- FINAL_MAP_CONFIG


So the experiment path will look like this: 

Gemini branch:
  EXP 1 → EXP 2 → EXP 3 → FINAL_GEMINI_MAP_CONFIG

OpenAI branch:
  EXP 4 → EXP 5 → EXP 6 → FINAL_OPENAI_MAP_CONFIG

EXP 7:
  FINAL_GEMINI_MAP_CONFIG vs FINAL_OPENAI_MAP_CONFIG


----------------------------------------------------------------------------------------

### EXP A — Bootstrap confidence intervals

Run after:
- map_scorer
- map_hybrid_blend, if used
- map_theta
- map_polarity_flag

Purpose:
Check whether the selected winner from each sweep is meaningfully better than nearby alternatives.

The key phrase for your notes is:

Bootstrap is not a new tuning axis. It is an uncertainty check around the sweep winner.

### EXP B — Agreement-Based Cascading

This experiment evaluates whether agreement-based cascading is useful as a routing mechanism and whether it provides a better cost-quality tradeoff than simpler baselines.

---

#### EXP B.1 — Routing usefulness: ABC vs matched random escalation

Question:
Does agreement choose better chunks to escalate than chance?

Comparison:
- Agreement-based cascade
- Matched random cascade

Matched random cascade:
Randomly escalates the same number / fraction of chunks as the ABC cascade.

Decision rule:
If ABC achieves higher strict F1 than matched random escalation at the same escalation rate, then agreement is useful as a routing signal.

Purpose:
This tests the routing mechanism itself, not just whether stronger models improve performance.

---

#### EXP B.2 — Cost-quality comparison: cascade vs cheap-only vs strong-model-only

Question:
Does cascading achieve similar F1 to a stronger model at lower cost?

Comparison:
1. Cheap-only baseline
   - Use only L1 cheap voters.
   - No escalation.
   - Measures how much the cascade improves over the cheapest setup.

2. Agreement-based cascade
   - The actual proposed system.
   - Uses agreement to decide which chunks escalate.

3. Strong-model-only baseline
   - Use Sonnet / L3 / strongest model for every chunk.
   - Measures the quality ceiling and the cost of using the strong model everywhere.

Decision rule:
If the cascade reaches similar strict F1 to strong-model-only at lower estimated cost, then cascading is cost-effective.

---

#### Optional baseline

4. Matched random cascade
   - Also useful in the cost-quality table, but its main role is EXP 8.1.
   - Same escalation rate as ABC, randomly selected chunks.


##### Caveats:
Strong-model-only is only valid if strong-model outputs exist for every chunk. If the voter cache contains Sonnet/L3 outputs only for chunks that ABC escalated, then we cannot compute a fair Sonnet-only baseline from the existing cache. In that case, we either need a separate all-Sonnet baseline run or we should not report Sonnet-only results.

Matched random escalation should be repeated over many random seeds and reported as mean ± confidence interval, because one random sample is not meaningful.


### EXP C — Agreement score vs accuracy

Purpose:

Check whether agreement score actually predicts correctness.

Method:

Bin chunks by agreement score:
  low agreement
  medium agreement
  high agreement

Then measure silver-match quality in each bin.

Expected thesis claim if it works:

Higher voter agreement correlates with higher extraction correctness, supporting agreement as a confidence signal.

This is important because it validates the premise behind cascading.


### EXP D — Matcher threshold sensitivity

#### Question

Are the MAP sweep conclusions stable under reasonable changes to the silver-matching similarity threshold?

The MAP sweep evaluates pipeline findings by matching them against silver findings using an embedding similarity threshold. If the chosen threshold is too influential, the selected scorer or θ/reject_θ pair may be an artifact of the evaluation setup rather than a genuinely better configuration.

#### Method

Repeat scoring / replay with several nearby matcher thresholds:

- sim_threshold = 0.50
- sim_threshold = 0.55
- sim_threshold = 0.60

For each threshold, compare the ranking of sweep cells:

- scorer_kind
- hybrid weights, if used
- theta / reject_theta
- polarity conflict flag

#### Metrics

Primary:
- strict_f1

Secondary:
- f1
- precision
- recall
- selected winner stability

#### Decision rule

If the same or very similar configurations remain near the top across matcher thresholds, the result is robust.

If different configurations win at different matcher thresholds, then the final selection should be described as sensitive to the evaluation matcher, and the chosen operating point should be selected more conservatively.

#### Interpretation

Stable result:
The same scorer and threshold region performed best across sim_threshold values 0.50–0.60, suggesting that the MAP configuration choice is not an artifact of the exact matcher threshold.

Unstable result:
The winning configuration changed across matcher thresholds, so we treat the MAP winner as uncertain and report this sensitivity as a limitation.


### EXP E — Recall-Gap / Missed-Finding Audit

#### Question

What kinds of findings does the selected MAP configuration miss, and are these misses true extraction failures or artifacts of the silver labels / matcher?

This experiment explains the false negatives behind the MAP-stage F1 score.

---

#### When to run

Run this after the final MAP configuration has been selected:

1. scorer_kind selected
2. hybrid weights selected, if hybrid is used
3. theta / reject_theta selected
4. agreement weights selected, if tuned
5. polarity conflict flag decided

This is not a tuning experiment. It is a diagnostic analysis of the chosen MAP configuration.

---

#### Inputs

- `eval/data/silver_findings.jsonl`
  - silver/reference findings
- selected MAP replay output
  - pipeline findings for the chosen MAP configuration
- per-case matching output
  - which silver findings matched pipeline findings
- `eval/data/source_cases.jsonl`
  - source text for manual inspection

---

#### Main object of analysis

False negatives:

A false negative is a silver finding that was not matched by any pipeline finding under the selected matcher threshold.

For each false negative, inspect:
- the source text
- the silver finding
- the closest pipeline finding, if any
- the similarity score to the closest pipeline finding
- all pipeline findings for that case

---

#### Manual classification categories

Each missed finding should be assigned one category:

1. `true_miss`
   - The silver finding is valid and supported by the source text, but the pipeline did not extract it.

2. `too_broad_pipeline`
   - The pipeline extracted a related finding, but it was too broad or vague to match the silver finding.

3. `too_specific_silver`
   - The silver finding is more specific than what the source text clearly supports.

4. `category_mismatch`
   - The pipeline extracted the same idea under the wrong category.

5. `entity_mismatch`
   - The pipeline extracted a related claim, but with the wrong subject or outcome entity.

6. `polarity_mismatch`
   - The pipeline found the relevant claim but got the direction/polarity wrong.

7. `matcher_failure`
   - The pipeline finding is semantically equivalent, but the matcher failed to match it.

8. `unsupported_silver`
   - The silver finding is not actually supported by the source text.

9. `atomicity_mismatch`
   - The pipeline and silver split/merge the same information differently.

10. `unclear`
   - Cannot confidently classify.

---

#### Sampling plan

Audit a manageable sample, not necessarily all false negatives.

Suggested sample:
- 50 false negatives total
- include near-threshold misses
- include far misses
- include random misses

Near-threshold misses are especially useful because they reveal matcher sensitivity.

---

#### Output

Create a CSV or spreadsheet with columns:

- `case_id`
- `source_text`
- `silver_finding`
- `closest_pipeline_finding`
- `closest_similarity`
- `all_pipeline_findings`
- `miss_category`
- `notes`
- `reviewer`

Final report:

| Miss category | Count | Percent | Interpretation |
|---|---:|---:|---|
| true_miss | ... | ... | MAP recall limitation |
| atomicity_mismatch | ... | ... | schema/matching issue |
| unsupported_silver | ... | ... | silver-label noise |
| matcher_failure | ... | ... | evaluation artifact |

---

#### Decision / interpretation

If most false negatives are `true_miss`, then the MAP extractor has a genuine recall problem.

If many false negatives are `matcher_failure` or `atomicity_mismatch`, then the reported recall may underestimate semantic coverage.

If many false negatives are `unsupported_silver` or `too_specific_silver`, then the silver labels are noisy and recall estimates should be interpreted cautiously.

If many false negatives are `category_mismatch`, `entity_mismatch`, or `polarity_mismatch`, then the extraction schema or normalization logic needs improvement.

---

#### Thesis use

This experiment turns the raw F1 result into an error analysis.

Example thesis wording:

A manual audit of unmatched silver findings showed that the main sources of recall loss were true missed findings, atomicity mismatches, and matcher failures. This suggests that part of the measured recall gap reflects genuine MAP-stage limitations, while another part reflects evaluation and schema-alignment artifacts.


--------

## DEFERRED: 

### EXP 3 - Grounding floor in GroundingConfig, which value is the best?