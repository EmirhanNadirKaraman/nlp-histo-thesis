## In what order do we do the MAP-stage experiments now?

The MAP-stage experiments are organized into two calibration branches:

1. Gemini branch — the current/default embedder.
2. OpenAI branch — the same calibration procedure repeated with OpenAI embeddings.

The final comparison is between the best fully calibrated Gemini configuration and the best fully calibrated OpenAI configuration.

Important:
Do not compare OpenAI using Gemini-tuned θ / reject_θ values. Switching the embedder changes similarity geometry, so each embedder needs its own calibrated operating point.

---

## Core MAP calibration

### Gemini branch

### EXP 1 — Tuned scorer comparison under Gemini

Default embedder: Gemini

#### Question

Which agreement scoring strategy gives the best MAP-stage performance under the current Gemini embedding setup, after each strategy is allowed to use its own best θ / reject_θ setting?

#### Compared scorer configurations

1. EmbeddingSimilarityStrategy

2. HybridStructuredSimilarity with multiple blend-weight variants:
   - hybrid_default
   - hybrid_balanced
   - hybrid_embedding_heavy
   - hybrid_category_heavy
   - hybrid_entity_heavy
   - hybrid_evidence_heavy

#### Inputs

- `eval/data/source_cases.jsonl`
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- Gemini embedding cache
- θ / reject_θ grid
- scorer configurations:
  - embedding
  - hybrid variants

#### Method

For each candidate scorer configuration:

1. Run the same θ / reject_θ grid.
2. Select that candidate's best θ / reject_θ pair by strict_f1.
3. Compare the best tuned row from each scorer configuration.

This avoids unfairly comparing scorers at a threshold pair that may be optimal for only one scorer.

#### Metrics

Primary:
- strict_f1

Secondary:
- f1
- precision
- recall
- escalation_rate / cost proxy

#### Output

- BEST_GEMINI_SCORER
- BEST_GEMINI_HYBRID_WEIGHTS, if hybrid wins
- BEST_GEMINI_THETA
- BEST_GEMINI_REJECT_THETA
- Gemini scorer-comparison sweep CSV

---

### EXP 2 — Agreement soft-weight sweep under the selected Gemini config

#### Question

Can tau / count_alpha / reuse_weight / contradiction_weight improve the selected Gemini-based scorer configuration?

#### Inputs

- BEST_GEMINI_SCORER
- BEST_GEMINI_HYBRID_WEIGHTS, if hybrid won EXP 1
- BEST_GEMINI_THETA
- BEST_GEMINI_REJECT_THETA
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- Gemini embedding cache
- agreement soft-weight grid:
  - tau
  - count_alpha
  - reuse_weight
  - contradiction_weight

#### Method

Run a local sweep around the selected scorer configuration from EXP 1.

#### Decision rule

Adopt a non-default agreement soft-weight setting only if it improves strict_f1 meaningfully without causing excessive escalation/cost.

#### Output

- BEST_GEMINI_AGREEMENT_WEIGHTS
- Updated BEST_GEMINI_SCORER_CONFIG
- Agreement soft-weight sweep CSV

---

### EXP 3 — Polarity conflict escalation ablation under the selected Gemini config

#### Question

Does forced escalation on polarity conflict improve results?

#### Inputs

- BEST_GEMINI_SCORER_CONFIG from EXP 1–2
- BEST_GEMINI_THETA
- BEST_GEMINI_REJECT_THETA
- BEST_GEMINI_AGREEMENT_WEIGHTS
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- Gemini embedding cache

#### Compared settings

1. Best Gemini config with:
   - force_escalate_on_polarity_conflict = true

2. Best Gemini config with:
   - force_escalate_on_polarity_conflict = false

#### Metrics

Primary:
- strict_f1

Secondary:
- precision
- recall
- f1
- escalation_rate
- n_polarity_conflict_chunks
- polarity_conflict_rate

#### Decision rule

Keep the flag enabled unless disabling it clearly improves strict_f1 without harming semantic safety. This flag is partly a safety mechanism, not only a tuning knob.

#### Output

- BEST_GEMINI_POLARITY_FLAG
- FINAL_GEMINI_MAP_CONFIG
- Polarity-flag ablation CSV

---

## OpenAI branch

The OpenAI branch repeats the same calibration procedure as the Gemini branch, but with OpenAI embeddings.

Important:
OpenAI must get its own θ / reject_θ calibration. Do not reuse Gemini-tuned thresholds.

---

### EXP 4 — Tuned scorer comparison under OpenAI

Embedder: OpenAI

#### Question

Which agreement scoring strategy gives the best MAP-stage performance under the OpenAI embedding setup, after each strategy is allowed to use its own best θ / reject_θ setting?

#### Compared scorer configurations

1. EmbeddingSimilarityStrategy

2. HybridStructuredSimilarity with multiple blend-weight variants:
   - hybrid_default
   - hybrid_balanced
   - hybrid_embedding_heavy
   - hybrid_category_heavy
   - hybrid_entity_heavy
   - hybrid_evidence_heavy

#### Inputs

- `eval/data/source_cases.jsonl`
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- OpenAI embedding cache
- θ / reject_θ grid
- scorer configurations:
  - embedding
  - hybrid variants

#### Method

For each candidate scorer configuration:

1. Run the same θ / reject_θ grid.
2. Select that candidate's best θ / reject_θ pair by strict_f1.
3. Compare the best tuned row from each scorer configuration.

#### Metrics

Primary:
- strict_f1

Secondary:
- f1
- precision
- recall
- escalation_rate / cost proxy

#### Output

- BEST_OPENAI_SCORER
- BEST_OPENAI_HYBRID_WEIGHTS, if hybrid wins
- BEST_OPENAI_THETA
- BEST_OPENAI_REJECT_THETA
- OpenAI scorer-comparison sweep CSV

---

### EXP 5 — Agreement soft-weight sweep under the selected OpenAI config

#### Question

Can tau / count_alpha / reuse_weight / contradiction_weight improve the selected OpenAI-based scorer configuration?

#### Inputs

- BEST_OPENAI_SCORER
- BEST_OPENAI_HYBRID_WEIGHTS, if hybrid won EXP 4
- BEST_OPENAI_THETA
- BEST_OPENAI_REJECT_THETA
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- OpenAI embedding cache
- agreement soft-weight grid:
  - tau
  - count_alpha
  - reuse_weight
  - contradiction_weight

#### Method

Run a local sweep around the selected scorer configuration from EXP 4.

#### Decision rule

Adopt a non-default agreement soft-weight setting only if it improves strict_f1 meaningfully without causing excessive escalation/cost.

#### Output

- BEST_OPENAI_AGREEMENT_WEIGHTS
- Updated BEST_OPENAI_SCORER_CONFIG
- Agreement soft-weight sweep CSV

---

### EXP 6 — Polarity conflict escalation ablation under the selected OpenAI config

#### Question

Does forced escalation on polarity conflict improve results under the OpenAI-based configuration?

#### Inputs

- BEST_OPENAI_SCORER_CONFIG from EXP 4–5
- BEST_OPENAI_THETA
- BEST_OPENAI_REJECT_THETA
- BEST_OPENAI_AGREEMENT_WEIGHTS
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- OpenAI embedding cache

#### Compared settings

1. Best OpenAI config with:
   - force_escalate_on_polarity_conflict = true

2. Best OpenAI config with:
   - force_escalate_on_polarity_conflict = false

#### Metrics

Primary:
- strict_f1

Secondary:
- precision
- recall
- f1
- escalation_rate
- n_polarity_conflict_chunks
- polarity_conflict_rate

#### Output

- BEST_OPENAI_POLARITY_FLAG
- FINAL_OPENAI_MAP_CONFIG
- Polarity-flag ablation CSV

---

## Final embedder comparison

### EXP 7 — Embedder branch comparison: Gemini vs OpenAI

#### Question

Does Gemini or OpenAI give better MAP-stage performance after each embedder is allowed to use its own calibrated configuration?

#### Inputs

- FINAL_GEMINI_MAP_CONFIG
- FINAL_OPENAI_MAP_CONFIG
- Gemini branch sweep outputs
- OpenAI branch sweep outputs
- optional bootstrap confidence intervals for top configurations

#### Compared configurations

1. FINAL_GEMINI_MAP_CONFIG
   - selected from EXP 1–3

2. FINAL_OPENAI_MAP_CONFIG
   - selected from EXP 4–6

#### Metrics

Primary:
- strict_f1

Secondary:
- f1
- precision
- recall
- escalation_rate / cost proxy
- runtime / cache behavior, if relevant

#### Decision rule

Choose the embedder whose best calibrated configuration gives the best strict_f1.

If the two configurations are statistically tied, choose the cheaper / faster / more stable embedder.

#### Output

- BEST_EMBEDDER
- FINAL_MAP_CONFIG
- Final branch-comparison table

---

## Overall experiment path

Gemini branch:

EXP 1 → EXP 2 → EXP 3 → FINAL_GEMINI_MAP_CONFIG

OpenAI branch:

EXP 4 → EXP 5 → EXP 6 → FINAL_OPENAI_MAP_CONFIG

Final comparison:

EXP 7 → FINAL_GEMINI_MAP_CONFIG vs FINAL_OPENAI_MAP_CONFIG → FINAL_MAP_CONFIG

---

## Validation and analysis experiments

These experiments are not new tuning axes. They validate, stress-test, or explain the selected MAP configuration.

---

### EXP A — Bootstrap confidence intervals

#### Purpose

Check whether the selected winner from each sweep is meaningfully better than nearby alternatives.

Bootstrap is not a new tuning axis. It is an uncertainty check around the sweep winner.

#### Run after

- EXP 1 / EXP 4 scorer-comparison sweeps
- EXP 2 / EXP 5 agreement soft-weight sweeps, if used
- EXP 3 / EXP 6 polarity-flag ablations
- EXP 7 final embedder comparison

#### Inputs

- sweep CSVs
- per-case matching outputs
- `eval/data/silver_findings.jsonl`
- `eval/data/source_cases.jsonl`

#### Output

- confidence intervals for strict_f1 / f1 / precision / recall
- list of statistically tied candidates
- final recommendation using lower escalation / lower cost as tie-breaker

---

### EXP B — Agreement-Based Cascading

This experiment evaluates whether agreement-based cascading is useful as a routing mechanism and whether it provides a better cost-quality tradeoff than simpler baselines.

---

#### EXP B.1 — Routing usefulness: ABC vs matched random escalation

##### Question

Does agreement choose better chunks to escalate than chance?

##### Inputs

- FINAL_MAP_CONFIG
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- selected embedding cache
- ABC cascade replay outputs
- random seeds

##### Comparison

- Agreement-based cascade
- Matched random cascade

Matched random cascade:
Randomly escalates the same number / fraction of chunks as the ABC cascade.

##### Decision rule

If ABC achieves higher strict_f1 than matched random escalation at the same escalation rate, then agreement is useful as a routing signal.

##### Output

- ABC-vs-random comparison CSV
- mean ± confidence interval over random seeds
- routing-usefulness conclusion

---

#### EXP B.2 — Cost-quality comparison: cascade vs cheap-only vs strong-model-only

##### Question

Does cascading achieve similar F1 to a stronger model at lower cost?

##### Inputs

- FINAL_MAP_CONFIG
- `eval/data/silver_findings.jsonl`
- `eval/data/map_primer/voter_cache.json`
- model cost table / cost proxy
- strong-model outputs for every chunk, if available

##### Comparison

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

4. Optional: matched random cascade
   - Useful in the cost-quality table, but its main role is EXP B.1.

##### Caveats

Strong-model-only is only valid if strong-model outputs exist for every chunk. If the voter cache contains Sonnet/L3 outputs only for chunks that ABC escalated, then we cannot compute a fair Sonnet-only baseline from the existing cache. In that case, we either need a separate all-Sonnet baseline run or we should not report Sonnet-only results.

Matched random escalation should be repeated over many random seeds and reported as mean ± confidence interval, because one random sample is not meaningful.

##### Decision rule

If the cascade reaches similar strict_f1 to strong-model-only at lower estimated cost, then cascading is cost-effective.

##### Output

- cost-quality comparison table
- strict_f1 vs estimated cost
- conclusion about whether cascading is cost-effective

---

### EXP C — Agreement score vs accuracy

#### Purpose

Check whether agreement score actually predicts correctness.

#### Inputs

- FINAL_MAP_CONFIG
- per-chunk agreement scores
- per-case / per-chunk matching outputs
- `eval/data/silver_findings.jsonl`

#### Method

Bin chunks by agreement score:

- low agreement
- medium agreement
- high agreement

Then measure silver-match quality in each bin.

#### Output

- agreement-bin table
- strict_f1 / precision / recall per bin
- conclusion about whether agreement is a useful confidence signal

#### Expected thesis claim if it works

Higher voter agreement correlates with higher extraction correctness, supporting agreement as a confidence signal.

---

### EXP D — Matcher threshold sensitivity

#### Question

Are the MAP sweep conclusions stable under reasonable changes to the silver-matching similarity threshold?

The MAP sweep evaluates pipeline findings by matching them against silver findings using an embedding similarity threshold. If the chosen threshold is too influential, the selected scorer or θ / reject_θ pair may be an artifact of the evaluation setup rather than a genuinely better configuration.

#### Inputs

- FINAL_MAP_CONFIG
- top candidate configurations from EXP 1–7
- per-case matching outputs or replay outputs
- `eval/data/silver_findings.jsonl`
- matcher thresholds:
  - sim_threshold = 0.50
  - sim_threshold = 0.55
  - sim_threshold = 0.60

#### Method

Repeat scoring / replay with several nearby matcher thresholds.

For each threshold, compare the ranking of sweep cells:

- scorer_kind
- hybrid weights, if used
- theta / reject_theta
- polarity conflict flag
- embedder branch, if EXP 7 is run

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

#### Output

- matcher-threshold sensitivity table
- winner-stability summary
- robustness conclusion

---

### EXP E — Recall-Gap / Missed-Finding Audit

#### Question

What kinds of findings does the selected MAP configuration miss, and are these misses true extraction failures or artifacts of the silver labels / matcher?

This experiment explains the false negatives behind the MAP-stage F1 score.

#### When to run

Run this after FINAL_MAP_CONFIG has been selected.

This is not a tuning experiment. It is a diagnostic analysis of the chosen MAP configuration.

#### Inputs

- `eval/data/silver_findings.jsonl`
  - silver/reference findings
- selected MAP replay output
  - pipeline findings for the chosen MAP configuration
- per-case matching output
  - which silver findings matched pipeline findings
- `eval/data/source_cases.jsonl`
  - source text for manual inspection

#### Main object of analysis

False negatives:

A false negative is a silver finding that was not matched by any pipeline finding under the selected matcher threshold.

For each false negative, inspect:
- the source text
- the silver finding
- the closest pipeline finding, if any
- the similarity score to the closest pipeline finding
- all pipeline findings for that case

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

#### Sampling plan

Audit a manageable sample, not necessarily all false negatives.

Suggested sample:
- 50 false negatives total
- include near-threshold misses
- include far misses
- include random misses

Near-threshold misses are especially useful because they reveal matcher sensitivity.

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

#### Decision / interpretation

If most false negatives are `true_miss`, then the MAP extractor has a genuine recall problem.

If many false negatives are `matcher_failure` or `atomicity_mismatch`, then the reported recall may underestimate semantic coverage.

If many false negatives are `unsupported_silver` or `too_specific_silver`, then the silver labels are noisy and recall estimates should be interpreted cautiously.

If many false negatives are `category_mismatch`, `entity_mismatch`, or `polarity_mismatch`, then the extraction schema or normalization logic needs improvement.

#### Thesis use

This experiment turns the raw F1 result into an error analysis.

Example thesis wording:

A manual audit of unmatched silver findings showed that the main sources of recall loss were true missed findings, atomicity mismatches, and matcher failures. This suggests that part of the measured recall gap reflects genuine MAP-stage limitations, while another part reflects evaluation and schema-alignment artifacts.

---

## Deferred experiments

### Deferred EXP — Grounding floor

Question:
Which grounding_floor value is best?

Reason for deferral:
`grounding_floor` is structurally inactive under the current legacy path because `AgreementContext` is not supplied to the agreement scorer. This experiment only becomes meaningful after either:

1. the router path becomes production, or
2. AgreementContext is plumbed into the legacy path.

### Deferred EXP — max_tokens

Question:
Does changing max_tokens improve MAP extraction quality or failure behavior?

Reason for deferral:
Changing max_tokens affects generated voter outputs and must be included in the voter cache/hash. This is not part of the current replay-based MAP sweep.

### Deferred EXP — voter profile cheap vs real

Question:
Should the MAP cascade use the cheap or real voter profile?

Reason for deferral:
Changing the voter profile changes the model roster and raw voter outputs. It requires a new primer/voter cache, so it is not part of the current replay sweep.