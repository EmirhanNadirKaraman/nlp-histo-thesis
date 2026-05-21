# Summarization — Calibration & Experiment Plan

The full staged plan for the summarization-pipeline calibration sweep.
Companion to [`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md);
mirrors its structure but covers MAP cascade, downstream thresholds, and
cross-run differentials instead of PDF-detection variants.

For the per-knob inventory and freeze tier rules see
[`CALIBRATION_EXECUTION_PLAN.md`](CALIBRATION_EXECUTION_PLAN.md);
for the model-agnostic stage-experiment battery see
[`STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md);
for the no-API proxy-metrics harness see
[`CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md).

---

## 1. Purpose & scope

This plan exists to calibrate the ABC cascade (`theta`, `reject_theta`),
the agreement scorer choice, and the downstream `grounding_threshold` /
`relate.{entailment,contradiction}_threshold` / `ResolveConfig` weights —
then freeze the picked values as the thesis-cited summarization
configuration. It also wires in the model-agnostic stage experiments
(structural replays + cross-run differentials) so the thesis can show
that downstream stages are deterministic where they claim to be and that
cascade choice affects MAP-side outputs but not the rule taxonomy.

**Explicit non-goals:**

- PDF extraction (frozen separately under
  [`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md);
  treat the PDF tier as Tier 0 and do not move it).
- Pipeline rewrites; no stage logic changes ship in the same PR as the
  sweep results.
- Manual labelling beyond the 100-row grounding sample at
  `eval/results/grounding_manual_sample.jsonl`. Silver labels at
  `eval/data/silver_findings.jsonl` are treated as a provisional
  placeholder (§3 note).

## 2. Freeze model (Tier 0 / 1 / 2)

Three tiers, frozen in order. Each tier blocks calibration of the next.
Condensed from `CALIBRATION_EXECUTION_PLAN.md` §3; consult that doc for
the full label-invalidation matrix.

- **Tier 0 — substrate, frozen first.** PDF config, sentence segmenter
  (`en_core_sci_sm`), DB schema (Alembic head),
  `compute_pipeline_config_hash()`, `synonyms.yaml`,
  `UMLS_THRESHOLD=0.85`. *Must not move during this plan.*
- **Tier 1 — MAP contract, frozen second.** `MAP_SCHEMA_VERSION`,
  `MAP_PROMPT_VERSION`, `Finding` Pydantic schema, per-voter MAP output
  persistence (`sum_map_voter_outputs`). *Must be frozen before priming.*
  - **B-056 caveat:** the batch runner currently does not write
    `sum_map_voter_outputs` (the sync runner does).
    `eval/silver/map_theta_sweep.py` is unblocked because it uses its
    own primer cache (`eval/data/map_primer/voter_cache.json`), not the
    production DB table. The M2 / M3 / Rs1 / R1 thesis numbers are
    produced against the harness cache and are therefore valid; what is
    not yet true is that a production batch run would let you
    reconstruct the same per-voter outputs from the DB. Treat B-056 as
    a separate pre-freeze fix (§9); the experiment plan does not block
    on it.
- **Tier 2 — sweepable thresholds.** `grounding.threshold`,
  `relate.{entailment,contradiction}_threshold`, `ResolveConfig`
  weights, `map.{theta, reject_theta}`, agreement-scorer choice.
  *Sweepable in any order after Tier 1 is locked.*

For which artifacts a knob change invalidates, refer to the §5 label
invalidation matrix in `CALIBRATION_EXECUTION_PLAN.md`.

## 3. Global defaults table (current production)

> **Silver-label status.** The current `eval/data/silver_findings.jsonl`
> (401 cases) is treated as a **provisional placeholder** of the
> canonical shape that the regenerated thesis-final silver will share.
> All M2 / M3 numbers in this plan are produced against the placeholder;
> when the new silver lands (per §10) the sweeps re-run unchanged and
> the thesis cites the regenerated numbers. The plan's structure,
> decision criteria, and command surface stay invariant under that swap.

```
map.theta                           = 0.80
map.reject_theta                    = 0.20
map.chunk_size / chunk_overlap      = 10 / 2
grounding.threshold                 = None  (disabled)
relate.entailment_threshold         = 0.50
relate.contradiction_threshold      = 0.50
resolve.grounding_weight            = 0.60
resolve.finding_bonus_max / scale   = 0.10 / 5
resolve.support_boost_per_rel/cap   = 0.08 / 0.20
resolve.single_study_pen            = 0.10
resolve.contradict_pen_per_rel/cap  = 0.15 / 0.30
agreement scorer (sync)             = EmbeddingScorer
agreement scorer (batch)            = SemanticAgreementScorer + EmbeddingSimilarityStrategy
voter cascade profile               = `real`
NLI model                           = configs/nli_models.yaml::active_spec
embedder                            = OpenAI text-embedding-3-small (Gemini optional)
```

These are the **baseline**. Every variant in §5 overrides exactly the
knob relevant to its decision; everything else inherits this table.

## 4. Stages overview

| Stage | Slug | Variants | API $? | Yields |
|---|---|---|---|---|
| M1 | `map_baseline` | 1 | yes (prime once) | primer cache + baseline F1 at default θ=0.8, reject_θ=0.2 |
| M2 | `map_theta_reject_joint` | 7 × 4 grid | no (offline replay) | best (θ, reject_θ) on silver |
| M3 | `map_scorer_compare` | 3 | no (offline replay) | best scorer ∈ {Embedding, Semantic+EmbeddingSim, HybridStructured} |
| M4 | `map_profile_compare` | 2 | yes (re-prime `cheap`) | cheap vs real cascade Pareto — *optional* |
| G1 | `grounding_threshold` | 8 | no | best `grounding.threshold` on 100 manual labels (P/R/F1) |
| R1 | `relate_threshold_joint` | 7 × 7 grid | no | best (entailment, contradiction) thresholds on `raw_pairs.jsonl` |
| Rs1 | `resolve_weight_stability` | 18 ±50 % perturbations | no | top-k Kendall τ / Jaccard vs default — robustness number |
| stage_structural | `stage_structural` | 8 deterministic replays (M1, G3, GR1, C1, C3, R2, R3, Rs3 — IDs from `STAGE_EVAL_EXPERIMENTS.md`) | no | pass/fail audit |
| X1 | `xrun_map_align` | 1 | yes (run `cheap` + `real` once each) | embedding-aligned MAP F1 between profiles |
| X2 | `xrun_normalize_triples` | 1 | (reuses X1 runs) | NORMALIZE Jaccard between profiles |
| X3 | `xrun_group_canon_setdiff` | 1 | (reuses X1 runs) | `group_id` / `canonical_id` set agreement |
| X4 | `xrun_relate_kappa` | 1 | (reuses X1 runs) | RELATE Cohen's κ across the 4-label scheme |
| X5 | `xrun_resolve_rank` | 1 | (reuses X1 runs) | RESOLVE top-k Jaccard + Kendall τ |

The "API $?" column drives the execution order in §8 — Day 1 = `$0`
work, Day 2 = primer batch + X1 runs (paid, then overnight), Day 3 =
`$0` sweeps + writeup.

## 5. Per-stage specs

Each section follows the same eight-field template:
**Goal · Knobs varied · Baseline · Metric · Decision criterion · Output
files · Commands · API cost · Thesis claim**.

### Stage M1 — `map_baseline`

- **Goal:** capture current-production MAP F1 against silver as the
  floor; lock today's `voter_cache.json` as the substrate for M2 / M3.
- **Knobs varied:** none.
- **Baseline:** §3.
- **Metric:** P / R / F1 on `eval/data/silver_findings.jsonl` (dev
  split) at θ=0.8, reject_θ=0.2.
- **Decision criterion:** record, not pick. Establishes the bar M2 / M3
  must beat.
- **Output files:**
  - `eval/data/map_primer/primer.json`
  - `eval/data/map_primer/voter_cache.json`
  - `eval/reports/map_baseline_<ts>.csv` (single row, current defaults
    — falls out of M2 sweep at `(0.80, 0.20)` cell)
- **Commands:** see §6 primer workflow.
- **API cost:** prime batch — six provider×model batch jobs across 401
  dev cases; rough estimate **$20–$60** depending on chunk count and
  L3 escalations. Tighten the estimate with the smoke prime in §8 / Day 2.
- **Thesis claim:** "MAP-stage F1 at production defaults is X on the
  silver dev split; this is the bar all calibration variants must clear."

### Stage M2 — `map_theta_reject_joint`

- **Goal:** pick a `(θ, reject_θ)` operating point on silver F1, with
  cascade-composition awareness (% kept@L1 / @L2 / @L3 / rejected).
- **Knobs varied:**
  - θ ∈ `{0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}` (7)
  - reject_θ ∈ `{-1.0 (never reject), 0.10, 0.20, 0.30}` (4)
  - Grid = 28 cells. **Constraint:** `reject_θ < θ` always — skip cells
    violating it (or pin reject_θ=-1 in those cells).
- **Baseline:** §3 (default cell is `(0.80, 0.20)`).
- **Metric:** dev-split P / R / F1 / strict-F1 vs silver (matcher.py) +
  cascade composition columns.
- **Decision criterion:** highest F1 cell whose cascade composition
  does not inflate the L3 escalation rate by more than 5 pp over the
  baseline cell (cost-quality guardrail).
- **Output files:**
  - `eval/reports/map_theta_reject_sweep_<ts>.csv` (one row per cell)
  - `eval/reports/map_theta_reject_sweep_<ts>.md` (best-cell narrative)
- **Commands:**
  - **Code change first** (~20 LOC in
    `eval/silver/map_theta_sweep.py`): parametrize `reject_theta` in
    `_replay_theta`, outer-product loop in `run_sweep`, add
    `reject_theta` CSV column. Tracked as the
    *ABC P1 — joint theta sweep* TODO in `THESIS.md`.
  - Then:
    `python -m eval.silver.map_theta_sweep sweep --embedder gemini --split dev`
- **API cost:** `$0` (reads `voter_cache.json` from M1).
- **Thesis claim:** "I chose `(θ=X, reject_θ=Y)` because it maximises
  silver-set F1 without inflating L3 escalation cost."

### Stage M3 — `map_scorer_compare`

- **Goal:** pick between `EmbeddingScorer`,
  `SemanticAgreementScorer(EmbeddingSimilarityStrategy)`, and
  `HybridStructuredSimilarity` at the M2-winning `(θ*, reject_θ*)`.
- **Knobs varied:** `agreement_scorer` ∈ {`EmbeddingScorer`,
  `SemanticAgreementScorer`, `HybridStructuredSimilarity`}.
- **Baseline:** §3 + M2 winning thresholds.
- **Metric:** dev-split F1 + cascade composition + Spearman ρ between
  scorer outputs (sanity check that the scorers aren't trivially
  correlated).
- **Decision criterion:** best F1 unless within 1 pp — then prefer the
  cheaper scorer (Embedding < Semantic < HybridStructured by inference
  cost).
- **Output files:** `eval/reports/map_scorer_compare_<ts>.csv` (3 rows
  × M2 winner cell + diagnostic).
- **Commands:** extend the sweep harness with a
  `--scorer {embedding,semantic,hybrid}` flag (~15 LOC), then:
  - `python -m eval.silver.map_theta_sweep sweep --embedder gemini --split dev --scorer embedding`
  - `… --scorer semantic`
  - `… --scorer hybrid`
- **API cost:** `$0` (same primer).
- **Thesis claim:** "Among the three pluggable agreement scorers I
  retained X because it yielded F1 within Y of the next-best at lower
  cost / on cleaner pairwise diagnostics."

### Stage M4 — `map_profile_compare` (optional, defer unless needed)

- **Goal:** quantify the `cheap` vs `real` voter-profile gap on the
  same silver set — relevant only if a cost-quality Pareto becomes the
  question.
- **Knobs varied:** voter profile ∈ {`cheap`, `real`}.
- **Baseline:** §3.
- **Metric:** F1 delta + cost-per-paper delta from `cost_report.json`.
- **Decision criterion:** record; do not freeze on this run unless
  `cheap` is within 1 pp F1 of `real`.
- **Output files:** `eval/reports/map_profile_compare_<ts>.csv`.
- **Commands:** re-prime with the `cheap` profile into a parallel
  primer dir, then sweep:
  ```bash
  NLP_HISTO_PROFILE=cheap python -m eval.silver.map_theta_sweep prime \
      --primer-dir eval/data/map_primer_cheap --split dev
  NLP_HISTO_PROFILE=cheap python -m eval.silver.map_theta_sweep collect \
      --primer-dir eval/data/map_primer_cheap
  python -m eval.silver.map_theta_sweep sweep \
      --primer-dir eval/data/map_primer_cheap --embedder gemini --split dev
  ```
- **API cost:** one extra prime batch — **$5–$15** for the cheap
  profile.
- **Thesis claim (optional):** "Switching to the cheap cascade costs
  X pp F1 while saving Y % on per-paper LLM spend."

### Stage G1 — `grounding_threshold` (Layer-A frozen artifacts + manual labels)

- **Goal:** pick `grounding.threshold` operating point against
  hand-labelled supported / partial / unsupported.
- **Knobs varied:** threshold ∈
  `{0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95}`.
- **Baseline:** §3 (currently disabled; the sweep selects the value
  that turns it on).
- **Metric:** two reports —
  (a) retention table from `eval/sweeps/grounding.py` (already shipped,
  no labels needed) and
  (b) **P / R / F1 + rejection rate** by joining
  `grounding_manual_sample.jsonl` (once labelled) with
  `grounding_score`.
- **Decision criterion:** highest F1 subject to recall ≥ 0.90 against
  `supported` labels (proposed; tune after first read of the data).
- **Output files:**
  - `eval/results/grounding_sweep.csv` (retention, no labels)
  - `eval/results/grounding_sweep.md`
  - `eval/results/grounding_manual_pr.csv` (P / R / F1 vs manual
    labels — **new**, needs `scripts/eval/compute_grounding_manual_pr.py`)
- **Commands:**
  - Fill labels — manual editor work on
    `eval/results/grounding_manual_sample.jsonl`, set each row's
    `label ∈ {supported, partial, unsupported}` and optional `notes`.
    Schema is stable; 100 rows total.
  - Run retention sweep — `python eval/sweeps/grounding.py`
  - Run P / R / F1 sweep:
    ```bash
    python scripts/eval/compute_grounding_manual_pr.py \
        --sample eval/results/grounding_manual_sample.jsonl \
        --thresholds 0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95 \
        --out eval/results/grounding_manual_pr.csv
    ```
    (new ~50-LOC script — loops over the threshold grid and computes
    counts).
- **API cost:** `$0`.
- **Distinction:** manual-label-based numbers (real evidence,
  thesis-citable) vs proxy retention (debugging / sanity only, do not
  cite as accuracy).
- **Thesis claim:** "I set `grounding.threshold = X` because it
  achieves Y % F1 against 100 hand-labelled findings while rejecting
  Z % of unsupported claims at recall ≥ 0.90."

### Stage R1 — `relate_threshold_joint`

- **Goal:** pick joint `(entailment, contradiction)` thresholds for
  RELATE.
- **Knobs varied:** entailment ∈ `[0.30, 0.90]` step 0.10 ×
  contradiction ∈ `[0.30, 0.90]` step 0.10 (49 cells).
- **Baseline:** §3 (default cell `(0.50, 0.50)`).
- **Metric:** per-class precision + relation count distribution +
  Spearman ρ vs default ranking.
- **Decision criterion:** the cell that retains macro-F1 within 1 pp
  of the maximum **while** keeping CONTRADICT count ≥ 1 per paper on
  average (so the contradiction code path stays exercised).
- **Output files:** `eval/results/relate_sweep.csv`.
- **Commands:** new `eval/sweeps/relate.py` following the
  `grounding.py` pattern (Layer A; reads
  `out/summaries/.../relate/{pmcid}/raw_pairs.jsonl`).
- **API cost:** `$0` after `raw_pairs.jsonl` is populated.
  **Precondition:** confirm `raw_pairs.jsonl` is being written; if
  not, run RELATE once with the persistence path enabled before
  sweeping.
- **Thesis claim:** "RELATE thresholds
  `(entailment=X, contradiction=Y)` chosen on N NLI pairs; relation-
  class distribution stays within Z % of the default."

### Stage Rs1 — `resolve_weight_stability`

- **Goal:** validate that the RESOLVE top-k ranking is robust to
  ±50 % weight perturbation on each of the nine `ResolveConfig` weights.
- **Knobs varied:** each of the nine weights perturbed individually at
  ±50 % from default → 18 variants + 1 baseline.
- **Baseline:** §3 defaults.
- **Metric:** Spearman ρ + top-k Jaccard at k ∈ {5, 10, 20} vs default
  ranking, per pmcid then aggregated.
- **Decision criterion:** ρ ≥ 0.85 **and** top-10 Jaccard ≥ 0.70
  across all 18 perturbations = "RESOLVE weights are not load-bearing"
  → freeze defaults. Any perturbation breaking the band gets flagged
  for follow-up.
- **Output files:** `eval/results/resolve_sweep.csv`.
- **Commands:** new `eval/sweeps/resolve.py` (Layer A; reads
  `final_rules.jsonl`, replays the scoring formula).
- **API cost:** `$0`.
- **Thesis claim:** "RESOLVE top-k rankings are robust to ±50 %
  weight perturbation (top-10 Jaccard ≥ 0.70 across all 9 weights),
  justifying frozen defaults."

### Stage `stage_structural` — deterministic replay battery (no API, no labels)

Bundle the Phase-1 model-agnostic replays from
`STAGE_EVAL_EXPERIMENTS.md` §5 implementation roadmap into one runner.
Keep the original IDs end-to-end:

- **M1** — `verbatim_support` containment (precision = TP / (TP+FP)).
- **G3** — GROUNDING span-offset precision.
- **GR1** — `group_id` determinism replay.
- **C1** — `canonical_id` determinism replay.
- **C3** — `member_normal_ids` ⊆ `member_ids` closure.
- **R2** — RELATE pre-NLI gate audit.
- **R3** — RELATE symmetry (SUPPORT ↔ SUPPORT, CONTRADICT ↔
  CONTRADICT, SCOPE_QUALIFY asymmetric).
- **Rs3** — `FinalRule.canonical_id` ⊆ `canonical_rules.jsonl`.

> **Naming note.** M1 here is the structural verbatim-containment
> experiment from `STAGE_EVAL_EXPERIMENTS.md`, **not** the M1
> `map_baseline` stage in §4. The two M1s coexist because the
> experiment battery and this plan number stages independently — cite
> both with their `STAGE_EVAL_EXPERIMENTS.md` IDs in the thesis to
> disambiguate.

- **Goal:** pass / fail audit on deterministic stage logic. All
  expected at 100 %; any failure is a bug.
- **Output files:** `eval/results/stage_structural_<ts>.{csv,md}` —
  one row per experiment ID (M1 / G3 / GR1 / C1 / C3 / R2 / R3 / Rs3).
- **Commands:** `python -m eval.stage_experiments --run <run_id>`
  (~150 LOC; bundles existing helper imports — no new logic).
- **API cost:** `$0`.
- **Thesis claim:** "All deterministic stage-level invariants hold on
  the production run (8/8 at 100 % precision)."

### Stages X1–X5 — `xrun_*` (cross-run differentials between `cheap` and `real`)

Use the same paper set on both `cheap` and `real` profiles. Both runs
produce all summarization artifacts; the X-series compares them.

- **X1** (`xrun_map_align`): embedding 1:1 greedy match on (claim,
  joined evidence text); standard P/R/F1; per-field mismatch breakdown.
- **X2** (`xrun_normalize_triples`): set of
  `(subject_cui, outcome_normalised, relation_type)` per run; Jaccard +
  macro P/R/F1.
- **X3** (`xrun_group_canon_setdiff`): exact `group_id` /
  `canonical_id` set comparison — cleanest cross-cascade signal.
- **X4** (`xrun_relate_kappa`): Cohen's κ on relation labels across
  the 4-label scheme.
- **X5** (`xrun_resolve_rank`): join `final_rules.jsonl` on
  `canonical_id`; Kendall τ + top-k Jaccard.

- **Goal:** empirical drift estimate between cascade profiles (no
  silver needed).
- **Baseline:** §3 + the two profiles.
- **Metric:** as above per row.
- **Decision criterion:** record. X3 ≤ 0.5 = upstream MAP drift
  dominates and X1 / X2 diagnose the source. X3 ≥ 0.8 = profile choice
  is largely cosmetic from GROUP onward.
- **Output files:** `eval/results/x{1..5}_<ts>.csv`.
- **Commands:**
  ```bash
  # 1. Run cheap profile on a shared PMCID subset (~10 papers for tractable cost).
  NLP_HISTO_PROFILE=cheap python -m pipeline.stages.summarization.runner …

  # 2. Run real profile on the same subset.
  NLP_HISTO_PROFILE=real  python -m pipeline.stages.summarization.runner …

  # 3. Diff the two runs.
  python -m eval.xrun_experiments --run-a <id_cheap> --run-b <id_real>
  ```
  Bundle is a new ~250-LOC script (read-only over two run dirs).
- **API cost:** depends — `real` on the silver corpus could be tens of
  dollars; `cheap` is a small fraction. Reuse M1's primer outputs
  where possible (if M4 ran, both profiles already submitted voters).
- **Thesis claim:** "The summarization pipeline is identifier-stable
  from GROUP onward (X3 set agreement = X %), so cascade choice
  affects MAP-side findings (X1 F1 = Y %) but not the downstream rule
  taxonomy."

## 6. Primer workflow

### 6.1 Pre-flight checklist (read-only, no API)

Before re-priming, verify every axis that affects cache validity:

1. **Voter set** —
   `pipeline/stages/summarization/batch/voter_configs.py::make_l1_voters / make_l2_voters / make_l3_voter`
   match the production voter list as of today.
2. **MAP schema/prompt versions** — `MAP_SCHEMA_VERSION` and
   `MAP_PROMPT_VERSION` in
   `pipeline/stages/summarization/models.py` are the values to stamp
   into the thesis.
3. **NLI spec** — `configs/nli_models.yaml` active spec is the model
   M3 / G1 will score against.
4. **Embedder** — confirm `--embedder` choice (`openai` or `gemini`)
   and that the matching cache exists at the default path
   (e.g. `eval/data/embedding_cache_gemini.json`).
5. **Source case set** — `eval/data/source_cases.jsonl` and
   `silver_findings.jsonl` line counts agree (currently 401 each;
   `wc -l` both).
6. **Git working tree clean** — `primer.json` stamps `git_commit`;
   commit pending changes before priming so the cache traces to a real
   revision.

If any of 1–4 drifted, **re-prime**. The existing primer at
`eval/data/map_primer/primer.json` is from 2026-05-07 — almost
certainly stale.

### 6.2 Prime (paid)

```bash
# Submits six provider×model batch jobs covering all L1/L2/L3 voters
# for every chunk of every dev case. Saves submission state;
# safe to re-run (idempotent on existing primer.json).
python -m eval.silver.map_theta_sweep prime --split dev
```

Outputs:

- `eval/data/map_primer/primer.json` (phase = `submitted`, with job
  IDs).
- Stdout: per-job submission ID + per-(provider, model) request count.

**Estimate the bill before pressing enter:** dry-run with
`--n-cases 5 --primer-dir eval/data/map_primer_smoke` first; multiply
observed cost by `401 / 5 ≈ 80`.

### 6.3 Collect (free polling; retrieval already paid at submit time)

```bash
# Polls job statuses every 30 s, retrieves results, builds voter_cache.json.
# Re-run until it prints "Voter cache written. Ready to sweep."
python -m eval.silver.map_theta_sweep collect
```

Outputs:

- `eval/data/map_primer/voter_cache.json` once all six jobs are
  `completed`.
- `primer.json` updated to `phase = "complete"`.

If a job fails:

```bash
python -m eval.silver.map_theta_sweep retry-failed
# then re-run collect
python -m eval.silver.map_theta_sweep collect
```

`retry-failed` re-submits failed jobs with identical `custom_id`s so
partial results aren't lost.

### 6.4 Cache validation (read-only, no API)

```bash
# 1. Confirm the cache exists and parsed cleanly.
python -c "
import json
c = json.load(open('eval/data/map_primer/voter_cache.json'))
assert all('l1' in e and 'l2' in e and 'l3' in e for e in c.values()), 'shape error'
print(f'{len(c)} cases  ·  L1/L2/L3 slots all present')
"

# 2. Sanity-replay the baseline cell to confirm the sweep harness reads it.
python -m eval.silver.map_theta_sweep sweep --split dev --embedder gemini
# Expect one CSV row per theta in THETA_GRID; the baseline F1
# is the row at theta=0.80.
```

If validation fails, do not press on — `rebuild-cache` recovers from
raw blobs when only the parsing layer is suspect:

```bash
python -m eval.silver.map_theta_sweep rebuild-cache
```

### 6.5 Money-vs-free at a glance

| Command | Spends $? |
|---|---|
| `prime` | **yes** (batch jobs submitted at this step) |
| `collect` | no (retrieval is paid at submit time) |
| `rebuild-cache` | no |
| `retry-failed` | yes (only the failed (provider, model) jobs) |
| `sweep` (any flags) | no |
| Anything in `eval/sweeps/*.py` | no (Layer A by construction) |
| Anything in `scripts/eval/compute_*.py` | no |

## 7. Manual grounding-label workflow

### 7.1 Schema (already encoded in the sample)

Each row in `eval/results/grounding_manual_sample.jsonl` carries:

```
sample_id              str   # stable id
pmcid, chunk_id        str   # provenance
claim                  str   # what the MAP voter asserted
chunk_text             str   # the premise (~10-sentence chunk)
grounding_score        float # cached NLI score
producer_threshold     float # threshold this run used (0.5)
kept_at_producer       bool  # was it kept at run time?
score_bucket           str   # very_low / low / near_threshold_low/high / medium / high
label                  null  ← FILL THIS
label_options          ["supported", "partial", "unsupported"]
notes                  ""    ← optional free text
```

Annotation guideline (1 sentence each):

- **supported** — every assertion in `claim` is directly entailed by
  `chunk_text`. Partial omissions of qualifiers are OK as long as the
  claim is not stronger than the source.
- **partial** — the chunk supports a weaker version of the claim, or
  supports the claim only when combined with extrapolation the source
  does not make.
- **unsupported** — chunk does not entail the claim; or the claim
  contradicts the chunk; or there is no anchor in the chunk for a key
  entity.

### 7.2 Generate / validate (no labels touched)

The sample is already on disk. Re-run only if the underlying summaries
change:

```bash
# Regenerate the sample (same seed → byte-identical labels):
python scripts/eval/sample_grounding_for_manual_labeling.py \
    --input out/summaries \
    --out eval/results/grounding_manual_sample.jsonl \
    --seed 42 --threshold 0.5 --n 100

# Validate row count + label-coverage:
python -c "
import json
rows = [json.loads(l) for l in open('eval/results/grounding_manual_sample.jsonl') if not l.startswith('{\"_meta\"')]
labelled = sum(1 for r in rows if r.get('label'))
print(f'{labelled}/{len(rows)} labelled')
"
```

### 7.3 Fill labels (human, ~1.5–2 hours)

Open the JSONL in a text editor; set
`"label": "supported"` (or `"partial"` / `"unsupported"`) on each row.
Optional `notes` for edge cases. Do not change any other field. Commit
the labelled file.

### 7.4 Score against labels

```bash
python scripts/eval/compute_grounding_manual_pr.py \
    --sample eval/results/grounding_manual_sample.jsonl \
    --thresholds 0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95 \
    --out eval/results/grounding_manual_pr.csv
```

Outputs:

- `eval/results/grounding_manual_pr.csv` — one row per threshold;
  columns `threshold | tp | fp | fn | tn | precision | recall | f1 |
  rejection_rate`.
- `eval/results/grounding_manual_pr.md` — best-threshold narrative +
  the table.

**Treat `partial` as `supported`** for the kept / rejected ground truth
(so the metric measures whether the threshold rejects fully
unsupported claims). Document that choice in the report header;
revisit if the data motivates a stricter or looser collapse.

### 7.5 Manual vs proxy boundary

- **`eval/results/grounding_manual_pr.csv`** = real evidence — cite in
  the thesis.
- **`eval/results/grounding_sweep.csv`** (retention-only) = debugging
  / sanity only — do not cite as accuracy.
- The report files themselves already carry the "this is not accuracy"
  disclaimer from `eval/sweeps/README.md`.

## 8. Walking through the plan (3-day realistic execution)

Prioritise highest thesis-value, lowest-cost work first. The shape is
"`$0` work and labels on Day 1, paid primer overnight on Day 2, `$0`
sweeps on Day 3."

### Day 1 — `$0` work (~5 hours)

1. **Pre-flight** (§6.1) — 15 min, read-only.
2. **Structural battery (M1, G3, GR1, C1, C3, R2, R3, Rs3)** — ~1.5 h
   to write the bundle (`eval/stage_experiments.py`) + 5 min runtime.
   Failures here mean a real bug — fix before priming.
3. **Fill grounding labels** (~1.5–2 h) — §7.3.
4. **Run `eval/sweeps/grounding.py`** for retention table — 1 min, no
   labels needed.
5. **Run G1 manual-label P / R sweep** (§7.4) — 5 min once labels
   filled.
6. **Open `Rs1` runner skeleton** — 1 h to write `eval/sweeps/resolve.py`
   (Layer-A clone of `grounding.py`).
7. **Run Rs1 weight-perturbation sweep** — 5 min.

**Day 1 deliverables:** filled grounding labels +
`grounding_manual_pr.csv` + `stage_structural_*.csv` +
`resolve_sweep.csv`. Three of the seven thesis numbers locked.

### Day 2 — paid primer + overnight (~3 active hours)

1. **Smoke prime** with `--n-cases 5 --primer-dir eval/data/map_primer_smoke`
   to estimate cost. Verify against `out/summaries/cost/*/cost_report.json`.
   — 30 min.
2. **Full prime** (§6.2) — submit time ~5 min, batch jobs run on
   provider clock (~hours).
3. *(concurrent)* Write `eval/sweeps/relate.py` and the
   joint-θ / reject_θ patch to `map_theta_sweep.py` — ~2 hours. Both
   are mechanical extensions.
4. *(concurrent)* Write `eval/xrun_experiments.py` bundle for X1–X5.
   ~1.5 hours.
5. **End of day:** kick off `collect` in a loop; let batch jobs finish
   overnight.

**Day 2 deliverables:** primer cache, R1 sweep code, M2 / M3 joint
sweep code, X-series bundle code. No metric numbers yet.

### Day 3 — `$0` sweeps + writeup (~5 hours)

1. **Cache validation** (§6.4) — 5 min.
2. **Run M2 joint sweep** — 10 min.
3. **Run M3 scorer comparison** — 10 min.
4. **Run R1 sweep on `raw_pairs.jsonl`** (precondition: confirm the
   file is populated; if not, run RELATE locally on the dev set first).
   — 30 min.
5. **Run X1–X5** — requires running the pipeline twice (`cheap` and
   `real`) on a smaller PMCID subset (~10 papers) for tractable cost.
   May spill to Day 4.
6. **Aggregate winning thresholds into
   `configs/profiles/profile-summ-frozen-<date>.yaml`** (§9) — 30 min.
7. **Write thesis appendix tables** — pull rows from the CSVs into the
   thesis manuscript with citations to the CSV files + git commit.

**Day 3 deliverables:** all stage CSVs filled, frozen profile YAML
committed, thesis-citable numbers ready.

## 9. Freezing the production config

After all stages have winners:

0. **Pre-freeze fix — B-056.** Reconcile the production batch runner
   so it persists per-voter MAP outputs to `sum_map_voter_outputs` the
   same way the sync runner does. Until this lands, the thesis claim
   of "I can replay any past production run's cascade decisions
   offline" is true only for sync runs. Plan-level execution does not
   block on B-056 (the harness primer cache is sufficient for
   M2 / M3 / Rs1 / R1), but **freezing the profile YAML implicitly
   claims production-side reproducibility**, so the fix belongs here.

1. **Create `configs/profiles/profile-summ-frozen-<date>.yaml`** per
   the schema sketched in `CALIBRATION_EXECUTION_PLAN.md` §8. Fields
   it must carry:
   - `profile_name`, `created_at`, `git_commit`.
   - `parent_profile`:  `profile-pdf-frozen-<date>` (or whichever PDF
     freeze profile is current).
   - `overrides`: only the knobs picked by this plan
     (`map.theta`, `map.reject_theta`, `grounding.threshold`,
     `relate.entailment_threshold`, `relate.contradiction_threshold`,
     agreement scorer name).
   - `voter_profile: real`.
   - `pipeline_config_hash`: auto-stamped.
   - Stage schema/prompt versions (`MAP_SCHEMA_VERSION`,
     `MAP_PROMPT_VERSION`, `CANONICALIZE_DIRECTION_POLICY_VERSION`,
     `MAP_AGREEMENT_POLICY_VERSION`).
   - `input_artifacts`: `eval/data/source_cases.jsonl`,
     `eval/data/silver_findings.jsonl`,
     `eval/data/map_primer/voter_cache.json`.
   - `label_artifacts`:
     `eval/results/grounding_manual_sample.jsonl`.
   - `metrics_artifacts`: every CSV produced by this plan.
   - `random_seed: 42` (matcher seed; document any other seeds).

2. **Bake the winners** into
   `pipeline/stages/summarization/config.py` defaults — only after the
   YAML profile is committed and the thesis cites it. Do not pre-bake.

3. **Add a row to `docs/STRUCTURE.md ## Pipeline changelog`** linking
   back to the relevant Decisions-log entry in `docs/THESIS.md`
   ("Summ-side calibration freeze: θ=X, reject_θ=Y, grounding=Z;
   `profile-summ-frozen-…yaml`").

4. **Update `docs/HOW_TO_RUN.md`** if any new command (e.g.
   `compute_grounding_manual_pr.py`, `eval/sweeps/relate.py`) becomes
   part of the documented reproducible run.

## 10. How to integrate future silver labels

The plan assumes today's `silver_findings.jsonl` shape persists. When
new silver labels are generated (e.g. an Opus-oracle pass after Tier 1
is frozen):

1. Drop them at `eval/silver/<silver_profile>/silver_findings.jsonl`
   per `CALIBRATION_EXECUTION_PLAN.md` §6 Phase 5 guidance — do not
   overwrite the existing file.
2. Re-run M2 + M3 with `--silver <path>` (extend the
   `map_theta_sweep.py` CLI to accept a path; currently hardcoded to
   `eval/data/silver_findings.jsonl`).
3. Compare new winners against old winners; if they drift > 1 pp F1,
   document the drift in `THESIS.md` Decisions log before re-freezing.
4. The G1 manual labels do not need regenerating unless the chunking
   changes — they live in chunk text, not silver-derived signal.

## 11. Appendix — experiment matrix

| Stage | Knobs | Grid | Metric | API $? | Priority |
|---|---|---|---|---|---|
| M1 | none | 1 cell | P / R / F1 vs silver | yes (prime) | P0 |
| M2 | θ, reject_θ | 7 × 4 | P / R / F1 vs silver + cascade composition | no | P0 |
| M3 | agreement scorer | 3 | F1 + Spearman | no | P0 |
| M4 | voter profile | 2 | F1 + $/paper | yes (re-prime cheap) | P2 |
| G1 | grounding.threshold | 8 | P / R / F1 vs manual labels | no | P0 |
| R1 | entailment_θ, contradiction_θ | 7 × 7 | macro F1 + class dist | no | P1 |
| Rs1 | 9 weights ±50 % | 18 | Kendall τ, top-k Jaccard | no | P1 |
| stage_structural (M1, G3, GR1, C1, C3, R2, R3, Rs3) | none | 8 deterministic | pass / fail | no | P0 |
| X1 | profile A vs B | 1 | embed-aligned F1 | yes (2 runs) | P1 |
| X2 | profile A vs B | 1 | triple Jaccard | (reuses X1) | P2 |
| X3 | profile A vs B | 1 | `group_id` / `canonical_id` set | (reuses X1) | P1 |
| X4 | profile A vs B | 1 | Cohen's κ | (reuses X1) | P2 |
| X5 | profile A vs B | 1 | Kendall τ, top-k Jaccard | (reuses X1) | P2 |

Priority legend: **P0** ship before defending; **P1** ship if Day 3
budget allows; **P2** defer unless an external question forces it.

The single sentence to keep in mind while executing this plan:
**Money is spent in `prime` (once) and in the `cheap` / `real`
cross-runs (X1–X5); every other number falls out of the cached primer
or the persisted run artifacts at `$0`.**

---

## See also

- [`CALIBRATION_EXECUTION_PLAN.md`](CALIBRATION_EXECUTION_PLAN.md) —
  Tier-0/1/2 freeze rules, knob registry, label-invalidation matrix.
- [`STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md) —
  model-agnostic stage-experiment specs (M1–Rs3 single-run, X1–X5
  cross-run).
- [`CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md) — Phase-1 proxy metrics
  harness (`scripts/eval/compute_proxy_metrics.py`).
- [`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md)
  — companion plan whose structure this one mirrors.
- [`HOW_TO_RUN.md`](HOW_TO_RUN.md) — reproducible command surface
  (update after new sweep scripts ship).
- [`THESIS.md`](THESIS.md) — TODOs + Decisions log
  (every freeze adds a Decisions-log row).
- [`STRUCTURE.md`](STRUCTURE.md) — pipeline changelog
  (every freeze adds a changelog row).
- [`BUGS.md`](BUGS.md#bug-56--batch-runner-omits-per-voter-map-persistence-code-path-absent)
  — B-056 (production batch path missing per-voter persistence;
  pre-freeze fix per §9).
