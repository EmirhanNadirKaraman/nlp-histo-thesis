# Calibration Execution Plan

## 1. Purpose

This document is the **order-of-operations plan** for calibrating the full
pipeline (PDF extraction → text assembly → MAP → grounding → normalize →
group → canonicalize → relate → resolve). It tells you *what to do, in what
order, and what to defer*. It does not replace the detailed inventory,
metrics, or experiment specs — those live in the companion docs below.

Use this file when deciding what to calibrate next. Use the companion docs
when you need the specifics of a knob, metric, or experiment.

## 2. Related Documents

| Doc | Contains |
|---|---|
| [`CALIBRATION_INVENTORY.md`](CALIBRATION_INVENTORY.md) | Full knob registry — every config field, env var, hardcoded constant, model, prompt, threshold. |
| [`CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md) | Phase-1 proxy metrics harness (`scripts/eval/compute_proxy_metrics.py`) — columns, sources, semantics. No LLM/NLI calls. |
| [`STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md) | Model-agnostic stage experiment battery (M1–Rs3 single-run, X1–X5 cross-run). |
| [`THESIS.md`](THESIS.md) | TODOs + Decisions log. Every calibration freeze adds a Decisions row referencing this plan. |
| [`BUGS.md`](BUGS.md) | Bug catalogue + per-bug write-ups. Calibration-blocking bugs live here. |

## 3. Pipeline Freeze Tiers

Three tiers. Freeze in order. Each tier blocks calibration of the next.

### Tier 0 — Stable substrate

**Freeze first. Everything downstream assumes these don't move.**

- PDF extraction `PipelineConfig` (Docling, TATR, masking, filtering,
  two-pass, cropping, text assembly).
- Sentence segmentation (`en_core_sci_sm`).
- DB schema (Alembic head).
- `compute_pipeline_config_hash()` definition and the fields it consumes.
- `synonyms.yaml` content and `UMLS_THRESHOLD` (0.85) — these silently
  reshape `subject_cui` / `outcome_cui` and therefore `normal_id` /
  `group_id`.

**Dependent labels:** every downstream label, because changes here move
`TextElement.unique_path` and/or CUI keys.

**What breaks if it changes:** PDF text labels stale; downstream silver
labels stale; group/canonical IDs change; cached pipeline runs invalid.

### Tier 1 — MAP output contract

**Freeze second. Required before any silver MAP labels or θ sweeps.**

- `MAP_SCHEMA_VERSION`, `MAP_PROMPT_VERSION` (`models.py:23-24`).
- `Finding` Pydantic schema (claim, verbatim_support, subject_entity,
  outcome_entity, relation_type, direction, scope, evidence).
- Provenance fields on each finding (pmcid, text_element_id, chunk_id,
  sentence span).
- Per-voter MAP output persistence end-to-end: `sum_map_voter_outputs`
  (migration 0013) populated in both sync and batch runners; cache key
  includes voter set + agreement policy + prompt + schema versions;
  replay path reconstructs per-voter outputs.

**Dependent labels:** silver MAP labels, θ / reject_θ sweep outputs,
agreement-scorer A/B results, X1 cross-run findings alignment.

**What breaks if it changes:** every MAP-level silver label invalidates;
agreement-scorer comparisons become apples-to-oranges; θ sweeps must
re-call LLMs.

### Tier 2 — Sweepable downstream thresholds

**Tune freely once Tiers 0 and 1 are frozen.**

- `grounding.threshold`.
- `relate.entailment_threshold`, `relate.contradiction_threshold`.
- `ResolveConfig` weights (both `relations_present` and
  `relations_absent` modes).
- Agreement scorer θ / reject_θ — sweepable only after per-voter replay
  is verified (Phase 0 of §6).

**Dependent labels:** none cascade upward. Grounding-threshold sweeps
preserve finding labels (the score is per-finding). RELATE threshold
sweeps preserve canonical IDs. RESOLVE weight sweeps preserve canonical
and relation IDs.

**What breaks if it changes:** only the artifact being swept.

## 4. Cross-Stage Dependency Rules

- **Upstream extraction changes can invalidate downstream labels.** A
  masking or filtering tweak renumbers `position_in_section`.
- **Chunking changes invalidate MAP-level labels** (`chunk_id` moves).
- **MAP schema/prompt changes invalidate MAP silver labels.** Bump
  versions deliberately.
- **Grounding-threshold changes usually preserve finding labels.** Only
  the keep/drop verdict moves; the underlying NLI score is per-finding.
- **`synonyms.yaml` / `UMLS_THRESHOLD` changes silently move
  `normal_id` / `group_id` / `canonical_id`.** Treat as Tier 0.
- **NLI model swap → human/silver grounding labels remain valid;
  cached NLI scores and cached keep/drop verdicts become stale.** A
  human (or strong-model) verdict on whether the verbatim_support
  actually entails the claim is independent of which cross-encoder
  produced the runtime score. What you lose is the score axis and the
  threshold calibration against it, not the human verdict.
- **Grounding-threshold sweeps are only comparable when the cached
  scores come from the same NLI model.** Comparing thresholds across
  NLI models requires re-scoring under the new model first.
- **RELATE-threshold changes** affect final rules but not MAP labels;
  Layer A sweepable if `sum_raw_nli_pairs` is populated.
- **RESOLVE-weight changes** are safe if RELATE output is frozen.

## 5. Label Invalidation Matrix

Read row-by-row: *if I change X, which artifacts stay useful?* "OK" =
still usable as-is. "stale" = invalidated.

Grounding labels are split into two columns because they invalidate
differently:

- *Human/silver grounding labels* — a human (or strong-model) verdict
  on whether the verbatim_support supports the claim. Independent of
  the NLI model.
- *Cached NLI scores & threshold* — per-finding scores produced at
  pipeline runtime by whichever NLI model was active, plus any
  threshold tuned against them.

| Change | PDF text/figure/table labels | Human/silver grounding labels | Cached NLI scores & threshold | NORMALIZE/GROUP set diffs | CANONICAL/RESOLVE diffs | Future silver MAP labels |
|---|---|---|---|---|---|---|
| PDF masking / filtering | **stale** | **stale** (text_element_id moves) | **stale** | **stale** | **stale** | **stale** |
| `tatr.threshold`, `tatr.render_dpi` | table P/R **stale**, text OK | OK | OK | OK | OK | OK |
| `chunk_size` / `chunk_overlap` | OK | **stale** (chunk_id moves) | **stale** | **stale** | OK if group_id holds | **stale** |
| MAP voter set / temperature | OK | OK | re-score new findings | partial | OK (group_id stable) | **stale** |
| MAP prompt / schema version bump | OK | OK | **stale** | **stale** | recheck | **stale** |
| Agreement scorer / θ / reject_θ | OK | OK | OK | **stale** | recheck | partial (same candidates) |
| `grounding.threshold` only | OK | OK | OK | partial | OK | OK |
| NLI model swap | OK | **OK** (verdict independent of model) | **stale** (re-score required, threshold must be re-calibrated) | OK | RELATE-side **stale** | OK |
| `synonyms.yaml` content | OK | OK | OK | **stale** | **stale** | OK |
| `UMLS_THRESHOLD` | OK | OK | OK | **stale** | **stale** | **stale** |
| `relate.entailment_threshold` / `contradiction_threshold` | OK | OK | OK | OK | **stale** | OK |
| `ResolveConfig` weights | OK | OK | OK | OK | OK (only `final_score` moves) | OK |

Practical reading of the NLI-swap row: keep human/silver grounding
labels forever; treat any cached score column and any threshold derived
from it as belonging to one specific NLI spec.

## 6. Recommended Calibration Order

### Phase 0 — Audit per-voter MAP persistence and replay

**Goal:** verify θ / reject_θ and agreement-scorer sweeps can replay
without re-calling LLMs.

Checks:

- `sum_map_voter_outputs` populated for a fresh single-paper run.
- Sync and batch paths persist equivalent information.
- Cache keys include voter set, agreement policy version, schema +
  prompt versions, NLI spec.
- Cached artifacts can reconstruct per-voter outputs (or, failing that,
  the path to regenerate them deterministically).
- Smallest regression test exists or is identified.

Deliverable: short audit note in `THESIS.md` Decisions log, verdict
"MAP replay safe" / "MAP replay not safe — gap is X".

### Phase 1 — Quick PDF/text substrate freeze

**Goal:** avoid PDF rabbit holes unless extraction is hurting downstream.

- Use existing labels in `eval/annotations/` and metrics in
  `eval/precision_recall.py` / `eval/recall.py`.
- Only sweep high-impact extraction knobs: `tatr.threshold`,
  `tatr.render_dpi`, two-pass thresholds if ghost-text recall is
  visibly bad.
- Freeze extraction config under a named profile before any downstream
  calibration starts.

### Phase 2 — Frozen-artifact downstream sweeps

**Goal:** cheap sweeps before expensive LLM sweeps.

- `grounding.threshold` sweep — already shipped in
  `eval/sweeps/grounding.py`. Pick operating point against manual labels.
- RELATE threshold sweep if `sum_raw_nli_pairs` is populated (add
  `eval/sweeps/relate.py` following the same pattern).
- RESOLVE weight sweep on frozen RELATE output (add
  `eval/sweeps/resolve.py`). Measure top-k stability under perturbation.
- Proxy metrics from `compute_proxy_metrics.py` recorded per profile.

### Phase 3 — Stage eval / proxy metrics

**Goal:** run the cheapest experiments from
[`STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md) for numerical
accuracy numbers.

- Structural replays: GR1, C1, C3, R2, R3, Rs3.
- Provenance/citation: M1 (verbatim containment), G3 (span offsets).
- Determinism / ID stability: GR1, C1, Rs3.
- Cost / runtime / memory: per-paper rows in `proxy_metrics.csv` and
  `cost_report.json`.

### Synthetic RELATE diagnostic set

A small, balanced, hand-authored RELATE dataset used as a sanity layer,
independent of pipeline noise. Built once, reused across NLI model
swaps and threshold sweeps. Not implementing the generator in this
plan — design only.

**Composition (pick one):**

- Option A: 100 SUPPORT / 100 CONTRADICT / 100 UNRELATED.
- Option B (if SCOPE_QUALIFY remains enabled): 75 SUPPORT / 75
  CONTRADICT / 75 UNRELATED / 75 SCOPE_QUALIFY.

**Purpose:**

- Sanity-check RELATE classification on clean inputs.
- Test threshold behavior in isolation (entailment vs contradiction
  cutoffs separately).
- Catch regressions when swapping NLI / LLM relation classifiers.
- Compare candidate relation-model backends head-to-head.
- Exercise the contradiction code path, which is rare in real pipeline
  data and easily breaks unnoticed.

**Warnings:**

- Synthetic pairs are **not** the final RELATE accuracy estimate. They
  are cleaner than real pipeline outputs and will overstate real-world
  performance.
- Real sampled canonical-rule pairs from `sum_relations` /
  `sum_raw_nli_pairs` are still required for actual calibration.
- Do not tune thresholds on synthetic data alone — use it as a floor
  ("performance below this on clean data ⇒ something is broken").

**Suggested path:** `eval/synthetic/relate_pairs_v1.jsonl`.

**Suggested fields per row:**

- `pair_id` — stable identifier (e.g. `synth_relate_v1_0001`).
- `claim_a` — first canonical-rule-style claim string.
- `claim_b` — second claim string.
- `gold_label` — `SUPPORT` | `CONTRADICT` | `UNRELATED` | `SCOPE_QUALIFY`.
- `category` — biological_insight / methodology / clinical / etc.
- `difficulty` — `easy` | `medium` | `hard` (human-judged).
- `phenomenon` — short tag (e.g. `negation`, `synonym`, `scope_subset`,
  `numerical_disagreement`).
- `notes` — free text, why this pair exists.

### Phase 4 — MAP cascade tuning

**Goal:** only after Phase 0 verifies replay safety.

- θ / reject_θ sweep replaying cascade decisions from per-voter outputs.
- Agreement scorer A/B (`SemanticAgreementScorer` vs `HybridScorer` vs
  `EmbeddingScorer`).
- Voter cascade swap (`cheap` ↔ `real`) — X1–X5 cross-run experiments
  from `STAGE_EVAL_EXPERIMENTS.md`.
- Cost/quality Pareto curve.

### Phase 5 — Silver labels

**Goal:** generate silver labels only once the relevant upstream contract
is frozen.

- Do **not** generate expensive MAP silver labels before MAP schema,
  prompt, provenance, and candidate universe are stable (i.e. before
  Phase 4 is done and Tier 1 is locked).
- Store labels under profile/versioned paths
  (`eval/silver/<profile_name>/`).

## 7. Minimum Viable Calibration Plan

Six items max.

1. Audit per-voter MAP persistence and replay (Phase 0).
2. Add or confirm calibration-profile / version tracking (§8).
3. Quick PDF/text substrate freeze under a named profile.
4. Hand-label `eval/results/grounding_manual_sample.jsonl`; run
   grounding-threshold sweep with labels for P/R.
5. Run RESOLVE and RELATE frozen-artifact sweeps where the persisted
   artifacts allow it.
6. Only then consider MAP θ / agreement-scorer sweeps and silver labels.

## 8. Calibration Profiles

A named, declarative profile sitting above `configs/run.yaml`. Cite the
profile in the thesis instead of an 8-char config hash. Not implemented
yet — sketch only.

**Proposed location:** `configs/profiles/<profile_name>.yaml`.

**Suggested fields:**

- `profile_name`
- `created_at`
- `git_commit`
- `parent_profile` (chain inheritance)
- `pipeline_extends` (base YAML, e.g. `configs/run.yaml`)
- `overrides` (sparse: only the keys this profile changes)
- `pipeline_config_hash` (auto-stamped)
- Stage schema/prompt versions (`MAP_SCHEMA_VERSION`,
  `MAP_PROMPT_VERSION`, `CANONICALIZE_DIRECTION_POLICY_VERSION`,
  `MAP_AGREEMENT_POLICY_VERSION`)
- Model names (voter list per level, NLI `hf_id`, embedding model, UMLS
  KB version)
- Thresholds (every YAML-surfaced θ)
- `paper_selection` (selection YAML + resolved PMCID list)
- `input_artifacts` (paths to summarization run output)
- `label_artifacts` (paths to label files)
- `metrics_artifacts` (paths to proxy / sweep / experiment outputs)
- `random_seed`
- `cost_runtime_summary` (total cost, runtime, peak RSS)
- `invalidation_assumptions` (which parent labels this profile inherits
  as still-valid, which it does not)

## 9. What Not To Do Yet

- **Do not generate silver MAP labels yet.** Tier 1 is not formally
  frozen.
- **Do not sweep θ / reject_θ before Phase 0 verifies per-voter replay.**
  Without replay, the sweep re-calls LLMs.
- **Do not deeply tune PDF extraction unless it is a proven bottleneck.**
  Spot-check, freeze, move on.
- **Do not YAML-ify every hardcoded constant at once.** Surface a Tier-2
  constant only the week you sweep it.
- **Do not touch `synonyms.yaml` or `UMLS_THRESHOLD` during unrelated
  sweeps.** They silently move `normal_id` / `group_id` hashes.
- **Do not calibrate RELATE/RESOLVE against unstable canonicalization.**
  If CANONICALIZE direction policy or RELATE comparability gate is about
  to change, wait.
- **Do not add duplicate evaluation scripts.** `eval/sweeps/_lib.py` and
  `scripts/eval/_lib.py` already abstract artifact loading; extend them
  rather than fork.
- **Do not write a new master calibration doc.** This plan + the three
  `CALIBRATION_*.md` companions are the surface.

## 10. Immediate Next Task (post-Phase-0)

Phase 0 audit is complete (see [`THESIS.md`](THESIS.md) Decisions log
2026-05-16 row and [`BUGS.md` B-056](BUGS.md#bug-56--batch-runner-omits-per-voter-map-persistence-code-path-absent)).
Sync per-voter persistence is verified end-to-end; the batch-path gap
does not block a θ sweep that uses
[`eval/silver/map_theta_sweep.py`](../eval/silver/map_theta_sweep.py),
which has its own primer cache.

**Next step: run a θ sweep on the existing primer cache.**

Pre-flight checklist (each item read-only; failure of any item means
re-prime before publishing the sweep result):

1. Confirm `eval/data/map_primer/primer.json` exists and has
   `phase == "complete"`.
2. Confirm `eval/data/map_primer/voter_cache.json` exists.
3. Confirm the primer metadata matches **current** production state on
   every axis that affects validity:
   - Voter set —
     [`pipeline/stages/summarization/batch/voter_configs.py`](../pipeline/stages/summarization/batch/voter_configs.py)
     (`make_l1_voters`, `make_l2_voters`, `make_l3_voter`) unchanged
     since the primer was built.
   - MAP `schema_version` and `prompt_version` (`models.py` constants
     `MAP_SCHEMA_VERSION`, `MAP_PROMPT_VERSION`) unchanged since the
     primer was built.
   - NLI spec — `configs/nli_models.yaml` active spec is the one the
     cached `grounding_score` values were produced against.
   - Embedder choice — the one passed to `--embedder` matches the
     cache file the sweep will read.
4. Confirm the embedding cache for the chosen embedder exists at the
   default path (e.g. `eval/data/embedding_cache_gemini.json` for
   `--embedder gemini`). Cold cache is acceptable but means small
   embedding-API spend on first run.
5. If any of (1)–(3) fail, **re-prime** before publishing the sweep —
   stale cascade composition or stale schemas invalidate the result.
   Re-priming is a `prime` + `collect` cycle — L1+L2+L3 **voter** batches
   (~\$3.66 batch on the 273-case dev set). If the **source cases** changed
   it *also* needs a fresh **Opus silver** pass (`eval.silver.generate`,
   `claude-opus-4-7` — ~\$6.35 batch, the larger half); the prime does **not**
   run Opus. Budget both with
   `python scripts/estimate_selection_cost.py --source-cases … --profile real`
   (prints prime + Opus + total, sync and batch — see
   [HOW_TO_RUN §5](HOW_TO_RUN.md#5-cost-estimation)).

**Command (once the checklist passes):**

```bash
python -m eval.silver.map_theta_sweep sweep --embedder gemini --split dev
```

Writes `eval/reports/map_theta_sweep_<timestamp>.csv` (one row per θ in
`THETA_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]`) plus a
console table marking best F1.

**Scope notes:**

- This is a **θ-only** sweep. `reject_theta` is hardcoded to `-1.0` in
  `_replay_theta` (`map_theta_sweep.py:590`) so the sweep isolates the
  escalation threshold; REJECT decisions are never taken during replay.
- Joint (θ, reject_θ) sweep is a small follow-up patch (~15–25 LOC in
  the same file: parametrize `reject_theta` in `_replay_theta`, outer
  product loop in `run_sweep`, extra CSV column). Tracked as the
  `ABC P1 — joint theta sweep` TODO in [`THESIS.md`](THESIS.md). **Not
  part of the first sweep.**
- `db_replay.py` (reading per-voter rows from `sum_map_voter_outputs`
  for an arbitrary production run) is **deferred**. `map_theta_sweep.py`
  uses its own primer cache and answers the "best θ on silver" question
  without needing the DB path. Add `db_replay.py` only when comparing
  production profiles or auditing past runs becomes the need.

**Constraints during this step:**

- Read-only against pipeline code; no refactor, no calibration-profile
  implementation, no batch voter-persistence fix (B-056).
- Do not run `prime` / `all` unless the checklist above requires
  re-priming.

## 11. Per-Stage Tune-or-Freeze Walk (pipeline call order)

For each stage in execution order: what is worth tuning, what to freeze
without tuning, and when in the calibration order (§6) to do it. This
section is the operational complement to the phase-ordered plan in §6 —
read it when deciding what to touch in a specific stage.

The recommended calibration order is **not** top-to-bottom of this list
— see §11.x at the end for the actual sweep sequence.

### PDF extraction pipeline

#### 11.1 Layout extraction (Docling)

- **Tune:** `do_ocr` (flip on only if a paper visibly fails text-layer
  extraction); `images_scale` (default `2.0` is usually right).
- **Freeze without tuning:** `do_table_structure`, `ocr_engine`,
  `accelerator_device`, model version.
- **When:** Phase 1 only if extraction recall is visibly hurting
  downstream. Otherwise leave on defaults and move on.

#### 11.2 Table detection (TATR / Docling / Hybrid)

- **Tune:** `table_detector` (`HYBRID` is the default; `TATR` alone for
  recall, `DOCLING` alone for precision); `tatr.threshold` (sweep
  `0.50 / 0.75 / 0.90 / 0.95 / 0.99` on the existing
  `eval/annotations/` set); `tatr.render_dpi` (`150` default — bump to
  `200`/`250` if small or faint tables are missed).
- **Freeze without tuning:** `tatr.model_name`, `tatr.device`.
- **When:** Phase 1, **first** PDF-side sweep. Has manual labels;
  cheap; isolated.

#### 11.3 Region masking

- **Tune:** `mask_header_footer_sidebar`; `expand_box_px` (default `2`;
  bump to 3–4 if glyph remnants leak).
- **Freeze without tuning:** `mask_tables`, `mask_figures`,
  `merge_overlapping_boxes`, the hardcoded sidebar / column heuristics
  (`_SIDEBAR_MAX_W=150`, `_COLUMN_GAP_MIN=50`).
- **When:** Phase 1, after table-detection threshold. Spot-check — do
  not deep-sweep. Masking changes invalidate every downstream label
  (§5 matrix).

#### 11.4 Re-extraction (Docling on masked PDF)

- **Tune:** Nothing distinct from §11.1. Override `docling_text`
  separately from `docling` only if you want a different OCR setting
  for the masked pass.
- **Freeze without tuning:** All else; defaults inherit from §11.1.
- **When:** No standalone tuning.

#### 11.5 Artifact filtering

- **Tune:** `apply_ner_filtering`; `apply_paragraph_relevance_filtering`
  (measure text-count delta against held-out paragraphs).
- **Freeze without tuning:** Internal filter rules in
  `parsers/layout_utils.filter_artifacts`.
- **When:** Phase 1, after masking. Spot-check; filter changes shift
  every TextElement.

#### 11.6 Two-pass invisible-text (when `two_pass.enabled=True`)

- **Tune:** `blank_brightness_threshold` (`245`),
  `blank_dark_pixel_max_fraction` (`0.02`), `max_chars_per_bbox_pt`
  (`15`, R3). Sweep on synthetic ghost-text fixtures only — not real
  papers.
- **Freeze without tuning:** `enabled=True`, `render_dpi=150`,
  `min_anchor_word_count`, `max_white_char_fraction=1.0` (R-color stays
  disabled per Decisions log), `header_mask_margin_pt`.
- **When:** Phase 1, only if `scripts/verify_ghost_text_detection.py`
  shows a regression.

#### 11.7 Text assembly

- **Tune:** `pre_filter_relevance`.
- **Freeze without tuning:** `write_raw_text` (audit-only); the
  assembly logic in `parsers/layout_utils.extract_text`.
- **When:** No standalone tuning — falls out of filtering / two-pass
  decisions.

#### 11.8 Cropping

- **Tune:** `merge_figures_by_caption` / `merge_tables_by_caption`
  (default `False`); `subfigure_proximity_pts` (`20`).
- **Freeze without tuning:** `image_format`, `dpi` (`200`),
  `min_figure_pts`, footnote-proximity knobs.
- **When:** Lowest PDF-side priority. Defer indefinitely unless a
  thesis figure needs better merging.

**Freeze the PDF tier as a profile** after §11.8 lands
(`configs/profiles/profile-pdf-frozen-<date>.yaml`). Nothing downstream
should run against a moving PDF target.

### Tier-0 substrate (between PDF and MAP)

#### 11.9 Sentence segmentation

- **Tune:** Nothing.
- **Freeze without tuning:** `en_core_sci_sm`. Do not swap segmenters.

#### 11.10 UMLS

- **Tune:** Nothing unless a domain expert flags systematic CUI
  miss-mapping.
- **Freeze without tuning:** `UMLS_THRESHOLD = 0.85` (`umls_utils.py:11`);
  the 16-TUI junk filter; `en_core_sci_lg`. Changing any of these
  silently moves `normal_id` / `group_id` / `canonical_id`.
- **When:** Treat as Tier-0. Only revisit if NORMALIZE precision is the
  bottleneck and you have evidence.

#### 11.11 Synonyms

- **Tune:** Add hand-curated entries to `synonyms.yaml` as needed.
  Content, not threshold.
- **Freeze without tuning:** Dedup logic.
- **When:** Pre-Phase-2, but quietly and continuously — every entry
  shifts downstream IDs.

### Summarization pipeline

#### 11.12 Chunking

- **Tune:** `map.chunk_size` (default `10`); `map.chunk_overlap`
  (default `2`).
- **Freeze without tuning:** Sentence segmenter (already Tier 0).
- **When:** **Freeze first; do not sweep yet.** Any change invalidates
  every MAP-level label and the existing primer (`chunk_id` moves).

#### 11.13 MAP cascade

- **Tune:** `map.theta` (default `0.8`); `map.reject_theta` (default
  `0.2`); agreement scorer choice (`SemanticAgreementScorer` ↔
  `HybridScorer` ↔ `EmbeddingScorer`); voter cascade profile (`cheap` ↔
  `real`).
- **Freeze without tuning:** Per-voter `temperature` (`0.1`),
  `max_tokens` (`16384`), retry budget (`2`), router on/off (keep on);
  `MAP_SCHEMA_VERSION` and `MAP_PROMPT_VERSION` (bump deliberately and
  re-prime when you do).
- **When:** Phase 4, last. Most expensive sweep. Run
  `eval/silver/map_theta_sweep.py` with a fresh primer that matches
  current voter config + schema/prompt versions.

#### 11.14 GROUNDING

- **Tune:** `grounding.threshold` (currently `None` / disabled; sweep
  `[0.3, 0.95]` against the 100 manual labels in
  `eval/results/grounding_manual_sample.jsonl` once hand-filled).
- **Freeze without tuning:** NLI model — sweep at the NLI-model layer
  (§11.19), not as a grounding knob. Token budgets
  (`_MODEL_MAX_TOKENS=512`, `_PREMISE_BUDGET_FLOOR=64`).
- **When:** Phase 2 — **cheapest meaningful sweep.** Already shipped at
  `eval/sweeps/grounding.py`. Run before MAP cascade tuning.
  Frozen-artifact, zero LLM calls.

#### 11.15 NORMALIZE

- **Tune:** `synonyms.yaml` content (continuous, Tier-0).
- **Freeze without tuning:** `UMLS_THRESHOLD`, junk-TUI set, polarity
  trigger lists (`_NEGATIVE_TRIGGERS`, `_POSITIVE_TRIGGERS`), dedup key
  composition.
- **When:** No standalone sweep. Any change is Tier-0 invalidation.

#### 11.16 GROUP

- **Tune:** Nothing yet.
- **Freeze without tuning:** Group key (`subject_entity × outcome_entity
  × relation_type × category`), `_SCOPE_FIELDS`,
  `_SCOPE_FIELDS_DEFAULTS`. Changes break group-id stability across
  runs.
- **When:** No tuning. Structural stage.

#### 11.17 CANONICALIZE

- **Tune:** Nothing yet.
- **Freeze without tuning:** `_DIRECTION_ALIASES`, `_CATEGORY_ALIASES`,
  `_RELATION_TYPE_ALIASES`, `POLARITY_BEARING_DIRS`. The per-direction
  no-folding policy is a Decisions-log row (B-049) — do not second-guess
  without new measurement.
- **When:** No tuning. Structural stage.

#### 11.18 RELATE

- **Tune:** `relate.entailment_threshold` (`0.50`);
  `relate.contradiction_threshold` (`0.50`). Sweep jointly over
  `[0.3, 0.9]` step `0.05` once `sum_raw_nli_pairs` is populated.
- **Freeze without tuning:** NLI model (sweep separately at NLI layer);
  comparability gate (category × relation_type × normalized subject ×
  normalized outcome — all four required).
- **When:** Phase 2, after GROUNDING. Frozen-artifact sweep, no LLM
  calls. Add `eval/sweeps/relate.py` following the `grounding.py`
  pattern.

#### 11.19 RESOLVE

- **Tune:** The nine `ResolveConfig` weights — `grounding_weight`,
  `finding_bonus_max` / `_scale`, `support_boost_per_rel` / `_cap`,
  `single_study_pen`, `contradict_pen_per_rel` / `_cap`, and the three
  relations-absent counterparts. Sweep ±50 % around defaults.
- **Freeze without tuning:** Score formula itself; mode-selection logic
  (`relations_present` vs `relations_absent`).
- **When:** Phase 2, last frozen-artifact sweep. Measure top-k
  stability under weight perturbation against frozen RELATE output. No
  labels needed; metric is Spearman ρ / top-k Jaccard vs default-weight
  ranking.

### Cross-cutting layer

#### 11.20 NLI model (affects GROUNDING + RELATE simultaneously)

- **Tune:** Active model in `configs/nli_models.yaml`
  (`pubmedbert_mednli` ↔ `deberta_zeroshot_v2` ↔ future candidates).
- **Freeze without tuning:** Per-model `batch_size` (in the registry).
- **When:** After GROUNDING (§11.14) and RELATE (§11.18) thresholds are
  characterised at the current default — then swap NLI, re-score,
  re-threshold both. Human/silver labels survive the swap; only cached
  scores and thresholds need re-calibration (§5 matrix).

### 11.x Sweep order (not the execution order above)

Read top-to-bottom of §11.1–§11.20 for **execution** order. The actual
**calibration** order — what to sweep first to avoid wasted work — is:

1. PDF stages §11.1–§11.8 under existing manual labels → freeze as
   `profile-pdf-frozen-<date>` (Phase 1).
2. **GROUNDING threshold** (§11.14) on frozen artefacts (Phase 2;
   already shipped) — pick operating point once 100 manual labels are
   filled in.
3. **RESOLVE weights** (§11.19) sweep on frozen RELATE output
   (Phase 2) — no labels needed.
4. **RELATE thresholds** (§11.18) on `sum_raw_nli_pairs` (Phase 2) — if
   those rows are populated.
5. Structural / proxy experiments per
   [`STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md) (Phase 3).
6. **MAP θ / reject_θ + agreement scorer + voter cascade** (§11.13)
   (Phase 4) — most expensive; do last; needs fresh primer matching
   current schema.
7. **NLI model swap** (§11.20) as a separate cross-cutting experiment
   after 2 and 4 are characterised.

§11.12 (chunking), §11.15 (NORMALIZE), §11.16 (GROUP), §11.17
(CANONICALIZE) **do not sweep** at this stage — frozen by definition.
They change only when a bug or a new design call forces them.
