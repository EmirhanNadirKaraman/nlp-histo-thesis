# Summarization pipeline — calibration surface (derived from code)

**Scope:** what is *tunable* in the summarization pipeline, read **only from the
code** (no `.md` consulted). Branch `pdf-extraction-eval`, 2026-05-24.

**How to read the tables**
- *Prod value* = what an actual run uses. Operational defaults come from
  `configs/run.yaml` (mapped onto `SummarizationConfig`); where run.yaml is
  silent the dataclass default in `config.py` applies.
- *In config?* = exposed as a field on `SummarizationConfig` / a nested config
  (i.e. tunable without editing source).
- *Harness?* = an existing offline sweep / optimizer already targets it.

> Two defaults can disagree: the `config.py` dataclass default and the stage
> constructor default. The runner always passes the config value, so the
> **config/run.yaml value is what production uses**; constructor defaults only
> bite when a stage is built directly (tests, sweeps, `corpus_relate`). See
> [§4 Divergences](#4-config--stage-default-divergences).

---

## 1. MAP (ABC cascade)

`current_stages/map_stage.py`, scorer in `agreement/`, wired in `runner.py`.

| Knob | Where | Prod value | In config? | Harness? |
|---|---|---|---|---|
| `theta` (KEEP if deferral ≥ θ) | `config.py:31` / run.yaml:82 | **0.8** | ✅ `map.theta` | ✅ `map_theta_sweep.py` |
| `reject_theta` (REJECT if ≤) | `config.py:35` / run.yaml:83 | **0.2** | ✅ `map.reject_theta` | ✅ `map_theta_sweep.py` |
| Scorer choice | `runner.py:211` | `SemanticAgreementScorer(EmbeddingSimilarityStrategy)` | ⚠️ chosen in code, not a config field | ✅ sweep compares `embedding` vs `hybrid` |
| `tau` (weak-match floor) | `config.py:169` / run.yaml:123 | **0.15** | ✅ `agreement.tau` | ✅ `map_theta_sweep` ScorerSpec |
| `count_alpha` (count-mismatch penalty) | `config.py:173` / run.yaml:124 | **0.25** | ✅ `agreement.count_alpha` | ✅ ScorerSpec |
| `reuse_weight` | `config.py:175` / run.yaml:125 | **0.15** | ✅ `agreement.reuse_weight` | ✅ ScorerSpec |
| `contradiction_weight` | `config.py:178` / run.yaml:126 | **0.20** | ✅ `agreement.contradiction_weight` | ✅ ScorerSpec |
| `chunk_size` | `config.py:37` | 10 | ✅ `map.chunk_size` | ⛔ (held fixed by sweep) |
| `chunk_overlap` | `config.py:40` | 2 | ✅ `map.chunk_overlap` | ⛔ |
| `enable_router` (cascade path) | `config.py` (`RoutingConfig`) / run.yaml `summarization.routing.enable_router` | **false** (legacy `AgreementChecker`) | ✅ `routing.enable_router` (moved from `map.*` in layout v2, 2026-05-26) | ⚠️ sweep replays legacy path only |
| `router_single_voter_policy` | `config.py` (`RoutingConfig`) / run.yaml `summarization.routing.router_single_voter_policy` | `escalate` | ✅ `routing.router_single_voter_policy` (moved from `map.*` in layout v2) | ✅ `eval/silver/run_summarization_sweeps.py --stage map_routing_policy` (only when router on) |
| `legacy_single_voter_policy` | `config.py` (`RoutingConfig`) / run.yaml `summarization.routing.legacy_single_voter_policy` | `keep` | ✅ `routing.legacy_single_voter_policy` (added 2026-05-25, moved to `RoutingConfig` in layout v2) | ✅ `eval/silver/run_summarization_sweeps.py --stage map_routing_policy`, `eval/silver/map_theta_sweep.py --legacy-single-voter-policy {keep,escalate,both}` |
| Voter roster (L1/L2/L3 models) | `batch/voter_configs.py` | `make_l1/l2/l3_voters()` | ⚠️ code | indirectly (cost/quality axis) |
| Voter `max_tokens` | `config.py:23` | 16384 | partial (`DEFAULT_MAX_TOKENS`) | ⛔ |
| `chunk_workers` | `config.py:43` | 5 | ✅ | n/a (perf only) |

Hidden MAP knobs (live in prod, **not** in `SummarizationConfig`) → [§3](#3-hidden--orphan-knobs).

---

## 2. GROUNDING (NLI entailment filter)

`helpers/grounding_filter.py`, `nli_config.py`.

| Knob | Where | Prod value | In config? | Harness? |
|---|---|---|---|---|
| `threshold` (keep if entail ≥) | `config.py:77` / run.yaml:98 | **0.5** | ✅ `grounding.threshold` (None disables) | ✅ `eval/sweeps/grounding.py`, `pipeline_sweep.run_grounding_sweep` |
| NLI model | `nli_config.py` / `configs/nli_models.yaml` | registry `default`, env `NLP_HISTO_NLI_MODEL` | ⚠️ env/yaml | ⛔ (registry exists, no model-comparison run) |
| `batch_size` | `nli_config.py` / env | 16 | env | n/a (perf only) |
| Window budget `_MODEL_MAX_TOKENS` / `_PREMISE_BUDGET_FLOOR` | `grounding_filter.py:191,192` | 512 / 64 | ❌ hardcoded | n/a (correctness, not quality) |

Same NLI model + batch_size feed RELATE (shared singleton).

---

## 3. NORMALIZE

`current_stages/normalize_stage.py`, `umls_utils.py`.

| Knob | Where | Prod value | In config? | Harness? |
|---|---|---|---|---|
| `UMLS_THRESHOLD` (link confidence) | `umls_utils.py:11` | **0.85** | ❌ hardcoded | ⛔ |
| Direction keyword triggers | `normalize_stage.py:219,243` | `_NEGATIVE_TRIGGERS` / `_POSITIVE_TRIGGERS` lists | ❌ hardcoded | ⛔ |
| Synonym map | `synonyms.yaml` + `normalize.extra_synonyms` | curated | ✅ (overrides only) | ⛔ |
| `JUNK_SEMANTIC_TYPES` | `umls_utils.py:16` | fixed set | ❌ hardcoded | ⛔ |
| Dedup key composition | `normalize_stage.py:281` | (te_id, subj, out, rel, dir, cat) | ❌ design choice | ⛔ |

---

## 4-bis. GROUP / CANONICALIZE

Both **deterministic, no numeric thresholds.**

- GROUP (`group_stage.py`): grouping key = (subject, outcome, relation_type,
  category); direction deliberately excluded. `scope_heterogeneity` is a
  *reported metric*, not a gate. Only "knob" is key composition (design choice).
- CANONICALIZE (`canonicalize_stage.py`): predicate = highest
  `mean_grounding_score` member (`_pick_best_predicate_deterministic`), no LLM,
  no threshold. Direction split governed by `POLARITY_BEARING_DIRS` (design).

Nothing to sweep here unless we reintroduce a scored/LLM predicate selector.

---

## 4. RELATE

`current_stages/relate_stage.py`.

| Knob | Where | Prod value | In config? | Harness? |
|---|---|---|---|---|
| `entailment_threshold` (→ SUPPORT) | `config.py:85` / run.yaml:101 | **0.50** | ✅ `relate.entailment_threshold` | ✅ `pipeline_sweep.run_relate_sweep` |
| `contradiction_threshold` (→ CONTRADICT) | `config.py:88` / run.yaml:102 | **0.50** | ✅ `relate.contradiction_threshold` | ✅ `run_relate_sweep` |
| Comparability gate | `relate_stage.py:225` (`_should_compare`) | CUI-first, else normalized-string | ❌ design choice | ⛔ |
| `same_polarity` guard | `relate_stage.py:162` | pos/partial vs neg/absent sets | ❌ design | ⛔ |
| NLI model | shared w/ GROUNDING | — | env/yaml | ⛔ |

> RELATE is **built for offline sweeps**: `relate()` returns `raw_pairs`
> (`RawNLIPair`) for *every* eligible pair with all four NLI scores, so
> thresholds replay with zero NLI re-inference. Note the offline classifier
> (`_classify_pair_offline`) still supports `SCOPE_QUALIFY`, which production
> dropped (B-006) — the sweep surface is wider than the live stage.

---

## 5. RESOLVE (deterministic scoring)

`current_stages/resolve_stage.py` — all weights in `ResolveConfig`.

| Knob | `config.py` | Prod | In config? | Harness? |
|---|---|---|---|---|
| `grounding_weight` | :102 / run.yaml:105 | 0.60 | ✅ | ⛔ |
| `grounding_default` | :105 / run.yaml:106 | 0.50 | ✅ | ⛔ |
| `finding_bonus_max` / `finding_bonus_scale` | :108/:111 | 0.10 / 5 | ✅ | ⛔ |
| `support_boost_per_rel` / `_cap` | :114/:117 | 0.08 / 0.20 | ✅ | ⛔ |
| `single_study_pen` | :120 | 0.10 | ✅ | ⛔ |
| `contradict_pen_per_rel` / `_cap` | :123/:126 | 0.15 / 0.30 | ✅ | ⛔ |
| relations-absent variants | :130–137 / run.yaml:114 | 0.80 / 0.15 / 0.05 | ✅ | ⛔ |

**All exposed, none calibrated** — there is no gold notion of a "good final
ranking." This is the hardest stage to calibrate and the clearest candidate for
a synthetic/fake dataset ([§6](#6-gaps--candidates-for-new-experiments)).

---

## Cross-cutting

| Knob | Where | Prod | In config? | Harness? |
|---|---|---|---|---|
| `contradiction_similarity_threshold` (ContradictionDetector) | `config.py:199` / run.yaml:130 | **0.7** (None disables) | ✅ | ⛔ |
| Agreement embedder | `agreement/providers.py` | OpenAI `text-embedding-3-small` (Gemini available) | ⚠️ code | sweep `--embedder` flag |
| Polarity-conflict hard-fail policy | `agreement/polarity_conflict.py:84` | escalate iff comparable `{positive, negative}` pair | ❌ policy in code | ⛔ (breadth tracked B-025) |

---

## 3. Hidden / orphan knobs

Live in production but **cannot be set from `SummarizationConfig`** — to sweep
them today you must edit source. Promotion candidates:

| Knob | Where | Value | Why it's orphaned |
|---|---|---|---|
| `grounding_floor` | `embedding_similarity.py:49` | **0.50** (inert) | `from_config` deliberately skips it (:61); no `AgreementConfig` field. **Only active when an `AgreementContext` is supplied (router path); inert on the legacy/production path** — see [Addendum](#addendum--map-sweep-validity--the-grounding_floor-caveat-verified-from-code-2026-05-24). |
| `_NUMERIC_RATIO_THRESHOLD` | `embedding.py:31` | 2.0 | Module constant; live via `_numeric_contradiction_ratio` inside the prod scorer. |
| `_NEG_WINDOW` | `embedding.py:26` | 3 | Negation lookback window for `_polarity`; live in prod. |
| Polarity vocab (`_POSITIVE`/`_NEGATIVE`/`_NEGATIONS`) | `embedding.py:16–25` | fixed sets | Drives contradiction penalty; live in prod. |
| Hybrid signal weights (`w_category/embedding/entity/evidence`) | `hybrid_structured.py:87` | 0.25/0.40/0.25/0.10 | Only matter if hybrid scorer is selected; `from_config` leaves them at defaults. |
| `UMLS_THRESHOLD` | `umls_utils.py:11` | 0.85 | see §3 NORMALIZE. |
| Direction trigger lists | `normalize_stage.py:219,243` | fixed | see §3 NORMALIZE. |

---

## 4. Config ↔ stage-default divergences

Same knob, two different defaults in two files. Masked in production (runner
passes the config value) but a trap for any sweep that constructs a stage
directly.

| Stage | `config.py` default | Stage constructor default | Prod (run.yaml) |
|---|---|---|---|
| MAP `theta` | 0.8 | `MapStage(theta=0.7)` (`map_stage.py:160`) | **0.8** |
| RELATE `entailment_threshold` | 0.50 | `RelateStage(...=0.55)` (`relate_stage.py:312`) | **0.50** |
| RELATE `contradiction_threshold` | 0.50 | `RelateStage(...=0.65)` (`relate_stage.py:313`) | **0.50** |

`corpus_relate` and the relate-sweep test inherit the **stale 0.55/0.65**
constructor defaults — worth reconciling before relying on a directly-built
stage in an experiment.

---

## 5. Existing calibration infrastructure (don't reinvent)

A "fake dataset for scoring" **already exists for the MAP findings**:

- **Silver dataset** — `eval/silver/`. `generator.py` calls Claude Opus 4.7
  (`extract_findings` tool) to produce reference findings per source case;
  `schemas.py` (SilverFinding), `split.py` (dev/test), `matcher.py`
  (embedding-similarity match of pipeline→silver findings). This is the gold
  for P/R/F1 of MAP / GROUNDING / RELATE outputs.

| Target | Harness | Method |
|---|---|---|
| MAP θ / reject_θ / scorer / agreement weights | `eval/silver/map_theta_sweep.py` | batch-prime L1+L2+L3 once, replay cascade offline over a grid vs silver F1, + deferral-safety (`early_accept_rate/precision`, `escalate_rate`) |
| GROUNDING threshold | `eval/sweeps/grounding.py` (frozen-artifact), `pipeline_sweep.run_grounding_sweep` | retention vs threshold from persisted `grounding_score`, no NLI re-run |
| RELATE ent/con thresholds | `pipeline_sweep.run_relate_sweep` + `_classify_pair_offline` | re-classify `raw_pairs` offline |
| Hybrid agreement (emb+ner) thresholds | `agreement/calibration/` (`DatasetCollector`, `GoldLabeler`, `ThresholdOptimizer`) | LP picks `keep_emb/keep_ner/reject_emb` under cost + precision/recall constraints — a **distinct** mechanism for `CascadedCompositeScorer`, not the prod `SemanticAgreementScorer` path |
| Eval **matcher** similarity threshold | `eval/silver/sweep.py` | calibrates the *measurement* (finding-match cutoff), not the pipeline |

> `scripts/eval/run_all_sweeps.py` is **PDF-extraction** (Docling/TATR), not
> summarization — don't reuse it here.

---

## 6. Gaps & candidates for new experiments

Calibratable but with **no harness today** (rough order of value):

1. **RESOLVE weights** (§5) — no gold for ranking. Needs a synthetic/fake
   dataset: constructed CanonicalRules with known "should-rank-above" pairs, or
   silver-derived rule importance. Hardest, highest design cost.
2. **NORMALIZE `UMLS_THRESHOLD` + direction triggers** — needs (a) surface→canonical
   gold pairs, (b) claim→direction gold. Both are buildable as small fake
   datasets and are isolated/deterministic (cheap to sweep).
3. **`grounding_floor`** and the other §3 orphans — first promote to
   `AgreementConfig`, then fold into the existing `map_theta_sweep` ScorerSpec
   grid (the replay machinery already exists).
4. **`contradiction_similarity_threshold`** (ContradictionDetector) — needs a
   small labeled contradiction set.
5. **Polarity-conflict policy breadth** (B-025) — does adding `absent`/`partial`
   to the hard-fail pair help? Replayable on the MAP voter cache.
6. **NLI model comparison** — the registry (`configs/nli_models.yaml`) holds
   candidates; an entailment-labeled set would let us pick a model, not just a
   threshold.

**Where a fake dataset is the natural unlock:** RESOLVE ranking, NORMALIZE
entity-canonicalization + direction inference, and the ContradictionDetector —
none have an organic gold the way MAP findings do via silver.

---

## Addendum — MAP sweep validity & the `grounding_floor` caveat (verified from code, 2026-05-24)

Answers to "is the current MAP sweep affected by the `MapStage` constructor drift,
and is it a complete agreement-surface calibration?" — all checked in source.

1. **The sweep never instantiates `MapStage`.** `map_theta_sweep._replay`
   (`:650`) builds `AgreementChecker(scorer, theta, reject_theta)` directly
   (`:668`) and walks L1→L2→L3 from `voter_cache`. So the `MapStage(theta=0.7)`
   constructor drift (§4) **cannot affect the sweep**.
2. **θ / reject_θ always come from the grid.** `run_sweep` iterates
   `THETA_GRID × REJECT_THETA_GRID` and passes each explicitly (`:759–761`,
   skip when `reject_theta ≥ theta`). `AgreementChecker`'s own `theta=0.7`
   default is overridden too. The production point (θ=0.8, reject_θ=0.2) is in
   the grid.
3. **Scorer is built through `AgreementConfig` / `ScorerSpec`** — `_build_scorer`
   → `SemanticAgreementScorer(EmbeddingSimilarityStrategy.from_config(AgreementConfig()))`
   (`:143–158`), identical to `runner.py:211`. Default `AgreementConfig()` weights
   equal run.yaml (0.15 / 0.25 / 0.15 / 0.20).
4. **`grounding_floor` is frozen at 0.50 AND inert — not merely frozen.** The
   legacy path calls `agreement.compute(voters, source_text=...)` with **no
   `AgreementContext`** (`decision.py:112`); the sweep does the same
   (`:690, :703`). With `context=None`, `SemanticAgreementScorer` forwards
   `ctx_sub=None` (`semantic_scorer.py:154–159`), so `EmbeddingSimilarityStrategy`
   keeps `grounding_factor=1.0` and **never reads `grounding_floor`**
   (`embedding_similarity.py:203–210`). The `_pass_frac`/`_mean_ev` fallback only
   **tie-breaks `best_index`** (`semantic_scorer.py:177, 247`), never the deferral
   score. `grounding_floor` is live only on the **router path**
   (`enable_router=true`); production runs `enable_router=false`.
5. **Verdict:** valid baseline for **θ / reject_θ / scorer / agreement-weights**
   on the production legacy path. Honest label = **"grounding penalty OFF"**, not
   "grounding_floor=0.50" (its value is a no-op today). **Keep the sweep.** A
   naive "expose `grounding_floor` + add a column + re-sweep" would produce
   **identical rows across all floor values**.

**To actually calibrate `grounding_floor`** (follow-up; no voter re-runs):
- Add `grounding_floor` to `AgreementConfig`; have
  `EmbeddingSimilarityStrategy.from_config` forward it (today it doesn't).
- In `_replay`, build an `AgreementContext`: compute per-voter
  `grounding_pass_fraction` via **NLI grounding on the cached voter findings**
  (local model, no LLM voter calls). Persist these scores next to
  `voter_cache.json` so they're computed once, not per grid cell. This couples in
  the GROUNDING threshold (pass/fail per finding).
- CSV: add `grounding_floor` (+ `nli_model_id`) to `fieldnames`/row.
- **Caches stay valid:** `grounding_floor` changes neither voter LLM outputs
  (`voter_cache.json`) nor embeddings (`EmbeddingCache`) — it is a post-embedding
  multiply. No re-prime, no re-embed.
- Note: supplying a context makes the *legacy* replay diverge from
  production-legacy, so this is really a **grounding-aware-agreement** experiment,
  coupled to the router/legacy decision (B-062) — not a standalone knob sweep.

---

*Next step (separate): design the per-stage experiments — metric, dataset,
grid, and dev/test discipline — starting from the gaps in §6, plus the
`grounding_floor` follow-up above.*
