# SILVER_REPLAY_DESIGN.md

End-to-end and per-stage evaluation via **silver MAP findings replayed through stages 2–6**.

Companion to `STAGE_EVAL_DESIGN.md` (which covers per-stage *isolated* judges). This doc covers the *cumulative* per-stage evaluation that falls out of one architectural move: take silver MAP findings already produced by `eval/silver/generator.py`, push them through the same NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE code paths the pipeline uses, and compare the pipeline's output to the silver-replay output at every stage exit.

The result is **end-to-end F1 plus a stage-by-stage F1 curve from one silver source** — without hand-curating any silver gold at NORMALIZE/GROUP/CANONICALIZE/RELATE/RESOLVE.

## §1. Why this approach

### 1.1 What silver replay gives you

A single Opus-extracted silver MAP set, when replayed through stages 2–6, produces reference outputs at every downstream stage. Because stages 2–5 are deterministic (NORMALIZE / GROUP / CANONICALIZE) or model-deterministic given fixed weights (RELATE NLI scores; RESOLVE arithmetic), the silver-replay output at stage N is *exactly what the pipeline would have produced if its MAP had emitted the silver findings*.

Compare that to the pipeline's actual output at the same stage exit, and the gap is the **cumulative error accruing from MAP up to and including stage N**.

```
                  pipeline MAP   ─▶ NORMALIZE ─▶ GROUP ─▶ CANON ─▶ RELATE ─▶ RESOLVE ─▶ pipeline FinalRules
                                       │           │         │         │          │
silver MAP findings  ─────────────▶ NORMALIZE ─▶ GROUP ─▶ CANON ─▶ RELATE ─▶ RESOLVE ─▶ silver FinalRules
                                       │           │         │         │          │
                                       ▼           ▼         ▼         ▼          ▼
                              cumulative F1 at each stage exit
```

### 1.2 What it doesn't give you

This framework measures **cumulative drift**, not **isolated stage error**. Example: if NORMALIZE F1 looks bad in this framework, the cause is partly NORMALIZE itself and partly whatever errors MAP injected upstream. To attribute error to individual stages, use the per-stage judges from `STAGE_EVAL_DESIGN.md` (Q1, N-merge, N-split, Gr-purity, etc.).

The two frameworks are complementary:

| Framework | Question answered | Cost |
| --- | --- | --- |
| **Silver replay** (this doc) | How much does pipeline drift from a silver reference at each stage exit? End-to-end F1. | One silver MAP run + 0 extra judge calls for stages 2–6 (alignment uses embedding cache from `eval/silver/matching/matcher.py`). |
| **Per-stage judges** (`STAGE_EVAL_DESIGN.md`) | Where is each individual stage's error rate? | One judge call per emitted item per stage. |

Both run on the same 15-paper selection. Both feed the final report.

### 1.3 Why not hand-curated gold

- 15 papers × ~100 paragraphs × hours of expert read-time = multiple days of human work.
- One human curator's judgement is hard to reproduce.
- Hand curation cannot scale to threshold sweeps.
- Silver replay can be re-run for every cascade tier (Haiku / Sonnet / Opus pipeline) at zero additional silver cost.

### 1.4 Honest caveats

- **Silver is silver, not gold.** Silver MAP carries Opus's biases. Use the existing Q1 + Q3 + Q5 judges to bound Opus-silver quality on the same 15-paper set. Report E2E F1 with the caveat "silver = Opus".
- **Conditional metric.** E2E F1 measures pipeline behavior *given perfect MAP*. It does not bound pipeline performance against true clinical ground truth.
- **Silver MAP doesn't have grounding scores.** RESOLVE uses `mean_grounding_score` as a base. We assign a sentinel value (§4.2) and document the bias.

## §2. Inputs we already have

- `eval/silver/generator.py` — produces `SilverCaseResult.findings: list[SilverFinding]` per paragraph, using Opus + the existing tool-schema (`eval/silver/data/schemas.py:SilverFinding`).
- `eval/silver/data/schemas.py:SilverFinding` fields: `claim, subject_entity, outcome_entity, relation_type, direction, category, confidence, verbatim_support, scope, source_sentence_ids`.
- `eval/silver/matching/matcher.py` — embedding-based alignment with cache + `compute_metrics` for P/R/F1, used today for MAP-only comparison. Reusable for every downstream stage.
- `eval/silver/matching/embedders.py` — OpenAI / Gemini embedders with disk cache.
- `eval/silver/evaluate.py` — existing CLI that does MAP-level silver eval. The new replay extends the *same metric framework* to later stages.
- `pipeline/stages/summarization/current_stages/{normalize,group,canonicalize,relate,resolve}_stage.py` — the deterministic stages, callable independently of `KnowledgeExtractionRunner`.

## §3. Architecture

Three components, one orchestrator.

```
eval/silver/replay/
├── __init__.py
├── adapter.py            # SilverFinding → Finding (pipeline schema)
├── runner.py             # SilverReplayRunner — drives stages 2-6
├── compare.py            # pipeline-vs-silver alignment per stage
└── metrics.py            # cumulative F1 tables, sweeps
```

The orchestrator entrypoint:

```bash
python -m eval.silver.replay \
  --pipeline-run-id <int>                 # the KnowledgeExtractionRunner run to evaluate
  --silver-run <path>                     # silver MAP output dir (from eval/silver/generator.py)
  --output eval/reports/replay_<date>     # where to write per-stage CSVs + alignments
```

### 3.1 Adapter (`adapter.py`)

`SilverFinding → Finding` conversion. Required because NORMALIZE consumes `Finding` objects, not silver schemas.

Mapping:

| `SilverFinding` field | `Finding` field | Notes |
| --- | --- | --- |
| `claim` | `claim` | direct |
| `subject_entity` | `subject_entity` | direct |
| `outcome_entity` | `outcome_entity` | direct |
| `relation_type` | `relation_type` | enum-coerce; map unknown → `unclear` and log to enum-observation file just like MAP does |
| `direction` | `direction` | enum-coerce; unknown → `unclear`; also handle missing `no_direction` if silver schema lacks it |
| `category` | `category` | direct |
| `verbatim_support` | `verbatim_support` | direct |
| `scope` | `scope` | direct (same `FindingScope` shape) |
| `confidence` | (dropped) | not consumed by NORMALIZE |
| `source_sentence_ids` | → `evidence` | needs **sentence_id → text_element_id** resolution from `TextElement` (see §3.2 caveat) |

Synthesised fields:

- `grounding_score`: assign `1.0` (silver is "perfectly grounded by construction"). Rationale: silver findings were extracted with `verbatim_support` constraints in the silver prompt. Document the bias.
- `finding_id`: compute via `compute_finding_id(pmcid, chunk_id, position_in_chunk, claim)` — same function the pipeline uses (`pipeline/stages/summarization/models.py:29-44`). We need a synthetic `chunk_id` and `position_in_chunk` because silver has no chunks. Use `chunk_id = "silver_" + sha8(pmcid + te_id)` and `position_in_chunk = 0..N` over the per-paragraph silver findings.

### 3.2 Evidence-string synthesis

Pipeline evidence format: `"[S{i}|{pmcid}|{te_id}]"`. NORMALIZE parses this to extract `(sentence_id, text_element_id)` for span building (`normalize_stage.py:354-379`).

`SilverFinding.source_sentence_ids: list[int]` carries `sentence_id`s but **not** `text_element_id`s. Resolution:

1. Silver was generated per-paragraph (per `eval/silver/generator.py`). The case being judged has a known `te_id`.
2. Adapter receives `(silver_case, te_id)` together. For each `source_sentence_id`, emit `f"[S{sid}|{pmcid}|{te_id}]"`.

If a silver finding spans multiple paragraphs (cross-paragraph aggregate finding), it gets multiple evidence strings, one per `(sentence_id, te_id)` pair. The adapter accepts an optional `cross_paragraph_te_ids: dict[int, int]` for this case; default is single-paragraph.

### 3.3 SilverReplayRunner (`runner.py`)

Wraps the post-MAP stages exactly as `KnowledgeExtractionRunner` does, but starts from a `list[Finding]` (silver-adapted) and writes to a separate artifact root.

```python
class SilverReplayRunner:
    def __init__(self, *, config, artifact_root: Path, ...): ...

    def replay_paper(self, pmcid: str, silver_findings: list[Finding]) -> ReplayResult:
        # 1. NORMALIZE
        normal = self._normalize.process(silver_findings, pmcid)
        # 2. GROUP — filter groupable first
        groupable = [nf for nf in normal if is_groupable(nf)]
        groups = self._group.group(groupable, pmcid)
        # 3. CANONICALIZE
        canon = self._canonicalize.canonicalize(groups, {nf.normal_id: nf for nf in normal})
        # 4. RELATE
        relations, raw_pairs, skipped = self._relate.relate(canon, pmcid)
        # 5. RESOLVE
        finals = self._resolve.resolve(canon, relations)
        return ReplayResult(normal, groups, canon, relations, raw_pairs, finals)
```

Persistence mirrors `pipeline/stages/summarization/persistence.py` so the output tree is *identical* in shape to a pipeline run — only the root differs:

```
<artifact_root>/silver_replay_<run_id>/
  manifest.json                        # extra.replay_source = silver_run_id, extra.pipeline_config_hash
  normalize/{pmcid}/normal_findings.jsonl
  group/{pmcid}/groups.jsonl
  canonicalize/{pmcid}/canonical_rules.jsonl
  relate/{pmcid}/{relations,raw_pairs,skipped_pairs}.jsonl
  resolve/{pmcid}/final_rules.jsonl
```

This means **`eval/silver/compare.py` can use the same loader for pipeline output and silver-replay output**.

### 3.4 Determinism guarantee

Stages 2–6 are deterministic under the following conditions, all of which we honor:

- NORMALIZE: pure Python; UMLS linker cached on first load. **Caveat:** scispaCy thread-safety — replay must run single-threaded or with the same shared linker singleton as the pipeline did.
- GROUP: SHA8 hashes, no randomness.
- CANONICALIZE: argmax with stable insertion-order tie-break.
- RELATE: NLI scores are deterministic for a fixed model + fixed inputs + fixed batch size + same hardware-precision (fp32 on CPU, fp16 on GPU may drift). Document the hardware setting.
- RESOLVE: arithmetic.

For thesis reproducibility, pin: scispaCy model + linker version, NLI model hash, hardware (CPU vs CUDA vs MPS), batch size, Python + numpy + torch versions, in `manifest.extra.runtime`.

## §4. Per-stage comparison and metrics

Same matcher (`eval/silver/matching/matcher.py`) at every stage; only the embedding input string changes.

### 4.1 MAP exit (sanity check)

Already implemented in `eval/silver/evaluate.py`. Compares pipeline MAP findings vs silver MAP findings. Embedding string:

```
claim | subject_entity | outcome_entity | relation_type | direction | category
```

Greedy one-to-one match above threshold (default 0.55). Output: P, R, F1, strict_F1.

This is the **anchor** that bounds silver quality before we use it downstream.

### 4.2 NORMALIZE exit

Embedding string per `NormalFinding`:

```
predicate_text | subject_entity | outcome_entity | subject_cui | outcome_cui | relation_type | direction | category
```

Adding CUI to the embedding input rewards correct entity linking. Threshold: same default (0.55), tunable.

Metrics:

- **NORMALIZE-P**: |pipeline ∩ silver-replay| / |pipeline|.
- **NORMALIZE-R**: |pipeline ∩ silver-replay| / |silver-replay|.
- **NORMALIZE-F1**: harmonic mean.
- **NORMALIZE-strict-F1**: penalize matches where any of `{category, relation_type, direction, subject_cui, outcome_cui}` mismatches.
- **NORMALIZE-merge-ratio**: |silver-replay NormalFindings| / |silver MAP findings|. Compare to pipeline's |pipeline NormalFindings| / |pipeline MAP findings|. Big ratio gap means the two MAP outputs have different duplicate structure.

### 4.3 GROUP exit

Embedding string per `FindingGroup`:

```
subject_entity | outcome_entity | relation_type | category | sorted(direction_counts.keys())
```

Metrics:

- **GROUP-P, GROUP-R, GROUP-F1** as above.
- **GROUP-size-distribution**: median + p90 of `len(member_ids)` for pipeline vs silver-replay.
- **GROUP-purity@k**: of the top-k largest groups, agreement on member set (Jaccard against the matched silver group's members).

Member-set agreement requires cross-referencing `member_ids` → NormalFindings. Use the NORMALIZE alignment from §4.2 to map silver-replay `normal_id` → pipeline `normal_id`, then compute Jaccard on each matched group pair.

### 4.4 CANONICALIZE exit

Embedding string per `CanonicalRule`:

```
predicate_text | subject_entity | outcome_entity | relation_type | direction | category
```

Metrics:

- **CANON-P, CANON-R, CANON-F1**.
- **direction-bin agreement**: when a pipeline rule matches a silver-replay rule by `(subject, outcome, relation_type, category)` but not by `direction`, count it as a bin-split disagreement. Report `bin_agreement_rate`.
- **predicate-pick agreement**: among matched pairs, semantic equivalence of `predicate_text` (cosine ≥ 0.85). Captures "did the argmax-grounding heuristic pick a comparable surface form".

### 4.5 RELATE exit

Two passes:

1. **Relation-set comparison** (matches above). Embedding string per `Relation`:

```
relation_type | direction_a | direction_b | rule_a.subject_entity | rule_a.outcome_entity | rule_b.subject_entity | rule_b.outcome_entity
```

Match pipeline relations to silver-replay relations. Output: RELATE-P, RELATE-R, RELATE-F1 **per class** (SUPPORT-F1, CONTRADICT-F1, SCOPE_QUALIFY-F1).

2. **Raw-pair score alignment** (the more thesis-interesting one). For pairs that exist in both pipeline and silver-replay raw_pairs (matched by `(rule_id_a, rule_id_b)` after CANONICALIZE alignment), compare NLI score distributions:

   - Spearman ρ between pipeline NLI scores and silver-replay NLI scores. Should be ≥ 0.95 because the NLI model is identical — anything lower means input drift through CANONICALIZE.

Skipped-pair audit: report `|skipped_pipeline ∪ skipped_silver|` and the intersection. Disagreement on gate decisions is itself signal.

### 4.6 RESOLVE exit (end-to-end FinalRule F1)

This is the headline number. Embedding string per `FinalRule`:

```
predicate_text | subject_entity | outcome_entity | relation_type | direction | category
```

Metrics:

- **E2E-P, E2E-R, E2E-F1** — alignment as above.
- **E2E-strict-F1** — penalize matched pairs that disagree on `is_contradicted` or have `final_score` differing by more than 0.10.
- **Top-k F1** (k ∈ {5, 10, 20}) — restrict to top-k by `final_score` on each side; compute F1 on the intersection. Tests whether RESOLVE puts the most credible rules at the top.
- **Spearman ρ** between pipeline `final_score` and silver-replay `final_score` on matched pairs — measures scoring consistency given identical inputs.
- **Score-bucket calibration** — bucket pipeline final_score into deciles; for each decile report the fraction of pipeline rules that match a silver-replay rule. A well-calibrated RESOLVE shows monotone increase across deciles.

### 4.7 Summary table per paper

Each replay run writes `eval/reports/replay_<date>/per_paper.csv`:

| pmcid | stratum | MAP-F1 | NORM-F1 | GROUP-F1 | CANON-F1 | RELATE-F1 | E2E-F1 | E2E-top10-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

And `corpus.csv` with the same columns aggregated (mean + bootstrap 95% CI over 1000 resamples).

## §5. Sampling, thresholds, and sweeps

- **Paper set**: the selected-15 (5 related + 5 diverse + 5 hard). Report each stratum separately *and* combined so the "hard" bucket can carry the regression-detection claim.
- **Embedding threshold**: default 0.55 from `eval/silver/matching/matcher.py`. Sweep ∈ {0.45, 0.50, 0.55, 0.60, 0.65} and report curves so the headline F1 isn't a single arbitrary cutoff.
- **Cascade tiers** (summarization model):
  - Run silver replay once with Opus silver MAP (the silver source itself).
  - Run pipeline three times with Haiku / Sonnet / Opus cascades.
  - Each combination yields a per-stage F1 row → 3 rows per paper for the report table.

## §6. Implementation order

In a separate PR sequence from `STAGE_EVAL_DESIGN.md`'s per-stage judges. Both can ship in parallel.

1. **Adapter** (`adapter.py`) + unit tests (round-trip `SilverFinding ↔ Finding`, evidence-string parse-back, `compute_finding_id` stability).
2. **SilverReplayRunner** (`runner.py`). Wraps post-MAP stages; reuses `persistence.py` writers. Pin determinism settings (§3.4).
3. **MAP-exit sanity check** — run replay, confirm `eval/silver/evaluate.py` still passes against the new path (it should be byte-identical for stage 1).
4. **NORMALIZE-exit comparison** (`compare.py:compare_normalize`) — reuses `compute_sim_matrix`, `match_from_matrix`, `compute_metrics` from `eval/silver/matching/matcher.py`. Validate on 2 papers.
5. **GROUP-exit** — adds member-set Jaccard via NORMALIZE alignment chain.
6. **CANONICALIZE-exit** — adds direction-bin and predicate-pick agreement.
7. **RELATE-exit** — both relation-set match and raw-pair Spearman.
8. **RESOLVE-exit** — E2E-F1 + top-k F1 + score-bucket calibration.
9. **Report writer** (`metrics.py`) — per-paper CSV, corpus CSV with bootstrap CIs, threshold-sweep table, stratum split.
10. **Wire into `eval.silver.evaluate` CLI** as a `--mode replay` subcommand, or keep `python -m eval.silver.replay` as its own entrypoint.

Steps 1–3 are scaffolding; the first usable signal is at step 4.

## §7. Outputs (final form)

After a full Haiku-Sonnet-Opus run on selected-15:

```
eval/reports/replay_<YYYYMMDD>/
  per_paper.csv                          # pmcid × stage F1
  corpus.csv                             # corpus aggregates + 95% CIs
  per_stratum.csv                        # related/diverse/hard × stage F1
  threshold_sweep.csv                    # F1 across embedding thresholds
  alignment_normalize.jsonl              # matched pairs (pipeline_id, silver_id, sim)
  alignment_group.jsonl
  alignment_canonicalize.jsonl
  alignment_relate.jsonl
  alignment_resolve.jsonl
  raw_pair_score_correlation.csv         # Spearman ρ per paper (RELATE)
  resolve_score_calibration.csv          # decile × match-rate (RESOLVE)
  manifest.json                          # silver_run_id, pipeline_run_id, pipeline_config_hash,
                                         # cascade tiers compared, embedding model, threshold,
                                         # determinism pins (scispaCy, NLI model, hardware)
```

This is everything the final `eval/reports/final_<YYYYMMDD>.md` (item 17) needs for the "per-stage F1 curve" section, including the variance / CI numbers the per-stage judges by themselves don't provide.

## §8. Open questions deferred to implementation

- **Cross-paragraph silver findings** — adapter §3.2 assumes single-paragraph. Silver findings that aggregate across paragraphs need careful `te_id` resolution; defer until we hit a real one.
- **MPS / CUDA determinism** for the NLI model — record actual fp precision and document drift if observed.
- **scispaCy version pinning** — bump in `requirements.txt` with a comment; replay must use the same version as the pipeline run being evaluated.
- **What to do when silver-replay produces zero relations** (e.g. tiny paper, ≤ 1 canonical rule): treat RELATE-F1 as undefined for that paper and exclude from corpus aggregate; record in `per_paper.csv` as `NaN`.
- **Should we replay the *grounding filter* too?** Today's spec skips it (silver findings are assumed grounded). Alternative: run grounding on silver MAP and report grounding-rejection rate as a diagnostic. Defer to step 2 of implementation.

## §9. Relationship to STAGE_EVAL_DESIGN.md

This doc and `STAGE_EVAL_DESIGN.md` cover orthogonal axes of evaluation. Both run on the same 15-paper selection and feed the same final report. Decision matrix:

| You want to know… | Use |
| --- | --- |
| Is each MAP finding correctly grounded and labelled? | `STAGE_EVAL_DESIGN.md` §10.1 (Q1) |
| What MAP missed on a paragraph? | `STAGE_EVAL_DESIGN.md` §10.3 (Q3) |
| Is each RELATE label correct? | `STAGE_EVAL_DESIGN.md` §10.2 (Q2) |
| Did NORMALIZE merge correctly? Split correctly? | `STAGE_EVAL_DESIGN.md` §3.6 (N-2) |
| How does pipeline output drift from silver at each stage? | **This doc** |
| End-to-end FinalRule F1? | **This doc** §4.6 |
| How does scoring calibration look across deciles? | **This doc** §4.6 |
| Pipeline vs cascade-tier per-paper variance with CIs? | **This doc** §7 |

Use both. The two together give isolated per-stage error rates (judges) and cumulative end-to-end drift (this doc); neither alone tells the full story.
