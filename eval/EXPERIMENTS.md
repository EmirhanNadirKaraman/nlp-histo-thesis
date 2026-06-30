# Summarization Pipeline — Evaluation Experiments

> **HISTORICAL (planning roadmap, ~2026-05-29).** This file is the original
> forward-looking experiment plan using the `M-/N-/Gr-/C-/R-/Rs-` priority IDs.
> The experiment program was later realized under a **different taxonomy** —
> the `E01`–`E14` series in `eval/silver/experiments/` (run via
> `python -m eval.silver.experiments.<Exx>.<mod>`), with results in
> `eval/reports/RESULTS.md`. Use those for the current experiment set; this file
> is kept for provenance and is **not** a current registry. Several items below
> marked "Not yet implemented" were never built under these IDs.

Judge model: `claude-opus-4-7` (same as existing Q1–Q5 harness).  
All batch-mode costs use 50% Anthropic batch discount.  
Baseline sample: **15 papers, ~80 post-MAP findings/paper**.

---

## Key Insights

### Cache invalidation cascades downstream

The `LlmJudgeCache` key includes `request_inputs` — the actual content passed to
Opus. Each experiment's inputs are the *outputs of its upstream stage*. This means:

- Changing **GROUNDING threshold** → free (G-1 is pre-filter; MAP findings unchanged)
- Changing **NORMALIZE** (e.g. UMLS match count, dedup logic) → invalidates N-1, N-2,
  Gr-1, C-1, C-2, R-1, R-2, Rs-1 — everything downstream pays again
- Changing **MAP** → invalidates the entire cache

Stabilize upstream stages before running downstream experiments or you will pay
for the same Opus calls multiple times.

The only free re-runs regardless of stage changes:
- G-2 threshold sweep (offline, uses cached NLI scores)
- Rs-2 weight sweep (offline, uses cached CanonicalRule + Relation data)

### Run a cheap diagnostic pass before deep optimization

Before committing to optimizing any single stage, run M-1/2/3 (~$18 total,
already built) to get a full-pipeline quality read. You might spend weeks
perfecting NORMALIZE only to find MAP recall is the real bottleneck. The
diagnostic pass reveals where the biggest signal loss is before you invest.

### Go stage-by-stage after the diagnostic pass

Once you have the full-pipeline picture, work upstream to downstream:

1. **MAP** — fix first if precision or recall is bad; everything else is built on it
2. **GROUNDING** — set threshold once via offline sweep; revisit last
3. **NORMALIZE + GROUP** — the blind spots; stabilize before running anything downstream
4. **CANONICALIZE → RELATE → RESOLVE** — only run after upstream is stable

### Some stages cannot be meaningfully improved by iteration

- **GROUP** is deterministic key-bucketing — error rate is fixed by whatever
  NORMALIZE produces; no logic to tune
- **NORMALIZE** is bounded by UMLS coverage — entity errors caused by gaps in the
  UMLS index cannot be fixed without changing the index or adding a fallback
- For these stages, the experiment tells you the error floor; accept it and move on
  rather than trying to perfect something that is externally limited

---

## Priority Matrix

| ID | Stage | Experiment | Priority | Est. cost (batch) |
|----|-------|------------|----------|-------------------|
| M-1 | MAP | Extraction F1 (Q5) | P1 — run now | ~$9 |
| M-2 | MAP | Precision + field accuracy (Q1) | P1 — run now | ~$5 |
| M-3 | MAP | Recall gap detection (Q3) | P1 — run now | ~$4 |
| G-1 | GROUNDING | Pre/post filter accuracy | P1 | ~$20 |
| N-1 | NORMALIZE | Entity normalization accuracy | P1 | ~$2 |
| N-2 | NORMALIZE | False merge audit | P1 | ~$1 |
| Gr-1 | GROUP | Non-groupable rejection audit | P1 | ~$1 |
| C-1 | CANONICALIZE | Direction conflict resolution | P2 | <$1 |
| N-3 | NORMALIZE | Missed dedup audit | P2 | ~$1 + embedding |
| Gr-2 | GROUP | Group membership correctness | P2 | ~$1 |
| C-2 | CANONICALIZE | Predicate text quality | P2 | ~$1 |
| R-1 | RELATE | Label precision (Q2) | P2 — already built | ~$3 |
| R-2 | RELATE | False negative (UNRELATED recall) | P2 | ~$3 |
| Rs-1 | RESOLVE | Score ranking quality | P3 | ~$1 |
| Rs-2 | RESOLVE | Weight sensitivity sweep | P3 | $0 (offline) |
| G-2 | GROUNDING | Threshold sweep | P3 — already built | $0 (offline) |

---

## Stage 0: MAP

MAP has the most coverage of any stage. Run these first to establish a quality
baseline before evaluating downstream stages.

### M-1 — Extraction F1 (Q5) · P1 · ~$9 batch

**Already implemented:** `eval/llm_judge/tests/q5_f1.py`

**What it tests:** paragraph-level precision/recall/F1. Opus generates silver
findings for a paragraph, then aligns them against pipeline findings using
`match_type ∈ {match, partial, unmatched}`.

**Metrics:** `micro_precision`, `micro_recall`, `micro_f1`, `total_tp/fp/fn`,
`total_partial_matches`.

**Known issue (BUGS.md #5):** partial matches reduce FP but don't contribute to
TP, so F1 is pessimistic when most matches are partial. Fix before treating F1
as ground truth.

```
python -m eval.llm_judge --mode batch --tests q5 --n 15
```

---

### M-2 — Precision + field accuracy (Q1) · P1 · ~$5 batch

**Already implemented:** `eval/llm_judge/tests/q1_precision.py`

**What it tests:** for each MAP finding, Opus judges whether it is grounded in
the verbatim support and whether each field (`relation_type`, `direction`,
`category`, `scope_*`) is correct.

**Metrics:** `grounded_rate`, `relation_type_accuracy`, `direction_accuracy`,
`category_accuracy`, `entity_error_rate`, `scope_error_rate`,
`score_calibration` (NLI score bin → Opus grounded rate).

```
python -m eval.llm_judge --mode batch --tests q1 --n 15 --q1-findings 30
```

---

### M-3 — Recall gap detection (Q3) · P1 · ~$4 batch

**Already implemented:** `eval/llm_judge/tests/q3_recall.py`

**What it tests:** Opus reads a paragraph and the pipeline's extracted findings,
then identifies any findings MAP missed. Stratified over paragraphs with and
without extractions.

**Metrics:** `gap_rate_overall`, `gap_rate_by_stratum`, `total_missing_findings`.

```
python -m eval.llm_judge --mode batch --tests q3 --n 15
```

---

## Stage 1a: GROUNDING Filter

### G-1 — Pre/post filter accuracy · P1 · ~$20 batch

**Not yet implemented.**

**What it tests:** whether the NLI filter is making correct keep/drop decisions.
Runs Opus on the *full pre-filter* Finding set (before any dropping), asking
"is this finding grounded in its verbatim support?". Compares Opus's judgment
to the NLI filter's decision.

**Metrics:**
- `filter_precision`: fraction of kept findings Opus also calls grounded
- `filter_recall`: fraction of dropped findings Opus calls *not* grounded
- `false_positive_rate`: Opus-grounded findings that were dropped by NLI
- `false_negative_rate`: Opus-hallucinated findings that survived the filter

**How:** extend Q1 with `grounding_threshold=None` (or 0.0) so all scored
findings are passed through, including those the filter would drop. The
`grounding_score` field is always written on every finding regardless of
keep/drop, so the NLI decision can be reconstructed from
`grounding_score < threshold`.

**Cost driver:** pre-filter set is ~3× larger than post-filter (~200–250
findings/paper vs ~80), hence ~$20 for 15 papers.

---

### G-2 — Threshold sweep (offline) · P3 · $0

**Already implemented:** `eval/silver/pipeline_sweep.py`

**What it tests:** precision/recall/F1 tradeoff at grounding thresholds from
`None` to `0.70`. Uses cached NLI scores — no API calls.

Run after G-1 to pick the threshold that maximises F1 against Opus silver labels.

---

## Stage 2: NORMALIZE

> **Blind spot — no evaluation exists.** NORMALIZE does two lossy operations
> with zero visibility: entity normalization and deduplication. Errors here
> propagate silently through GROUP, CANONICALIZE, RELATE, and RESOLVE.

### N-1 — Entity normalization accuracy · P1 · ~$2 batch

**Not yet implemented.**

**What it tests:** whether UMLS normalization and synonym merging produce correct
canonical entity strings. Opus sees the raw Finding entity string
(pre-normalization) and the NormalFinding entity string (post-normalization) and
judges whether the mapping is correct.

**Requires instrumentation:** `NormalizeStage` currently does not persist
pre-normalization entity strings. Need to log `raw_subject_entity` /
`raw_outcome_entity` before normalization runs, either in the trace or as extra
fields on NormalFinding.

**Metrics:**
- `normalization_accuracy`: fraction of entities correctly mapped
- `error_breakdown` by normalization type (UMLS lookup, synonym merge,
  passthrough)

**Sample size:** ~100 NormalFindings across 5 papers is sufficient (~$2 batch).

---

### N-2 — False merge audit · P1 · ~$1 batch

**Not yet implemented.**

**What it tests:** whether the dedup step incorrectly collapses distinct findings
into one NormalFinding. A false merge permanently loses signal — the merged
finding count inflates `finding_count` in CANONICALIZE and `final_score` in
RESOLVE.

**How:** query `sum_normal_findings` + `sum_normal_finding_spans` where span
count > 1. Show Opus all source verbatim spans and the merged NormalFinding.
Ask: "do all these evidence spans support the same claim, or were distinct
findings incorrectly merged?"

**Metrics:** `false_merge_rate`, `avg_spans_per_false_merge`.

**Sample size:** ~50 multi-span NormalFindings (~$1 batch).

---

### N-3 — Missed dedup audit · P2 · ~$1 batch + embedding cost

**Not yet implemented.**

**What it tests:** whether the dedup step *fails* to merge findings that express
the same claim differently. Missed dedup inflates rule counts and degrades GROUP
grouping quality.

**How:** embed all NormalFindings per paper using the same embedding model
(`text-embedding-3-small`). Find high-cosine pairs (> 0.85) that were NOT
merged. Show to Opus: "are these two findings actually the same claim?"

**Metrics:** `missed_merge_rate` among high-cosine pairs.

---

## Stage 3: GROUP

### Gr-1 — Non-groupable rejection audit · P1 · ~$1 batch

**Not yet implemented.**

**What it tests:** whether findings dropped at the `is_groupable()` gate are
correct exclusions or real signal being silently lost. A NormalFinding is
rejected if `subject_entity=None`, `outcome_entity=None`, or
`relation_type=unclear`. The `non_groupable_rate` is tracked in
`rejection_summary` but never audited for correctness.

**How:** read `sum_rejected_findings` where `stage='group_non_groupable'`. Show
Opus each rejected finding with its verbatim support. Ask: "is this a valid
finding that should have been kept?"

**Metrics:**
- `false_rejection_rate`: Opus says "should have been kept" / total rejected
- Breakdown by reason (`no_subject`, `no_outcome`, `unclear_relation`)

**Sample size:** ~30 rejected findings per paper, 5 papers → ~150 findings
(~$1 batch). Cheapest high-signal experiment in this list.

---

### Gr-2 — Group membership correctness · P2 · ~$1 batch

**Not yet implemented.**

**What it tests:** whether large FindingGroups (> 3 members) contain findings
that genuinely belong together, or whether the deterministic
`(subject, outcome, relation_type, category)` key is over-grouping distinct
facts.

**How:** query `sum_finding_groups` + members. Filter groups with ≥ 4 members.
Show Opus all member NormalFindings and ask: "do all these belong in the same
group, or should any be separated?"

**Metrics:** `over_grouping_rate`, `avg_false_members_per_large_group`.

---

## Stage 4: CANONICALIZE

### C-1 — Direction conflict resolution · P2 · <$1 batch

**Not yet implemented.**

**What it tests:** for `is_conflicted=True` CanonicalRules (groups where members
disagree on direction), whether CANONICALIZE picks the correct direction given
the evidence.

**How:** query `sum_canonical_rules` where `is_conflicted=True`. Join to member
findings via `sum_group_members` → `sum_normal_findings`. Show Opus all member
findings and the chosen `direction`. Ask: "is the chosen direction correct, or
should it be different?"

**Metrics:** `conflict_resolution_accuracy`.

**Sample size:** conflicted rules are a small minority — likely < 30 across 15
papers, so cost is negligible.

---

### C-2 — Predicate text quality · P2 · ~$1 batch

**Not yet implemented.**

**What it tests:** whether the canonicalized `predicate_text` faithfully and
concisely represents the finding group. Predicate text is the NLI input in
RELATE and the final rule surface form in RESOLVE.

**How:** sample CanonicalRules, show Opus the member NormalFinding
`predicate_text` values and the chosen canonical text. Ask: "is this predicate
text accurate and representative?"

**Metrics:** `predicate_accuracy`, common failure modes (too vague, wrong
direction embedded, entity dropped).

---

## Stage 5: RELATE

### R-1 — Label precision (Q2) · P2 · ~$3 batch

**Already implemented:** `eval/llm_judge/tests/q2_relations.py`

**What it tests:** whether SUPPORT / SCOPE_QUALIFY / CONTRADICT labels on
persisted Relations are correct. Opus makes a blind judgment given both rule
texts and the NLI scores.

**Metrics:** `accuracy`, confusion matrix, `nli_score_calibration`.

**Known issue (BUGS.md #2):** silent skip if `canonical_id` format diverges from
`rule_id_a`. Verify this is not silently dropping cases before reading results.

```
python -m eval.llm_judge --mode batch --tests q2 --n 15 --q2-relations 15
```

---

### R-2 — False negative audit (UNRELATED recall) · P2 · ~$3 batch

**Not yet implemented.**

**What it tests:** whether the NLI thresholds cause RELATE to classify real
relations as UNRELATED. Q2 only measures precision of persisted relations — it
cannot detect false negatives.

**How:** read `relate_raw_pairs` from per-paper result JSONs. Sample pairs where
the NLI score is just below the entailment or contradiction threshold. Show Opus
both rule texts + scores. Ask: "should this be SUPPORT, CONTRADICT,
SCOPE_QUALIFY, or UNRELATED?"

**Metrics:** `false_negative_rate` among near-threshold pairs.

---

## Stage 6: RESOLVE

### Rs-1 — Score ranking quality · P3 · ~$1 batch

**Not yet implemented.**

**What it tests:** whether the hand-tuned `final_score` formula (weights in
`ResolveConfig`) produces rankings that match expert judgment. Weights are not
empirically validated (PIPELINE.md DES-8).

**How:** for each paper, show Opus the top-10 FinalRules sorted by `final_score`
(scores hidden). Ask Opus to rank them by importance. Compare ranks.

**Metrics:** Spearman rank correlation between Opus rank and pipeline rank.

---

### Rs-2 — Weight sensitivity sweep (offline) · P3 · $0

**Not yet implemented. Requires Rs-1 silver ranks first.**

**What it tests:** whether different weight combinations improve Spearman
correlation against Rs-1 silver labels. Five free parameters:
`grounding_weight`, `finding_bonus_max`, `support_boost_per_rel`,
`contradict_pen_per_rel`, `single_study_pen`.

**How:** deterministic offline replay of the RESOLVE formula over a grid of
weight combinations using cached CanonicalRule + Relation data. No API calls.

**Metrics:** Spearman correlation vs weight grid → optimal weights.

---

## Recommended Running Order

```
# Week 1 — MAP baselines (all already implemented)
python -m eval.llm_judge --mode batch --tests q1,q3,q5 --n 15

# Week 1 — grounding pre/post (new, highest cost)
# implement G-1, then:
python -m eval.llm_judge --mode batch --tests g1 --n 15

# Week 2 — NORMALIZE + GROUP blind spots (new, low cost)
# implement N-1, N-2, Gr-1

# Week 3 — CANONICALIZE, RELATE false negatives, RESOLVE
# implement C-1, C-2, R-2, Rs-1
```

---

## Cost Summary

### Full run — 15 papers (~80 post-MAP findings/paper)

| Phase | Experiments | Est. batch cost |
|-------|-------------|-----------------|
| Week 1 | M-1, M-2, M-3, G-1 | ~$38 |
| Week 2 | N-1, N-2, Gr-1 | ~$4 |
| Week 3 | C-1, C-2, N-3, Gr-2, R-1, R-2 | ~$10 |
| Week 4 | Rs-1, Rs-2, G-2 | ~$1 |
| **Total** | | **~$53** |

G-1 is the cost driver (~$20) because it evaluates the full pre-filter finding
set (~3× larger than post-filter).

---

### Pilot run — 200 sentences (~1–2 papers)

200 sentences ≈ 40–80 MAP findings pre-filter, ~30–60 post-filter, shrinking
further at each downstream stage.

| Phase | Experiments | Est. batch cost |
|-------|-------------|-----------------|
| P1 all | M-1, M-2, M-3, G-1, N-1, N-2, Gr-1 | ~$3–5 |
| P2 all | C-1, C-2, R-1, R-2 | ~$1–2 |
| **Total** | | **~$4–7** |

**Is 200 sentences enough?**

It depends on what you are trying to decide:

- **Catching obvious bugs / directional signal** — yes. If a stage is badly
  broken (e.g. 50% grounded rate, high false merge rate) you will see it clearly
  at this scale.
- **Tuning thresholds with confidence** — no. At n=50 findings, a 95% confidence
  interval on any proportion metric (grounded_rate, accuracy, etc.) is roughly
  ±14%. You cannot reliably distinguish 82% from 96%, which means threshold
  decisions based on this data will be noisy.

For threshold decisions you need **200–300 findings** (not sentences), which is
closer to 5–8 papers (~$15–25 for P1 experiments).

**Recommended workflow:**

1. **Pilot at 200 sentences (~$4–7)** — run all P1 experiments to catch bugs and
   get a rough error-rate picture at each stage. Iterate on obvious fixes.
2. **Intermediate run at 5 papers (~$13–18)** — once a stage looks reasonable,
   re-run at this scale to set thresholds with tighter confidence intervals before
   moving downstream.
3. **Full run at 15 papers (~$53)** — final validation before treating results as
   ground truth for threshold calibration.
