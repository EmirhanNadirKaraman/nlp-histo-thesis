# Paper selection algorithm

End-to-end specification of `eval/paper_selection/`. Produces the YAML
selection files that feed `scripts/run_paper.py --from-selection`, e.g.
`configs/paper_selection/calibration_set_v1.yaml`. No LLM / API calls
anywhere in this stage — everything is computed offline from
post-extraction DB rows (or a JSONL export).

---

## 1. Why this exists

The summarisation pipeline is evaluated on a *calibration set* — a small,
audited corpus chosen so that downstream metrics are interpretable:

* **Related** — `k_related` papers that overlap in disease / biomarker
  vocabulary, so cross-paper RELATE has a real chance of firing
  SUPPORT / CONTRADICT edges.
* **Diverse** — `k_diverse` papers that *cover* the entity space, so
  the corpus stresses the entity-normalisation and grouping stages.
* **Hard** — `k_hard` papers that are individually difficult to extract
  (table-heavy / entity-dense / long), so the pipeline's failure modes
  surface inside the sample.

Default `k = 5` per bucket → a 15-paper calibration set. Three buckets
are mutually exclusive by default; `--allow-overlap` relaxes that.

---

## 2. Pipeline

```
RawPaper (DB / JSONL)
  │   loaders.py:DBLoader | JSONLLoader
  ▼
PaperFingerprint                        (offline feature vector)
  │   fingerprints.py:build_fingerprints
  ▼
SelectionResult{related, diverse, hard, rationale}
  │   selectors.py    (greedy, default)
  │   ilp_selectors.py (PuLP-CBC, optional)
  ▼
configs/paper_selection/{version}.yaml          ← minimal pmcid roster
configs/paper_selection/{version}_rationale.json ← full per-pick reasoning
configs/paper_selection/{version}_summary.csv    ← flat one-row-per-paper
  │   export.py:write_calibration_set
```

Loader output (`loaders.RawPaper`) carries text elements + NER entities
+ layout counters straight from the ingestion DB. The fingerprinter
turns this into a `PaperFingerprint`; the selectors operate only on
fingerprints (no DB access).

---

## 3. Fingerprint (`fingerprints.py` → `models.PaperFingerprint`)

For each paper the fingerprinter produces:

| Field group | What | Source |
|---|---|---|
| Workload | `n_text_elements`, `n_paragraphs`, `n_sentences`, `n_chunks`, `n_tokens_est` | Sentence regex (`.!?` + whitespace) + chunk size mirroring MAP stage |
| Layout | `n_tables`, `n_figures`, `n_captions` | DB layout counters |
| Entity vocab — categorised | `disease_entities`, `biomarker_entities`, `gene_entities`, `tissue_entities`, `method_entities`, `outcome_entities` | UMLS semantic-type bucketing (TUI sets in `fingerprints.py`) + regex extractors (`CD-N`, `Ki-67`, gene-like uppercase symbols) + curated keyword dictionaries |
| Entity vocab — flat | `entities` | union of everything observed, lowercased + surface-normalised |
| UMLS provenance | `umls_cuis`, `semantic_types` | scispaCy + UMLS linker output |
| Frequency | `entity_counts` | dict surface-form → count |
| Source stats | `source_stats` | mean paragraph length, abbreviation density, very-short / very-long paragraph counts, optional `layout_noise_score` |

A generic-term filter prevents broad words ("patient", "cell", "tissue")
from leaking into the bucketed entity sets. The flat `entities` field
keeps everything; bucketed fields are derived from `entities` via
TUI filters + regex/keyword classifiers.

`PaperFingerprint.is_empty()` and `.has_useful_entities()` are the
eligibility hooks used by the selectors.

---

## 4. Metrics (`metrics.py`)

Three pluggable Protocols. Default implementations:

### 4.1 Relatedness — `PairMetric`

Weighted Jaccard across entity buckets:

```
rel(a, b) = 0.30·J(disease_a, disease_b)
          + 0.20·J(biomarker_a, biomarker_b)
          + 0.15·J(gene_a, gene_b)
          + 0.15·J(umls_cuis_a, umls_cuis_b)
          + 0.10·J(tissue_a, tissue_b)
          + 0.06·J(method_a, method_b)
          + 0.04·J(outcome_a, outcome_b)
          + 0.02·J(entities_a, entities_b)
          − 0.10  if disease_a ∩ disease_b == ∅
```

`J(·, ·)` is Jaccard. The disease-overlap penalty is the strongest single
signal that two papers aren't comparable — without it the score can be
inflated by a few coincidental tissue / method matches.

### 4.2 Diversity — `SetMetric`

Reward broad coverage, penalise pairwise relatedness:

```
coverage = 0.30·|⋃ diseases|/(3n) + 0.20·|⋃ methods|/(2n)
         + 0.10·|⋃ outcomes|/(2n) + 0.10·|⋃ tissues|/(2n)
         + 0.20·|⋃ biomarkers|/(3n)

div(S) = coverage − 0.40·mean_pairwise_rel − 0.20·n_near_dup_pairs
```

`near_dup` threshold is `rel(i,j) ≥ 0.65`. The `n` denominators
normalise so adding redundant papers stops paying for itself.

### 4.3 Hardness — `PaperMetric` → `HardnessBreakdown`

Three sub-scores, each clipped to `[0, 1]`:

```
content_complexity  = 0.4·min(ent_density/4.0, 1)
                    + 0.4·min(unique_ent_density/1.5, 1)
                    + 0.2·min(abbr_density/0.5, 1)

layout_complexity   = 0.30·min(table_density/0.05, 1)
                    + 0.20·min(caption_density/0.10, 1)
                    + 0.25·min(short_para_rate/0.30, 1)
                    + 0.25·min(long_para_rate/0.10, 1)
                    [optional 0.10 layout_noise blend]

relation_complexity = min((|biomarkers|+|genes|+|diseases|)/30, 1)

normalized_hardness = 0.45·content + 0.25·layout + 0.30·relation
absolute_hardness   = normalized · log1p(n_chunks)
workload_factor     = log1p(n_chunks) / log1p(30)
```

`normalized_hardness` is per-unit difficulty (good for ranking across
papers regardless of length). `absolute_hardness` weights it by chunk
count so a giant paper at moderate density still ranks high. Both are
emitted in the rationale + summary CSV.

Reference values (`ref_chunks=30`, `ref_entity_density=4.0`, etc.) and
all weights are exposed via `HardnessConfig` / `HardnessWeights` dataclasses
so the formulas are overridable without forking.

---

## 5. Strategies

Two interchangeable implementations of the same bucket APIs.

### 5.1 Greedy (`selectors.py`, default)

Deterministic for a given fingerprint list + metric config.

* **Related** — seed = paper with highest mean relatedness to all
  candidates; then iteratively add the candidate with the highest mean
  relatedness to the already-picked set, requiring `min_disease_overlap`
  (default 1) shared disease entity with at least one picked paper.
  Falls back to no-overlap if the constraint exhausts candidates.
* **Diverse** — seed = paper with largest combined
  disease/method/biomarker/outcome vocab; then greedy farthest-point:
  add the paper with the largest `Diversity.score(chosen + {p}) −
  Diversity.score(chosen)`.
* **Hard** — composed (not optimised): take `hard_high_normalized`
  (default 2) papers from the top-normalized order, plus
  `hard_high_absolute` (default 2) from the top-absolute order, plus
  `hard_medium_control` (default 1) from the middle band
  (40–65th percentile of `normalized_hardness`). Pad with next-hardest
  if any sub-pool is too small. Composition is auditable in the
  rationale: every pick carries the sub-pool reason string.

### 5.2 ILP (`ilp_selectors.py`, opt-in via `--strategy ilp`)

PuLP-formulated mixed-integer programs solved by the bundled CBC. Same
`(papers, rationale)` shape as greedy — the rest of the pipeline is
strategy-agnostic. PuLP is an optional dependency; without it the CLI
errors with an actionable message (or falls back to greedy if
`--ilp-fallback-greedy`).

Two scaling levers across all three buckets:

1. **Candidate pruning.** Each bucket has its own pre-score
   (`_related_prescore`, `_diverse_prescore`, `_hard_prescore`) and a
   per-bucket cap (`related=80`, `diverse=100`, `hard=120`). Top-N
   survives into the ILP. `--candidate-limit` overrides all three.
2. **Edge sparsification.** Pair variables (`y_ij`) are *not* dense.
   For related, only the union of each node's top-`M` neighbours
   above `related_pair_threshold=0.01` is retained — caps `y` count
   at `≈ |pool|·M`. For diverse, pairs are retained only if they cross
   `near_duplicate_threshold` (or `pairwise_threshold` when the
   pairwise penalty is non-zero — otherwise dead variables).

Solver knobs: `time_limit_seconds=30.0` per-bucket default,
`accept_feasible=True` (accept best feasible solution when CBC times
out before proving optimality, as long as exactly `k` papers are
selected), `solver_verbose=False`. If the solver returns infeasible /
unbounded / no usable solution, `_prescore_fallback` returns the
top-`k` by pre-score with an `ilp_solution_quality="prescore_fallback"`
flag in the rationale.

#### Related ILP

```
maximise   Σ_{(i,j)∈E_kept} rel(i,j)·y_ij
         + α · Σ paper_quality(i)·x_i
         − β · Σ length_penalty(i)·x_i
subject to Σ x_i = k
           y_ij ≤ x_i  ∀(i,j)∈E_kept
           y_ij ≤ x_j  ∀(i,j)∈E_kept
           x_i ∈ {0,1}, y_ij ∈ [0,1]
```

Pair coefficients are non-negative, so `y_ij ≤ x_i` and `y_ij ≤ x_j`
suffice for the LP to push `y` to its integer vertex without an explicit
binary cast. `α = related_quality_weight = 0.05`, `β =
related_length_penalty = 0.02`. `paper_quality(i)` rewards balanced
structured-entity coverage (max 0.2 per bucket, six buckets ⇒ max ~1.2);
`length_penalty(i) = min(n_sentences/350, 2.0)`.

The rationale entry carries `ilp_pair_reward`, `ilp_quality_reward`,
`ilp_length_penalty`, `ilp_objective`, `ilp_n_pair_edges`,
`ilp_solution_quality`. The summary CSV's `selection_reason` strings of
the form `"ILP related: quality=1.200, length_penalty=0.486"` originate
here.

#### Diverse ILP

Per-concept coverage variables `z_c ∈ [0, 1]` activate when any
selected paper contains concept `c`:

```
maximise   Σ_c eff_reward(c)·z_c
         − Σ rel(i,j)·y_ij  (pairwise_penalty, off by default)
         − Σ y_ij           (near_duplicate_penalty=1.0, over pairs with rel ≥ 0.65)

subject to Σ x_i = k
           z_c ≤ Σ_{i: c ∈ p_i} x_i      ∀c
           y_ij ≥ x_i + x_j − 1           ∀(i,j) retained
           x_i ∈ {0,1}, z_c,y_ij ∈ [0,1]
```

Per-bucket concept caps (`disease=80`, `biomarker=80`, `gene=80`,
`tissue=50`, `method=50`, `outcome=50`, optional `cui=30`) prevent one
rich bucket — most often raw UMLS CUIs — from drowning the coverage
objective. Concepts within a bucket are ranked by coverage count
descending then by name (deterministic). Effective per-concept reward
caps total bucket reward at `diverse_bucket_reward_caps[b]`:
`eff_reward(c) = min(base_weights[b], cap[b] / n_retained_concepts[b])`.

Rare-concept rescue: the candidate pre-score is augmented by
`Σ 1 / freq(c)` over concepts the paper contains. Rare-concept
papers survive the prescore cutoff even when their overall
vocabulary is small.

CUIs are off by default (`diverse_include_cuis=False`); the raw count
dwarfs structured buckets and degrades coverage diagnostics when on.

#### Hard ILP

No pair variables, only composition constraints:

```
maximise   Σ absolute_hardness(i)·x_i
subject to Σ x_i = k
           Σ_{i ∈ TopNorm}    x_i ≥ effective_hard_high_normalized
           Σ_{i ∈ TopAbs}     x_i ≥ effective_hard_high_absolute
           Σ_{i ∈ MediumBand} x_i ≥ effective_hard_medium_control
           x_i ∈ {0,1}
```

* `TopNorm` = top-30 by `normalized_hardness` after prescore cap.
* `TopAbs` = top-30 by `absolute_hardness`.
* `MediumBand` = 40th–65th percentile of `normalized_hardness`.

The *effective* values are clamped to each sub-pool's actual size and
surfaced in the rationale (`effective_hard_high_normalized` etc.) so
that "the constraint was relaxed because the sub-pool was too small"
is auditable. If `hard_high_normalized + hard_high_absolute +
hard_medium_control > k`, all three composition constraints are
skipped — the model degenerates to "pick `k` by absolute hardness".

---

## 6. Validation (`run_select.py:_validate_selection`)

After selection, three sanity assertions before the result is written:

1. **Count** — `|set(pmcids)| == k_related + k_diverse + k_hard` when
   not allowing overlap.
2. **Useful entities** — every non-`hard` pick must have
   `has_useful_entities()`; hard-bucket picks may skip this (a paper
   can be hard precisely *because* its entities are sparse / noisy).
3. **Ordering invariants** — `mean_pairwise_rel(related) >
   mean_pairwise_rel(diverse)` (related should be tighter); and
   `mean_hardness(hard) > {mean_hardness(related), mean_hardness(diverse)}`.

Failures land as warnings on stdout, not exceptions — the writer always
runs (with the warnings logged alongside the rationale) so an imperfect
selection is still inspectable.

---

## 7. Outputs (`export.py:write_calibration_set`)

Three files per run, written into `--export-dir` (default
`configs/paper_selection/`) under `--output-version`:

| File | Purpose |
|---|---|
| `{version}.yaml` | Minimal `bucket → [pmcid, …]` roster. Read by `scripts/run_paper.py --from-selection`. |
| `{version}_rationale.json` | Full per-paper rationale: bucket, rank, `selection_reason`, ILP diagnostics, hardness breakdown, top entities per bucket, n_sentences/chunks/tables/figures. |
| `{version}_summary.csv` | Flat one-row-per-paper summary with workload counters + normalized/absolute hardness + selection_reason — useful for spreadsheet review. |

`{version}.yaml` is the only file consumed downstream. The other two are
for thesis evidence (you can show your supervisor *why* PMC9826086
landed in `related` and not `hard`).

---

## 8. How to run

```bash
# DB-backed, ILP strategy, default cohort size (5+5+5), config snapshot v1
python -m eval.paper_selection.run_select \
    --strategy ilp \
    --output-version calibration_set_v1

# JSONL fallback (no DB)
python -m eval.paper_selection.run_select \
    --jsonl out/papers_export.jsonl \
    --strategy ilp \
    --output-version offline_set

# Greedy (no PuLP required) — useful for quick smoke runs
python -m eval.paper_selection.run_select \
    --strategy greedy \
    --output-version smoke_v2

# Dry run — print + validate, write nothing
python -m eval.paper_selection.run_select --strategy ilp --dry-run
```

Common knobs (`run_select.py --help` lists everything):

| Flag | Default | Purpose |
|---|---|---|
| `--k-related / --k-diverse / --k-hard` | 5 each | Bucket sizes |
| `--max-sentences-related / --max-sentences-diverse` | 350 | Length cap on candidates |
| `--min-sentences` | 20 | Lower bound on candidates |
| `--max-text-elements` | None | Drop textbook-chapter outliers globally |
| `--allow-overlap` | false | Allow PMCIDs across buckets |
| `--candidate-limit` | None | Override per-bucket prescore caps |
| `--time-limit-seconds` | 30 | Per-bucket CBC budget (`0` = unlimited) |
| `--solver-verbose` | off | Stream CBC progress |
| `--ilp-fallback-greedy` | off | Silently fall back if PuLP missing |
| `--pmcid` (repeatable) | None | Restrict candidate pool |

PuLP install: `pip install pulp` (CBC is bundled with PuLP — no separate
solver install on macOS / Linux).

---

## 9. Why this design

| Choice | Rationale |
|---|---|
| Three buckets (related / diverse / hard) | Three independent failure modes of the summarisation pipeline. A single homogeneous sample can pass each metric in isolation but mask the failure surface. |
| ILP over greedy as the default for thesis runs | Greedy farthest-point is order-sensitive at the margins. ILP gives a globally optimal pick under the stated objective + constraints, which is the defensible position in the thesis. Greedy stays as a fast fallback. |
| PuLP-CBC (not Gurobi / CPLEX) | CBC is bundled with PuLP, has no licence requirement, and solves all three buckets within `time_limit_seconds=30` at the corpus sizes we work with. Switching solvers later only requires changing `_solve`. |
| Candidate pruning + edge sparsification | Dense O(n²) pair variables blow up the model. Bucket-specific caps + top-M edge retention keep the LP relaxation tight while bounding solve time. The pre-score fallback guarantees we always return *some* defensible selection if the solver times out. |
| Concept caps per bucket | One bucket (raw CUIs in particular) can otherwise dominate the diversity objective and starve structured buckets. Hard caps + per-bucket reward caps make the objective comparable across buckets. |
| Fingerprints are pure dataclasses | Selection runs offline with no DB / LLM access. Tests construct fingerprints directly without DB fixtures. |
| Output triple (YAML + JSON + CSV) | YAML is consumed by the pipeline; JSON keeps the full audit trail (ILP objective, sub-pool sizes, reasons strings); CSV is for human spreadsheet review during thesis sanity checks. |
