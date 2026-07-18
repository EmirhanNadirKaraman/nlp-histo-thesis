# EXPERIMENTS — registry

Maps every thesis experiment to **which report file it produces**, the command to
reproduce it, the dataset it runs on, and the thesis section / RQ it supports. This
is the experiment-level counterpart to `docs/thesis/C_claim_evidence_map.md` (which
maps *claims* → evidence). All values are being **rerun** on the new datasets
(`related15` / `heldout15`); old numbers from `docs/thesis/build/` are superseded.

## Report layout convention

Each experiment writes into its own **E-numbered** subfolder under `eval/reports/`
(gitignored), so the directory sorts by experiment number:

```
eval/reports/E06_family_refine/
  checkpoint.csv          resumable-sweep progress (one row per completed cell)
  checkpoint.meta.json    signature guard (grid/finalists/weights/split/seed)
  sweep_<UTC-ts>.csv       final output of a completed sweep stage
eval/reports/E12_voter_loo/
  <UTC-ts>.csv            one-shot output
```

The checkpoint is deleted on successful completion; the timestamped `sweep_*.csv` /
`*.csv` is the durable artifact. Latest: `ls -t eval/reports/E##_*/*.csv | head -1`.

> ⚠️ **In-flight exception:** the `family_refine` run started *before* the folder
> change, so it writes the **legacy flat** paths
> `eval/reports/checkpoint_family_refine.csv` + `new_sweep_family_refine_<ts>.csv`.
> Once it finishes, migrate:
> `mkdir -p eval/reports/E06_family_refine && mv eval/reports/new_sweep_family_refine_*.csv eval/reports/E06_family_refine/sweep_$(date -u +%Y%m%dT%H%M%S).csv`
> Every *future* run self-files into the E-numbered subfolder automatically.

**Datasets:** `related15` = silver calibration set (454 cases, 15 papers, Opus-4.7
silver) · `heldout15` = reserved held-out test · `rubric27` = 27 human-annotated
PDFs (doc-extraction) · `corpus` = full corpus.

## Run order

Phase 1 (freeze config): E06 → **E06b → E06c** (voter-subset, optional) → E07 → E08.
Phase 2 (offline analyses on `related15`): E09, E12, E10+E11, E03+E13, E04.
Phase 3 (local eval): E01, E02.
Phase 4 (LLM, last): E14 (headline), E15.
Methodology-hardening re-analyses (sim-threshold sensitivity, paper-level bootstrap)
fold into Phase 2.

## Registry

| # | Experiment | Thesis § (RQ) | Dataset | Output (`eval/reports/`) | Status |
|---|---|---|---|---|---|
| E01 | Document-extraction sweep + rubric F1 | 9.1 (RQ1) | rubric27 | `E01_doc_extraction/figtable_extraction_sweep_rerun_27pdf_20260604_PR.md` | **done** — winner var18 (hybrid@0.99+footnote-expand1.2): tables strict-F1 40→83.8%, figures 84% |
| E02 | Provenance preservation + qualitative example | 9.2, 9.7 (RQ1) | corpus | `E02_provenance/provenance_20260618T120844.csv` | **done** — text-path 100%, tables+figures 100% page/bbox (B-075 fix + migration 0014 + backfill), caption 100% tables / 86% figures, crop 100%, xref 54-59% |
| E02b | Final-rule provenance carry-rate | 9.2 (RQ2) | related15 | `E02b_rule_provenance/rule_provenance_20260702T173213.csv` | **done 2026-07-02** — **1729/1729 (100%)** final rules trace to ≥1 source paragraph (per-run all 100%, 15 related15 runs); failure funnel empty; `member_normal_ids` 100% resolved (1801/1801), spans 100% carry `text_element_id` (1878/1878), 0 cross-paper. Rule-side counterpart to E02; empirically confirms `09_provenance_model` |
| E02c | Final-rule provenance carry-rate — held-out | 9.6 (RQ5) | heldout15 | `E02c_rule_provenance_heldout/rule_provenance_heldout_20260702T175134.csv` | **done 2026-07-02** — held-out **1273/1273 (100%)** final rules trace to a source paragraph (per-paper all 100%, funnel empty, 0 cross-paper; PMC8221904 emits 0); related15 JSON cross-check **1747/1747 (100%)** on shipped 5-voter config. JSON summaries + DB read (no re-run); resolves via canonical `representative_text_element_id`. Provenance **generalizes**. Config note: related15 JSON=1747 (5-voter) vs E02b DB=1729 (superseded 6-voter) — 100% under both |
| E03 | Grounding-filter retention + sensitivity | 9.3 (RQ2) | related15 | `E03_grounding/sweep_20260622T174122.csv` | **done 2026-06-22 (5-voter)** — filter-off strict_f1 0.7160 (= cascade headline); @0.5 strict_f1 0.6520, retention 83.8% (1923/2294), precision flat (0.872→0.875); ~14% <0.3. Filter = faithfulness guard (trades recall, not precision) |
| E04 | Knowledge-extraction per-stage cardinalities | 9.2 (RQ2) | related15 (frozen corpus) | `E04_cardinalities/cardinalities_20260622T210109.csv` | **done 2026-06-22 (5-voter)** — 2294 MAP→1923 grounded (83.8%)→1747 canonical=1747 final; ≈1 finding→1 rule (0.939); relations sparse (13 intra + 12 cross = 25, only 3 CONTRADICT). Supersedes 6-voter (2280→1729) |
| **E05** | **Cascade structure screen** (Pareto finalists) | 9.4 (RQ3) | related15 | `E05_structure_screen/` | **done** |
| **E06** | **Cascade family-refine** (weights × θ) | 9.4 (RQ3) | related15 | `E06_family_refine/sweep_*.csv` | **done** |
| E06b | Voter-subset screen (configs × VOTER_SUBSET_GRID at fixed θ) | 9.4 ext (RQ3) | related15 | `E06b_voter_subset_screen/sweep_20260622T131114.csv` | **done 2026-06-22** — voter-aware per-chunk cost: drop-haiku sf1-neutral @ −18% cost (quality), two L1 drops dominate `all` (economy). Voter pruning **is** a cost lever (supersedes "every drop loses") |
| E06c | Voter-subset refine (dominating drops + `all` baselines × full θ) | 9.4 ext (RQ3) | related15 | `E06c_voter_subset_refine/sweep_20260622T143146.csv` | **done 2026-06-22** — drops dominate across θ; quality/drop-haiku owns max-F1 (0.7147 vs all 0.7131, −18% cost) → **adopt 5-voter `BEST_VOTER_SUBSET="drop_l2_2"`**, selected on calibration for cost + evaluator-independence (F1-equivalent) |
| E07 | θ / reject-θ selection (default gate) | 9.4 (RQ3) | related15 | `E07_map_theta/sweep_20260622T153848.csv` | **done 2026-06-22 (5-voter, keep)** — θ=0.9/reject=0.2 (0.7147) at the default gate; θ-curve flat at top (diminishing returns). The θ-selection record before the gate is chosen |
| E08 | Gates / hard-fail ablations (keep·escalate, polarity) | 9.4 (RQ3) | related15 | `E08_map_gates/sweep_20260622T154808.csv` | **done 2026-06-22 (5-voter)** — gates **NOT inert** (cf. 6-voter): escalate wins **0.7160** vs keep 0.7147 → adopt `escalate` (N=1-at-L2 more common with 2 L2 voters) |
| E08b | Shipped config's θ-curve (`map_theta_shipped`, escalate gate) | 9.4 (RQ3) | related15 | `E08b_map_theta_shipped/sweep_20260622T164928.csv` | **done 2026-06-22** — θ0.9/r0.2 → **0.7160** at the escalate gate; confirms θ argmax is gate-invariant. The curve E09 reads (characterization, not a θ re-selection) |
| E09 | Cost–quality frontier (per-tier $ vs F1) | 9.4 (RQ3) | related15 | `E09_cost_quality/frontier_20260622T165135.csv` | **done 2026-06-22 (5-voter)** — quality 0.7160@23.66, knee 0.7067@21.80, economy 0.5433@3.38; single-curve frontier (5-voter tier prices, L2=4.80); dominated by single-Sonnet (E10/E11) |
| E10 | Single-model + tier baselines (vs cascade, cost/chunk) | 9.4 (RQ3) | related15 | `E10_baselines/baselines_20260622T165331.csv` | **done 2026-06-22 (5-voter)** — cascade 0.7160@23.66 vs single-Sonnet 0.7129@18.0 (within noise, ~31% more cost); cheaper than old 6-voter (28.5) |
| E11 | Cascade-vs-Sonnet paired bootstrap | 9.4 (RQ3) | related15 | `E11_bootstrap/bootstrap_20260622T165546.csv` | **done 2026-06-22 (5-voter)** — beats all cheaper models (sig); vs Sonnet Δ**+0.0031** CI[−0.0002,+0.0068] **NOT sig** → not cost-justified vs Sonnet |
| E12 | **Leave-one-voter-out** (per-model contribution) | 9.4 ext (RQ3) | related15 | `E12_voter_loo/20260622T165730.csv` | **done 2026-06-22** — 6-voter diagnostic by design (`_make_voters` stays 6); verdict unchanged, no production change |
| E13 | NLI relation-classification ablation (predicate vs scope, RELATE) | 9.5 (RQ4) | synthetic 300 claim-pairs | `E13_nli_ablation/20260622T165812.csv` | **done 2026-06-22** — predicate_only acc 0.927/macroF1 0.927 @ frozen 0.50/0.50 (SUPPORT P=1.0); scope_aware 0.923; cascade-independent, unchanged |
| **E14** | **Held-out test** (frozen 5-voter config) | 9.6 (RQ5) | heldout15 | `E14_heldout/heldout_20260622T170151.csv` | **done 2026-06-22 (5-voter)** — strict_f1 **0.7128** (loose 0.884) vs related15 0.7160 → **gap −0.0032 ⇒ generalizes, no overfit**; escalate=0.990 @θ0.9 (≈single-Sonnet); operating point selected on related15, reported here. Held-out downstream funnel (5-voter, `cardinalities_20260622T214622.csv`, MAP=1672 matches E14 n_pipeline): **1672 MAP→1379 grounded (82.5%)→1273 canonical=1273 final**, rules/finding **0.949** (related15 0.939, near-identical), dedup −2.7% (related15 −3.3%), relations sparse (4 intra, 0 cross, 0 CONTRADICT) → **funnel shape generalizes at every stage**. Full-metric comparison: related15 loose 0.886 / P 0.872 / R 0.900 vs heldout loose 0.884 / P 0.880 / R 0.888 (every metric within ~1 pt). Descriptive θ-frontier (`--theta-frontier`, `heldout_20260625T082104.csv`): θ0.4→0.5453 … θ0.9→0.7128 reproduces the calibration diminishing-returns shape → **cost-quality trade-off generalizes** (fig `theta_strict_f1_heldout`). |
| E15 | Silver-vs-human validity | (threats) | rubric27 | `E15_silver_validity/` | pending (LLM) |

## Reproduce

```bash
# E05 structure_screen (done) — recompute finalists from its CSV without rerunning:
python -m eval.silver.analysis.run_new_summarization_sweeps --stage structure_screen \
  --from-csv eval/reports/E05_structure_screen/sweep_<ts>.csv --top-k 2 --keep-within 0.005 --pareto-cap 3

# E06 family_refine (running) — resumable; re-run to resume:
caffeinate -i python3 -m eval.silver.analysis.run_new_summarization_sweeps --stage family_refine

# E06b/E06c voter-subset (optional, between E06 and E07). family_refine prints
# VOTER_SUBSET_SCREEN_CONFIGS → paste → screen prints VOTER_SUBSET_REFINE_CONFIGS → paste:
python -m eval.silver.analysis.run_new_summarization_sweeps --stage family_refine --from-csv eval/reports/E06_family_refine/sweep_<ts>.csv   # re-print screen configs (no rerun)
python -m eval.silver.analysis.run_new_summarization_sweeps --stage voter_subset_screen   # then paste refine configs
python -m eval.silver.analysis.run_new_summarization_sweeps --stage voter_subset_refine   # then pin BEST_VOTER_SUBSET

# E07 map_theta (θ-selection, default gate) / E08 map_gates / E08b shipped θ-curve at the chosen gate:
python -m eval.silver.analysis.run_new_summarization_sweeps --stage map_theta          # E07 (keep) → pin BEST_THETA/BEST_REJECT_THETA
python -m eval.silver.analysis.run_new_summarization_sweeps --stage map_gates          # E08 → pin BEST_SINGLE_VOTER_POLICY (escalate)
python -m eval.silver.analysis.run_new_summarization_sweeps --stage map_theta_shipped  # E08b → shipped θ-curve (E09 reads this)
python -m eval.silver.experiments.E07_map_theta.plot_theta_curve              # §03 figure: strict-F1 vs θ (keep+escalate) → eval/reports/E07_map_theta/theta_strict_f1.{pdf,png}

# E02b final-rule provenance carry-rate (read-only DB walk of the sum_* chain, no API):
python -m eval.silver.experiments.E02b_rule_provenance.rule_provenance_metric

# E02c held-out final-rule provenance carry-rate (reads out/summaries/heldout15 JSONs + DB, no API):
python -m eval.silver.experiments.E02c_rule_provenance_heldout.rule_provenance_heldout

# E03 grounding-threshold sweep (offline — replays cached voter_cache, no API):
python -m eval.silver.experiments.E03_grounding.grounding_sweep_related15

# E09 cost-quality frontier (offline CSV analysis of the E08b shipped θ-curve, no API):
python -m eval.silver.experiments.E09_cost_quality.cost_quality_frontier

# E04 knowledge-extraction funnel (done on haiku_only cache; offline aggregation):
python -m eval.silver.experiments.E04_cardinalities.cardinalities   # --summaries-dir for a frozen rebuild

# E12 leave-one-voter-out (full):
python -m eval.silver.experiments.E12_voter_loo.voter_loo   # add --cases N for a quick smoke

# E14 held-out generalization (offline replay over the heldout15 primer; no voter API):
python -m eval.silver.experiments.E14_heldout.heldout_eval                   # frozen θ0.9/r0.2 → strict_f1 0.7128
python -m eval.silver.experiments.E14_heldout.heldout_eval --theta-frontier  # + descriptive θ-sweep θ∈{0.4..0.9}
python -m eval.silver.experiments.E14_heldout.plot_theta_curve_heldout       # §06 fig: heldout θ-curve vs related15 → eval/reports/E14_heldout/theta_strict_f1_heldout.{pdf,png}

# E02 provenance metric (corpus DB queries, read-only, no API):
python -m eval.silver.experiments.E02_provenance.provenance_metric

# E15 silver labels on the rubric set (for silver-vs-human):
python -m eval.silver.generation.generate --batch --source eval/data/source_cases_rubric27.jsonl
```

E10/E11/E13/E01–E04 still need their harnesses ported from the `docs/thesis` EXP
definitions (or `eval/run.py` for E01). Mark each `done` here and tick the matching
`THESIS.md` TODO when it lands.
