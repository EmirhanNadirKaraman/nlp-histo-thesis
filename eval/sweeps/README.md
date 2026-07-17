# eval/sweeps — calibration sweep reports

Reproducible, single-command reports that show how a threshold or knob
changes the *retention* of summarisation outputs over a fixed corpus of
frozen pipeline artifacts.

> **This directory does NOT measure accuracy, precision, or recall. The
> outputs are threshold sensitivity / retention reports. Computing
> accuracy / precision / recall requires joining with gold or silver
> labels (Phase 2 of the eval framework).**

---

## Why this exists

The thesis sweep over `grounding_threshold` (and, later, other knobs)
needs to be **reproducible without bash glue**. A supervisor should be
able to clone the repo, run one Python command, and get the same CSV /
Markdown report we cite.

`eval/sweeps/` contains a small framework (`_lib.py`) plus one script per
knob (e.g. `grounding.py`, with a plotting companion `grounding_plot.py`).
Layer A scripts simulate a knob over already-persisted artifacts; they never
call APIs, never re-run NLI / UMLS, and never re-execute pipeline stages.
Re-runs are deterministic.

## Layer A vs Layer B

| | Layer A — frozen-artifact sweep | Layer B — full-pipeline replay |
|---|---|---|
| What runs | counting / threshold math over persisted scores | re-executes production stage classes locally |
| Where | this directory (`eval/sweeps/`) | future `eval/replay/` (not implemented) |
| Cost | seconds, $0 | minutes, $0 (after MAP outputs are cached) |
| Knobs supported | knobs whose input signal is already in the artifact | knobs that change upstream behaviour |
| Pre-requisites | the producer run must persist the signal | per-voter MAP outputs (B-052) + replay engine |
| Determinism | byte-identical given same input | byte-identical against a fresh production run on a tiny fixture |
| Invariant | never imports NLI / LLM / embedding libraries | reuses production stage classes; no duplicated logic |

The Layer A invariant is enforced by a subprocess import-safety test in
`tests/eval/sweeps/test_grounding.py` that asserts the script does not
load `transformers`, `torch`, `openai`, `anthropic`, `google.genai`,
`google.generativeai`, `vertexai`, `sentence_transformers`, or any
`pipeline.stages.summarization.llm_providers` symbol.

## Currently supported sweeps (Layer A)

| Knob | Script | Input signal | Where the signal lives | Status |
|---|---|---|---|---|
| `grounding_threshold` | `grounding.py` | NLI grounding scores per finding | `audit_trail.map_chunks[*].findings[*].grounding_score` + `rejection_summary.rejected[stage='grounding_map'][*].grounding_score` | **Shipped** |
| `final_score_cutoff` | _future_ | RESOLVE `final_score` per rule | `final_rules[*].final_score` | Not yet implemented |
| RESOLVE-weight tweaks (deterministic) | _future_ | per-rule grounding / support / contradict counts | `final_rules[*]` + `relations[*]` | Not yet implemented |

For these knobs the input signal is already persisted, so the sweep is a
pure post-hoc calculation. Adding one is a small follow-up PR (see
"How to add a new Layer A sweep" below).

## Not honestly sweepable yet

These knobs change upstream behaviour and would need a Layer B replay
engine. **No stub scripts exist for them** — adding empty placeholders
would be misleading.

| Knob | Why Layer A isn't enough | Blocking work |
|---|---|---|
| MAP `theta` / `reject_theta` | needs per-voter AuditableSummary outputs to re-score agreement at different gates | B-052: persist per-voter MAP outputs |
| Cascade scorer choice | same — re-runs the agreement gate | B-052 |
| NLI model swap | would need to re-score every premise/claim pair | full replay engine; can run locally (no API), but requires loading the NLI model |
| `chunk_size` / `chunk_overlap` | changes the input to MAP itself | full replay starting at MAP |
| Voter model set / cascade profile | changes MAP outputs | full replay starting at MAP |

When `eval/replay/` lands these will live there alongside their parity
tests.

## Reproducing the grounding sweep on the current corpus

After `out/summaries/` contains a Run B (grounding-enabled) baseline:

```bash
python eval/sweeps/grounding.py
head -40 eval/results/grounding_sweep.md
```

Outputs:

- `eval/results/grounding_sweep.csv` — one row per threshold; columns
  documented in the file's leading `#`-comment metadata block.
- `eval/results/grounding_sweep.md` — human-readable report with the
  disclaimer, metadata, interpretation, warnings (when applicable),
  retention table, per-paper detail.

Common variations:

```bash
# Custom threshold grid.
python eval/sweeps/grounding.py --thresholds 0.4,0.5,0.6

# Fail fast when the input dir mixes producer configurations
# (multiple pipeline_config_hashes or run_ids).
python eval/sweeps/grounding.py --strict-single-config

# Custom output paths.
python eval/sweeps/grounding.py \
    --out-csv eval/results/grounding_calv1_high_t.csv \
    --out-md  eval/results/grounding_calv1_high_t.md \
    --thresholds 0.7,0.8,0.9,0.95
```

## Reading the Warnings block

The Markdown report may emit a `## Warnings` section. The most common
warnings:

- **"Multiple pipeline_config_hashes detected …"** — the input
  `summaries/` directory contains artifacts from more than one producer
  configuration. The retention table mixes them. Either curate the
  input dir to a single producer config, or accept that the sweep
  reports the union. Use `--strict-single-config` to make this a hard
  failure (exit code 2) instead of a soft warning.
- **"Multiple run_ids detected …"** — same idea, but at the run level
  (e.g. an A-run and a B-run both wrote into the same directory).
- **"`{pmcid}: surviving + grounding_rejected = X but
  rejection_summary.map_findings_total = Y`"** — per-paper invariant
  check failed. The producer wrote a `map_findings_total` that doesn't
  match the visible findings + grounding-rejected entries. The sweep
  still runs but the count may undercount.
- **"All findings missing grounding_score …"** — typical when the input
  is a Run A baseline (grounding filter disabled and NLI scores not
  persisted). Re-run the producer with grounding enabled before
  running this sweep.

## How to add a new Layer A sweep

1. Add a `simulate_<knob>_sweep(records, values) -> list[dict]` function
   to `_lib.py` next to `simulate_threshold_sweep`. Keep the denominator
   semantics explicit in the column names.
2. Add `<knob>.py` next to `grounding.py` with the same CLI shape:
   `--input`, `--<knob>-values`, `--out-csv`, `--out-md`,
   `--strict-single-config`. Reuse `_lib.build_sweep_metadata`,
   `_lib.write_sweep_csv`, `_lib.write_sweep_markdown`.
3. Add `tests/eval/sweeps/test_<knob>.py` mirroring
   `test_grounding.py`: happy-path CLI, multi-config warning + strict
   flag, disclaimer assertion, denominator column assertion, and the
   subprocess import-safety check.
4. Update the **Currently supported sweeps** table above.

Hard rule when adding a new sweep: if the script imports anything from
`transformers`, `torch`, an LLM SDK, or `pipeline.stages.summarization.llm_providers`,
it does not belong here. That's Layer B. Move it to `eval/replay/`
(once that exists) instead.

## Disclaimer (repeated for emphasis)

> The reports in this directory measure **retention** — what fraction of
> the persisted findings survive a given threshold, and what their
> grounding-score distribution looks like. They do **not** measure
> accuracy, precision, recall, or F1. Those numbers require gold or
> silver labels and a separate evaluation harness.
