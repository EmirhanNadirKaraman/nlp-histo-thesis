# Silver-Label Evaluation

## Purpose

`eval/silver/` implements the **silver-label evaluation loop**:

1. obtain or load source cases;
2. generate Opus **silver labels** for them;
3. match pipeline findings against those silver findings;
4. evaluate metrics and threshold / cascade configurations;
5. produce inspection and reporting artifacts.

This is a **separate measurement track** from the **LLM-as-Judge** harness
(`eval/llm_judge/`), which the parent [`eval/README.md`](../README.md) documents; that
README lists this package as the "silver-label loop + thesis experiments E01–E14" track.
Both rely on Opus-derived **proxy labels, not clinical ground truth**.

Not every workflow here is offline, and not all are reproducible from a fresh clone — see
[Offline, cached, and paid execution](#offline-cached-and-paid-execution).

## Package structure

`eval/silver/` is **both** an importable Python package **and** a set of module entry
points. Entry points are invoked with `-m`, from the repository root:

```bash
python -m eval.silver.<module> --help
```

Several modules come as **deliberate CLI / library pairs** — a thin command-line front end
plus the reusable implementation it calls: `data/sample.py`/`data/sampler.py`,
`data/export_pipeline.py`/`data/exporter.py`,
`generation/generate.py`/`generation/generator.py`. These pairs are intentional, not
duplicated implementations.

```
eval/silver/
├── data/             DB-backed sampling + export (sampler, exporter, export pipeline)
├── generation/       silver-label generation CLI + library, and its prompts
├── reporting/        Markdown/CSV inspection reports and the HTML dashboard
├── analysis/         evaluation drivers, calibration sweeps, replay context, score analyses
├── bridges/          interoperability tools: primer→corpus bridge + one-case proof
├── experiments/      numbered thesis experiment packages (E01–E14) + corpus_stats
├── relation_pairs/   E13 claim-pair workflow (Anthropic Message Batches)
├── old_files/        superseded historical driver — still imported for its constants
├── tests/            the package's pytest suite (intentionally co-located)
└── __init__.py
```

The embedding adapters/caches and the finding-alignment matcher (formerly `matching/`), the
Pydantic schemas, the JSONL IO helpers, and the deterministic dev/test split now live in the
installed package under `src/nlp_histo/evaluation/` (import as `nlp_histo.evaluation.*`); the
harness depends on them from there. The remaining subpackages layer strictly downward —
`data` depends on nothing else here; `generation` and `reporting` depend on `data`; `analysis`
depends on `data`/`reporting` and `nlp_histo.evaluation`; `bridges` depends on `analysis`. The
import graph is acyclic, and no lower layer imports an upper one.

Facts worth keeping in mind:

* **`eval` is not installed** as part of the distribution (`pyproject.toml` deliberately
  excludes it). Commands are run as `python -m eval.silver.…` **from the repository
  root**; `database`, `pipeline` and friends come from the editable project install.
* **Data and artifact paths stay repository-root-relative** (`eval/data/…`, `out/…`,
  `configs/…`), so the working directory must be the repository root.
* **`tests/` is intentionally co-located** with the package it tests, rather than living
  under the top-level `tests/` tree.
* **`old_files/` stays** because active code still imports constants from it
  (`scripts/eval/run_summarization_experiments.py`), not merely for provenance.
* **`experiments/` and `relation_pairs/` are unchanged** by the package grouping.

## Main workflows

### Source-case preparation — `data/`

```bash
python -m eval.silver.data.sample --help
python -m eval.silver.data.export_pipeline --help
```

Select source cases and export pipeline findings for them. **Database-backed:** these
require access to the PostgreSQL instance. The Pydantic schemas
(`nlp_histo.evaluation.schemas`), JSONL helpers (`nlp_histo.evaluation.jsonl_utils`) and the
deterministic dev/test split (`nlp_histo.evaluation.split`) are pure offline utilities that
now live in the installed package.

### Silver-label generation — `generation/`

```bash
python -m eval.silver.generation.generate --help
```

Produces silver findings from source cases (`generate` CLI → `generator` library, with
prompts in `prompts`). **This workflow uses Anthropic / Opus and can make paid API
calls.**

### Matching and evaluation — `nlp_histo.evaluation.matching` + `analysis/`

```bash
python -m eval.silver.analysis.evaluate --help
```

`nlp_histo.evaluation.matching` provides the embedding adapters (`embedders`), the embedding
caches and the alignment + P/R/F1 metrics (`matcher`); `analysis/evaluate` drives them. The embedding
providers (OpenAI / Gemini) **can make paid API calls on a cache miss**; a warm local
embedding cache is not guaranteed.

### Calibration and replay — `analysis/` + `bridges/`

```bash
python -m eval.silver.analysis.sweep --help
python -m eval.silver.analysis.pipeline_sweep --help
python -m eval.silver.analysis.map_theta_sweep --help
python -m eval.silver.analysis.run_new_summarization_sweeps --help
python -m eval.silver.analysis.score_distribution --help
python -m eval.silver.analysis.escalation_breakdown --help
python -m eval.silver.bridges.bridge_populate_corpus --help
```

Calibration, replay and analysis utilities (`analysis/map_context` is a library, not a
CLI), plus the corpus bridges. They are effectively offline **only when the required
local primer and embedding-cache bundle already exists** — otherwise they fall back to
live embedding calls.

> `bridges/bridge_one_case_proof` has a `__main__` but **no argparse**, so it does not
> support `--help`: any argument is ignored and the proof runs. Read it before invoking.

### Inspection and reporting — `reporting/`

```bash
python -m eval.silver.reporting.dashboard --help
```

`reporting/inspect` writes the human-readable Markdown/CSV reports (library only) and
`reporting/dashboard` renders the standalone HTML dashboard. These modules do not
construct API clients or require database access — the input artifacts they read are
produced by the workflows above.

## Offline, cached, and paid execution

1. **Pure offline utilities** — schemas, JSONL helpers, deterministic splitting,
   inspection, dashboard generation.
2. **Cached replay** — matching, sweeps, and replay workflows, *when* the required primer
   and embedding caches are already present locally.
3. **Paid or external-service workflows** — Opus silver-label generation; embedding
   generation on cache miss; `relation_pairs/` Anthropic Message-Batches submission and
   collection; database-backed source-case preparation.

Note that most of `eval/data/` is **gitignored**. The primer and embedding SQLite caches
are **not guaranteed to exist in a fresh clone**, so a workflow that behaves as "offline"
in a populated local thesis environment may become **unavailable or paid** when those
caches are absent.

## Experiments and outputs

`experiments/` holds the numbered **thesis experiment packages E01–E14** (IDs are
non-contiguous), and `corpus_stats` belongs to the same experiment-record area.
Experiment IDs, configurations, and frozen outputs **should not be casually renumbered,
rewritten, or rerun**.

Authoritative experiment specifications, costs, and status live in
[`eval/EXPERIMENTS.md`](../EXPERIMENTS.md); broader evaluation orientation lives in
[`eval/README.md`](../README.md). Generated reports normally land under the gitignored
`eval/reports/` directory, so not every experiment output is a tracked file.

## Relation-pair workflow

`relation_pairs/` implements the **E13 claim-pair workflow**. Its submission and
collection steps use **Anthropic Message Batches** — they **may spend money and should
not be run casually**. E13's authoritative experiment record remains in
[`eval/EXPERIMENTS.md`](../EXPERIMENTS.md); the full procedure is not repeated here.

## Historical code

`old_files/` contains a **superseded historical driver** (`run_summarization_sweeps.py`).
It is **not** a maintained entry point and its self-documented commands no longer
resolve — but it is **not dead**: `scripts/eval/run_summarization_experiments.py` still
imports constants (e.g. `HYBRID_BLEND_GRID`) from it, which is why it stays in place. Its
contents should not be modernized or reactivated without an explicit reason.

## Tests

`eval/silver/tests/` is the active pytest suite for this package. The tests are intended
to run **without paid API calls**. Contributors should add or update tests when changing
active harness behaviour.

## Invocation guidance

```bash
python -m eval.silver.<module> --help
```

Before running anything, read the module and the experiment documentation — some entry
points will connect to PostgreSQL, initialize Anthropic or embedding clients, submit
batches, populate missing caches, or rerun frozen thesis experiments.

## Related documentation

- [`eval/README.md`](../README.md) — evaluation harness overview
- [`eval/EXPERIMENTS.md`](../EXPERIMENTS.md) — experiment specifications, costs, status
- [`eval/sweeps/README.md`](../sweeps/README.md) — frozen-artifact calibration sweeps
