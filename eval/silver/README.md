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
plus the reusable implementation it calls: `sample.py`/`sampler.py`,
`export_pipeline.py`/`exporter.py`, `generate.py`/`generator.py`. These pairs are
intentional, not duplicated implementations.

```
eval/silver/
├── *.py              top-level modules: sampling/export, silver generation, matching,
│                     evaluation, calibration sweeps, replay bridges, reporting
├── experiments/      numbered thesis experiment packages (E01–E14) + corpus_stats
├── relation_pairs/   E13 claim-pair workflow (Anthropic Message Batches)
├── old_files/        superseded historical drivers — reference only
└── tests/            the package's pytest suite
```

## Main workflows

### Source-case preparation

`sample`, `export_pipeline` — select source cases and export pipeline findings for them.
**Database-backed:** these require access to the PostgreSQL instance.

### Silver-label generation

`generate`, `generator`, `prompts` — produce silver findings from source cases.
**This workflow uses Anthropic / Opus and can make paid API calls.**

### Matching and evaluation

`embedders`, `matcher`, `evaluate` — embed and align pipeline findings against silver
findings, then compute metrics. The embedding providers (OpenAI / Gemini) **can make
paid API calls on a cache miss**; a warm local embedding cache is not guaranteed.

### Calibration and replay

`sweep`, `pipeline_sweep`, `map_theta_sweep`, `run_new_summarization_sweeps`,
`map_context`, `score_distribution`, `escalation_breakdown`, and the `bridge_*` modules
are calibration, replay, and analysis utilities. They are effectively offline **only when
the required local primer and embedding-cache bundle already exists** — otherwise they
fall back to live embedding calls.

### Inspection and reporting

`inspect`, `dashboard`, `split`, `schemas`, `jsonl_utils` — human-readable reports, the
HTML dashboard, deterministic dev/test splitting, Pydantic schemas, and JSONL helpers.
These modules do not themselves construct API clients or require database access (the
input artifacts they read are produced by the workflows above).

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

`old_files/` contains **superseded historical drivers**, retained for provenance and
reference only. It is **not** a maintained entry point, and its contents should not be
modernized or reactivated without an explicit reason.

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
