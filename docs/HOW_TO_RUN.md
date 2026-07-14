# How to run

Every command below was executed against this tree before being written down. Where a
command needs something this machine does not have (a live database, an API key, the
frozen thesis artifacts), that is stated explicitly and the command is documented up to
the point it was verified — argument parsing and preflight validation.

---

## 1. Requirements

* **Python 3.10 – 3.12** (`requires-python = ">=3.10,<3.13"`). 3.13 is not supported:
  the dependency set (torch, scispaCy) does not build there.
* **PostgreSQL** — only for the database-backed workflows (§5, §7, §8, §9).
* **API keys** — only for knowledge extraction (§9). Everything else is free.

## 2. Install

```bash
git clone <repository>
cd nlp-histo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install -r requirements.txt   # dependencies
python -m pip install -e . --no-deps        # the project itself, editable
```

This installs one package — `nlp_histo` — and one console command, `nlp-histo`.
Nothing else in the repository is installed: `eval/`, `scripts/` and `tests/` are
repository-only.

From a built wheel (no source tree needed):

```bash
python -m build                       # writes dist/nlp_histo-*.whl
python -m pip install dist/nlp_histo-*.whl
```

Verify:

```bash
nlp-histo --help
python -c "import nlp_histo; print(nlp_histo.__file__)"
```

Both work from **any** directory — the package does not need the repository as its
working directory in order to import.

## 3. The command surface

```
nlp-histo db init | check                   create / verify the schema
nlp-histo acquire download | unpack | organize   build the corpus from PMC
nlp-histo ingest                            PDF → text + figures + tables → DB
nlp-histo ner extract | merge | export      scispaCy/UMLS entities
nlp-histo knowledge                         LLM knowledge extraction   ⚠ COSTS MONEY
nlp-histo replay chapter9                   offline thesis replay      (free)
```

`--help` works for every command and subcommand without a database, an API key, a
model download, or the repository — it opens no connection and writes nothing.

## 4. Environment variables

| Variable | Purpose |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | database connection (usually via `.env`) |
| `NLP_HISTO_ENV_FILE` | explicit path to the `.env` file |
| `NLP_HISTO_ENTITY_CACHE` | entity-linking cache file (see §8) |
| `NLP_HISTO_OPENAI_EMBEDDING_CACHE`, `NLP_HISTO_GEMINI_EMBEDDING_CACHE` | embedding caches |
| `NLP_HISTO_MODEL_PRICES`, `NLP_HISTO_NLI_MODELS` | override the packaged defaults |
| `NLP_HISTO_DISABLE_UMLS`, `NLP_HISTO_SKIP_UMLS_ENRICHMENT` | kill-switches for low-RAM machines |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … | only for knowledge extraction |

`.env` is searched **upward from your working directory**, not next to the installed
package. Copy `.env.example` to `.env` and fill it in.

## 5. Database

```bash
createdb -U <admin-role> -O <db-user> <database-name>
cp .env.example .env      # set DB_* (DB_USER must be <db-user>)

nlp-histo db init         # create + verify the schema; safe to re-run
nlp-histo db check        # verify only; creates nothing
```

`db init` builds the schema from the ORM models. Alembic only migrates a database that
already exists — it does **not** initialise an empty one.

*Verified here:* `nlp-histo db check` ran against a live PostgreSQL server and reported
the schema current. `db init` was not re-run (the schema already exists); it is the same
code path with creation enabled.

## 6. Acquire the corpus

```bash
nlp-histo acquire download --pmcid-file target_pmc_ids.txt --output-dir files/tarballs
nlp-histo acquire unpack   --input-dir  files/tarballs      --output-dir files/corpus
nlp-histo acquire organize --input-dir  files/corpus \
                           --pdf-dir    files/organized_pdfs \
                           --xml-dir    files/organized_xmls
```

Every path is explicit — nothing is resolved against a hidden default. Re-running skips
work already done; pass `--overwrite` to force. A missing input fails immediately with
the offending path.

*Verified here:* `acquire organize` end-to-end on a temporary fixture. `download` was
not run against NCBI (no bulk downloading during the migration).

## 7. Ingest PDFs

```bash
nlp-histo ingest --pdf-dir files/organized_pdfs --out-root out
```

Outputs land under `out/` **relative to the directory you run from** — that is
deliberate, and `--out-root` overrides it. Requires PostgreSQL.

## 8. Named-entity recognition

```bash
nlp-histo ner extract --entity-cache /path/to/entity_linking_cache.json
nlp-histo ner merge
nlp-histo ner export
```

**Cache migration — read this if you have the old 30 MB cache.** The entity cache used
to live at `named_entity_recognition/entity_linking_cache.json`, next to the module.
That cannot survive installation (it would be written into `site-packages`), so it now
resolves as: explicit `--entity-cache` → `$NLP_HISTO_ENTITY_CACHE` →
`~/.cache/nlp-histo/entity_linking_cache.json`.

Your existing cache was **not moved or copied**. To keep using it:

```bash
export NLP_HISTO_ENTITY_CACHE=$PWD/named_entity_recognition/entity_linking_cache.json
```

Otherwise NER starts with a cold cache and recomputes the UMLS links.

## 9. Knowledge extraction — ⚠ this costs money

```bash
nlp-histo knowledge --profile cheap --pmcid PMC1448691 --sync --health-check no
nlp-histo knowledge --profile real  --all --source-cases eval/data/source_cases_related15.jsonl
```

`--profile` is required — there is no implicit default, precisely so a run cannot spend
money by accident. `cheap` avoids the premium tier.

* `--config PATH` — run configuration (default `configs/run.yaml`, relative to your
  working directory). It is *your* file, not an application default, so it is not
  packaged.
* `--source-cases PATH` — the frozen calibration selection used by `--all` / `--sample`.
  A repository artifact; not packaged. A missing file fails **before** any paid call.

*Verified here:* `--help`, flag parsing, and the missing-file preflight (it exits before
any provider is constructed). No API call was made during the migration.

## 10. Offline chapter-9 replay — free

```bash
nlp-histo replay chapter9 --artifact-root . --output-dir /tmp/replay-out
```

Offline: no API key, no database, no model inference, no cost.

`--artifact-root` is **required**. The replay used to infer the repository root from its
own file location, which is meaningless once installed. The root must contain:

```
out/summaries/summaries/
eval/data/map_primer/voter_cache.json
eval/data/embedding_cache_openai.sqlite
```

The embedding cache is *required*, not optional: without it the matcher would miss and
issue **paid** embedding calls in a workflow documented as free, so the command refuses
to start instead. Missing inputs are named in the error and nothing is written.

Outputs default to `<artifact-root>/out/thesis_results/chapter9_offline_replay/`.

## 11. Tests

```bash
python -m pytest              # full suite
ruff check .
```

## 12. Repository-only commands (must run from the repository root)

The thesis experiments are **not** installed and never enter the wheel. They import the
installed package and read frozen artifacts under `eval/data/` and `eval/reports/`, so
they must be run from the repository root:

```bash
python -m eval.silver.experiments.E04_cardinalities.cardinalities
python -m eval.silver.experiments.E14_heldout.heldout_eval --theta-frontier
python -m eval.silver.analysis.map_theta_sweep --help
python eval/sweeps/grounding.py
```

## 13. Known limitations

* **B-102** — `eval/silver/analysis/map_theta_sweep.py` still declares
  `PRIMER_DIR = Path("eval/data/map_primer")`, a working-directory-relative path. Any
  experiment driver that reads the primer must be run from the repository root. This is
  an experiment-side path bug, tracked separately; it does not affect the installed
  package or `nlp-histo replay chapter9`, which takes an explicit `--artifact-root`.
* `scripts/run_paper.py` and `scripts/thesis/run_chapter9_offline_replay.py` remain as
  thin compatibility wrappers around the installed commands. Prefer
  `nlp-histo knowledge` and `nlp-histo replay chapter9`.
