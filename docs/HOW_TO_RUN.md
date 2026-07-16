# How to run

## Verification status — read this first

This file previously opened by claiming *"every command below was executed against this
tree before being written down."* **That was not true**, and the claim is what hid the
defects: §9's two commands had been written by inspection, both were unrunnable (a
`--pmcid` flag that does not exist; two omitted required arguments), and the section
asserted "flag parsing" had been verified while it had not (BUGS.md B-105). A blanket
assurance is worth less than an honest inventory, so here is the inventory.

**Executed and verified 2026-07-16**, under a guard that raises on any billable host
(zero paid calls made):

| § | Command | Status |
|---|---|---|
| 2 | `nlp-histo --help`, `import nlp_histo` from any directory | verified |
| 3 | every command/subcommand `--help` (15) | verified, all exit 0 |
| 4 | the eight `NLP_HISTO_*` env vars | verified present in `src/` |
| 5 | `db check`, `db init` | verified against a live PostgreSQL — including `db init` against an **empty** database (21 tables created from the ORM, exit 0, ~1 s) |
| 6 | `acquire organize` missing-input failure | verified (exit 1, names the path) |
| 7 | `ingest` | **verified end-to-end** on one 8-page PDF into an isolated database — exit 0, 33 s, 14 text elements + 10 figures, matching the established corpus exactly |
| 9 | both `knowledge` commands, `--dry-run` | verified (exit 0, no provider contacted) |
| 10 | `replay chapter9` | verified — 9/9 CSVs byte-identical; exit 3 when UMLS is unreachable |
| 11 | `pytest`, `ruff check .` | verified — 1565 passed, 0 failed; ruff clean |
| 12 | E04, `sweeps/grounding.py`, E14, `map_theta_sweep --help` | verified free (0 cache misses, no paid host) |

**Not executed — do not read these as verified:**

* **§2** — the `venv` sequence and the `python -m build` wheel path. This machine runs a
  framework 3.12 install; imports and the console script resolve, but the documented
  install was not re-run from scratch.
* **§6** — `acquire download` / `unpack`. No bulk download was performed against NCBI.
* **§8 `ner`** — **dispatch and flag names confirmed by reading the forwarded parser**
  (`batch_ner.py:288` `--entity-cache`), and the entity-cache migration advice checked
  against `ner/cache_paths.py`, but **no `ner` command was run**.
* **§12** — `E14 --theta-frontier` (the plain E14 run is verified).
* **§13** — the B-102 limitation, carried forward from an earlier audit.

Where a command needs something this machine does not have, that is stated at the command
itself. **`--dry-run` (§9) resolves a paid invocation's full configuration for free** — a
cost constraint is a reason to find the free verification, not to skip it.

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
nlp-histo replay chapter9                   thesis replay from frozen artifacts (free)
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

*Verified 2026-07-16:* both ran against a live PostgreSQL server. `db check` reported
`schema is present and valid (21 tables)`. `db init` was exercised **both ways**: against
an existing schema it exits 0 (`OK: schema verified (21 tables)`), dropping nothing — and
against a genuinely **empty** database it created all 21 tables from the ORM in ~1 s
(`Database tables created successfully!` → `OK: schema verified (21 tables)`), with the
table set matching `database/models.py`.

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

`ingest` defines no options of its own — everything after it is forwarded to the
extraction runner. `nlp-histo ingest --help` shows this CLI's stub; **`nlp-histo ingest
-- --help` lists the runner's real options** (`--glob`, `--max-docs`, `--workers`,
`--detector`, …).

*Verified end-to-end 2026-07-16* — the command above, exactly as written, on one 8-page
main PDF, into an isolated artifact root and an empty isolated database (`DB_NAME` pointed
elsewhere via `NLP_HISTO_ENV_FILE`; the production corpus was untouched):

```
nlp-histo ingest --pdf-dir <root>/pdfs --out-root <root>/out
→ exit 0 in 33 s · ok=1 fail=0 skip=0 · no paid call (guard: 0 billable hosts)
DB:   documents=1 · text_elements=14 (all non-empty, avg 692 chars, all with path_string)
      figures=10 · tables=0
disk: out/text (1) · out/figures (10 PNGs) · out/json (1) · out/docling_full (2)
      out/run_metadata (2)
```

The counts match the established corpus for the same paper exactly (14 text elements,
10 figures, 0 tables), and `unique_path` has the documented
`{PMCID}/{section}/{position}` shape. Two things worth knowing, both matching production
rather than deviating from it: `documents.title` is left `NULL` (it is NULL for all 977
papers in the established corpus), and `ingest` writes no `pipeline_runs` row. Also note a
missing `--pdf-dir` does **not** error — the run finds 0 PDFs, writes a run manifest, and
exits 0.

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

As with `ingest`, `ner` forwards its options: `nlp-histo ner extract -- --help` lists the
real ones (`--entity-cache`, `--limit`, `--force`, …).

> **Partly verified (2026-07-16).** Confirmed by inspection: the 30 MB cache is still at
> `named_entity_recognition/entity_linking_cache.json` (the module's `.py` files moved to
> `src/nlp_histo/ner/`; the data file did not), `--entity-cache` exists
> (`batch_ner.py:288`), and the resolution order matches `ner/cache_paths.py` — explicit
> flag → `$NLP_HISTO_ENTITY_CACHE` → `$XDG_CACHE_HOME/nlp-histo/…` → `~/.cache/nlp-histo/…`.
> **No `ner` command was executed** (they write to the database), so the migration advice
> is reasoned from the code, not demonstrated.

## 9. Knowledge extraction — ⚠ this costs money

```bash
nlp-histo knowledge PMC1448691 --profile cheap --sync --health-check no
nlp-histo knowledge --all --profile real --batch --health-check no \
                    --source-cases eval/data/source_cases_related15.jsonl
```

The PMCID is a **positional** argument — there is no `--pmcid` flag.

**Three flags are required and have no defaults** — `--profile`, `--health-check`, and
exactly one of `--sync` / `--batch`. That is deliberate: it is what stops a paid run
starting by accident. `cheap` avoids the premium tier; `--batch` is async (roughly half
the price) and is the right mode for a full `--all` corpus run, `--sync` for a single
paper.

**Check any paid command with `--dry-run` first** — it prints the resolved config and
exits before a single API call, so you can confirm the whole invocation is right for free:

```bash
nlp-histo knowledge --profile real --all --batch --health-check no \
                    --source-cases eval/data/source_cases_related15.jsonl --dry-run
```

* `--config PATH` — run configuration (default `configs/run.yaml`, relative to your
  working directory). It is *your* file, not an application default, so it is not
  packaged.
* `--source-cases PATH` — the frozen calibration selection used by `--all` / `--sample`.
  A repository artifact; not packaged. A missing file fails **before** any paid call.

*Verified 2026-07-16:* both commands above were run **exactly as written** with
`--dry-run` appended, under a guard that raises on any billable host — each resolves its
full cascade config and exits 0 without contacting a provider. Also verified: `--help`,
and the missing-`--source-cases` preflight (exits non-zero before any provider is
constructed). No API call has been made.

Both commands in this section were previously wrong — they used a non-existent `--pmcid`
flag and omitted required arguments, so neither could run (BUGS.md B-105). They are now
checked by `--dry-run` rather than by inspection.

## 10. Chapter-9 replay — free (no paid API calls)

```bash
nlp-histo replay chapter9 --artifact-root . --output-dir /tmp/replay-out
```

No API key, no database, **no cost** — no paid provider is ever called (verified
2026-07-16: the only hosts contacted are `huggingface.co` and
`s3-us-west-2.amazonaws.com`, both free model downloads).

> **API-free does not mean offline.** Two separate things, and conflating them is what
> produced two defects here:
>
> * **API-free — guaranteed.** Every embedding is served from the frozen caches. The
>   replay runs strict **cache-only**: no embedding provider is constructed, every
>   required cache entry is validated before the run starts, and a miss aborts (exit 4)
>   rather than contacting OpenAI or Gemini. It cannot fall through to a paid call
>   (BUGS.md B-112, fixed in `8d0c5c5`).
> * **Offline — not supported.** **UMLS-dependent replay still requires network access
>   even when the model artifacts have already been downloaded**, because scispaCy
>   resolves its cache through a live S3 ETag lookup. That access is a free
>   model/metadata fetch from `s3-us-west-2.amazonaws.com` and incurs **no paid model or
>   API usage**. Without it the replay refuses to start (exit 3) rather than silently
>   skipping CUI work (BUGS.md B-107).

**It requires network on every run, and it runs model inference.** It uses two models —
scispaCy `en_core_sci_lg` + the UMLS KB (from S3) and `pritamdeka/PubMedBERT-MNLI-MedNLI`
(from HuggingFace) — and runs the NLI model locally (on MPS/CUDA where available). A cold
machine downloads several hundred MB; a warm one still needs the network, for a reason
worth knowing:

* **HuggingFace caches properly.** Once downloaded, the NLI model loads offline (verified
  2026-07-16 against a real DNS failure — no `HF_HUB_OFFLINE` needed).
* **scispaCy does not.** Its cache filename is `sha256(url).sha256(etag)`, and the ETag
  comes from a live `requests.head()` against S3 on **every** load
  (`scispacy/file_cache.py:119,126`). With no network it cannot compute the filename, so
  it cannot find a fully-downloaded 2.1 GB cache that is already on disk. **Pre-fetching
  does not make the replay offline-capable** — nothing you download changes this.

**If the linker is unreachable the replay now refuses to start**, exits **3**, and writes
nothing — naming the affected outputs (`06_exp_f_test_split`,
`12_real_profile_grounding_polarity`) and the real cause. It previously logged a single
`WARNING UMLS: linker unavailable`, exited 0, and wrote plausible-but-wrong tables; that
silent path is fixed (BUGS.md B-107). Exit codes: `0` ran · `2` artifact tree unusable ·
`3` UMLS linker unavailable · `4` an embedding cache is incomplete. **Every** non-zero
path refuses before anything is written.

`--artifact-root` is **required**, and it is a **repository-shaped artifact tree**, not a
flat bundle: the *code* comes from the installed package, the tree supplies only *data*.
Every input below is validated in one pass **before** anything is written — a missing or
invalid artifact lists all problems and exits non-zero, so an incomplete tree can never
produce a partial set of tables that reads like a result.

```
out/summaries/summaries/                        per-paper summary JSONs
out/summaries/cascade_decisions/                per-chunk cascade decisions
out/summaries/corpus_relations*.json            ≥1 variant, for the NLI-input A/B table
eval/data/map_primer/voter_cache.json           frozen voter outputs
eval/data/silver_findings_related15.jsonl       silver labels
eval/data/embedding_cache_openai.sqlite         frozen embedding cache — analyses 05, 10
eval/data/embedding_cache_gemini.sqlite         frozen embedding cache — analyses 06, 10, 12
scripts/eval/run_summarization_experiments.py   orchestrator, loaded by path
reports/stage6_PR.md                            frozen rubric report, parsed as data
```

**Both** embedding caches are required, and both are resolved from `--artifact-root`.
Analysis 05 loads the map context with the `openai` embedder; 06, 10 and 12 use `gemini`.
A cache that is absent — or present but incomplete — would make the run miss and embed at
run time, which is a **paid** call in a workflow documented as free. So the replay refuses
to start instead, and the check is stronger than "the files exist":

* **Every required entry is validated up front**, not just the SQLite header. An empty or
  partial cache is a perfectly valid database (and `make_embedding_cache` creates one on
  demand), so existence proves nothing. The required set is derived from the voter cache
  using the same extraction the run itself performs — 15 273 claims on this tree. Missing
  entries → **exit 4**, before the output directory is created, reporting counts and paths.
* **The run is strict cache-only.** No embedding provider is constructed at all, so an
  unexpected miss raises rather than billing. The guarantee is structural — it does not
  depend on an unset API key.

*Verified 2026-07-16 (BUGS.md B-112, fixed in commit `8d0c5c5`):* both caches complete
here — 15 273 unique claims, **0 cache misses**, zero paid hosts contacted. A tree without
the gemini cache exits 2; a tree whose gemini cache is present but empty exits 4
(`gemini: 15273 of 15273 required entries missing`). Neither writes anything. Before that
commit the gemini cache was used but unvalidated, and was resolved against the repository
rather than `--artifact-root` — a tree without it passed preflight and then issued ~15 000
paid Gemini calls.

**Size** (measured 2026-07-16 on this tree; `du` reports 1024-based MB):

| Scope | Size | What it includes |
|---|---|---|
| Required inputs | **≈ 1 585 MB (1.5 GB)** | exactly the artifacts listed above — nothing else under `out/` or `eval/` |
| `eval/data/` wholesale | **≈ 1 714 MB (1.7 GB)** | the whole directory: the two caches, the primers, silver labels, splits, relation pairs |

The two embedding caches are **1 551 MB** of the required set (gemini 1.1 GB + openai
467 MB); everything else together is under 35 MB. Copying `eval/data/` wholesale is the
simpler recipe and costs ~130 MB more than the minimum.

```bash
# reproduce the required-set figure
du -scm out/summaries/summaries out/summaries/cascade_decisions \
        out/summaries/corpus_relations*.json \
        eval/data/map_primer/voter_cache.json \
        eval/data/silver_findings_related15.jsonl \
        eval/data/embedding_cache_openai.sqlite \
        eval/data/embedding_cache_gemini.sqlite \
        scripts/eval/run_summarization_experiments.py \
        reports/stage6_PR.md | tail -1
du -scm eval/data | tail -1        # the wholesale figure
```

None of it is in the wheel. A symlink or hard-link tree pointing at the repository's
artifacts works and costs nothing. *(Earlier revisions of this file claimed "roughly
2.8 GB", which was unsourced and matched no measurable scope, and then "≈ 640 MB", which
was measured before the gemini cache became required.)*

**A complete run writes exactly 9 CSVs.** Two files in the historical results directory
do not regenerate, for reasons that predate this packaging work:

* `04_theta_heatmap.csv` — needs `eval/reports/exp_{1,4}_*_scorer_full_*.csv`, which are
  no longer in the tree; the analysis reports `status=missing` and writes nothing.
* `10_cascade_vs_sonnet_gap_ci_per_case.csv` — fails with
  `ValueError: too many values to unpack (expected 4)`: `map_theta_sweep._replay()` now
  returns 5 values and the replay unpacks 4. Verified to be a pre-existing mismatch at
  the branch point, untouched by the packaging migration, and not fixed here.

Outputs default to `<artifact-root>/out/thesis_results/chapter9_offline_replay/`.

## 11. Tests

```bash
python -m pytest              # full suite
ruff check .
```

Both are clean as of 2026-07-16: `ruff check .` passes, and `pytest` is **1565 passed,
0 failed** (~2–4 min). Two `tests/test_config_loader.py` failures found during that day's
verification were stale assertions pinning the pre-calibration agreement defaults, and
have been corrected to the calibrated E06/E08 winner that `configs/run.yaml` ships —
the config was right, the tests were not (BUGS.md B-110). Nine regression tests were
added alongside the B-107 fail-hard fix and the B-111 passthrough fix.

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

**These are free**, despite reading like paid commands: every embedding is served from
the frozen sqlite caches (`eval/data/embedding_cache_{openai,gemini}.sqlite`), which
were verified on 2026-07-16 to yield **0 cache misses**. No paid endpoint is contacted.
Two caveats before you run them:

* **They still require `GOOGLE_API_KEY` to be set**, even though they never call Google:
  `map_context.py` exits with `GOOGLE_API_KEY not set` before it reaches the cache
  (BUGS.md B-109). Any non-empty value satisfies it.
* **`eval/sweeps/grounding.py` overwrites the tracked `eval/results/grounding_sweep.md`**
  with numbers from whatever is currently in `out/summaries` — it does not reproduce the
  committed 5-paper version. Expect a dirty worktree; `git checkout eval/results/` to
  restore (BUGS.md B-108).

Each of these loads scispaCy + the UMLS KB (several GB of RSS). **Run them one at a
time** — concurrent runs will exhaust memory on a 16–32 GB machine.

## 13. Known limitations

* **B-102** — `eval/silver/analysis/map_theta_sweep.py` still declares
  `PRIMER_DIR = Path("eval/data/map_primer")`, a working-directory-relative path. Any
  experiment driver that reads the primer must be run from the repository root. This is
  an experiment-side path bug, tracked separately; it does not affect the installed
  package or `nlp-histo replay chapter9`, which takes an explicit `--artifact-root`.
* `scripts/run_paper.py` and `scripts/thesis/run_chapter9_offline_replay.py` remain as
  thin compatibility wrappers around the installed commands. Prefer
  `nlp-histo knowledge` and `nlp-histo replay chapter9`.
