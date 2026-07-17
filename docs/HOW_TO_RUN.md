# How to run

> **Never run this project before?** Read **`docs/REPRODUCE.md`** instead. It is the same
> material as one linear sequence — clone, install, restore, replay — written to be followed
> top to bottom without jumping between sections. Come back here when you want the detail on
> a specific command: this file is the reference, and it is authoritative if the two ever
> disagree.

**Already oriented? Read §0, then follow the numbered sections in order.** Each one says what
it produces and what success looks like. Nothing below costs money except §9, which is marked.

**Fastest useful result:** §0 (get the bundle) → §2 (install) → §10 (replay) → nine tables
identical to the thesis. No database, no API key, no cost. Budget an hour: the 1.2 GB
download and the ~4 GB dependency install dominate, and the replay itself is ~5 minutes.

| I want to… | Do this |
|---|---|
| reproduce the thesis tables | §0 Path A → §2 → §10 |
| query the corpus / run NER | §0 Path B → §2 → §5 → §8 |
| rebuild the corpus from PMC | §0 Path C → §2 → §5 → §6 → §7 → (§9 **costs money**) |
| just check the code is sane | §2 → §11 |

### What each step produces

Follow these in order. Every command is run **from the repository root**.

| § | Command | Produces | Success looks like | Time |
|---|---|---|---|---|
| 2 | `pip install -r requirements.txt` · `pip install -e . --no-deps` | the `nlp-histo` command | `nlp-histo --help` exits 0 | ~4 GB; **time not measured** |
| 5 | `nlp-histo db init` | 21 tables in your database | `OK: schema verified (21 tables)` | seconds |
| 6 | `nlp-histo acquire download` → `organize` | `files/organized_pdfs/`, `files/organized_xmls/` | `N succeeded · 0 failed` | ~3 s/paper |
| 7 | `nlp-histo ingest` | text + figures + tables in the database, `out/` | `ok=N fail=0` | ~30 s/paper |
| 8 | `nlp-histo ner extract` → `merge` → `export` | `entities` rows; `umls_entities_lg/`, `disease_entities_lg/` | `Summary: N Processed … 0 Errors`, then `✓ Saved N files` | slow on a cold cache |
| 9 | `nlp-histo knowledge` | `sum_*` tables, `out/summaries/` | per-paper JSON written | **⚠ costs money** |
| 10 | `nlp-histo replay chapter9` | 9 CSVs in `out/thesis_results/…` | exit 0, nine files | ~5 min |
| 11 | `pytest` · `ruff check .` | — | `1697 passed`, `All checks passed!` | 3–4 min |

**Exit codes are meaningful, not decorative.** Non-zero always means *stop and read*, never
"it mostly worked": `2` the artifact tree is unusable · `3` the UMLS model is unreachable ·
`4` an embedding cache is incomplete. Commands here fail loudly rather than producing
plausible-looking wrong output — if something is missing you get an itemised error, not a
partial result.

### Two things that will bite you if nobody says so

1. **Your shell's `DB_*` variables beat the `.env` file.** If `env | grep '^DB_'` prints
   anything, that wins, and a command you *think* is aimed at a scratch database will write
   to whichever one your shell names. Check before any write; `db init`/`db check` print the
   database they resolved — believe that, not your intent.
2. **Run the heavy commands one at a time.** `ingest`, `ner`, `replay` and `pytest` each
   load scispaCy plus the UMLS knowledge base — several GB of RAM each. Two at once will
   exhaust a 16–32 GB machine.

A note on the tone below: sections cite bug IDs like `B-112` and say when a command was
last verified. That is provenance for the maintainer — you can ignore it. If a command
surprises you, the citation tells you where the reasoning is written down
(`docs/BUGS.md`).

The full audit — what was executed, what was only inspected — is in the
**appendix at the end**.

---

## 0. What you need before you start

**A clean clone is not enough, and that is by design.** It gives you 172 MB of code,
tests, docs and configuration. It does **not** give you the frozen artifacts, the
database, or the PDFs — those are 1.5 GB of paid LLM output and embedding caches, a
485 MB database, and 5.2 GB of publisher PDFs that mostly cannot be redistributed (322 of
1093 carry no Creative Commons licence at all).

Verified on a fresh clone, 2026-07-16 — so you know exactly what does and does not work
before you start:

| Works from a clean clone alone | Needs the artifact bundle |
|---|---|
| §2 install · §3 `--help` · §4 env vars | §10 `replay chapter9` → exits **2**, naming every missing artifact |
| §5 `db init` / `db check` (your own PostgreSQL) | §12 experiments → no primer / caches |
| **§11 `pytest` → 1697 passed** | §7 `ingest` · §8 `ner` → no PDFs, no ingested documents |

Nothing fails silently — a missing artifact is a loud, itemised error. But a loud error
is not a result, so pick your path:

### Path A — reproduce the thesis tables (recommended, free)

You need **the replay bundle** — a 1.2 GB download that unpacks to ~1.5 GB, so budget both.
It is the frozen output of the paid pipeline, so no API key and **no database** are
required — `replay chapter9` never connects to PostgreSQL.

```
eval/data/embedding_cache_openai.sqlite       467 MB
eval/data/embedding_cache_gemini.sqlite       1.1 GB
eval/data/map_primer/voter_cache.json          22 MB
eval/data/silver_findings_related15.jsonl     1.1 MB
out/summaries/{summaries,cascade_decisions}/   10 MB
out/summaries/corpus_relations*.json
```

Unpack it over the clone so the paths match, then → **§10**. Expect 9 CSVs, byte-identical
to the published tables.

### Path B — work with the corpus (adds the database)

Restore the **corpus dump** (`nlp-histo-corpus.sql.gz`, 49 MB compressed → ~445 MB
restored, 15 s) into your own database (§5), point `.env` at it, then → §8 NER, §9
knowledge extraction, or any SQL against the 977 papers.

### Path C — rebuild from scratch

`files/target_pmc_ids.txt` (in the clone — it is the corpus *definition*) → **§6** acquires
the PDFs from NLM's AWS dataset → **§7** ingests them → **§9** re-runs knowledge
extraction. **§9 costs money.** You do not need the PDFs shipped to you: §6 fetches them.

### Getting the replay bundle

| | |
|---|---|
| **File** | `nlp-histo-replay-artifacts-ec11eec.tar.gz` |
| **Size** | 1,244,628,904 bytes (1.2 GB) |
| **Source commit** | `ec11eec` |
| **SHA-256** | `cded4299ac1a47cf8b857f5796731a27d3b66eac234a1acf306771e73d3e45d3` |

* **Primary — LRZ Sync+Share:** <https://syncandshare.lrz.de/getlink/fiBHdDWVJLKMxYP5JicFvd/nlp-histo-bundles>
* **Mirror — Google Drive:** <https://drive.google.com/drive/folders/1uo-iGOb3df11LqjQRbCwoyRsiVDMJkpY?usp=sharing>

**Both locations hold the identical archive** — verified 2026-07-16 by downloading from
each and comparing SHA-256 against the build machine: all three match
(`cded4299…`), as do the `.sha256` and `.manifest.json` sidecars.

**No account is needed.** Both are read-only links that serve anonymously — you do not
sign in, and you cannot write through either. Note what that means: **anyone holding a URL
can download the bundle.** It is unlisted rather than access-controlled, and it is not
publicly indexed, archived, or assigned a DOI. The bundle holds derived artifacts —
embeddings, summaries, cascade decisions — and **no publisher PDFs**, which is why link
sharing is acceptable here; the PDFs are not redistributable and are not in it (§6
re-acquires them instead).

```bash
# ── download ────────────────────────────────────────────────────────────────
# LRZ: the browser link is JavaScript-driven; this fetches the folder as one zip.
curl -L -o lrz-bundle.zip \
  "https://syncandshare.lrz.de/dl/fiBHdDWVJLKMxYP5JicFvd/nlp-histo-bundles.dir"
unzip lrz-bundle.zip

# Google Drive mirror: a 1.2 GB file triggers Drive's virus-scan interstitial, so a plain
# curl silently saves an HTML page named like the archive. Use the confirm parameter:
FILE_ID=1SOWdAhhy0ZFqwlMjG3X3EFvgNQyKhIxW
curl -L -o nlp-histo-replay-artifacts-ec11eec.tar.gz \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t"
# or just download it from the browser link above.

# ── validate BEFORE trusting it ─────────────────────────────────────────────
shasum -a 256 -c nlp-histo-replay-artifacts-ec11eec.tar.gz.sha256
#   → nlp-histo-replay-artifacts-ec11eec.tar.gz: OK
file nlp-histo-replay-artifacts-ec11eec.tar.gz    # must say "gzip compressed data",
                                                  # not "HTML document" (see Drive, above)

# ── extract over a clean clone so the paths line up ──────────────────────────
tar -xzf nlp-histo-replay-artifacts-ec11eec.tar.gz -C /path/to/nlp-histo

# ── artifact preflight (fails loudly and itemised if anything is missing) ────
cd /path/to/nlp-histo
nlp-histo replay chapter9 --artifact-root . --output-dir /tmp/replay-out
#   exit 0 → 9 CSVs;  exit 2 → artifact tree unusable;  exit 3 → UMLS unreachable;
#   exit 4 → an embedding cache is incomplete
```

The archive also carries `MANIFEST.json` (per-file path, byte size, SHA-256) if you want
to check individual artifacts rather than only the whole archive.

**Why checksum at all, given the preflight?** The replay already refuses an *incomplete*
cache — it validates every required entry, not merely that the file exists, and exits 4
rather than falling through to paid calls (B-112). What it cannot see is a cache that is
complete but *wrong*: every entry present, values corrupted in transit. That produces
plausible numbers. The checksum is what closes that gap.

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

From a built wheel — the package itself needs no source tree, but **the dependencies are
not in the wheel**:

```bash
python -m build                             # writes dist/nlp_histo-*.whl
python -m pip install -r requirements.txt   # STILL REQUIRED — see below
python -m pip install dist/nlp_histo-*.whl
```

`pyproject.toml` deliberately declares **no** `dependencies`: `requirements.txt` is the
pinned, tested set, and the wheel's only metadata requirement is `Requires-Python`. So
`pip install dist/nlp_histo-*.whl` on its own resolves nothing else — you get a package
that imports and prints help, and then fails on the first real command with
`ModuleNotFoundError: No module named 'sqlalchemy'`. Install `requirements.txt` into the
same environment (BUGS.md B-116).

Verify:

```bash
nlp-histo --help
python -c "import nlp_histo; print(nlp_histo.__file__)"
```

Both work from **any** directory — the package does not need the repository as its
working directory in order to import.

*Verified 2026-07-16* in a throwaway venv **outside** the repository (Python 3.12.0), from
the wheel alone — no source tree, no editable install, no `PYTHONPATH`:

```
python -m build                    → nlp_histo-0.1.0-py3-none-any.whl (483 KB)
pip install …/nlp_histo-0.1.0-py3-none-any.whl
  → "Would install nlp-histo-0.1.0" and nothing else — the wheel has no dependencies
import nlp_histo   → …/fresh-venv/lib/python3.12/site-packages/nlp_histo/__init__.py
nlp-histo --help + all 11 command/subcommand --help  → exit 0
packaged resources load: model_prices.json (1437 B), nli_models.yaml (1558 B)
nlp-histo db check → ModuleNotFoundError: sqlalchemy   ← the wheel carries no deps
```

Not exercised: installing `requirements.txt` into that fresh venv (a multi-GB
torch/docling resolution) — so the wheel's *packaging* is verified end-to-end, its
*dependency resolution* is not.

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
| `NLP_HISTO_ENV_FILE` | which `.env` file to read — **selects the file; does not win over already-set variables** (see below) |
| `NLP_HISTO_ENTITY_CACHE` | entity-linking cache file (see §8) |
| `NLP_HISTO_OPENAI_EMBEDDING_CACHE`, `NLP_HISTO_GEMINI_EMBEDDING_CACHE` | embedding caches |
| `NLP_HISTO_MODEL_PRICES`, `NLP_HISTO_NLI_MODELS` | override the packaged defaults |
| `NLP_HISTO_DISABLE_UMLS`, `NLP_HISTO_SKIP_UMLS_ENRICHMENT` | kill-switches for low-RAM machines |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … | only for knowledge extraction |

`.env` is searched **upward from your working directory**, not next to the installed
package. Copy `.env.example` to `.env` and fill it in.

**Precedence — a real variable already in your environment beats the file.** The order is
*environment variable* → *`.env` file* → *built-in default*, which is deliberate: it is how
you override a file value for one command (`DB_NAME=other nlp-histo db check`).

But it applies to `NLP_HISTO_ENV_FILE` too, and that surprises people:
`NLP_HISTO_ENV_FILE` chooses **which file is read**, not whose values win. If `DB_NAME` is
already exported — by `source .env`, direnv, docker-compose, CI secrets, or an IDE run
configuration — then pointing `NLP_HISTO_ENV_FILE` at a scratch config **will not** move
you off your production database, and nothing will say so. That nearly wrote a test ingest
into the 977-paper corpus during the §7 verification (BUGS.md B-113).

```bash
env | grep '^DB_'                    # what is already set?
env -u DB_NAME nlp-histo db check    # run with an inherited value cleared
nlp-histo db check                   # prints: Target: user@host:port/database
```

`db init` and `db check` echo the resolved target; `ingest` and `ner` do not — so check
with `db check` before pointing a write at a non-default database.

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

### Restoring the corpus dump

`db init` gives you an **empty** schema — the right thing if you intend to rebuild the
corpus yourself (§6 → §7, days of work). To get the 977 papers directly, restore the dump
instead; it carries the schema *and* the data, so **`db init` is not needed and should not
be run first** — restore into a database that is genuinely empty:

```bash
createdb -U <admin-role> -O <db-user> nlp_histo
gunzip -c nlp-histo-corpus.sql.gz | psql -d nlp_histo
```

Plain SQL, so any `psql` version reads it — deliberately not `-Fc`, which only a
`pg_restore` of the dumping version or newer can read, and the recipient's toolchain is
unknown (the reasoning is in `scripts/make_reproduction_bundle.py`). It prints a long
stream of `SET`/`CREATE TABLE`/`COPY`; `NOTICE` lines are normal, `ERROR` lines are not.
Then:

```bash
nlp-histo db check    # → OK: schema is present and valid (21 tables).
psql -d nlp_histo -c "SELECT count(*) FROM documents;"   # → 977
```

`db check` also prints `Note: 1 unrelated table(s) present and left untouched:
alembic_version`. That is expected — the dump carries Alembic's bookkeeping table, which is
not one of the ORM's 21; the count of 21 is correct.

The dump is built by `scripts/make_reproduction_bundle.py --with-db`. It carries
`--no-owner --no-privileges`, so it restores as whoever connects and needs no matching
`local_db_user` role; it contains no `CREATE DATABASE` (hence the `createdb` above), no
extensions, and no credentials.

*Verified 2026-07-17:* run end-to-end into a genuinely empty database owned by the
application role. **15 s**, exit 0, **zero `ERROR` and zero `NOTICE` lines**. Every table
matched the source database exactly — documents 977, text_elements 35 896, entities
1 792 440, figures 4 479, tables 1 960 — and `db check` reported `OK: schema is present and
valid (21 tables)`. The restored database is **~445 MB**; the 485 MB source figure above is
the same data plus accumulated write bloat, so a fresh restore is legitimately smaller.

## 6. Acquire the corpus

```bash
# AWS Open-Access dataset (default) — two steps, no tarball, no deprecation date
nlp-histo acquire download --pmcid-file files/target_pmc_ids.txt --output-dir files/corpus
nlp-histo acquire organize --input-dir  files/corpus \
                           --pdf-dir    files/organized_pdfs \
                           --xml-dir    files/organized_xmls
```

`download` defaults to `--source aws`: NLM publishes the OA Subset on AWS (free, no
login), and it writes `files/corpus/<PMCID>/` directly — the layout `unpack` produces —
so **`unpack` is not needed on this route**. AWS carries **no announced retirement date**,
unlike the FTP tree; that is not a guarantee of permanence, only the absence of a
published end-date.

The legacy FTP tarballs remain available with `--source ftp`, for resuming a corpus
started that way. **They are deleted in August 2026** (B-118), and that route still needs
all three steps:

```bash
nlp-histo acquire download --source ftp --pmcid-file files/target_pmc_ids.txt \
                           --output-dir files/tarballs
nlp-histo acquire unpack   --input-dir  files/tarballs --output-dir files/corpus
nlp-histo acquire organize --input-dir  files/corpus  --pdf-dir … --xml-dir …
```

**Both sources reach the same document ID.** AWS names its objects `PMC8395919.1.pdf`
(`.1` = article version), which would mint `PMC8395919.1` as a *new* document ID and
duplicate a paper the corpus already holds. So the AWS route reads the article's JATS
`self-uri` — the authoritative record of the publisher's filename, the same one the
tarball carried — and writes `corpus/<PMCID>/dermatopathology-08-00036.pdf` +
`<PMCID>.nxml`. `organize` then yields `PMC8395919_dermatopathology-08-00036.pdf` from
either route, so **resuming an FTP-built corpus with AWS is safe** and creates no
duplicate (BUGS.md B-119). If an article does not authoritatively name its PDF, that
article **fails** rather than acquiring a mismatched identifier.

Content equivalence was **spot-checked on one paper**, not proven for the corpus: for
`PMC8395919` the AWS PDF and the FTP tarball's PDF have the same SHA-256 and the same
5 733 574 bytes. That establishes equivalence for the tested article; it is not a claim
about all 1093.

Every path is explicit — nothing is resolved against a hidden default. Re-running skips
work already done; pass `--overwrite` to force. A missing input fails immediately with
the offending path.

`download` exits **1** when a non-empty request fetches nothing — it previously printed
`Done — 0 tarball(s)` and exited 0, so a wholly failed run read as success (BUGS.md
B-117). A *partial* result is still success: papers outside the OA subset are reported
per-PMCID and are expected.

> **Why the default is AWS.** NCBI moved every legacy FTP tree under
> `/pub/pmc/deprecated/` (their readme, updated 2026-04-10) while **its own OA API still
> advertises the pre-move paths** — so the advertised URL 404s for *every* paper
> (measured 0/5 across 2010–2025). `--source ftp` copes by trying the advertised URL
> first and the relocated one second, announcing the fallback when it uses it, but NCBI
> states those files **"will be removed in August 2026"**. AWS has no such expiry.
>
> The FTP code was never at fault, and that was tested rather than assumed: an FTP probe
> of the *original* advertised URL answers `550 … No such file or directory`, so both
> protocols fail identically and the `ftp://`→`https://` rewrite is sound. NCBI simply
> stopped serving what its API advertises (BUGS.md B-118).

`download` exits **1** if any requested paper fails, reporting succeeded/failed/skipped —
papers outside the OA subset are *skipped*, not failed. A downloaded file is validated as
a real `.tar.gz`; a 200 carrying an error page or a truncated stream is a failure, and
unusable files are discarded so a re-run retries them instead of skipping them (B-117).

*Verified end-to-end 2026-07-16*, both sources, one PMCID each, isolated from the
established corpus:

```
# aws (default)
acquire download --output-dir <iso>/corpus   → exit 0 · 3 s · PDF 5.7 MB (%PDF) + XML
acquire organize --input-dir  <iso>/corpus   → exit 0 · 1 PDF + 1 XML organized

# ftp (legacy, dies Aug 2026)
acquire download --source ftp → exit 0 · 7.2 MB · "↪ via NCBI's relocated legacy tree"
acquire unpack                → exit 0 · valid tar (17 members) · 1 PDF + 1 XML
acquire organize              → exit 0 · 1 PDF + 1 XML organized
```

Not a bulk fetch; `files/organized_pdfs` still holds its original 1132 PDFs.

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
`{PMCID}/{section}/{position}` shape.

Three things worth knowing, none of them faults in this run:

* **`ingest` writes no `pipeline_runs` row — by design.** That table tracks
  `KnowledgeExtractionRunner.process()` (§9) and is the FK root of the `sum_*` tables; the
  established database holds 33 rows against 977 ingested papers. Ingest's provenance is
  `out/run_metadata/run_{ISO}_{uuid}.json` plus per-document `{pmcid}_stats.json`.
* **`documents.title` is left `NULL`** — as are `journal` and `publication_year`, for all
  977 papers in the established corpus. Nothing consumes them (BUGS.md B-114).
* **A missing `--pdf-dir` does not error** — the run finds 0 PDFs, writes a run manifest,
  and exits 0.

> **⚠ Before running `ingest` against anything but your default database, prove the
> target.** An explicit `NLP_HISTO_ENV_FILE` does **not** override `DB_*` variables
> already exported in your shell, so a run aimed at a scratch database can silently write
> to your production corpus — this very nearly happened during the verification above.
> Check first: `env | grep '^DB_'`, and see BUGS.md B-113.

## 8. Named-entity recognition

```bash
nlp-histo ner extract            # add --entity-cache <path> only if you have a cache
nlp-histo ner merge
nlp-histo ner export
```

**If you received this project rather than built it, you do not have the entity cache** —
it is not in the bundle (it is 30 MB of the author's local UMLS-linking results, not a
thesis artifact). Omit `--entity-cache`; NER then starts cold and recomputes the links,
which is slow but correct. The rest of this section is for the maintainer's existing cache.
Note also that a restored corpus already holds its 1.79M entities, so `extract` will report
the documents as processed and skip them unless you pass `-- --force`.

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

`merge` and `export` filter on `entities.model_name`, which holds spaCy's
`nlp.meta["name"]` — **`core_sci_lg`**, not the package name `en_core_sci_lg` (the `en` is
`meta["lang"]`). The default is correct; pass `--model` only to select a different stored
model. A `--model` that matches nothing now **fails (exit 1)** naming what is available,
rather than reporting an empty result as success — that silent no-op was B-115, and it
meant these two commands had never emitted a file.

*Verified end-to-end 2026-07-16* — all three commands, documented form, on the ingested
document above, in the same isolated database:

```
nlp-histo ner extract --entity-cache <copy>   → exit 0 · 85 s · 1 processed, 0 errors
    DB: 865 entities · 749 with UMLS CUIs · across all 14 text elements
    cache: 280 790 → 280 790 entries (+0 new) — fully warm, no recomputation
nlp-histo ner merge                            → exit 0 · 749 occurrences · 762 files
nlp-histo ner export                           → exit 0 · 89 disease CUIs · 178 files
```

Spot-checked as meaningful rather than merely present: `C0002989 "Epithelioid hemangioma
of skin"` for a paper on cutaneous epithelioid angiomatous nodule. The run used a **copy**
of the entity cache; the original was verified unchanged (identical SHA-256).

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

Both are clean as of 2026-07-16: `ruff check .` passes, and `pytest` is **1693 passed,
0 failed** (~3 min). Two `tests/test_config_loader.py` failures found during that day's
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

*Verified 2026-07-16, all four, under a guard that raises on any billable host:* E04,
`eval/sweeps/grounding.py`, `map_theta_sweep --help`, and
`E14_heldout.heldout_eval --theta-frontier` (exit 0, 248 s, zero paid blocks, zero cache
misses; the only host contacted was `s3-us-west-2.amazonaws.com` for the free scispaCy
KB). E14's frontier CSV is **byte-identical** to the 2026-06-25 baseline — 7 rows, 0
differing cells — is monotonic in θ (0.40→0.5453 … 0.90→0.7128), and its θ0.90 row equals
the primary heldout15 number, matching `eval/reports/RESULTS.md`'s pinned *"heldout15
strict-F1 0.7128 vs calibration 0.7160, gap −0.0032"*. E14 now runs strict cache-only, so
that freedom from paid calls is structural rather than a property of the current cache.

**No API key of any kind is needed** — verified 2026-07-16 by running all three with every
provider credential unset. They previously exited with `GOOGLE_API_KEY not set` before
reaching the cache, demanding a key they never used (B-109, fixed).

One caveat before you run them:

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

---

## Appendix — verification status

Provenance for the maintainer. A first-time reader does not need this to run anything —
it records what was executed against this tree and what was only inspected, so no claim
above rests on assumption.

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
| 2 | wheel build + install, `nlp-histo --help`, `import nlp_histo` | **verified from a fresh venv outside the repository** — wheel-only, no source tree / editable / `PYTHONPATH`; all 11 entry points and both packaged resources load |
| 3 | every command/subcommand `--help` (15) | verified, all exit 0 |
| 4 | the eight `NLP_HISTO_*` env vars | verified present in `src/` |
| 5 | `db check`, `db init` | verified against a live PostgreSQL — including `db init` against an **empty** database (21 tables created from the ORM, exit 0, ~1 s) |
| 6 | `acquire download` / `unpack` / `organize` | **verified end-to-end** on one PMCID, **both sources** — aws (default, durable) and ftp (legacy, expires Aug 2026). Identical PDF bytes from each (B-118) |
| 7 | `ingest` | **verified end-to-end** on one 8-page PDF into an isolated database — exit 0, 33 s, 14 text elements + 10 figures, matching the established corpus exactly |
| 8 | `ner extract` / `merge` / `export` | **verified end-to-end** on that document — 865 entities (749 with UMLS CUIs) → 762 merged files → 89 disease CUIs. `merge`/`export` produced nothing before the B-115 fix |
| 9 | both `knowledge` commands, `--dry-run` | verified (exit 0, no provider contacted) |
| 10 | `replay chapter9` | verified — 9/9 CSVs byte-identical; exit 3 when UMLS is unreachable |
| 11 | `pytest`, `ruff check .` | verified — 1693 passed, 0 failed; ruff clean |
| 12 | E04, `sweeps/grounding.py`, E14 (incl. `--theta-frontier`), `map_theta_sweep --help` | verified free (0 cache misses, no paid host). `E14 --theta-frontier` reproduces **byte-identically** to the 2026-06-25 baseline |

**Not executed — do not read these as verified:**

* **§2's dependency resolution** — `pip install -r requirements.txt` was not run into the
  fresh venv (multi-GB torch/docling download; this machine is disk-constrained). The
  wheel build, install, imports, entry points and packaged resources **were** verified
  from a throwaway venv outside the repository.
* **§6's bulk path** — one PMCID per source was fetched, never the full 1093.
* **§8's entity-cache migration advice** — the paths and resolution order were checked
  against `ner/cache_paths.py`, and `ner extract` was run against a *copy* of the cache;
  the documented `export NLP_HISTO_ENTITY_CACHE=…` line itself was not exercised.
* **§13** — the B-102 limitation, carried forward from an earlier audit.

Where a command needs something this machine does not have, that is stated at the command
itself. **`--dry-run` (§9) resolves a paid invocation's full configuration for free** — a
cost constraint is a reason to find the free verification, not to skip it.

---
