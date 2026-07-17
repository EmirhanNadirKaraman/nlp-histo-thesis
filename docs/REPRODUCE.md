# Reproducing this thesis — a start-to-finish runbook

Read this top to bottom and run the commands as you meet them. Every step says what it
does, what you should see, and what to do if you see something else. You should not need
to jump around; where a detail would interrupt the flow it is in a note you can skip.

**Nothing here costs money except Step 15, which is marked and optional.**

By the end you will have reproduced the nine tables that Chapter 9 reports, and you will
have a working corpus database you can query.

Companion document: `docs/HOW_TO_RUN.md` is the *reference* — every command, every flag,
every caveat. This file is the *path*. If the two ever disagree, HOW_TO_RUN.md is
authoritative and this file has a bug.

---

## Before you start — begin the download

**Start this now and read on while it transfers**; it is 1.2 GB and you will not need it
until Step 4. Download both files, from either location (the two are identical — use
whichever is faster):

* **Primary — LRZ Sync+Share:** <https://syncandshare.lrz.de/getlink/fiBHdDWVJLKMxYP5JicFvd/nlp-histo-bundles>
* **Mirror — Google Drive:** <https://drive.google.com/drive/folders/1uo-iGOb3df11LqjQRbCwoyRsiVDMJkpY?usp=sharing>

| File | Size | What it is |
|---|---|---|
| `nlp-histo-replay-artifacts-ec11eec.tar.gz` | 1.2 GB | Frozen outputs of the paid LLM pipeline + the embedding caches |
| `nlp-histo-corpus.sql.gz` | 49 MB | The PostgreSQL corpus: 977 papers, 35,896 text elements, 1.79M entities |

Each comes with a `.sha256` file — download those too, and keep them beside the file they
name. Step 4 uses them.

Put all four in one directory. This guide calls it `~/nlp-histo-bundles`; anywhere works.

**Why these are not in the git repository:** the artifacts cost real money to generate and
are too large for git; the corpus PDFs they derive from are mostly not redistributable
(322 of the 1093 papers carry no Creative Commons licence). The repository holds the code
and the corpus *definition* — everything else you either receive or regenerate.

## What you need on your machine

* **Python 3.10, 3.11 or 3.12.** Not 3.13 — the dependency set (torch, scispaCy) does not
  build there. Check with `python3 --version`.
* **PostgreSQL**, running, and the ability to create a database. Check with `psql --version`.
* **About 10 GB of free disk**: ~4 GB of Python dependencies, 1.2 GB archive, 1.5 GB
  extracted, ~0.5 GB database, plus ~3 GB of models downloaded on first use.
* **An internet connection.** Steps 3 and 10 download packages and models. Neither costs
  money.

You do **not** need an API key. You do **not** need the PDFs.

---

## Step 1 — Get the code

```bash
git clone <repository-url> nlp-histo
cd nlp-histo
```

Everything from here runs **from this directory**. If a command misbehaves, the first
thing to check is that you are still in it.

## Step 2 — Create an isolated Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Your prompt should now show `(.venv)`. If you open a new terminal later, run
`source .venv/bin/activate` again before any `nlp-histo` command.

## Step 3 — Install the dependencies

```bash
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install -e . --no-deps
```

This is the slowest step: roughly 4 GB, including torch and the scispaCy model
`en_core_sci_lg`. `--no-cache-dir` keeps pip from doubling that on disk.

The two commands are deliberately separate. `requirements.txt` is the pinned, tested
dependency set; the project itself is installed `--no-deps` so pip cannot quietly upgrade
anything out from under it.

**Check it worked:**

```bash
nlp-histo --help
```

You should see a list of commands (`db`, `acquire`, `ingest`, `ner`, `knowledge`,
`replay`). If you get `command not found`, your virtualenv is not active — go back to
Step 2.

## Step 4 — Verify the two files you received

Do this **before** anything else touches them. A truncated download produces failures much
later and much more confusingly.

```bash
shasum -a 256 -c nlp-histo-replay-artifacts-ec11eec.tar.gz.sha256
shasum -a 256 -c nlp-histo-corpus.sql.gz.sha256
```

Both must print `OK`.

```bash
file nlp-histo-replay-artifacts-ec11eec.tar.gz
```

Must say **gzip compressed data**. If it says *HTML document*, your download was
intercepted — Google Drive shows a virus-scan warning page for files this large and some
tools save that page instead of the file. Download it again through a browser, or see
`docs/HOW_TO_RUN.md` §0 for the command-line form.

> The expected checksums, for reference:
> `cded4299ac1a47cf8b857f5796731a27d3b66eac234a1acf306771e73d3e45d3` (archive)
> `af80657dc8b512668b2fc1a120610290816797db18d1eb384140f2688b3a4f59` (corpus dump)

## Step 5 — Unpack the artifacts into the repository

```bash
tar -xzf /path/to/nlp-histo-replay-artifacts-ec11eec.tar.gz -C .
```

The archive carries its own directory structure (`eval/data/…`, `out/summaries/…`), so it
must be extracted **into the repository root** — the paths only line up there.

**Check it worked:**

```bash
ls eval/data/embedding_cache_openai.sqlite eval/data/map_primer/voter_cache.json
ls out/summaries/summaries | head -3
```

All three must exist. The archive also contains `MANIFEST.json`, listing every file with
its size and checksum, if you ever want to verify them individually.

## Step 6 — Create an empty database

Use whatever database name you like; this runbook uses `nlp_histo`.

```bash
createdb nlp_histo
```

If your PostgreSQL requires a specific role, use it — for example
`createdb -U postgres -O <your-role> nlp_histo`. The dump does not embed any role names,
so it restores under whatever user you connect as.

## Step 7 — Restore the corpus

```bash
gunzip -c /path/to/nlp-histo-corpus.sql.gz | psql -d nlp_histo
```

This restores plain SQL, so it works with any `psql` version. It prints a long stream of
`SET`, `CREATE TABLE` and `COPY` lines. A few `NOTICE` messages are normal; `ERROR` lines
are not.

**Check it worked:**

```bash
psql -d nlp_histo -c "SELECT count(*) FROM documents;"
psql -d nlp_histo -c "SELECT count(*) FROM text_elements;"
```

Expect **977** documents and **35,896** text elements. If you get `0`, the restore did not
take — re-read the output of the previous command for `ERROR` lines.

> **Not yet verified.** This restore has not been executed end-to-end by the author: doing
> so needs an empty database, and the machine it was built on has none to spare. The dump
> itself *was* checked — valid gzip, 22 `CREATE TABLE` statements, the data `COPY` blocks,
> no embedded role names, no credentials. If the restore misbehaves, that is a bug in this
> document; please report it with the exact output.

## Step 8 — Point the project at your database

```bash
cp .env.example .env
```

Open `.env` and set the five `DB_` values to match your PostgreSQL — at minimum
`DB_NAME=nlp_histo`, plus the host, port, user and password you use.

**Now the one trap in this whole runbook:**

```bash
env | grep '^DB_'
```

**This must print nothing.** If your shell already exports `DB_NAME` or friends — from a
`.bashrc`, from `source .env`, from direnv — those values **beat the `.env` file**, and
every command below will quietly use whichever database your shell names instead of the
one you configured. That is deliberate behaviour (it lets you override a single value for
a single command), but it surprises everyone the first time.

If it printed something, either `unset` those variables or open a fresh terminal.

## Step 9 — Confirm the connection

```bash
nlp-histo db check
```

You should see:

```
Target: <user>@<host>:5432/nlp_histo
OK: schema is present and valid (21 tables).
```

**Read the `Target:` line.** It is the database the project actually resolved — not the one
you meant. If it names something unexpected, go back to Step 8.

This is also the first command that touches PostgreSQL, so it is where a wrong password or
a stopped server shows up.

**The corpus is now yours to query.** It is an ordinary PostgreSQL database — `psql -d
nlp_histo` and any SQL you like. The 21 tables are described in `docs/STRUCTURE.md`; the
ones you probably want are `documents` (one row per paper), `text_elements` (its
hierarchical text), and `entities` (the UMLS concepts found in that text).

The structure worth knowing about is that `text_elements` keeps each paragraph's **section
path** as an array, so you can ask for text by where it sits in the paper rather than by
string-matching headings:

```bash
psql -d nlp_histo -c \
  "SELECT count(*) FROM text_elements WHERE path_list @> ARRAY['Methods'];"
```

The same thing from Python, which is how the project itself reads the corpus:

```python
from nlp_histo.database import get_db_connection, Document, TextElement

db = get_db_connection()
with db.session_scope() as session:            # commits on exit, rolls back on error
    print(session.query(Document).count())     # 977

    methods = session.query(TextElement).filter(
        TextElement.path_list.contains(['Methods'])   # 'Methods' anywhere in the path
    )
    print(methods.count())                     # 86
    print(methods.first().unique_path)         # PMC10092619_HIS-82-254/Methods/0
```

`unique_path` is `{document_id}/{section hierarchy}/{position}`, so a row tells you exactly
where in which paper it came from. Note the document ID is composite
(`PMC10092619_HIS-82-254`), not a bare accession — the publisher's filename is part of it.

## Step 10 — Reproduce the thesis tables

This is the main result.

```bash
nlp-histo replay chapter9 --artifact-root . --output-dir out/replay-check
```

Takes about five minutes. On first run it downloads two models (~3 GB: the scispaCy UMLS
knowledge base and a biomedical NLI model). That download is free, and it is cached — later
runs skip it.

**What you should see:** it works through twelve analyses and ends with `done.`. Exit code
0.

```bash
ls out/replay-check/*.csv | wc -l
```

**Expect 9.**

Two of the twelve analyses do not produce a CSV, by design and for reasons that predate
this work: `04_theta_heatmap` reports `status=missing` (its inputs are not in the tree) and
`10_cascade_vs_sonnet_gap_ci` fails with a known pre-existing error. Nine CSVs is success.

**If the exit code is not 0**, it will be one of these, and the message says which:

| Exit | Meaning |
|---|---|
| 2 | The artifact tree is incomplete — Step 5 did not land. It lists exactly what is missing. |
| 3 | The UMLS model could not be downloaded. It needs the network even when cached; see the note below. |
| 4 | An embedding cache is incomplete — the archive did not extract fully. Re-do Steps 4 and 5. |

Nothing here fails silently: a non-zero exit always names the cause.

> **On Step 10 needing the network.** This command makes no paid calls and needs no API
> key. It does need to reach `s3-us-west-2.amazonaws.com` and `huggingface.co`, even on a
> second run, because scispaCy resolves its cache through a live lookup. If you are
> offline, the command stops with exit 3 rather than quietly producing different numbers.

**Comparing against the published tables:** the nine CSVs should be byte-identical to the
ones in `out/thesis_results/chapter9_offline_replay/` if you have them. For example:

```bash
diff out/replay-check/01_provenance_carry_rate.csv \
     out/thesis_results/chapter9_offline_replay/01_provenance_carry_rate.csv
```

No output means identical.

## Step 11 — Run the test suite

```bash
python -m pytest -q
ruff check .
```

Expect **1693 passed** and `All checks passed!`. Takes three to four minutes. This needs
none of the artifacts — it is a check on the code alone, and it is the fastest way to tell
whether your environment is sound.

---

At this point you have reproduced the thesis result. Everything below is optional.

---

## Step 12 (optional) — Named-entity recognition over the corpus

Requires Step 7 (the database).

```bash
nlp-histo ner extract
nlp-histo ner merge
nlp-histo ner export
```

`ner extract` reads the papers from the database and writes entities back to it. `merge`
and `export` then write JSON/TXT files grouped by UMLS concept, into `umls_entities_lg/`
and `disease_entities_lg/` in your current directory.

The restored corpus already contains 1.79M entities, so `extract` will report that
documents are already processed and skip them. To force it, add `-- --force` — but be aware
that a cold run recomputes every UMLS link and is slow. The author's 30 MB entity cache is
not part of what you received; without it the first run has nothing to reuse.

**What you should see:** `Summary: N Processed | 0 Errors`, then `✓ Saved N files`.

If `merge` or `export` reports *"No entities are stored under model …"*, they are being
asked for a model the corpus does not contain — the message lists what is available.

## Step 13 (optional) — The evaluation experiments

Requires Step 5 (the artifacts). Free — every embedding is served from the caches you
extracted, and no provider is ever contacted.

```bash
python -m eval.silver.experiments.E14_heldout.heldout_eval --theta-frontier
python -m eval.silver.experiments.E04_cardinalities.cardinalities
```

E14 is the headline generalisation result: expect `strict_f1_optimal = 0.7128` and a
generalisation gap of `-0.0032`.

Run these **one at a time** — each loads scispaCy plus the UMLS knowledge base, several GB
of memory apiece.

> `eval/sweeps/grounding.py` also runs, but it overwrites a tracked file
> (`eval/results/grounding_sweep.md`) with numbers from whatever is currently in
> `out/summaries`. Use `git checkout eval/results/` afterwards to restore it.

## Step 14 (optional) — Rebuild the corpus from PubMed Central

You do not need this to reproduce anything; it is here if you want to see where the corpus
comes from. `files/target_pmc_ids.txt` in the repository lists all 1093 papers.

```bash
head -3 files/target_pmc_ids.txt > /tmp/three-papers.txt

nlp-histo acquire download --pmcid-file /tmp/three-papers.txt --output-dir files/corpus
nlp-histo acquire organize --input-dir files/corpus \
    --pdf-dir files/organized_pdfs --xml-dir files/organized_xmls
nlp-histo ingest --pdf-dir files/organized_pdfs --out-root out
```

Start with three papers, not all 1093 — the full run has never been exercised in one go.
Downloading takes about three seconds per paper and ingesting about thirty.

Papers are fetched from NLM's Open-Access dataset on AWS. If a download fails, the command
exits non-zero and says which paper and why; papers simply absent from the Open-Access
subset are reported and skipped, which is normal and not an error.

## Step 15 (optional) — ⚠ Knowledge extraction — **this costs money**

Every other command in this runbook is free. This one calls paid LLM APIs, and cost scales
with the number of papers.

Check the invocation for free first — `--dry-run` resolves the whole configuration,
prints the model cascade, and exits without contacting a provider:

```bash
nlp-histo knowledge PMC1448691 --profile cheap --sync --health-check no --dry-run
```

Remove `--dry-run` to actually run it. Three arguments are required and have no defaults —
`--profile`, `--health-check`, and one of `--sync`/`--batch` — precisely so that a paid run
cannot start by accident. Use `--profile cheap` and a single paper before anything larger.

---

## If something goes wrong

**`command not found: nlp-histo`** — the virtualenv is not active. `source .venv/bin/activate`.

**`ModuleNotFoundError: No module named 'sqlalchemy'`** (or similar) — Step 3's first
command did not complete. The project installs `--no-deps` on purpose, so it will import
and print help without its dependencies and only fail when it tries to do real work.

**The wrong database** — run `nlp-histo db check` and read the `Target:` line. If it is not
what you configured, `env | grep '^DB_'` will show why (Step 8).

**`replay chapter9` exits 2** — the artifacts are not in place. The error lists every
missing file. Re-do Step 5, extracting into the repository root.

**`replay chapter9` exits 3** — no network, or S3 is unreachable. This command needs the
network even though it is free and cached.

**`replay chapter9` exits 4** — an embedding cache is incomplete. Your download or
extraction was truncated: re-verify with Step 4.

**Out of memory** — you are running two heavy commands at once. `ingest`, `ner`, `replay`
and `pytest` each load scispaCy plus the UMLS knowledge base. Run them one at a time.

**Anything else** — `docs/HOW_TO_RUN.md` documents every command in detail, and
`docs/BUGS.md` records the known defects with their reasoning. If a command in *this* file
does not behave as described, that is a bug in this file: please report the exact command
and its output.
