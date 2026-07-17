# Reproducing this thesis — a start-to-finish runbook

Read this top to bottom and run the commands as you meet them. Every step says what it
does, what you should see, and what to do if you see something else.

There are **two tracks**, and you almost certainly want the first:

* **Track A — reproduce from the files provided (free).** You download the frozen artifacts
  and the corpus database from a link, and every command runs against those caches. No API
  key, no cost, and the thesis tables come out byte-identical. **Start here.**
* **Track B — rebuild the artifacts yourself.** You re-acquire the PDFs from PubMed Central,
  re-ingest them into a fresh database, and re-run the pipeline. This regenerates what
  Track A hands you. Its last step calls **paid** LLM APIs, and because those models are
  non-deterministic the results will *not* be byte-identical. Do this only if you want to
  see where the artifacts come from.

**Both tracks share the same setup (Steps 1–3).** After that, do Track A *or* Track B — not
both against the same database.

**Nothing costs money except the final step of Track B, which is clearly marked.**

**This file is self-contained** — every command and value you need to complete either track
is here; you do not need to open another document to finish. `docs/HOW_TO_RUN.md` is the
deeper reference (every flag, every option, the per-command verification history) for when
you want more than the path, and `docs/BUGS.md` records known defects with their reasoning.
If this file and `HOW_TO_RUN.md` ever disagree on a fact, `HOW_TO_RUN.md` is authoritative
and this file has a bug worth reporting.

## What you need on your machine

* **Python 3.10, 3.11 or 3.12.** Not 3.13 — the dependency set (torch, scispaCy) does not
  build there. Check with `python3 --version`.
* **PostgreSQL**, running, and the ability to create a database. Check with `psql --version`.
* **About 10 GB of free disk**: ~4 GB of Python dependencies, 1.2 GB archive, 1.5 GB
  extracted, ~0.5 GB database, plus ~3 GB of models downloaded on first use.
* **An internet connection.** Steps 3 and 7 download packages and models; neither costs
  money.

You do **not** need an API key for Track A. You do **not** need the PDFs for either track.

---

# Part 1 — Set up the code (both tracks)

## Step 1 — Get the code

```bash
git clone -b refactor/python-packaging \
  https://gitlab.lrz.de/00000000014B8E24/nlp-histo.git nlp-histo
cd nlp-histo
```

The `-b` is deliberate: the work lives on the `refactor/python-packaging` branch. Clone
without it and you may land on a branch that predates the current layout — no
`src/nlp_histo/`, and none of the commands below will exist.

Check you got the right thing:

```bash
ls src/nlp_histo/cli/main.py docs/REPRODUCE.md    # both must exist
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

---

# Track A — Reproduce from the files provided (free)

This is the recommended path. You download the artifacts and the corpus once, then every
command runs against them for free.

**Start the download now** — the artifact archive is 1.2 GB and you will not need it until
Step 5, so kick it off before you read on. There are **four files** to fetch (two archives
and their `.sha256` sidecars):

| File | Size | What it is |
|---|---|---|
| `nlp-histo-replay-artifacts-ec11eec.tar.gz` | 1.2 GB | Frozen outputs of the paid LLM pipeline + the embedding caches |
| `nlp-histo-replay-artifacts-ec11eec.tar.gz.sha256` | tiny | Checksum for the archive |
| `nlp-histo-corpus.sql.gz` | 49 MB | The PostgreSQL corpus: 977 papers, 35,896 text elements, 1.79M entities |
| `nlp-histo-corpus.sql.gz.sha256` | tiny | Checksum for the corpus dump |

Both locations hold the identical set — use whichever is faster:

* **Primary — LRZ Sync+Share:** <https://syncandshare.lrz.de/getlink/fiBHdDWVJLKMxYP5JicFvd/nlp-histo-bundles>
* **Mirror — Google Drive:** <https://drive.google.com/drive/folders/1uo-iGOb3df11LqjQRbCwoyRsiVDMJkpY?usp=sharing>

Put all four files in one directory. This guide calls it `~/nlp-histo-bundles`; anywhere
works, and the rest of Track A assumes that path.

**From a browser**, open either link and download all four files into `~/nlp-histo-bundles`.

**From the command line**, the LRZ link fetches the whole folder — all four files — as one
zip:

```bash
mkdir -p ~/nlp-histo-bundles && cd ~/nlp-histo-bundles
curl -L -o bundle.zip \
  "https://syncandshare.lrz.de/dl/fiBHdDWVJLKMxYP5JicFvd/nlp-histo-bundles.dir"
unzip bundle.zip        # yields the four files above
```

The Google Drive mirror is per-file, and a 1.2 GB file triggers Drive's virus-scan
interstitial — a plain `curl` silently saves that HTML warning page under the archive's
name. If you use Drive from the command line, pass the confirm token:

```bash
curl -L -o nlp-histo-replay-artifacts-ec11eec.tar.gz \
  "https://drive.usercontent.google.com/download?id=1SOWdAhhy0ZFqwlMjG3X3EFvgNQyKhIxW&export=download&confirm=t"
```

Either way, Step 5 verifies what you got before anything trusts it.

**No account is needed.** Both are read-only links that serve anonymously — you cannot write
through either. Anyone holding a URL can download the bundle; it is unlisted rather than
access-controlled. It contains only derived artifacts (embeddings, summaries, cascade
decisions) and **no publisher PDFs**, which is why sharing a link is acceptable.

**Why these are not in the git repository:** the artifacts cost real money to generate and
are too large for git; the corpus PDFs they derive from are mostly not redistributable
(322 of the 1093 papers carry no Creative Commons licence). The repository holds the code
and the corpus *definition* — everything else you either receive here or regenerate in
Track B.

## Step 4 — Confirm the code is sound while the download runs

The test suite needs none of the artifacts — it checks the code alone, so it is the fastest
way to confirm Step 3 produced a working environment. Run it now, while the download
continues in the background.

```bash
python -m pytest -q
ruff check .
```

Expect **1697 passed** and `All checks passed!`. Takes three to four minutes. If this
fails, fix the install before going further — every step below assumes it passes.

## Step 5 — Verify the two files you received

Do this **before** anything else touches them. A truncated download produces failures much
later and much more confusingly.

```bash
cd ~/nlp-histo-bundles
shasum -a 256 -c nlp-histo-replay-artifacts-ec11eec.tar.gz.sha256
shasum -a 256 -c nlp-histo-corpus.sql.gz.sha256
```

Both must print `OK`.

```bash
file nlp-histo-replay-artifacts-ec11eec.tar.gz
```

Must say **gzip compressed data**. If it says *HTML document*, your download was
intercepted — Google Drive shows a virus-scan warning page for files this large and some
tools save that page instead of the file. Re-download through a browser, or use the
`confirm=t` command in the download block above (Track A introduction).

> The expected checksums, for reference:
> `cded4299ac1a47cf8b857f5796731a27d3b66eac234a1acf306771e73d3e45d3` (archive)
> `af80657dc8b512668b2fc1a120610290816797db18d1eb384140f2688b3a4f59` (corpus dump)

## Step 6 — Unpack the artifacts into the repository

Step 5 left you in `~/nlp-histo-bundles`. Go back to the `nlp-histo` clone from Step 1
(wherever you created it) before extracting:

```bash
cd /path/to/nlp-histo        # the clone from Step 1
tar -xzf ~/nlp-histo-bundles/nlp-histo-replay-artifacts-ec11eec.tar.gz -C .
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

## Step 7 — Reproduce the thesis tables

This is the headline result, and it needs only the artifacts you just unpacked — **no
database, no API key.**

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
| 2 | The artifact tree is incomplete — Step 6 did not land. It lists exactly what is missing. |
| 3 | The UMLS model could not be downloaded. It needs the network even when cached; see the note below. |
| 4 | An embedding cache is incomplete — the archive did not extract fully. Re-do Steps 5 and 6. |

Nothing here fails silently: a non-zero exit always names the cause.

> **On this step needing the network.** It makes no paid calls and needs no API key. It
> does need to reach `s3-us-west-2.amazonaws.com` and `huggingface.co`, even on a second
> run, because scispaCy resolves its cache through a live lookup. If you are offline, the
> command stops with exit 3 rather than quietly producing different numbers.

**Comparing against the published tables.** The repository ships the frozen thesis output at
`out/thesis_results/chapter9_offline_replay/` (the CSVs are tracked in git — you already have
them from Step 1). Your nine CSVs should be byte-identical to these. Diff all nine at once:

```bash
for f in out/replay-check/*.csv; do
  diff "$f" "out/thesis_results/chapter9_offline_replay/$(basename "$f")" \
    && echo "OK  $(basename "$f")"
done
```

Every line should print `OK` with no `diff` output above it — that is the reproduction
confirmed. (The reference directory also holds `04_theta_heatmap.csv` and
`10_cascade_vs_sonnet_gap_ci_per_case.csv`, which your run does not produce — they are the
two analyses noted above whose inputs are not in the shipped tree. Your nine are the ones
that matter.)

## Step 8 (optional) — The evaluation experiments

Also artifact-based and free — every embedding is served from the caches you unpacked, and
no provider is ever contacted.

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

Everything above reproduces the published results. The rest of Track A gives you the corpus
database to query and, optionally, its named entities.

## Step 9 — Restore the corpus database

The steps so far never touched PostgreSQL. This one loads the 977-paper corpus so you can
query it.

```bash
createdb nlp_histo
```

Use whatever database name you like; this runbook uses `nlp_histo`. **If a database of that
name already exists on your server, choose another** — the restore expects an empty target
and this avoids writing into something you care about. If your PostgreSQL requires a
specific role to create databases, name it: `createdb -U postgres -O <your-role> nlp_histo`.
The dump embeds no role names, so it restores under whoever connects.

Then restore:

```bash
gunzip -c ~/nlp-histo-bundles/nlp-histo-corpus.sql.gz | psql -d nlp_histo
```

Plain SQL, so any `psql` version reads it. It takes about 15 seconds and prints a long
stream of `SET`, `CREATE TABLE` and `COPY` lines. Into an empty database it produces no
`ERROR` and no `NOTICE` output; an `ERROR` means something is wrong — most likely the
database was not empty.

**Check it worked:**

```bash
psql -d nlp_histo -c "SELECT count(*) FROM documents;"
psql -d nlp_histo -c "SELECT count(*) FROM text_elements;"
```

Expect **977** documents and **35,896** text elements. If you get `0`, the restore did not
take — re-read the output of the previous command for `ERROR` lines.

> *Verified 2026-07-17:* this exact command was run into a genuinely empty database owned
> by the application role. It took **15 seconds**, exited 0, and printed **no `ERROR` and no
> `NOTICE` lines at all**. Every table matched the source: documents 977, text_elements
> 35,896, entities 1,792,440, figures 4,479, tables 1,960. `nlp-histo db check` then
> reported `OK: schema is present and valid (21 tables)`. The restored database is ~445 MB.

## Step 10 — Point the project at your database

```bash
cp .env.example .env
```

Open `.env` and set the five `DB_` values to match your PostgreSQL — at minimum
`DB_NAME=nlp_histo` (or whatever you named it in Step 9), plus the host, port, user and
password you use.

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

## Step 11 — Confirm the connection, and query the corpus

```bash
nlp-histo db check
```

You should see:

```
Target: <user>@<host>:5432/nlp_histo
Note: 1 unrelated table(s) present and left untouched: alembic_version
OK: schema is present and valid (21 tables).
```

The `Note:` line is expected, not a warning. The dump carries `alembic_version` — a
migration bookkeeping table that is not one of the ORM's 21 — so `db check` reports it and
leaves it alone. 21 tables is the correct count.

**Read the `Target:` line.** It is the database the project actually resolved — not the one
you meant. If it names something unexpected, go back to Step 10. This is also the first
command that touches PostgreSQL, so it is where a wrong password or a stopped server shows
up.

**The corpus is now yours to query.** It is an ordinary PostgreSQL database — `psql -d
nlp_histo` and any SQL you like. The ones you probably want are `documents` (one row per
paper), `text_elements` (its hierarchical text), and `entities` (the UMLS concepts found in
that text). The full set of 21 tables:

* **Document extraction (7):** `documents`, `text_elements`, `figures`, `tables`,
  `entities`, `text_element_figure_references`, `text_element_table_references`.
* **Run tracking (2):** `pipeline_runs`, `llm_judge_cache`.
* **Knowledge extraction (12), all prefixed `sum_`:** `sum_map_findings`,
  `sum_map_voter_outputs`, `sum_normal_findings`, `sum_normal_finding_spans`,
  `sum_finding_groups`, `sum_group_members`, `sum_canonical_rules`, `sum_relations`,
  `sum_final_rules`, `sum_rejection_summaries`, `sum_rejected_findings`,
  `sum_corpus_relations`.

(A Track A restore ships these already populated — the corpus was built with knowledge
extraction run, so `sum_final_rules` holds 1,729 rules, `sum_map_findings` 1,911, and so
on. Track B regenerates them from scratch.)

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

## Step 12 (optional) — Named-entity recognition over the corpus

```bash
nlp-histo ner extract
nlp-histo ner merge
nlp-histo ner export
```

`ner extract` reads the papers from the database and writes entities back to it. `merge`
and `export` then write JSON/TXT files grouped by UMLS concept, into `umls_entities_lg/`
and `disease_entities_lg/` in your current directory.

The restored corpus already contains 1.79M entities, so `extract` will report that
documents are already processed and skip them. To force a recompute, add `-- --force` — but
be aware that a cold run recomputes every UMLS link and is slow. The author's 30 MB entity
cache is not part of what you received; without it the first run has nothing to reuse.

**What you should see:** `Summary: N Processed | 0 Errors`, then `✓ Saved N files`.

If `merge` or `export` reports *"No entities are stored under model …"*, they are being
asked for a model the corpus does not contain — the message lists what is available.

**Track A is complete.** You have reproduced the thesis tables and have a working corpus
database. You can stop here.

---

# Track B — Rebuild the artifacts yourself

Everything in Track A ran against files you downloaded. Track B regenerates those files from
the corpus *definition* in the repository: it re-acquires the PDFs, rebuilds the database by
ingesting them, and re-runs the pipeline.

**Read this before starting:**

* **It does not need Track A.** Do the common setup (Steps 1–3), then come here. If you
  already did Track A, use a *different, empty* database below so you do not overwrite the
  restored corpus.
* **The results will not be byte-identical.** Track A reproduces the *published* numbers
  because it replays frozen outputs. Track B's final step calls LLMs, which are
  non-deterministic; a fresh run produces its own outputs, close but not equal.
* **The final step costs money.** Everything up to it is free.
* **Honesty about verification:** the single-paper acquisition and ingest below have been
  exercised; a full 1093-paper rebuild and the paid extraction have **not** been run
  end-to-end by the author. Start small.

## Step 13 — Create a fresh, empty database

Unlike Track A, you are not restoring a dump — you build the schema from the ORM and fill it
yourself.

```bash
createdb nlp_histo_rebuild
# if your role cannot create databases:
#   createdb -U postgres -O <your-role> nlp_histo_rebuild
```

Point `.env` at it (as in Step 10 — set `DB_NAME=nlp_histo_rebuild`, and check
`env | grep '^DB_'` prints nothing), then create the schema:

```bash
nlp-histo db init
```

This builds all 21 tables from `database/models.py` and verifies them. Expect `Database
tables created successfully!` then `OK: schema verified (21 tables)`. It is safe to re-run
and never drops anything. (Note: `db init` creates an **empty** schema — do not run it
before a Track A restore, which brings its own schema and data.)

## Step 14 — Acquire the PDFs from PubMed Central

`files/target_pmc_ids.txt` in the repository lists all 1093 papers — the corpus definition.
Start with three, because the full run has never been exercised in one go:

```bash
head -3 files/target_pmc_ids.txt > /tmp/three-papers.txt

nlp-histo acquire download --pmcid-file /tmp/three-papers.txt --output-dir files/corpus
nlp-histo acquire organize --input-dir files/corpus \
    --pdf-dir files/organized_pdfs --xml-dir files/organized_xmls
```

Papers are fetched from NLM's Open-Access dataset on AWS (~3 s/paper). If a download fails,
the command exits non-zero and says which paper and why; papers simply absent from the
Open-Access subset are reported and skipped, which is normal and not an error.

## Step 15 — Ingest the PDFs into the database

```bash
nlp-histo ingest --pdf-dir files/organized_pdfs --out-root out
```

This is the PDF → text + figures + tables pipeline (~30 s/paper). It populates `documents`,
`text_elements`, `figures` and `tables` — the same tables Track A's dump restored, now
built from the PDFs you just fetched. Expect `ok=N fail=0`.

Confirm:

```bash
psql -d nlp_histo_rebuild -c "SELECT count(*) FROM documents;"   # = the papers you ingested
```

## Step 16 — Rebuild the entities

```bash
nlp-histo ner extract
nlp-histo ner merge
nlp-histo ner export
```

Same commands as Track A Step 12, but here they run against your freshly ingested papers
with no prior entities, so `extract` does the full scispaCy + UMLS work rather than
skipping. This is the slow, cold path — every UMLS link is computed from scratch, because
the author's entity cache is not shipped.

## Step 17 — ⚠ Knowledge extraction — **this costs money**

Every other command in either track is free. This one calls paid LLM APIs, and cost scales
with the number of papers. It regenerates the summaries and `sum_*` tables — the LLM outputs
that Track A's archive contained.

Check the invocation for free first — `--dry-run` resolves the whole configuration, prints
the model cascade and the required API keys, and exits without contacting a provider:

```bash
nlp-histo knowledge PMC1448691 --profile cheap --sync --health-check no --dry-run
```

Remove `--dry-run` to actually run it. Three arguments are required and have no defaults —
`--profile`, `--health-check`, and one of `--sync`/`--batch` — precisely so that a paid run
cannot start by accident. Use `--profile cheap` and a single paper before anything larger.

Because the voters are non-deterministic, the rules this produces will resemble but not
equal the frozen ones in Track A's archive. That is expected; the frozen archive exists
precisely so the *published* numbers can be reproduced exactly (Track A), which a fresh paid
run cannot guarantee.

---

## If something goes wrong

**`command not found: nlp-histo`** — the virtualenv is not active. `source .venv/bin/activate`.

**`ModuleNotFoundError: No module named 'sqlalchemy'`** (or similar) — Step 3's first
command did not complete. The project installs `--no-deps` on purpose, so it will import
and print help without its dependencies and only fail when it tries to do real work.

**The wrong database** — run `nlp-histo db check` and read the `Target:` line. If it is not
what you configured, `env | grep '^DB_'` will show why (Step 10, or Step 13 for Track B).

**`replay chapter9` exits 2** — the artifacts are not in place. The error lists every
missing file. Re-do Step 6, extracting into the repository root.

**`replay chapter9` exits 3** — no network, or S3 is unreachable. This command needs the
network even though it is free and cached.

**`replay chapter9` exits 4** — an embedding cache is incomplete. Your download or
extraction was truncated: re-verify with Step 5.

**Out of memory** — you are running two heavy commands at once. `ingest`, `ner`, `replay`
and `pytest` each load scispaCy plus the UMLS knowledge base. Run them one at a time.

**Anything else** — `docs/HOW_TO_RUN.md` documents every command in detail, and
`docs/BUGS.md` records the known defects with their reasoning. If a command in *this* file
does not behave as described, that is a bug in this file: please report the exact command
and its output.
