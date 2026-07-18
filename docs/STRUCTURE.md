# Repository structure

The boundary that matters: **what ships in the wheel, and what does not.**

```
src/nlp_histo/     installed library + public CLI   → ships
eval/              thesis experiments + frozen artifacts   → repository-only
scripts/           developer utilities + 2 compatibility wrappers   → repository-only
tests/             test suite
docs/              tracked documentation (this file, REPRODUCE.md, HOW_TO_RUN.md)
configs/           user-editable run configuration (run.yaml)
```

Nothing under `src/nlp_histo/` reaches back into the repository tree. Everything outside
it may import *from* it. That direction is enforced by tests
(`tests/evaluation/test_library_boundary.py`, `tests/test_runtime_paths.py`).

## The installed package

```
src/nlp_histo/
├── cli/            the `nlp-histo` console command (argparse; handlers import lazily)
├── workflows/      knowledge.py (LLM extraction), replay.py (chapter-9 offline replay)
├── acquisition/    downloader.py, tarballs.py, organizer.py, selector.py
├── pipeline/
│   ├── stages/pdf_text_extraction/    PDF → hierarchical text + figures + tables → DB
│   ├── stages/knowledge_extraction/   MAP → GROUNDING → NORMALIZE → GROUP →
│   │                                  CANONICALIZE → RELATE → RESOLVE
│   └── utils/                         memory logging
├── ner/            scispaCy + UMLS entity extraction, cache_paths.py
├── database/       SQLAlchemy ORM (21 tables), connection, init_db
├── parsers/        layout_utils.py, text_processing.py — shared parsing hub
├── evaluation/     REUSABLE evaluation library only: schemas, jsonl_utils, split,
│                   matching/{embedders,matcher}
└── resources/      immutable packaged defaults: model_prices.json, nli_models.yaml
```

**The wheel's entire non-Python surface is three files:** `synonyms.yaml`,
`model_prices.json`, `nli_models.yaml`. No PDFs, no caches, no experiment artifacts, no
`.env`.

## Repository-only

* **`eval/`** — the thesis experiments: E01–E14 drivers, sweeps, calibration, silver
  generation, the LLM judge, and the frozen artifacts under `eval/data/`,
  `eval/reports/`, `eval/results/`. They hardcode experiment identifiers and read
  repository artifacts, so packaging them would force the thesis datasets into the
  wheel. They import `nlp_histo.evaluation` and run with `python -m eval.…` **from the
  repository root**.
* **`scripts/`** — developer utilities. **`scripts/` is not a Python package** (no
  `__init__.py`); run them as file paths, `python scripts/foo.py`. `scripts/run_paper.py`
  and `scripts/thesis/run_chapter9_offline_replay.py` are thin wrappers around the
  installed CLI and carry a removal plan.

## Path semantics

| Kind | Where it lives | How it resolves |
|---|---|---|
| Immutable defaults (prices, NLI registry) | inside the package | `importlib.resources`; `NLP_HISTO_MODEL_PRICES` / `NLP_HISTO_NLI_MODELS` override |
| User config (`.env`, `configs/run.yaml`) | your working tree | `.env` searched upward from the cwd; `--config PATH` |
| Frozen experiment artifacts | `eval/data/`, `eval/reports/` | explicit path (`--artifact-root`, `--source-cases`) — never packaged |
| Writable caches (entity, embedding) | user cache dir | explicit arg → env var → `~/.cache/nlp-histo/` — never `site-packages` |
| Generated outputs (`out/…`) | wherever you run | **deliberately** relative to your working directory; `--out-root` overrides |

## Database

Schema lives in `src/nlp_histo/database/models.py` — 21 tables (7 document-extraction,
2 run-tracking/judge-cache, 12 `sum_*` knowledge-extraction).

**The ORM creates the schema, not Alembic.** `nlp-histo db init` calls
`create_tables()`. Alembic migrates a database that already exists; its revisions assume
the ORM-created schema and will not initialise an empty database.

## Providers

Knowledge extraction uses a three-tier voter cascade over **LangChain** chat models —
LangChain is a production dependency, not a legacy import. Providers in the roster:
Gemini (Flash-Lite / Flash), OpenAI (GPT-4o-mini / 4.1-nano / 4.1-mini) and Anthropic
Claude (Haiku at L2, Sonnet at L3). Prices for these live in the packaged
`model_prices.json`.

The test suite makes **no** paid calls: voters are faked and the NLI/embedding models are
local or cached.
