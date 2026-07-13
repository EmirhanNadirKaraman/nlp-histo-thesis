# NLP Histopathology: Auditable Knowledge Extraction from Medical Literature

A complete pipeline for extracting **structured, traceable medical knowledge** from histopathology papers. The system downloads papers from PubMed Central, parses PDFs into a hierarchical database, performs named entity recognition with UMLS linking, and uses LLM-based knowledge extraction to generate clinical rules—all with full provenance tracking back to source sentences.

## Key Innovation: Full Audit Trail

Every claim in the final output traces back through:
```
Clinical Rule → Summary Claim → Sentence ID → PMCID + text_element_id → Database Record
```

Citation format: `[S1|PMC123456|789]` where:
- `S1` = sentence index in processing chunk
- `PMC123456` = PubMed Central paper identifier
- `789` = database `text_elements.id`

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  target_pmc_ids.txt                                                         │
│         │                                                                   │
│         ▼ file_downloader.py                                                │
│  histopathology_papers/*.tar.gz                                             │
│         │                                                                   │
│         ▼ tarball_extractor.py                                              │
│  processed_corpus/PMC*/                                                     │
│         │                                                                   │
│         ▼ pdf_organizer.py                                                  │
│  organized_pdfs/*.pdf                                                       │
│         │                                                                   │
│         ▼ PipelineRunner (pipeline/…/pdf_text_extraction; Docling + masking) │
│  ┌──────┴──────────────────────────────────────────────────┐                │
│  │  PostgreSQL Database                                    │                │
│  │  ├── documents (PMCID, title, journal, year)            │                │
│  │  ├── text_elements (hierarchical paths, text)           │                │
│  │  ├── figures (cropped images, captions)                 │                │
│  │  └── tables (cropped images, captions)                  │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              LLM KNOWLEDGE-EXTRACTION PIPELINE (production)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reads text_elements from Postgres and calls run_ner_on_db internally       │
│  for UMLS entity enrichment where needed. It does NOT consume the JSON      │
│  produced by the historical standalone NER utilities (see below).           │
│                                                                             │
│  Production path = agreement-based cascading (ABC) over a multi-provider    │
│  voter pool (DeepSeek, Gemini, Mistral, Claude Haiku, Claude Sonnet 4.6),   │
│  via direct provider APIs (no LangChain on this path).                      │
│                                                                             │
│   MAP ─► GROUNDING ─► NORMALIZE ─► GROUP ─► CANONICALIZE ─► RELATE ─► RESOLVE │
│    │         │            │          │           │            │         │    │
│   ABC       NLI         UMLS +    (subject,   predicate     NLI pair-  final │
│  cascade  entailment   synonym    outcome,    selection +   wise rel.  rule  │
│  voting    filter      dedup      relation)   surface form  detection  score │
│                                                                             │
│  Every FinalRule traces back: CanonicalRule → NormalFinding → source         │
│  paragraph → source document (provenance recorded at generation time).      │
│  Results persist to the sum_* Postgres tables via knowledge_extraction/persistence. │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    Auditable Summaries + Clinical Rules
                    (JSON with full provenance to source)

  ── Historical side-branch (gen-1) — NOT part of the current flow ──
  Three standalone NER CLIs read/write the Postgres `entities` table and
  export per-concept files; nothing downstream consumes their JSON
  (retained for reference — see "Historical standalone NER utilities"):
    batch_ner.py              -> populate `entities` (scispaCy + UMLS linker)
    merge_entities_by_umls.py / export_disease_entities.py
                              -> alternative exporters (all concepts | disease-only)
  The exported disease-entity JSON may then be analyzed by the archived
  cost estimator legacy/langchain-summarization/count_tokens.py (token
  stats + retired LangChain map-reduce-rules cost model; not this pipeline).
```

## Project Structure

> Detailed maps live in `docs/readmes/other_readmes/STRUCTURE.md` (architecture) and
> `docs/readmes/other_readmes/REPOSITORY_GUIDE.md` (file-by-file). This tree is the orientation
> summary.

```
nlp-histo/
├── file-selector/                      # Stage 1: data acquisition (PMC download, extract, organize)
│
├── pipeline/                           # Stage 2: the production pipelines (modular, current)
│   ├── stages/pdf_text_extraction/     #   PDF → hierarchical text + figures/tables → Postgres
│   │   ├── runner.py                   #     PipelineRunner (8-step orchestrator)
│   │   ├── config.py                   #     PipelineConfig + sub-configs
│   │   ├── components/                 #     layout extractor, masker, two-pass ghost-text, croppers…
│   │   ├── table_detectors/            #     Docling / TATR / Hybrid
│   │   └── outputs/                    #     text/DB writers, stats + run-manifest writers
│   ├── stages/knowledge_extraction/    #   Text → auditable clinical rules (3-tier ABC LLM cascade)
│   │   ├── runner.py                   #     KnowledgeExtractionRunner (MAP→…→RESOLVE)
│   │   ├── stages/                     #     map / normalize / group / canonicalize / relate / resolve
│   │   ├── agreement/ routing/ batch/  #     voter scorers, MAP router, async batch dispatch
│   │   └── grounding/ costing/ observability/
│   └── utils/                          #   cross-pipeline utilities (memory logging)
│
├── parsers/                            # Shared parsing hub (layout_utils.py, text_processing.py)
│
├── database/                           # SQLAlchemy ORM (models.py) + connection mgmt; schema via Alembic
├── alembic/                            # Schema migrations (head: 0014)
│
├── named_entity_recognition/           # live: ner.py + enums.py (used by the KE pipeline) + retained gen-1 standalone CLIs (batch_ner, merge/export)
│
├── eval/                               # Evaluation harness (measures the two pipelines)
│   ├── llm_judge/  silver/             #   Opus silver labels + matching + MAP-cascade calibration
│   ├── silver/experiments/             #   numbered thesis experiments E01…E14 (non-contiguous) + corpus_stats/
│   ├── silver/relation_pairs/          #   E13 claim-pair generation (Message-Batches workflow)
│   ├── paper_selection/                #   calibration-set builder (greedy / ILP)
│   └── reports/                        #   per-experiment results + RESULTS.md
│
├── configs/                            # run.yaml, model_prices.json, nli_models.yaml, paper_selection/*.yaml
├── scripts/                            # runners (run_paper.py), inspectors, eval helpers (+ legacy ingests)
├── tests/                              # pytest suite (77 test files, ~1,080 test functions; summarisation-heavy)
├── docs/readmes/                       # project docs — HOW_TO_RUN.md + other_readmes/ (STRUCTURE, REPOSITORY_GUIDE, BUGS, …) + thesis_review/
│
├── legacy/langchain-summarization/     # LEGACY summarisation stack (superseded by pipeline/…/summarization); incl. count_tokens.py cost estimator
├── legacy/pdf_parsers/                 # LEGACY research/comparison PDF parsers (Docling/Nougat/Marker/PyMuPDF4LLM/pdffigures/Ensemble; NOT the production path)
├── files/                              # Input PDFs/XMLs (not in repo)
├── out/                                # Runtime outputs (cached layouts, summaries, run metadata)
├── requirements.txt
└── .env.example                        # DB + API-key template
```

## Quick Start

### 1. Install Dependencies and the Project

Supported Python versions: **3.10 – 3.12**. (3.10 is the floor — langchain and torch
require it, and the Pydantic schemas use PEP-604 unions. 3.13 is not supported —
scispaCy caps at `<3.13`.)

```bash
# 1. Dependencies — requirements.txt is the tested source of truth
python -m pip install -r requirements.txt
# requirements.txt covers the production pipelines (PDF extraction +
# the multi-provider knowledge_extraction cascade, which uses direct provider APIs).
# The langchain/* packages are only needed for the legacy
# legacy/langchain-summarization/ prototype, which is no longer the production path.

# 2. The project itself, in editable mode
python -m pip install -e . --no-deps
```

`--no-deps` is deliberate: `pyproject.toml` does **not** duplicate the dependency list,
so `requirements.txt` stays the single tested manifest. Installing the project makes
`pipeline`, `database`, `parsers` and `named_entity_recognition` importable as normal
packages, without the per-file `sys.path` bootstraps that direct-run scripts would
otherwise need.

`eval` is **intentionally not installed**. It is run from the repository root as
`python -m eval.…` (a top-level distribution package literally named `eval` would be a
generic-name collision).

> **Run commands from the repository root.** Editable installation fixes *imports*; it
> does not make the project location-independent. The pipeline's `PathConfig` and the
> evaluation harness read repository-root-relative data paths (`out/`, `files/`,
> `configs/`, `eval/data/`), so a different working directory will not find them.

### 2. Set Up Database

Prerequisite: a running PostgreSQL server (check with `pg_isready`). The database
itself must exist **before** the schema can be initialized — `database.init_db`
creates tables, not the database.

```bash
# 1. Create the (empty) PostgreSQL database, owned by the application role
createdb -U <admin-role> -O <db-user> <database-name>

# 2. Configure credentials
cp .env.example .env
# Set DB_USER=<db-user>, DB_NAME=<database-name>, and the remaining DB_* values:
#   DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# (DB_PASSWORD must be present; leave it empty for peer/trust authentication.)

# 3. Create and verify the schema
python -m database.init_db
```

**Ownership matters.** `<admin-role>` is any PostgreSQL role permitted to create
databases; `<db-user>` must be the same role you put in `DB_USER` in `.env`. The
`-O <db-user>` flag makes the application role the **owner** of the new database,
which is what lets `database.init_db` create the ORM tables. The application role
itself therefore does **not** need the `CREATEDB` privilege — only the
administrative role does. Substitute your own role and database names; do not
copy `<admin-role>` literally, and do not assume the role is called `postgres`.

*(Alternative: an administrator may instead grant the configured role sufficient
schema-creation rights on the new database. Assigning ownership with `-O` is the
recommended and simpler setup.)*

`python -m database.init_db` connects as `DB_USER`, creates the ORM-managed schema
(`database/models.py`), and then verifies it. On success it reports the number of
tables verified.

It is **safe to run again**: it only ever creates *missing* tables and never drops,
truncates, or alters existing objects. If an existing table has drifted from the
models (for example a missing column), the command **fails with a clear message
instead of attempting an automatic repair**, and creates nothing.

Verification modes:

```bash
python -m database.init_db --check-only   # verify only; creates nothing
python -m database.init_db --smoke        # initialize, then a minimal ORM round trip
```

`--smoke` inserts one `Document` + `TextElement`, reads them back through the ORM,
and rolls the transaction back — leaving **no rows** behind. The two flags cannot be
combined (`--check-only` is strictly read-only).

> **Alembic is not the fresh-database initializer in this repository.** Do not run
> `alembic upgrade head` merely to initialize a newly created empty database — it
> cannot build the schema from empty (see [Schema ownership](#schema-ownership)).
> `database.init_db` uses the current SQLAlchemy ORM schema.

### 3. Run Data Pipeline

```bash
# Download papers (requires target_pmc_ids.txt)
cd file-selector && python file_downloader.py

# Extract and organize
python tarball_extractor.py
python pdf_organizer.py

# Process and ingest to database (production pipeline; see HOW_TO_RUN.md §2 for flags)
cd .. && python -m pipeline.stages.pdf_text_extraction.runner
```

### 4. (Historical) standalone NER utilities — gen-1

These are **retained standalone utilities that predate the current integrated
knowledge-extraction flow**. They are **not a required step** for the production
pipeline (§5), which calls `run_ner_on_db` internally where NER enrichment is
needed and does not consume their JSON. They can populate/read the Postgres
`entities` table and export per-concept files, and may require scispaCy/UMLS
models, database access, and local setup; depending on the command they write
database rows or local files.

The two exporters are **alternative** consumers of the `entities` table (run
either — not necessarily both).

```bash
cd named_entity_recognition

# Populate the `entities` table (scispaCy + UMLS linking) — writes DB rows
python batch_ner.py

# Export per-UMLS-concept JSON/TXT (all concepts) — reads DB, writes files
python merge_entities_by_umls.py

# ...or the disease-filtered subset — reads DB, writes files
python export_disease_entities.py
```

> The exported disease-entity JSON can be analyzed by the archived cost
> estimator `legacy/langchain-summarization/count_tokens.py`, which reports
> token statistics and estimates costs for the retired LangChain
> MAP → REDUCE → RULES workflow — not the current production pipeline.

### 5. Run Knowledge Extraction Pipeline

The production summariser is `pipeline/stages/knowledge_extraction/`, driven via
`scripts/run_paper.py` (sync or async batch). It needs three direct-API keys in
`.env`: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`. See
[`docs/readmes/HOW_TO_RUN.md`](docs/readmes/HOW_TO_RUN.md) §3 for the full recipe.

`run_paper.py` requires a mode (`--sync` or `--batch`), a `--profile NAME`, and
`--health-check yes|no`. Valid profiles are `cheap`, `real`, `real_5`,
`haiku_only` (defined in `pipeline/stages/knowledge_extraction/batch/voter_configs.py`
via `get_profile`); see HOW_TO_RUN.md §3.

```bash
# Single paper, sync (live) mode — pmcid is positional
python scripts/run_paper.py PMC1234567 --sync --profile cheap --health-check no

# Single paper, async batch mode (cheaper)
python scripts/run_paper.py PMC1234567 --batch --profile real --health-check no

# A whole calibration set from a YAML selection
python scripts/run_paper.py --from-selection configs/paper_selection/related15.yaml --batch --profile real --health-check no
```

> The old notebook stack under `legacy/langchain-summarization/` is legacy and kept for
> reference only.

## Output Examples

### Database Text Element

```json
{
  "id": 789,
  "document_id": 42,
  "unique_path": "PMC123456/Methods > Immunohistochemistry/0",
  "path_list": ["Methods", "Immunohistochemistry"],
  "text_content": "CD30 staining was performed using clone Ber-H2..."
}
```

### Extracted Clinical Rule

```json
{
  "rule_id": "R1",
  "type": "Diagnostic",
  "condition": "IF CD30 positive AND ALK negative",
  "action": "THEN consider primary cutaneous ALCL",
  "confidence": "High",
  "evidence_chain": [
    {
      "sentence_id": "S3",
      "pmcid": "PMC123456",
      "text_element_id": 789,
      "verbatim": "CD30+ ALK- phenotype is characteristic of pcALCL"
    }
  ]
}
```

### Audit Trail

Every summary includes:
- `chunks_processed`: Number of text chunks analyzed
- `total_sentences_cited`: Sentences used as evidence
- `unique_pmcids`: Source papers referenced
- `unique_text_element_ids`: Database IDs for full traceability
- `evidence_conflicts`: Flagged contradictions with sources

## Technology Stack

| Component | Technology |
|-----------|------------|
| PDF Parsing | Docling, PyMuPDF, Marker, Nougat |
| Database | PostgreSQL + SQLAlchemy |
| NER | scispacy (en_core_sci_lg) + UMLS linker |
| LLM Pipeline | Multi-provider agreement-based cascade — DeepSeek, Gemini, Mistral, Claude Haiku, Claude Sonnet 4.6 (direct provider APIs) |
| NLI backbone | PubMedBERT (MNLI/MedNLI-tuned, default `pubmedbert_mednli`) for grounding + relation classification |
| Structured Output | Pydantic schemas |
| Evaluation | Opus silver-label judge + grounding/relation NLI; UMLS concept matching |

## Database Schema

```
documents
├── id, pmcid (unique), title, journal, publication_year
├── filename, file_path, text_source

text_elements
├── id, document_id (FK), unique_path (unique)
├── path_list (PostgreSQL array), path_string, depth
├── text_content, position_in_section
└── references (JSON: figure/table mentions)

entities
├── id, text_element_id (FK)
├── entity_text, entity_label (CHEMICAL, DISEASE, etc.)
├── umls_cui, umls_score, canonical_name
├── semantic_types (UMLS TUI codes)
└── start_char, end_char, model_name

figures / tables
├── id, document_id (FK), figure_id/table_id
├── caption_text, image_path, page_number
└── bounding_box coordinates

text_element_figure_references (junction)
text_element_table_references (junction)
```

The schema above is the document-extraction core (7 tables). The full schema also has
`pipeline_runs` and `llm_judge_cache`, plus the knowledge-extraction-persistence tables
(`sum_map_findings`, `sum_map_voter_outputs`, `sum_normal_findings`,
`sum_normal_finding_spans`, `sum_finding_groups`, `sum_group_members`,
`sum_canonical_rules`, `sum_relations`, `sum_final_rules`,
`sum_rejection_summaries`, `sum_rejected_findings`, `sum_corpus_relations`)
written via `pipeline/stages/knowledge_extraction/persistence.py`. See
`database/models.py` for the authoritative definitions.

### Schema ownership

The current runtime schema is defined by the SQLAlchemy models in `database/models.py`.
`create_tables()` calls `Base.metadata.create_all()` and is the mechanism that
**initializes a new database**; `python -m database.init_db` is the maintained command
that wraps it with configuration validation and schema verification.

Alembic currently manages **incremental historical changes**, not complete
empty-database initialization: the migration chain assumes the core ORM-created tables
already exist, and it does **not** currently reproduce the ORM schema exactly. Treat
`database/models.py` as authoritative.

- `alembic current` — inspect the recorded revision of an existing database.
- `alembic upgrade head` — intended **only** for an existing database that was
  initialized under the project's historical setup and is known to be behind the
  migration head. It is **not** the fresh-database initialization command.

## Key Features

### Medical-Grade PDF Parsing
- **Table/Figure Masking**: White-rectangle masking before text re-extraction
- **Hierarchical Structure**: Preserves section nesting (e.g., Methods > 2.1 Staining)
- **Paragraph Stitching**: Reconnects text split by tables/figures
- **Reference Detection**: Finds "see Figure 3" mentions and links to extracted images
- **Parallel Processing**: ThreadPoolExecutor for batch PDF processing
- **Blacklist System**: Tracks failed PDFs to skip on reruns

### Named Entity Recognition
- **UMLS Linking**: Maps extracted entities to UMLS concepts (CUI)
- **Persistent Cache**: on-disk entity-linking cache (`named_entity_recognition/entity_linking_cache.json`, ~30 MB) avoids redundant UMLS lookups
- **Semantic Types**: Filters entities by UMLS semantic types (diseases, chemicals, etc.)
- **Token Counting (historical)**: `legacy/langchain-summarization/count_tokens.py` estimates costs for the archived LangChain map-reduce-rules prototype, not the current pipeline

### Auditable LLM Pipeline
- **Structured Pydantic schemas** ensure consistent output format
- **Caching** avoids redundant LLM calls for identical inputs
- **Conflict detection** flags contradictory evidence from different sources
- **Hallucination detection** verifies claims against source sentences

## Configuration

### Environment Variables (.env)

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_histo
DB_USER=postgres
DB_PASSWORD=your_password

# Knowledge_extraction cascade — direct provider APIs (see .env.example)
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

## License

See LICENSE file for details.
