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
│                       NAMED ENTITY RECOGNITION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  batch_ner.py → Extract medical entities (scispacy + UMLS linker)           │
│         │                                                                   │
│         ▼                                                                   │
│  merge_entities_by_umls.py → Group sentences by UMLS concept (CUI)          │
│         │                                                                   │
│         ▼                                                                   │
│  export_disease_entities.py → Filter to disease-related entities only       │
│         │                                                                   │
│         ▼                                                                   │
│  JSON files per concept with full database provenance:                      │
│  {cui, canonical_name, sentences: [{pmcid, text_element_id, sentence}]}     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           LLM KNOWLEDGE-EXTRACTION PIPELINE (knowledge extraction stage)            │
├─────────────────────────────────────────────────────────────────────────────┤
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
│   └── pdf_parsers/                    #   research-only alternative parsers (NOT the production path)
│
├── database/                           # SQLAlchemy ORM (models.py) + connection mgmt; schema via Alembic
├── alembic/                            # Schema migrations (head: 0014)
│
├── named_entity_recognition/           # scispaCy + UMLS entity extraction (ner.py, batch_ner.py, …)
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
├── legacy/langchain-summarization/     # LEGACY summarisation stack (superseded by pipeline/…/summarization)
├── files/                              # Input PDFs/XMLs (not in repo)
├── out/                                # Runtime outputs (cached layouts, summaries, run metadata)
├── requirements.txt
└── .env.example                        # DB + API-key template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# requirements.txt covers the production pipelines (PDF extraction +
# the multi-provider knowledge_extraction cascade, which uses direct provider APIs).
# The langchain/* packages are only needed for the legacy
# legacy/langchain-summarization/ prototype, which is no longer the production path.
```

### 2. Set Up Database

```bash
# Create PostgreSQL database
createdb -U postgres nlp_histo

# Configure credentials
cp .env.example .env
# Edit .env with DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Initialize schema (Alembic-managed)
alembic upgrade head
```

### 3. Run Data Pipeline

```bash
# Download papers (requires target_pmc_ids.txt)
cd file-selector && python file_downloader.py

# Extract and organize
python tarball_extractor.py
python pdf_organizer.py

# Process and ingest to database (production pipeline; see HOW_TO_RUN.md §2 for flags)
cd .. && python pipeline/stages/pdf_text_extraction/runner.py
```

### 4. Run NER Pipeline

```bash
cd named_entity_recognition

# Extract medical entities with UMLS linking
python batch_ner.py

# Group by UMLS concept
python merge_entities_by_umls.py

# Export disease entities only
python export_disease_entities.py

# (Optional) Estimate LLM costs
python count_tokens.py
```

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

The schema above is the document-extraction core (7 tables). The full schema is
**21 Alembic-managed tables** (`alembic upgrade head`): the 7 above, plus
`pipeline_runs` and `llm_judge_cache`, plus 12 knowledge-extraction-persistence tables
(`sum_map_findings`, `sum_map_voter_outputs`, `sum_normal_findings`,
`sum_normal_finding_spans`, `sum_finding_groups`, `sum_group_members`,
`sum_canonical_rules`, `sum_relations`, `sum_final_rules`,
`sum_rejection_summaries`, `sum_rejected_findings`, `sum_corpus_relations`)
written via `pipeline/stages/knowledge_extraction/persistence.py`. See
`database/models.py` for the authoritative definitions.

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
- **Token Counting**: Estimates LLM costs before running knowledge_extraction

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
