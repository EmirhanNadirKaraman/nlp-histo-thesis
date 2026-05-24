# NLP Histopathology: Auditable Knowledge Extraction from Medical Literature

A complete pipeline for extracting **structured, traceable medical knowledge** from histopathology papers. The system downloads papers from PubMed Central, parses PDFs into a hierarchical database, performs named entity recognition with UMLS linking, and uses LLM-based summarization to generate clinical rules—all with full provenance tracking back to source sentences.

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
│         ▼ latest_ingest.py (Docling + masking)                              │
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
│                    LLM SUMMARIZATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MAP (GPT-4o-mini)                                                  │    │
│  │  Chunks of 10 sentences → Atomic findings with citations            │    │
│  │  Categories: morphology | IHC | molecular_genetics | staging |      │    │
│  │              treatment | prognosis                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  REDUCE (GPT-4o)                                                    │    │
│  │  Consolidate chunk analyses → Master clinical brief                 │    │
│  │  Sections: clinical_significance | histopathological_features |     │    │
│  │            management_outcomes | risk_factors_associations          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RULE EXTRACTION (GPT-4o)                                           │    │
│  │  Generate IF-THEN clinical rules with evidence chains               │    │
│  │  Types: Diagnostic | Prognostic | Management                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  EVALUATION (evaluator.py)                                          │    │
│  │  Verify citations against source text                               │    │
│  │  Detect hallucinations via numeric + UMLS concept matching          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    Auditable Summaries + Clinical Rules
                    (JSON with full provenance to source)
```

## Project Structure

```
nlp-histo/
├── file-selector/                      # Stage 1: Data Acquisition
│   ├── file_downloader.py              # Download tarballs from PMC
│   ├── tarball_extractor.py            # Extract PDFs from tarballs
│   └── pdf_organizer.py                # Organize into flat directories
│
├── parsers/                            # Stage 2: Document Parsing
│   ├── pdf_parsers/
│   │   ├── docling_parser.py           # Docling-based PDF parsing
│   │   ├── pymupdf4llm_parser.py       # PyMuPDF4LLM parser
│   │   ├── marker_parser.py            # Marker PDF parser
│   │   ├── nougat_parser.py            # Nougat OCR-based parser
│   │   ├── ensemble_parser.py          # Ensemble combining parsers
│   │   ├── pdffigures_parser.py        # PDFFigures2 integration
│   │   ├── deduplicator.py             # Header/footer removal
│   │   └── base_parser.py              # Base parser interface
│   └── text_processing.py              # Citation removal, paragraph stitching
│
├── scripts/                            # Stage 3: Processing Pipeline
│   ├── latest_ingest.py                # Main PDF→Database pipeline
│   ├── docling_files/
│   │   └── mask_tables.py              # PDF masking with white rectangles
│   ├── visualize_docling_full.py       # Table reconstruction visualization
│   ├── process_pdffigures_results.py   # PDFFigures2 result processing
│   ├── create_tui_gin_index.py         # GIN index for semantic type search
│   └── copy_relevant_files.py          # File utility script
│
├── database/                           # Database Layer
│   ├── models.py                       # SQLAlchemy ORM models
│   └── db_connection.py                # Connection mgmt; schema via Alembic (alembic/ at repo root)
│
├── named_entity_recognition/           # NER Pipeline
│   ├── batch_ner.py                    # Parallel NER + UMLS linking
│   ├── ner.py                          # Core NER logic with scispacy
│   ├── merge_entities_by_umls.py       # Group sentences by UMLS CUI
│   ├── export_disease_entities.py      # Filter to disease entities only
│   ├── count_tokens.py                 # Token counting + cost estimation
│   ├── enums.py                        # UMLS disease semantic types
│   └── entity_linking_cache.json       # Persistent UMLS cache (~66MB)
│
├── langchain-summarization/            # LLM Summarization Pipeline
│   ├── langchain_summarization.ipynb   # Main pipeline notebook
│   ├── evaluator.py                    # Hallucination detection
│   ├── price-estimator/
│   │   └── estimator.py                # Cost estimation utilities
│   ├── test_results_50_docs/           # Input: sentences by UMLS concept
│   └── summarization_results/          # Output: summaries and rules
│       ├── summaries/                  # Final JSON summaries
│       └── rules/                      # Extracted clinical rules
│
├── files/                              # Data Directories
│   ├── organized_pdfs/                 # Flat organized PDFs
│   ├── figures/                        # Cropped figure images
│   └── tables/                         # Cropped table images
│
├── out/                                # Output Directories
│   ├── complete_pipeline/              # Layout JSON from Docling
│   ├── masked_pdfs/                    # PDFs with masked tables/figures
│   ├── text/                           # Extracted text files
│   └── failed_pdfs_blacklist.json      # Blacklisted failed PDFs
│
├── disease_entities/                   # Disease entities grouped by CUI
├── umls_entities/                      # All entities grouped by CUI
│
├── requirements.txt                    # Python dependencies
└── .env.example                        # Database config template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# For summarization pipeline
pip install langchain langchain-openai tiktoken python-dotenv
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

# Process and ingest to database
cd .. && python scripts/latest_ingest.py
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

### 5. Run Summarization Pipeline

```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_key

# Run Jupyter notebook
cd langchain-summarization
jupyter notebook langchain_summarization.ipynb

# Verify outputs for hallucinations
python evaluator.py
```

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
| LLM Pipeline | LangChain + OpenAI (GPT-4o-mini, GPT-4o) |
| Structured Output | Pydantic schemas |
| Evaluation | Hallucination detection via UMLS concept matching |

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
- **Persistent Cache**: 66MB cache avoids redundant UMLS API calls
- **Semantic Types**: Filters entities by UMLS semantic types (diseases, chemicals, etc.)
- **Token Counting**: Estimates LLM costs before running summarization

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

# OpenAI (for summarization)
OPENAI_API_KEY=your_key
```

## License

See LICENSE file for details.
