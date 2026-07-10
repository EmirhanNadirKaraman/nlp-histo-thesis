# LangChain Summarization & Rule Extraction

> **⚠️ ARCHIVED / SUPERSEDED.** This LangChain + OpenAI (GPT-4o) prototype is an
> early summarisation stack and is **no longer the production path**. The current
> summariser is the multi-provider agreement-based-cascading pipeline in
> `pipeline/stages/summarization/` (see `pipeline/stages/summarization/PIPELINE.md`).
> Nothing in the live tree imports this directory; it is retained for provenance
> only. Commands below (including `cp .env.example .env` — there is no
> `.env.example` in this folder, only the repo-root one) are not maintained.

Auditable AI system for extracting clinical rules from histopathology literature with full traceability.

## Overview

This system processes disease-related medical concepts through an AI pipeline that:
1. Generates concise summaries using LLMs
2. Extracts structured clinical rules
3. Maintains complete audit trails for verification

## Features

- ✅ **Full Traceability**: Every rule links back to summary → sentences → source documents
- ✅ **Structured Output**: JSON format with metadata and provenance
- ✅ **Batch Processing**: Handles multiple UMLS concepts efficiently
- ✅ **LLM-Powered**: Uses GPT-4 via LangChain for high-quality extraction

## Setup

### 1. Install Dependencies

```bash
pip install langchain langchain-openai python-dotenv tiktoken
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Run Data Preparation (if not done)

```bash
# From project root, run the NER pipeline notebook first
cd /Users/emir/Documents/GitHub/nlp-histo/langchain-knowledge_extraction
jupyter notebook test_pipeline_50_docs.ipynb
```

This creates the input data in `test_results_50_docs/relevant_texts/`

## Usage

### Run the Summarization Notebook

```bash
jupyter notebook langchain_summarization.ipynb
```

The notebook will:
1. Load disease-related text files
2. Generate summaries using GPT-4
3. Extract structured clinical rules
4. Save results with audit trails

### Output Structure

```
summarization_results/
├── summaries/              # Human-readable summaries
│   └── C0001234_Disease_summary.txt
├── rules/                  # Structured rules (JSON)
│   └── C0001234_Disease_rules.json
├── audit_trails/           # Complete traceability
│   └── C0001234_Disease_audit.json
└── processing_index.json   # Master index
```

## Key Notebooks

1. **`test_pipeline_50_docs.ipynb`** - Prepares disease-related texts (NER pipeline)
2. **`langchain_summarization.ipynb`** - Main summarization & rule extraction

## Example Output

### Summary
```
Concept: Neoplasms

Key findings about neoplastic processes in histopathology, including
diagnostic criteria, treatment approaches, and prognostic indicators...
```

### Extracted Rule
```json
{
  "type": "Diagnostic Criterion",
  "condition": "When atypical cells show nuclear pleomorphism",
  "action": "Consider neoplastic process and perform immunohistochemistry",
  "confidence": "High"
}
```

### Traceability
```
Rule → Summary → Source Sentences → PMCID Documents
```

## Configuration

Edit the notebook to adjust:
- **Model**: Change `gpt-4` to `gpt-3.5-turbo` for faster/cheaper processing
- **Temperature**: Adjust `0.3` for more/less creative outputs
- **Batch Size**: Change `limit=10` to process more/fewer files

## Cost Estimation

Processing 10 concepts:
- GPT-4: ~$0.50-1.00 (depending on text length)
- GPT-3.5-Turbo: ~$0.05-0.10

## Troubleshooting

**No API key found**:
```bash
export OPENAI_API_KEY=your-key-here
# OR add to .env file
```

**Token limit exceeded**:
- The notebook automatically truncates long texts to 3000 chars
- Adjust this limit in the `process_document()` function

**No input files**:
- Run `test_pipeline_50_docs.ipynb` first to generate disease-related texts

## Next Steps

1. Run on full dataset (remove `limit=10`)
2. Fine-tune prompts for specific rule types
3. Integrate with clinical decision support systems
4. Export rules to knowledge graphs

## Contact

For issues or questions, check the main project repository.
