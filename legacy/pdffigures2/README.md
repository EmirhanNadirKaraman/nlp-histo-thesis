# PDFFigures2 research workflow (archived)

> **ARCHIVED / DORMANT.** A historical research workflow, retained for provenance.
> It is **not** part of production PDF ingestion
> (`pipeline/stages/pdf_text_extraction/`, which uses Docling directly) and **not**
> part of thesis reproduction. The historical state is recoverable from the
> `thesis-submission-2026-07-11` tag.

## What it does

- `run_pdffigures.sh` runs PDFFigures2 (`pdffigures2.jar`) in batch over PDFs.
- `process_pdffigures_results.py` masks the detected figure/table regions and
  re-extracts text using the archived research Docling parser
  (`legacy.pdf_parsers.docling_parser`).

## Running

Both commands are expected to be launched from the **repository root**:

```bash
bash   legacy/pdffigures2/run_pdffigures.sh
python legacy/pdffigures2/process_pdffigures_results.py
```

The scripts retain **historical machine-specific paths** (a hardcoded `JAVA_HOME`
and absolute `/Users/emir/...` input/output directories in `run_pdffigures.sh`)
and may require manual editing before reuse.

## Bundled JAR

`pdffigures2.jar` is retained here to preserve the exact historical workflow.
Its `META-INF/MANIFEST.MF` reports:

- Main-Class: `org.allenai.pdffigures2.FigureExtractorBatchCli`
- Version: `0.1.0`
- Vendor: `org.allenai`

**No `LICENSE` or `NOTICE` file was found inside the JAR.** Attribution/licensing
material should be verified before redistribution.
