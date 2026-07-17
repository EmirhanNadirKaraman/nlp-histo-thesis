# legacy/

Quarantined code that predates the current pipeline. **None of this is on the
production path and nothing in the live tree imports it** — these files are kept
only for reference and provenance. For all current work use
`pipeline/stages/pdf_text_extraction/runner.py` (`PipelineRunner`).

| File | What it was | Superseded by |
|------|-------------|---------------|
| `latest_ingest.py` | Legacy monolithic PDF→DB pipeline (~1400 LOC) | `PipelineRunner` |
| `combined_pipeline.py` | Early Docling + TATR combined pipeline | `PipelineRunner` |
| `merged_pipeline.py` | Larger experimental layout + NER + summarization pipeline. Its header/footer detection was ported to `pipeline/stages/pdf_text_extraction/components/region_masker.py` (`_detect_header_footer_elements`). | `PipelineRunner` |
| `docling_files/mask_tables.py` | PDF table/figure masking helper, used only by `latest_ingest.py` | `components/region_masker.py` |
| `PDF_Processing_Pipeline.ipynb` | Demo notebook driving `mask_tables` | `PipelineRunner` / `scripts/run_paper.py` |

Moved here from `scripts/` and `notebooks/` on 2026-06-19 (REFACTOR_PLAN A5) to
keep the live `scripts/` directory free of dead pipelines.

Note: `latest_ingest.py` and `mask_tables.py` still reference
`scripts/visualize_docling_full.py` via `sys.path`. That helper deliberately
stays in `scripts/` (live pipeline components cite it for provenance), so these
two would need `scripts/` on `PYTHONPATH` if ever re-run.
