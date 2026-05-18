# Per-variant PDF-extraction P/R/F1

Computed by `scripts/eval/score_pdf_variants.py`.

Source labels are read from `eval/annotations/<variant>/<mode>.json` if a per-variant file exists, else from the legacy `annotations_<mode>.json`.

| variant | kind | P | R | F1 | TP | FP | FN | labelled | unlabelled | emitted |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| baseline | tables | 68.9% | 72.1% | 70.5% | 31 | 14 | 12 | 48 | 16 | 64 |
| baseline_evalcfg | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| baseline_evalcfg | tables | 64.6% | 72.1% | 68.1% | 31 | 17 | 12 | 51 | 14 | 65 |
| detector_docling | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| detector_docling | tables | 88.2% | 69.8% | 77.9% | 30 | 4 | 13 | 37 | 15 | 52 |
| detector_tatr | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| detector_tatr | tables | 73.2% | 69.8% | 71.4% | 30 | 11 | 13 | 43 | 22 | 65 |
| no_two_pass | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| no_two_pass | tables | 68.9% | 72.1% | 70.5% | 31 | 14 | 12 | 48 | 16 | 64 |
| tatr_090 | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| tatr_090 | tables | 68.9% | 72.1% | 70.5% | 31 | 14 | 12 | 48 | 24 | 72 |
| tatr_095 | figures | 81.8% | 100.0% | 90.0% | 72 | 16 | 0 | 88 | 27 | 115 |
| tatr_095 | tables | 68.9% | 72.1% | 70.5% | 31 | 14 | 12 | 48 | 20 | 68 |
