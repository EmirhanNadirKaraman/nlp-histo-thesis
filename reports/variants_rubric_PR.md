# Per-variant PDF-extraction — rubric-based P/R/F1

Rubric: `/Users/emir/Documents/GitHub/nlp-histo/eval/label_rubric.yaml`

Per-dim semantics:
* **Crop F1** — figure / table output correctness with GT-derived recall.
* **Mask F1** — text-safety (was the region correctly masked from body text?).  Same FN source as crop (missed_figures / missed_tables).
* **Caption / Footnote precision** — accuracy on detected items (no recall denominator).  TP/FP counts in parens.
* **Strict F1** — TP iff every dim scores ≥ 1.0.
* **Icon count** — emitted crops labelled `icon` (mask-OK but crop-FP).  Helps quantify how much of crop-precision loss is icons specifically.

| variant | kind | crop P | crop R | crop F1 | mask P | mask R | mask F1 | caption P | caption tp/fp | footnote P | footnote tp/fp | strict P | strict R | strict F1 | strict tp/fp/fn | emitted | unlabelled | unrecog | icons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| baseline | tables | 77.6% | 88.4% | 82.6% | 81.6% | 93.0% | 87.0% | 89.7% | 35/4 | 52.6% | 20/18 | 34.7% | 39.5% | 37.0% | 17/32/26 | 50 | 0 | 0 | 0 |
| baseline_evalcfg | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| baseline_evalcfg | tables | 74.0% | 86.0% | 79.6% | 78.0% | 90.7% | 83.9% | 89.5% | 34/4 | 51.4% | 19/18 | 32.0% | 37.2% | 34.4% | 16/34/27 | 51 | 0 | 0 | 0 |
| baseline_footnote_relaxed | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| baseline_footnote_relaxed | tables | 77.6% | 88.4% | 82.6% | 81.6% | 93.0% | 87.0% | 89.7% | 35/4 | 92.1% | 35/3 | 65.3% | 74.4% | 69.6% | 32/17/11 | 50 | 0 | 0 | 0 |
| detector_docling | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| detector_docling | tables | 97.3% | 83.7% | 90.0% | 100.0% | 86.0% | 92.5% | 88.9% | 32/4 | 47.2% | 17/19 | 40.5% | 34.9% | 37.5% | 15/22/28 | 38 | 0 | 0 | 0 |
| detector_tatr | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| detector_tatr | tables | 78.0% | 90.7% | 83.9% | 82.0% | 95.3% | 88.2% | 90.0% | 36/4 | 53.8% | 21/18 | 36.0% | 41.9% | 38.7% | 18/32/25 | 51 | 0 | 0 | 0 |
| docling_recon | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| docling_recon | tables | 90.0% | 83.7% | 86.7% | 92.5% | 86.0% | 89.2% | 91.9% | 34/3 | 48.6% | 18/19 | 40.0% | 37.2% | 38.6% | 16/24/27 | 43 | 2 | 0 | 0 |
| no_two_pass | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| no_two_pass | tables | 77.6% | 88.4% | 82.6% | 81.6% | 93.0% | 87.0% | 89.7% | 35/4 | 52.6% | 20/18 | 34.7% | 39.5% | 37.0% | 17/32/26 | 50 | 0 | 0 | 0 |
| tatr_090 | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| tatr_090 | tables | 72.2% | 90.7% | 80.4% | 75.9% | 95.3% | 84.5% | 90.0% | 36/4 | 51.3% | 20/19 | 31.5% | 39.5% | 35.1% | 17/37/26 | 55 | 0 | 0 | 0 |
| tatr_095 | figures | 81.8% | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 93.1% | 67/5 | 100.0% | 65/0 | 73.9% | 100.0% | 85.0% | 65/23/0 | 88 | 0 | 0 | 16 |
| tatr_095 | tables | 71.7% | 88.4% | 79.2% | 75.5% | 93.0% | 83.3% | 89.7% | 35/4 | 52.6% | 20/18 | 32.1% | 39.5% | 35.4% | 17/36/26 | 54 | 0 | 0 | 0 |
