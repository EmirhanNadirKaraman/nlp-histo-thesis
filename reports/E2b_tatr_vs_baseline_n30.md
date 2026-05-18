# PDF-extraction run comparison

* **A (baseline):** `out/sweeps/baseline/run_metadata/run_20260517T214919Z_8525e08e.json`
* **B (variant):**  `out/sweeps/detector_tatr/run_metadata/run_20260517T220705Z_bb15883b.json`

## Overview

| field | A | B |
|---|---|---|
| run_id | 20260517T214919Z_8525e08e | 20260517T220705Z_bb15883b |
| config_digest | 424591746b68 | 83da244b2b0f |
| started_at | 2026-05-17T21:49:19Z | 2026-05-17T22:07:05Z |
| finished_at | 2026-05-17T22:01:32Z | 2026-05-17T22:20:40Z |
| host | Emirhans-MacBook-Pro.local | Emirhans-MacBook-Pro.local |
| python | 3.12.0 | 3.12.0 |
| git.sha | 550b266a94bc9ae52768803034b0a42294c5d1c2 | 550b266a94bc9ae52768803034b0a42294c5d1c2 |
| git.branch | pdf-extraction-eval | pdf-extraction-eval |
| git.dirty | True | True |

⚠️  Config digests differ — A and B were run with different knobs.

## Batch summary

| metric | A | B | Δ (B − A) |
|---|---|---|---|
| n_attempted | 30 | 30 | 0 |
| n_with_stats | 30 | 30 | 0 |
| n_ok | 30 | 30 | 0 |
| n_failed | 0 | 0 | 0 |
| total_wall_seconds | 1463.0 | 1628.1 | 165.044 |
| mean_wall_seconds | 48.768 | 54.270 | 5.502 |
| reason_in_header_zone_sum | 78 | 78 | 0 |

## Counts (summed across documents)

| key | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| after_filter | 3753 | 3753 | 0 | 0.000 |
| figures_cropped | 115 | 115 | 0 | 0.000 |
| layout_elements_full | 5146 | 5146 | 0 | 0.000 |
| layout_elements_masked | 3779 | 3779 | 0 | 0.000 |
| tables_cropped | 64 | 65 | 1 | 1.600 |
| text_rows | 1946 | 1946 | 0 | 0.000 |
| two_pass_kept | 5024 | 5024 | 0 | 0.000 |
| two_pass_rejected | 122 | 122 | 0 | 0.000 |
| two_pass_scored | 5146 | 5146 | 0 | 0.000 |

## Reason histogram (NodeScorer R0/R1/R2/R3/R-color)

| code | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| R1_blank_pixels | 88 | 88 | 0 | 0.000 |
| R3_dense_text | 34 | 34 | 0 | 0.000 |

## Per-document status changes

_No status changes._

## Per-document count differences

| pmcid | field | A | B | Δ (B − A) |
|---|---|---|---|---|
| PMC11863705_main | n_tables | 6 | 7 | 1 |
