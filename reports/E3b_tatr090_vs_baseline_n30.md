# PDF-extraction run comparison

* **A (baseline):** `out/sweeps/baseline/run_metadata/run_20260517T214919Z_8525e08e.json`
* **B (variant):**  `out/sweeps/tatr_090/run_metadata/run_20260517T223645Z_41e12385.json`

## Overview

| field | A | B |
|---|---|---|
| run_id | 20260517T214919Z_8525e08e | 20260517T223645Z_41e12385 |
| config_digest | 424591746b68 | 89ccc472fb84 |
| started_at | 2026-05-17T21:49:19Z | 2026-05-17T22:36:45Z |
| finished_at | 2026-05-17T22:01:32Z | 2026-05-17T22:49:56Z |
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
| total_wall_seconds | 1463.0 | 1577.9 | 114.868 |
| mean_wall_seconds | 48.768 | 52.597 | 3.829 |
| reason_in_header_zone_sum | 78 | 78 | 0 |

## Counts (summed across documents)

| key | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| after_filter | 3753 | 3753 | 0 | 0.000 |
| figures_cropped | 115 | 115 | 0 | 0.000 |
| layout_elements_full | 5146 | 5146 | 0 | 0.000 |
| layout_elements_masked | 3779 | 3779 | 0 | 0.000 |
| tables_cropped | 64 | 72 | 8 | 12.500 |
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
| PMC11755463_dermatopathology-12-00002 | n_tables | 1 | 2 | 1 |
| PMC12192205_dermatopathology-12-00018 | n_tables | 0 | 1 | 1 |
| PMC7122038_978-1-4614-4800-6_Chapter_25 | n_tables | 14 | 17 | 3 |
| PMC7124083_978-1-61779-403-2_Chapter_21 | n_tables | 0 | 1 | 1 |
| PMC8221903_main | n_tables | 0 | 1 | 1 |
| PMC8352662_main | n_tables | 3 | 4 | 1 |
