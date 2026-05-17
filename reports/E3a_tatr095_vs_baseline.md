# PDF-extraction run comparison

* **A (baseline):** `out/sweeps/baseline/run_metadata/run_20260517T164702Z_ba8f9982.json`
* **B (variant):**  `out/sweeps/tatr_095/run_metadata/run_20260517T165203Z_b58b0545.json`

## Overview

| field | A | B |
|---|---|---|
| run_id | 20260517T164702Z_ba8f9982 | 20260517T165203Z_b58b0545 |
| config_digest | 424591746b68 | a42fa3e065aa |
| started_at | 2026-05-17T16:47:02Z | 2026-05-17T16:52:03Z |
| finished_at | 2026-05-17T16:48:28Z | 2026-05-17T16:53:29Z |
| host | Emirhans-MacBook-Pro.local | Emirhans-MacBook-Pro.local |
| python | 3.12.0 | 3.12.0 |
| git.sha | 25e62f71327735067b54a0d0bcbec001e0df98a8 | 25e62f71327735067b54a0d0bcbec001e0df98a8 |
| git.branch | pdf-extraction-eval | pdf-extraction-eval |
| git.dirty | False | False |

⚠️  Config digests differ — A and B were run with different knobs.

## Batch summary

| metric | A | B | Δ (B − A) |
|---|---|---|---|
| n_attempted | 5 | 5 | 0 |
| n_with_stats | 5 | 5 | 0 |
| n_ok | 5 | 5 | 0 |
| n_failed | 0 | 0 | 0 |
| total_wall_seconds | 164.516 | 164.149 | -0.367 |
| mean_wall_seconds | 32.903 | 32.830 | -0.073 |
| reason_in_header_zone_sum | 29 | 29 | 0 |

## Counts (summed across documents)

| key | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| after_filter | 337 | 337 | 0 | 0.000 |
| figures_cropped | 18 | 18 | 0 | 0.000 |
| layout_elements_full | 544 | 544 | 0 | 0.000 |
| layout_elements_masked | 348 | 348 | 0 | 0.000 |
| tables_cropped | 6 | 7 | 1 | 16.700 |
| text_rows | 117 | 117 | 0 | 0.000 |
| two_pass_kept | 469 | 469 | 0 | 0.000 |
| two_pass_rejected | 75 | 75 | 0 | 0.000 |
| two_pass_scored | 544 | 544 | 0 | 0.000 |

## Reason histogram (NodeScorer R0/R1/R2/R3/R-color)

| code | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| R1_blank_pixels | 64 | 64 | 0 | 0.000 |
| R3_dense_text | 11 | 11 | 0 | 0.000 |

## Per-document status changes

_No status changes._

## Per-document count differences

| pmcid | field | A | B | Δ (B − A) |
|---|---|---|---|---|
| PMC10047158_dermatopathology-10-00017 | n_tables | 0 | 1 | 1 |
