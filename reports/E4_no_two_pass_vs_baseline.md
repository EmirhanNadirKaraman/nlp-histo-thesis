# PDF-extraction run comparison

* **A (baseline):** `out/sweeps/baseline/run_metadata/run_20260517T164702Z_ba8f9982.json`
* **B (variant):**  `out/sweeps/no_two_pass/run_metadata/run_20260517T165529Z_501ebda0.json`

## Overview

| field | A | B |
|---|---|---|
| run_id | 20260517T164702Z_ba8f9982 | 20260517T165529Z_501ebda0 |
| config_digest | 424591746b68 | 56ba69966f12 |
| started_at | 2026-05-17T16:47:02Z | 2026-05-17T16:55:29Z |
| finished_at | 2026-05-17T16:48:28Z | 2026-05-17T16:56:53Z |
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
| total_wall_seconds | 164.516 | 163.751 | -0.764 |
| mean_wall_seconds | 32.903 | 32.750 | -0.153 |
| reason_in_header_zone_sum | 29 | 0 | -29 |

## Counts (summed across documents)

| key | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| after_filter | 337 | 440 | 103 | 30.600 |
| figures_cropped | 18 | 18 | 0 | 0.000 |
| layout_elements_full | 544 | 544 | 0 | 0.000 |
| layout_elements_masked | 348 | 1176 | 828 | 237.900 |
| mask_regions | 0 | 220 | 220 | — |
| tables_cropped | 6 | 6 | 0 | 0.000 |
| text_rows | 117 | 129 | 12 | 10.300 |
| two_pass_kept | 469 | 0 | -469 | -100.000 |
| two_pass_rejected | 75 | 0 | -75 | -100.000 |
| two_pass_scored | 544 | 0 | -544 | -100.000 |

## Reason histogram (NodeScorer R0/R1/R2/R3/R-color)

| code | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| R1_blank_pixels | 64 | 0 | -64 | -100.000 |
| R3_dense_text | 11 | 0 | -11 | -100.000 |

## Per-document status changes

_No status changes._

## Per-document count differences

| pmcid | field | A | B | Δ (B − A) |
|---|---|---|---|---|
| PMC10047213_dermatopathology-10-00018 | n_text_rows | 20 | 21 | 1 |
| PMC10047408_dermatopathology-10-00016 | n_text_rows | 37 | 43 | 6 |
| PMC10047897_dermatopathology-10-00015 | n_text_rows | 23 | 25 | 2 |
| PMC10082646_main | n_text_rows | 23 | 26 | 3 |
