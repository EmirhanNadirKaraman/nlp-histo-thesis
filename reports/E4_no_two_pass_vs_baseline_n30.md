# PDF-extraction run comparison

* **A (baseline):** `out/sweeps/baseline/run_metadata/run_20260517T214919Z_8525e08e.json`
* **B (variant):**  `out/sweeps/no_two_pass/run_metadata/run_20260517T225013Z_907d22d1.json`

## Overview

| field | A | B |
|---|---|---|
| run_id | 20260517T214919Z_8525e08e | 20260517T225013Z_907d22d1 |
| config_digest | 424591746b68 | 56ba69966f12 |
| started_at | 2026-05-17T21:49:19Z | 2026-05-17T22:50:13Z |
| finished_at | 2026-05-17T22:01:32Z | 2026-05-17T23:02:49Z |
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
| total_wall_seconds | 1463.0 | 1508.2 | 45.193 |
| mean_wall_seconds | 48.768 | 50.275 | 1.506 |
| reason_in_header_zone_sum | 78 | 0 | -78 |

## Counts (summed across documents)

| key | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| after_filter | 3753 | 6096 | 2343 | 62.400 |
| figures_cropped | 115 | 115 | 0 | 0.000 |
| layout_elements_full | 5146 | 5146 | 0 | 0.000 |
| layout_elements_masked | 3779 | 14404 | 10625 | 281.200 |
| mask_regions | 0 | 1610 | 1610 | — |
| tables_cropped | 64 | 64 | 0 | 0.000 |
| text_rows | 1946 | 2331 | 385 | 19.800 |
| two_pass_kept | 5024 | 0 | -5024 | -100.000 |
| two_pass_rejected | 122 | 0 | -122 | -100.000 |
| two_pass_scored | 5146 | 0 | -5146 | -100.000 |

## Reason histogram (NodeScorer R0/R1/R2/R3/R-color)

| code | A | B | Δ (B − A) | Δ % |
|---|---|---|---|---|
| R1_blank_pixels | 88 | 0 | -88 | -100.000 |
| R3_dense_text | 34 | 0 | -34 | -100.000 |

## Per-document status changes

_No status changes._

## Per-document count differences

| pmcid | field | A | B | Δ (B − A) |
|---|---|---|---|---|
| PMC10123624_dermatopathology-10-00020 | n_text_rows | 12 | 19 | 7 |
| PMC10296831_dermatopathology-10-00026 | n_text_rows | 23 | 26 | 3 |
| PMC10297671_dermatopathology-10-00024 | n_text_rows | 18 | 19 | 1 |
| PMC11649516_HIS-86-236 | n_text_rows | 49 | 56 | 7 |
| PMC11674653_dermatopathology-11-00035 | n_text_rows | 35 | 48 | 13 |
| PMC11755463_dermatopathology-12-00002 | n_text_rows | 16 | 15 | -1 |
| PMC11791726_HIS-86-485 | n_text_rows | 45 | 51 | 6 |
| PMC11863705_main | n_text_rows | 39 | 131 | 92 |
| PMC11863984_main | n_text_rows | 35 | 65 | 30 |
| PMC11984985_main | n_text_rows | 5 | 6 | 1 |
| PMC12045760_HIS-86-1053 | n_text_rows | 43 | 47 | 4 |
| PMC12129649_NEUP-45-177 | n_text_rows | 59 | 80 | 21 |
| PMC12192205_dermatopathology-12-00018 | n_text_rows | 48 | 47 | -1 |
| PMC12337196_main | n_text_rows | 16 | 23 | 7 |
| PMC4945808_dpa-0003-0055 | n_text_rows | 22 | 17 | -5 |
| PMC6851684_HIS-75-329 | n_text_rows | 29 | 55 | 26 |
| PMC7122038_978-1-4614-4800-6_Chapter_25 | n_text_rows | 1039 | 1155 | 116 |
| PMC7124083_978-1-61779-403-2_Chapter_21 | n_text_rows | 19 | 20 | 1 |
| PMC7150024_main | n_text_rows | 30 | 28 | -2 |
| PMC7152403_main | n_text_rows | 56 | 61 | 5 |
| PMC7317439_NEUP-40-302 | n_text_rows | 34 | 36 | 2 |
| PMC7436514_HIS-77-169 | n_text_rows | 13 | 14 | 1 |
| PMC7543760_main | n_text_rows | 40 | 45 | 5 |
| PMC8008316_dermatopathology-08-00010 | n_text_rows | 12 | 10 | -2 |
| PMC8221903_main | n_text_rows | 28 | 29 | 1 |
| PMC8352662_main | n_text_rows | 65 | 78 | 13 |
| PMC8416649_main | n_text_rows | 27 | 45 | 18 |
| PMC8420544_CYT-33-93 | n_text_rows | 59 | 74 | 15 |
| PMC9906788_main | n_text_rows | 12 | 13 | 1 |
