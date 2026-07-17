# Grounding manual-label sample

> **This is a labeling sample, not an accuracy report. It contains placeholder rows ready for hand-labeling. Until the `label` field is populated, none of these numbers measure accuracy, precision, or recall.**

## Sweep metadata

- schema_version: `v1`
- sample_type: `grounding`
- tool: `scripts/eval/sample_grounding_for_manual_labeling.py`
- created_at: `2026-05-16T15:52:45Z`
- git_commit: `983e9b7f2d150e16fb292f54b5531c07cfffebaf-dirty`
- input_dir: `out/summaries`
- seed: `42`
- threshold: `0.5`
- requested_n: `100`
- actual_n: `100`
- total_findings_scanned: `965`
- missing_score_count: `0`
- pipeline_config_hashes: `149023b87374cbc2`
- run_ids: `grounding_compare_calv1_runB_20260516T163007`

## Bucket allocation

| bucket | range | available | target | sampled |
|---|---|---|---|---|
| very_low | [0.00, 0.30) | 87 | 10 | 41 |
| low | [0.30, 0.45) | 7 | 15 | 7 |
| near_threshold_low | [0.45, 0.50) | 2 | 25 | 2 |
| near_threshold_high | [0.50, 0.55) | 2 | 25 | 2 |
| medium | [0.55, 0.75) | 15 | 15 | 15 |
| high | [0.75, 1.00] | 852 | 10 | 33 |
| missing | (no grounding_score) | 0 | 0 | 0 |

## Kept vs rejected (at --threshold 0.5)

- kept (`grounding_score >= 0.5`): 50
- rejected (`grounding_score <  0.5`): 50

## Notes

- Sampled rows live in `eval/results/grounding_manual_sample.jsonl` (first line is the meta header; subsequent lines are the samples).
- To hand-label, set the `label` field on each row to one of `supported` / `partial` / `unsupported`. The schema is stable across re-runs with the same seed.
- Re-running the sampler with the same seed and same input produces an identical JSONL — safe to regenerate.
