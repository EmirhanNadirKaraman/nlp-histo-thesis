# Archived bug diagnostics

This directory holds **historical diagnostic scripts for mitigated bugs**,
retained for provenance. They are **not** maintained commands and should not
be treated as part of any normal workflow. The original submission state is
available from the `thesis-submission-2026-07-11` tag.

## `diagnose_b055.py` — B-055 (post-MAP persistence)

Reproduces / verifies **B-055**, which concerned post-MAP persistence.

- It depends on historical local batch handles under
  `out/summaries/batch_handles.prepatch/`. That directory is **gitignored** and
  is **not present in a fresh clone**, so the script is not runnable without
  those local artifacts.
- When executed it may **create and delete a scratch `pipeline_runs` database
  row**.
- Do **not** treat it as a normal maintained command.

This README can be expanded when further archived diagnostics are added here.
