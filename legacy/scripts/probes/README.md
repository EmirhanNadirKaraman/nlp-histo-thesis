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

## `_citation_probe.py`, `_residual_citation_probe.py` — B-080 (citations)

Historical diagnostics for **B-080** (now **mitigated**), which concerned
hallucinated / cross-document citations in cached MAP voter output.

- `_citation_probe.py` — citation-failure prevalence over the related15
  `voter_cache.json`. **Offline** (reads a local JSON, runs the
  `ProvenanceValidator`; no scorer / embedder / API / DB).
- `_residual_citation_probe.py` — characterizes the residual cross-document
  citation fails on the clean cache. **Offline** (same local-cache-only shape).

## `_replay_contamination.py`, `_gemini_roundtrip_probe.py` — B-081 (Gemini batch)

Historical diagnostics for **B-081** (now **mitigated**), which concerned the
Gemini batch provider and cross-paper contamination in the selected set.

- `_replay_contamination.py` — selected-set contamination rate via the real
  frozen-config replay. It loads the Gemini map context and **may contact
  Gemini for embeddings if its embedding cache is absent** — not guaranteed
  offline.
- `_gemini_roundtrip_probe.py` — live custom_id round-trip check. It
  **performs real Gemini batch calls (needs `GOOGLE_API_KEY`) and may incur
  cost**. Its paid work is behind `main()`.

**None of these is a maintained normal command.** They are retained only for
provenance. This README can be expanded when further archived diagnostics are
added here.
