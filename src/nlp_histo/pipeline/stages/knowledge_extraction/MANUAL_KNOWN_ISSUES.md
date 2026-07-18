# Manual Known Issues — Summarization Pipeline

Issues logged manually. Read before editing any pipeline stage.  
Each issue has a **severity**, the **file:location** to touch, a description, and a suggested fix.

---

## Bugs (code is wrong today)

---

## High-Impact Accuracy Risks

---

## Lower-Impact / Design Choices

### DES-3 — RESOLVE scoring weights are hand-tuned, not empirically validated
**Severity:** Low  
**File:** `config.py` `ResolveConfig` (weights `grounding_weight` / `support_boost_per_rel` / `contradict_pen_per_rel`, etc.), consumed in `resolve_stage.py`  
**Symptom:** Weights were chosen to produce intuitively reasonable distributions, not fit
to a gold-labeled dataset.  
**Fix options:**
- Proxy labels from citation count of supporting papers.
- LLM-as-judge confidence rating.
- Agreement across papers as weak supervision proxy.  
See iPad notes for more details.

### DES-4 — `best_cui` returns only one CUI — loses valid secondary matches
**Severity:** Low  
**File:** `umls_utils.py` (`best_cui`), `normalize_stage.py` (`_umls_canonical`)  
**Symptom:** `best_cui` picks a single top non-junk CUI and discards all remaining valid
candidates. An entity like "CD30" may have 3–5 valid UMLS hits above threshold (e.g. the
protein, the gene, the receptor ligand). Only the highest-scoring one reaches downstream
stages; the others are silently dropped.  
**Fix:** Return the top-N (e.g. 3 or 5) non-junk CUIs sorted by score descending instead of
a single CUI. Downstream impact: `_umls_canonical` in `normalize_stage.py` currently returns
`str | None`; changing to `list[str]` requires updating `NormalFinding.cui` (models.py),
GROUP grouping key, CANONICALIZE predicate selection, and any DB column that stores a single
CUI. Do not attempt without a schema migration.

---

## Summary Table

| ID | Severity | Stage | One-line description | Status |
|----|----------|-------|----------------------|--------|
| DES-3 | Low | RESOLVE | Scoring weights hand-tuned — no gold label dataset to validate against | Open |
| DES-4 | Low | NORMALIZE | `best_cui` returns one CUI — top-N (3–5) valid matches would improve coverage | Open |
