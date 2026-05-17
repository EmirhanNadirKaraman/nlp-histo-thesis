# THESIS_MATERIAL.md

Working notebook for Stage-1 PDF-extraction stabilization.  Records every
configuration we run, what we observed, and which decisions came out of it.

Cross-references:
* TODOs and the formal Decisions log live in [`THESIS.md`](THESIS.md).
* Bugs (with evidence and fix) live in [`BUGS.md`](BUGS.md).
* Reproducible commands for each comparison live in
  [`HOW_TO_RUN.md`](HOW_TO_RUN.md#21-reproducible-sweep-runs).
* Pipeline architecture changes are appended to the changelog at the bottom
  of [`STRUCTURE.md`](STRUCTURE.md#pipeline-changelog).

Routing reminder: bug-driven rationales belong in `BUGS.md`; permanent design
calls belong in `THESIS.md`'s Decisions log.  **Only sweep observations and
their interpretation belong here.**

---

## Configurations compared

| run_id | config_digest | detector | tatr_threshold | render_dpi | two_pass | other knobs | notes |
|--------|---------------|----------|----------------|------------|----------|-------------|-------|
| _(add a row per sweep)_ |  |  |  |  |  |  |  |

How to populate: after a sweep run, read the latest
`out/sweeps/{name}/run_metadata/run_*.json`.  Copy the values from
`run_id` (top), `config_digest` (top), and the relevant sub-blocks under
`config`.  Add the resulting `summary` snippet to **Observations**.

---

## Observations

One subsection per comparison.  Each entry should answer:

1. What was changed (knob + values)?
2. What measurable output changed (counts, reason histogram, wall time)?
3. What did NOT change?  (Be explicit — invariance is also a finding.)
4. Working hypothesis or interpretation.

Template:

```
### {short title} ({YYYY-MM-DD})

* **Compared:** `{run_id_A}` (digest `…`) vs `{run_id_B}` (digest `…`).
* **Changed:** `{knob}` `{value_A}` → `{value_B}`.
* **Result:**
  - `counts_sum.{key}`: A=… B=… (Δ=…)
  - `reason_histogram_sum`: A=… B=… (Δ=…)
  - mean wall: A=…s B=…s
* **Invariant:** {what stayed the same}
* **Interpretation:** {one paragraph}
* **Implications for the thesis:** {one sentence}
```

_(no entries yet)_

---

## Open questions

Items the sweeps surface but don't resolve.  Once an item turns into a
permanent decision, move the rationale to `THESIS.md` and link back from
here; once it turns into a defect, file in `BUGS.md` and link back here.

* _(none yet — populated as sweeps run)_

---

## Decisions made (cross-references into `THESIS.md`)

For each Stage-1 design call, add a row pointing at the Decisions-log entry.

| Date | Decision (short) | Link |
|------|------------------|------|
| _(populated when sweeps lead to a permanent call)_ | | |
