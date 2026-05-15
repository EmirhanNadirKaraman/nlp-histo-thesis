# MAP prompt + schema audit (2026-05-15)

Follow-up review of the MAP stage prompt (`pipeline/stages/summarization/prompts.py`) and Pydantic schema (`pipeline/stages/summarization/models.py`) prompted by the discovery of [B-018](BUGS.md#bug-18--relation_type-prognosis-noun-form-coerced-to-unclear-instead-of-prognostic). B-018 was one instance of a systematic class of failures; this document catalogues the rest.

Scope: read-only review. No code changes here — each issue ends with a recommended fix the user can take separately.

---

## Empirical baseline

Aggregate of `logs/enum_observations.jsonl` (production runs through 2026-05-15, all profiles):

| Field          | Raw value            | Reason             | Count |
|----------------|----------------------|--------------------|-------|
| relation_type  | `prognosis`          | unknown_value      | 16    |
| relation_type  | `molecular_genetics` | unknown_value      | 6     |
| relation_type  | `treatment`          | unknown_value      | 2     |
| relation_type  | `staging`            | unknown_value      | 2     |
| relation_type  | `associates_with`    | unknown_value      | 1     |
| direction      | `null`               | null_to_no_direction | 22  |
| direction      | `maybe`              | unknown_value      | 1     |
| category       | `demographic` / `demographics` | alias_repair | 84   |

Plus 4 records in `logs/bad_findings.jsonl`, all with `category="expression"` — entire Finding dropped before reaching `enum_observations.jsonl`.

**Read:** every category value the LLM emits into `relation_type` ends up coerced to `unclear` and silently dropped at `is_groupable()`. Cross-field bleed is bidirectional (category→relation_type *and* relation_type→category) and systematic — not isolated to the `prognosis`/`prognostic` pair.

---

## Why this matters — `category` and `relation_type` are both load-bearing

Both fields are mandatory joint grouping keys at every downstream stage:

| Stage          | Use of (category, relation_type) |
|----------------|----------------------------------|
| NORMALIZE      | Both carried into `NormalFinding`. |
| GROUP          | `group_id = sha8(pmcid) ⊕ sha8(subject) ⊕ sha8(outcome) ⊕ relation_type ⊕ sha8(category)`. Both feed the hash (`group_stage.py:70, 139`). |
| CANONICALIZE   | Both passed to `CanonicalRule`. |
| RELATE         | Pre-filter: pair rejected if either field differs (`relate_stage.py:13–14, 217–218`). |
| CORPUS_RELATE  | Same gate cross-paper; `category_mismatch` / `relation_type_mismatch` are early-rejection reasons (`corpus_relate.py:113–126`). |
| RESOLVE        | Both flow to `FinalRule`. |

Semantically the fields are orthogonal:

* `category` — *what evidence domain* (morphology / IHC / molecular_genetics / staging / treatment / prognosis / demographic).
* `relation_type` — *what predicate* connects subject → outcome (has_feature / expression / prognostic / comparative / demographic / treatment_response / unclear).

But the prompt presents them as 1:1 paired labels (`prompts.py:110–148`), which is exactly why LLMs collapse them. A mislabel on either field fragments groups, prevents pairing in RELATE, and produces duplicate `FinalRule`s instead of one consolidated rule with cross-paper SUPPORT evidence.

---

## Ranked issues

### Issue 1 — Prompt has no `relation_type` mapping for `staging` or `molecular_genetics` (HIGH)

`prompts.py:110–148` shows field-mapping examples for every `category` value except `staging` and `molecular_genetics`. The LLM is asked to pick a relation_type but never shown which. Result: it emits the category name (`relation_type="staging"`, `relation_type="molecular_genetics"`) — which the logs confirm (6 + 2 occurrences).

**Recommended fix (prompt-only, requires `MAP_PROMPT_VERSION` bump):**

Add to the Field-mapping examples block:

```
  Staging (has_feature, or prognostic when stage drives outcome):
    "Stage III disease shows extranodal involvement"
      → relation_type: has_feature, direction: positive
    "Advanced stage associated with worse OS"
      → relation_type: prognostic, direction: negative

  Molecular genetics (expression for variant calls, has_feature for structural):
    "MYD88 L265P mutation present in 90% of cases"
      → relation_type: expression, direction: positive
    "Translocation t(11;14) characteristic of mantle cell lymphoma"
      → relation_type: has_feature, direction: positive
```

Open design call: whether `staging`/`molecular_genetics` deserve their own `relation_type` enum values. The 1:1 prompt mapping (Issue context above) is the deeper reason these fields collapse — splitting the predicate axis from the domain axis with truly orthogonal labels is the cleaner long-term fix.

### Issue 2 — `treatment` is the same-stem footgun as `prognosis`/`prognostic` (MEDIUM)

`category="treatment"` (noun) sits next to `relation_type="treatment_response"` (compound). Same lexical-confusion pattern as B-018, observed 2× in logs.

**Recommended fix:** extend `_RELATION_TYPE_ALIASES` in `models.py:167`:

```python
_RELATION_TYPE_ALIASES: dict[str, str] = {
    "prognosis": "prognostic",
    "treatment": "treatment_response",
}
```

Requires `MAP_SCHEMA_VERSION` bump.

### Issue 3 — `category="expression"` and other category-side bleed isn't catalogued (MEDIUM)

`AuditableSummary._drop_invalid_findings` (`models.py:337`) catches the whole Finding when `category` fails Literal validation and writes it to `bad_findings.jsonl` — but only the *Pydantic error message*, not the structured `field_name / raw_value / valid_values / coerced_value / reason` shape used by `enum_observations.jsonl`. Harder to mine, no aggregate query.

**Recommended fix:** in `_drop_invalid_findings`, before falling through to the drop, attempt `log_enum_observation(field_name="category", raw_value=raw_category, valid_values=…, coerced_value=None, reason="invalid_literal_dropped")` for every dropped finding whose `category` failed validation. Pure observability — no semantic call.

Do **not** alias `expression → IHC`. The LLM's misuse of `expression` is signal that the rubric needs sharper category guidance, not a silent mapping. Same reason holds for `molecular_genetics` claims that could also be `expression`-category.

### Issue 4 — Null `subject_entity` / `outcome_entity` silently drops findings at GROUP (HIGH, not enum-related but in scope)

Prompt invites null: "Output null only when no subject can be identified" (`prompts.py:59, 65`). `is_groupable()` (`group_stage.py:49`) then drops every finding with either field null. The runner tracks this as `non_groupable_subject_null` / `non_groupable_outcome_null` counters (`runner.py:1837`) but the per-paper breakdown is invisible in the run logs — only the aggregate count surfaces.

**Recommended fix:**

* Tighten the prompt: "If the subject is unclear, copy the most specific noun phrase from the claim. Never emit null for `subject_entity` or `outcome_entity` — emit your best guess instead." This shifts the false-positive cost from "silent drop" to "noisy group" — the latter is recoverable downstream via dedup; the former is not.
* Optionally: emit a structured log line in `is_groupable()` so per-paper null-subject vs null-outcome vs unclear-relation counts are mineable from the JSONL stream.

Prompt change requires `MAP_PROMPT_VERSION` bump.

### Issue 5 — `scope.scope_parsed` is LLM-set but trivially derivable (LOW) — **FIXED 2026-05-15 ([B-045](BUGS.md#bug-45--scope_parsed-is-llm-set-but-trivially-derivable))**

`prompts.py:159` asks the LLM to compute "true if at least one sub-field is non-null". This is a one-line `any(...)` in Python — putting it in the LLM prompt adds one more thing the model can get wrong (and one more reason to use up output tokens).

**Recommended fix:** add a `@model_validator(mode="after")` on `FindingScope` that sets `scope_parsed = any(v is not None for k, v in self if k != "scope_parsed")`. Remove the `scope_parsed` line from the prompt schema-as-instructions block. Cache invalidation: `MAP_SCHEMA_VERSION` bump.

### Issue 6 — `direction=absent` vs `direction=negative` ambiguity in expression contexts (LOW) — **FIXED 2026-05-15 ([B-047](BUGS.md#bug-47--direction-absent-vs-negative-ambiguity-on-expression-claims))**

Prompt example for `"BCL2 was negative"` → `direction=absent` (`prompts.py:138`), but:

* `negative`: "not expressed / decreased / less frequent / worse outcome / inverse"
* `absent`: "explicitly not present / not seen / lacking / negative staining"

Both reasonably apply to `"BCL2 was negative"`. For `relation_type=expression` the two collide; for `relation_type=prognostic` they're clean. The 1 occurrence of `direction="maybe"` in the logs suggests the model is also reaching for labels outside the enum when the rubric isn't sharp.

**Recommended fix (prompt-only):** add an explicit rule under the `direction` definition:

```
For relation_type=expression:
  "negative staining"  → direction: absent
  "decreased intensity" / "reduced expression" → direction: negative
For other relation_types:
  prefer negative; use absent only when text says "absent" / "not present" / "lacking".
```

Defer until you can mine the frequency of `direction=absent` and `direction=negative` co-occurring on expression findings — if both labels appear in the same FindingGroup with different polarity, RELATE will misfire.

### Issue 7 — `Rule.type` is Title-Case (`"Diagnostic"|"Prognostic"|"Management"`); everything else lowercase (LOW) — **FIXED 2026-05-15 ([B-048](BUGS.md#bug-48--ruletype-title-case-inconsistent-with-lowercase-convention))**

`models.py:422`. Inconsistent with `Finding.confidence` (lowercase, post-B-016). Different stage, separate prompt — so bleed risk is low — but the inconsistency invites the next refactor to reintroduce a casing alias map.

**Recommended fix:** align with the lowercase convention used by `Finding.confidence` and `Finding.category`. Touches RULE prompt + RULE persistence; defer until the optional REDUCE+RULES block is next exercised.

### Issue 8 — `direction="maybe"` (single occurrence) (LOW) — **FIXED 2026-05-15 ([B-046](BUGS.md#bug-46--direction-hedging-words-coerce-to-unclear-instead-of-alias-repair))**

Telemetry signal: the model reaches for "maybe" when it can't find `unclear`. One occurrence is below the action threshold, but if it recurs:

```python
_DIRECTION_ALIASES = {"maybe": "unclear", "possibly": "unclear"}
```

Mirror the alias-repair pattern from B-018 / Issue 2.

---

## Recommended action plan

Grouped by what gets invalidated when applied:

| Bump            | Issues       | Effort    | Estimated impact |
|-----------------|--------------|-----------|------------------|
| `MAP_PROMPT_VERSION` | 1, 4, 6  | 30 min, prompt-only | High — recovers ~10 findings/paper now coerced; tightens subject/outcome yield. |
| `MAP_SCHEMA_VERSION` | 2, 5     | 10 min, code-only | Medium — handles `treatment` and removes one LLM-error surface. |
| No bump (observability only) | 3 | 5 min | Medium — enables data-driven decision on `category="expression"`. |
| Defer              | 7, 8     | n/a       | Low. |

If running them as a single change: bump both versions in the same commit so the next calibration run picks up everything at once and re-reuses the same `--profile cheap` budget (~$0.22).

---

## Cross-references

* [B-018 (BUGS.md)](BUGS.md#bug-18--relation_type-prognosis-noun-form-coerced-to-unclear-instead-of-prognostic) — the fix that prompted this audit.
* [B-016 (BUGS.md)](BUGS.md#bug-16--demographic-spelling-and-confidence-casing-divergence) — earlier instance of the same class (singular vs plural).
* [B-015 (BUGS.md)](BUGS.md#bug-15--raw-llm-enum-values-lost-on-coercion) — raw-value capture mechanism that makes this audit possible.
* `pipeline/stages/summarization/PIPELINE.md` — stage-by-stage architecture (use as the entry point when implementing any of the recommended fixes).
