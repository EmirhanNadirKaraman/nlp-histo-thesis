# Calibration & Swap Inventory

Every runtime knob that affects pipeline output, organized by stage. Companion to
[`CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md) (the measurement harness),
[`STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md) (model-agnostic
experiments), and [`../eval/EXPERIMENTS.md`](../eval/EXPERIMENTS.md) (Opus-judge
experiments).

This document answers **"what can we vary in an experiment?"** It is **not** a
plumbing-cleanup audit — for that see plan
`/Users/emir/.claude/plans/you-are-reviewing-the-zany-pnueli.md` (Tier 1 shipped;
Tier 2 backlog mirrored in §10 below).

Status legend:

- **YAML** — surfaceable in `configs/run.yaml` today; safe to sweep.
- **ENV** — only swappable via an environment variable.
- **HARD** — hardcoded constant; must be surfaced before sweeping (Tier 2).
- **CODE** — selected by editing Python (e.g. swapping a scorer class).
- **DEAD** — present in config but not consumed (kept for traceability).

Measurement column points at either a `compute_proxy_metrics.py` column
(`docs/CALIBRATION_EVAL.md`), an experiment ID in `eval/EXPERIMENTS.md`
(M-1 … Rs-2), or a section of `docs/STAGE_EVAL_EXPERIMENTS.md`. `—` means
no proxy or experiment is designed yet.

---

## 1. PDF text extraction

Source: `pipeline/stages/pdf_text_extraction/config.py`.

### 1.1 Docling layout extractor (`DoclingConfig`)

| ID | Knob | Default | Range / Options | Wired via | Affects | Measurement |
|---|---|---|---|---|---|---|
| P-D-01 | `do_table_structure` | `True` | bool | YAML | Table structure inference on/off | recall against TATR-only baseline |
| P-D-02 | `do_ocr` | `False` | bool | YAML | Force OCR on text-native PDFs | character recall vs ground truth XML |
| P-D-03 | `force_full_page_ocr` | `False` | bool | YAML | OCR entire page regardless of text layer | same as P-D-02 |
| P-D-04 | `ocr_engine` | `EASYOCR` | `easyocr` \| `tesseract` \| `rapidocr` | YAML | OCR engine swap (only when `do_ocr=True`) | char-level accuracy |
| P-D-05 | `images_scale` | `2.0` | float ≥1.0 | YAML | Image resolution multiplier; higher = better OCR, slower | OCR accuracy vs runtime |
| P-D-06 | `accelerator_device` | `"cpu"` | `cpu` \| `cuda` \| `mps` | YAML | Inference device | wall-time only |
| P-D-07 | `reconstruct_tables_from_lists` | `False` | bool | YAML | Treat list-like layouts as tables | table recall/precision |
| P-D-08 | `export_intermediate_json` | `True` | bool | YAML | Cache full Docling JSON to disk | cold/warm run time |
| P-D-09 | `timeout_sec` | `300` | int | YAML | Advisory only — not enforced (B-035) | — |

### 1.2 TATR table detector (`TATRConfig`)

| ID | Knob | Default | Range | Wired via | Affects | Measurement |
|---|---|---|---|---|---|---|
| P-T-01 | `threshold` | `0.99` | `[0.0, 1.0]` | YAML | Detection score cutoff | table P/R against manual annotations in `eval/annotations/` |
| P-T-02 | `device` | `"cpu"` | `cpu` \| `cuda` \| `mps` | YAML | Inference device | wall-time only |
| P-T-03 | `model_name` | `"microsoft/table-transformer-detection"` | HF model id | YAML | Swap to a different TATR checkpoint | table P/R |
| P-T-04 | `render_dpi` | `150` | int | YAML | DPI when rasterising pages for TATR; bump for small/faint tables (B-034) | table recall vs runtime |

### 1.3 Region masking (`MaskingConfig`)

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| P-M-01 | `enabled` | `True` | YAML | Toggle masking pass entirely | text/table P/R |
| P-M-02 | `mask_tables` | `True` | YAML | Whiten table regions before Pass 2 | table-text bleed rate |
| P-M-03 | `mask_figures` | `True` | YAML | Whiten figure regions | figure-text bleed rate |
| P-M-04 | `mask_header_footer_sidebar` | `True` | YAML | Strip page furniture | header/footer leak rate |
| P-M-05 | `merge_overlapping_boxes` | `True` | YAML | Pre-merge bboxes before masking | mask count vs accuracy |
| P-M-06 | `expand_box_px` | `2` | YAML | Mask padding (glyph remnants) | masking precision |

### 1.4 Filtering (`FilteringConfig`)

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| P-F-01 | `enabled` | `True` | YAML | Toggle entire filter step | text count delta |
| P-F-02 | `apply_ner_filtering` | `True` | YAML | NER-based irrelevant-block drop | text element count |
| P-F-03 | `apply_paragraph_relevance_filtering` | `True` | YAML | Heuristic relevance filter | text element count |

### 1.5 Cropping (`CroppingConfig`)

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| P-C-01 | `enabled` / `save_figure_crops` / `save_table_crops` | `True` ×3 | YAML | Crop emission toggles | output count |
| P-C-02 | `image_format` | `"png"` | YAML | jpeg/png/etc. | disk usage |
| P-C-03 | `dpi` | `200` | YAML (>0) | Output crop DPI | OCR-on-crop accuracy |
| P-C-04 | `min_figure_pts` | `50` | YAML | Minimum side; smaller skipped | figure precision |
| P-C-05 | `merge_figures_by_caption` | `False` | YAML | Merge same-numbered PICTUREs | figure count |
| P-C-06 | `merge_tables_by_caption` | `False` | YAML | Merge same-numbered TABLEs | table count |
| P-C-07 | `subfigure_proximity_pts` | `20` | YAML | Edge-gap for subfigure clustering | merge precision |
| P-C-08 | `expand_tables_with_footnotes` | `False` | YAML | Absorb footnotes into table crop | table completeness |
| P-C-09 | `footnote_proximity_pts` | `20.0` | YAML | Adaptive footnote gap | merge precision |
| P-C-10 | `text_footnote_proximity_pts` | `8.0` | YAML | Non-adaptive TEXT gap | merge precision |

### 1.6 Text assembly (`TextAssemblyConfig`)

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| P-A-01 | `write_raw_text` | `False` (YAML `true`) | YAML | Dump pre-assembly elements to `out/text_raw/` | none (audit-only) |
| P-A-02 | `pre_filter_relevance` | `True` | YAML | `is_relevant_para` before stitch vs. post-stitch boilerplate filter | text recall |

### 1.7 Two-pass invisible-text pipeline (`TwoPassConfig`)

Full description in [`../memory/two_pass_pipeline.md`](#) (project memory).

| ID | Knob | Default | Range | Wired via | Affects | Measurement |
|---|---|---|---|---|---|---|
| P-2P-01 | `enabled` | `True` | bool | YAML | Replace standard Steps 1/3/4 with pixel-render pipeline | ghost-text recall (run `scripts/verify_ghost_text_detection.py`) |
| P-2P-02 | `render_dpi` | `150` | int | YAML | Pixel render resolution | ghost detection accuracy vs runtime |
| P-2P-03 | `blank_brightness_threshold` | `245.0` | `[0, 255]` | YAML | Mean-luminance cutoff for "blank" classification | R1 P/R on ghost-text fixture |
| P-2P-04 | `blank_dark_pixel_max_fraction` | `0.02` | `[0, 1]` | YAML | Max dark-pixel fraction for blank | same |
| P-2P-05 | `min_char_coverage_threshold` | `0.05` | `[0, 1]` | YAML | R2 fallback char-coverage cutoff | R2 P/R |
| P-2P-06 | `min_text_chars_for_word_check` | `8` | int | YAML | Min length before R2 applies | R2 noise rate |
| P-2P-07 | `max_top_fraction_header` | `0.15` | `[0, 1]` | YAML | "Header zone" definition — wording only, not gate | none |
| P-2P-08 | `mask_figures` / `mask_tables` | `True` / `True` | bool | YAML | Pass-1 mask additions before Pass 2 | text-bleed metrics |
| P-2P-09 | `max_white_char_fraction` | `1.0` (disabled) | `[0, 1]` | YAML | R-color (white-text) rejection cutoff | R-color FP rate on inverted-banner headers |
| P-2P-10 | `max_chars_per_bbox_pt` | `15.0` | float ≥0 (`0` disables) | YAML | R3 dense-text-layer rejection | R3 P/R |
| P-2P-11 | `min_anchor_word_count` | `5` | int | YAML | Min words to qualify as body anchor | anchor false-positive rate |
| P-2P-12 | `header_mask_margin_pt` | `3.0` | float | YAML | Gap below mask before anchor | mask clip-rate |

### 1.8 Visualization / DB / Runtime / pipeline selector

| ID | Knob | Default | Wired via | Affects | Status |
|---|---|---|---|---|---|
| P-V-01 | `visualization.enabled` | `True` | YAML | Annotated PDFs to `out/visualization/` | YAML |
| P-V-02 | `visualization.save_tatr_visualization` | `True` | YAML | TATR-only viz output | YAML |
| P-V-03 | `visualization.save_combined_visualization` | `True` | YAML | Combined viz output | YAML |
| P-V-04 | `visualization.max_pages` | `None` | YAML | Cap pages drawn | YAML |
| P-DB-01 | `database.enabled` | `False` | YAML | DB ingestion on/off | YAML |
| P-DB-02 | `database.db_url` | `None` (auto from `.env`) | YAML/env | Connection string | YAML |
| P-R-01 | `runtime.log_level` | `INFO` | YAML | Logger verbosity | YAML |
| P-R-02 | `runtime.fail_fast` | `False` | YAML | Stop on first failure | YAML |
| P-R-03 | `runtime.skip_blacklisted` | `True` | YAML | Honor `out/failed_pdfs_blacklist.json` | YAML |
| P-R-04 | `runtime.skip_existing_in_db` | `True` | YAML | Skip docs already in DB | YAML |
| P-R-05 | `runtime.update_blacklist_on_failure` | `True` | YAML | Auto-blacklist on failure | YAML |
| P-R-06 | `runtime.blacklist_if_rows_exceed` | `None` | YAML | Blacklist after success if row count exceeds | **DEAD** (zany-pnueli Tier 1, slated for delete) |
| P-R-07 | `runtime.skip_existing_media_json` | `False` | YAML | Skip docs with existing media JSON | YAML |
| P-R-08 | `runtime.skip_existing_outputs` | `False` | YAML | Stage-level output cache (B-027) | YAML |
| P-R-09 | `runtime.multi_source_crops` | `False` | YAML | Produce three media JSONs | **DEAD** (zany-pnueli Tier 1) |
| P-R-10 | `runtime.save_error_traces` | `True` | YAML | Persist tracebacks | YAML |
| P-R-11 | `runtime.seed` | `42` (`None` opts out) | YAML | `random` / `numpy` / `torch` seed | YAML |
| P-R-12 | `runtime.num_workers` | `1` | YAML (≥1) | ParallelBatchRunner threads | YAML |
| P-TD | `table_detector` | `HYBRID` | `tatr` \| `docling` \| `hybrid` \| `vlm` | YAML | Detector swap | table P/R |
| P-DBL | `docling_text` (optional override) | `None` | YAML | Distinct DoclingConfig for Step 4 re-extract | text recall |

---

## 2. Summarization — MAP

Source: `pipeline/stages/summarization/config.py:MapConfig`, plus prompts and
cascade.

### 2.1 Cascade + chunking

| ID | Knob | Default | Range | Wired via | Affects | Measurement |
|---|---|---|---|---|---|---|
| S-M-01 | `map.theta` | `0.8` | `[0, 1]` | YAML | Deferral score ≥ θ → KEEP | escalation rate (cascade_decisions counts); experiment M-1/2/3 |
| S-M-02 | `map.reject_theta` | `0.2` | `[0, 1]` | YAML | Deferral score ≤ → hard REJECT | rejection rate, downstream finding count |
| S-M-03 | `map.chunk_size` | `10` sentences | int >0 | YAML | Sentences per MAP chunk | findings/paper, escalation rate |
| S-M-04 | `map.chunk_overlap` | `2` | `< chunk_size` | YAML | Sentence overlap between chunks | duplicate-finding rate |
| S-M-05 | `map.chunk_workers` | `5` | int ≥1 | YAML | Parallel chunk threads | wall-time only |

**Sweep path (2026-05-24):** the θ / reject_theta sweep runs through
`eval/silver/map_theta_sweep.py`, which **primes all L1+L2+L3 voters directly**
into `eval/data/map_primer/voter_cache.json` and replays the cascade offline.
This sidesteps the production per-voter-persistence gap (B-056: the batch runner
does not write `sum_map_voter_outputs`) — the primer path does **not** read the
production DB cache, so the sweep is runnable today. A *DB-replay* sweep (reusing
production voter outputs) would still be blocked by B-056.

### 2.2 Voter cascade (`batch/voter_configs.py`)

Two profiles, selected by `--profile` CLI flag or `$NLP_HISTO_PROFILE`. No
implicit default — unset raises `ValueError`.

| ID | Knob | Options | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| V-PROF | `cascade_profile` | `cheap` (2-tier OpenAI+Gemini) \| `real` (3-tier 3-provider) | ENV `NLP_HISTO_PROFILE` (Tier 2 → YAML) | Whole voter set | M-1 cost vs F1 |
| V-L1 | L1 voter set | `cheap`/`real` both use `gemini-2.5-flash-lite`, `gpt-4o-mini`, `gpt-4.1-nano` @ T=0.1 | CODE (`make_l1_voters`) | L1 agreement rate | M-1, M-2, M-3 |
| V-L2 | L2 voter set | `real`: `gemini-2.5-flash`, `gpt-4.1-mini`, `claude-haiku-4-5-20251001` @ T=0.1; `cheap`: `gpt-4.1-mini` only | CODE | L2 escalation rate | M-1, M-2, M-3 |
| V-L3 | L3 escalator | `real`: `claude-sonnet-4-6` @ T=0.0; `cheap`: `gpt-4.1-mini` @ T=0.1 | CODE | Tail correctness | M-2 field accuracy |
| V-TEMP | Per-voter temperature | L1/L2 = 0.1, L3 (real) = 0.0 | CODE | Output variance | M-2 stability across reruns |
| V-MAXTOK | Per-voter `max_tokens` | `DEFAULT_MAX_TOKENS = 16384` | HARD (`config.py:23`) | Truncation rate | truncation flag in trace; **Tier 2** |
| V-RETRY | `stop_after_attempt` | `2` | HARD (`prompts.py:430/438/445`) | LLM retry budget | error rate from `runs.jsonl` |

### 2.3 Agreement scorers (`agreement/`)

Selected by routing policy `scorer_name`. All return a deferral score ∈ `[0, 1]`
that the cascade decision step compares to θ.

| ID | Scorer | Class | Key knobs | Wired via | Measurement |
|---|---|---|---|---|---|
| AG-EMB | `EmbeddingScorer` (legacy) / `EmbeddingSimilarityStrategy` (production, via `SemanticAgreementScorer`) | `agreement/embedding.py`, `agreement/embedding_similarity.py` | `tau=0.15`, `count_alpha=0.25`, `reuse_weight=0.15`, `contradiction_weight=0.20` — now read from `AgreementConfig` (H-EMB-01) via `_align` | **YAML** (`summarization.agreement.*`) | Calibration data in `agreement/calibration/`; experiment M-3; swept by `map_theta_sweep` |
| AG-SEM | `SemanticAgreementScorer` | `agreement/semantic_scorer.py` | `theta` (optional override) | CODE | M-3 |
| AG-HYB | `HybridScorer` | `agreement/hybrid_scorer.py` | scorer-list, weights | CODE | M-3 |
| AG-NER | `NerScorer` | `agreement/ner_scorer.py` | NER overlap | CODE | M-3 |
| AG-LEX | `LexicalSimilarity` | `agreement/lexical_similarity.py` | lexical overlap | CODE | M-3 |
| AG-CAT | `CategoryJaccard` | `agreement/category_jaccard.py` | category set overlap | CODE | M-3 |
| AG-LLM | `LlmJudgeScorer` | `agreement/llm_judge.py` | judge model id | CODE | M-3 — most expensive |
| AG-POL | `PolarityConflict` | `agreement/polarity_conflict.py` | `_HARD_POLARITY_PAIR` (frozenset) | HARD | reason-code count in cascade_decisions |

### 2.4 Routing policy (`routing/policy.py:RoutingPolicySpec`)

ILP-selected policy parameters. Today produced by `select_policy.py`; the spec
fields are sweepable.

| ID | Knob | Default | Wired via | Affects |
|---|---|---|---|---|
| R-01 | `scorer_name` | varies | CODE | Which deferral score backs θ |
| R-02 | `theta` | varies | CODE | Same as S-M-01 (different per scorer) |
| R-03 | `single_voter_policy` | `"escalate"` | CODE | What to do when only one valid voter |
| R-04 | `reject_on_low_agreement` | `False` | CODE | Hard reject at low deferral |
| R-05 | `fabricated_threshold` | `0.25` | CODE | Fabrication-flag deferral cutoff |
| R-06 | `weak_threshold` | `0.60` | CODE | Weak-agreement deferral cutoff |
| R-07 | `escalation_model` | varies | CODE | L2 model when policy escalates |

### 2.5 Prompts + cache versioning

| ID | Knob | Value | File:line | Affects |
|---|---|---|---|---|
| S-V-01 | `MAP_SCHEMA_VERSION` | `"map_v10_nli_and_voter_config_in_key"` | `models.py:23` | MAP cache invalidation key |
| S-V-02 | `MAP_PROMPT_VERSION` | `"map_prompt_v5_expression_absent_vs_negative"` | `models.py:24` | MAP cache invalidation key |
| S-V-03 | Prompt body | `_REDUCE_SYSTEM` / `_REDUCE_USER` (`prompts.py:272-323`) and MAP map-chain prompt | source | Voter behavior — bump version on edit |

---

## 3. Summarization — GROUNDING

Source: `pipeline/stages/summarization/config.py:GroundingConfig` + helpers/grounding_filter.py.

| ID | Knob | Default | Range | Wired via | Affects | Measurement |
|---|---|---|---|---|---|---|
| S-G-01 | `grounding.threshold` | `None` (disabled) | `None` \| `[0, 1]` | YAML | NLI entailment cutoff to keep a claim | `grounding_rejection_rate` (proxy); experiment G-1 P/R; G-2 sweep |
| H-G-01 | `_MODEL_MAX_TOKENS` | `512` | int | HARD (`grounding_filter.py:191`) | NLI premise+hypothesis truncation cap | rejection rate on long premises |
| H-G-02 | `_PREMISE_BUDGET_FLOOR` | `64` | int | HARD (`grounding_filter.py:192`) | Min tokens reserved for premise | rare-case truncation |

---

## 4. Summarization — NORMALIZE

Source: `current_stages/normalize_stage.py`.

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| S-N-01 | `normalize.extra_synonyms` | `{}` | YAML | Surface→canonical overrides merged on top of `synonyms.yaml` | experiment N-1 |
| S-N-02 | `synonyms.yaml` content | bundled | source file | Same as S-N-01 baseline | N-1 |
| S-N-03 | `_NEGATIVE_TRIGGERS` / `_POSITIVE_TRIGGERS` | hardcoded tuples | HARD (`normalize_stage.py:219, 243`) | Polarity detection on borderline claims | experiment N-2 (false-merge audit) |
| H-U-01 | `UMLS_THRESHOLD` | `0.85` | HARD (`umls_utils.py:11`, mirrored in `ner.py:29`) | Min scispaCy linker score to accept a CUI | UMLS-CUI coverage / `entity_count` per finding |
| H-U-02 | `JUNK_SEMANTIC_TYPES` | 16 TUI codes | HARD (`umls_utils.py:16-33`) | Filtered UMLS semantic types | false-CUI rate |
| E-U-01 | `NLP_HISTO_DISABLE_UMLS` | unset | ENV (`umls_resources.py:50`) | Full UMLS bypass kill-switch | ablation: UMLS off vs on |
| E-U-02 | `NLP_HISTO_SKIP_UMLS_ENRICHMENT` | unset | ENV (`helpers/entity_linker.py:15`) | Skip CUI enrichment only | ablation |

---

## 5. Summarization — GROUP

Source: `current_stages/group_stage.py`.

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| S-Gr-01 | `_SCOPE_FIELDS` | hardcoded tuple | HARD (`group_stage.py:27`) | Which fields partition a group | experiments Gr-1, Gr-2 |
| S-Gr-02 | `_SCOPE_FIELDS_DEFAULTS` | hardcoded dict | HARD (`models.py:197`) | Defaults assumed when scope field absent | merge correctness |
| S-Gr-03 | Dedup key composition | `(text_element_id, subject, outcome, relation_type, direction)` — missing `category`/`scope` | HARD (`normalize_stage.py:_dedup_key`) | False-merge rate (B-053 candidate) | experiment N-2 |

---

## 6. Summarization — CANONICALIZE

Source: `current_stages/canonicalize_stage.py`.

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| S-C-01 | `_CATEGORY_VALID` enum | `(BIOLOGICAL_INSIGHT, …)` | HARD (`models.py:269`) | Allowed category surface forms | experiment C-2 |
| S-C-02 | `_CATEGORY_ALIASES` | hardcoded dict | HARD (`models.py:278`) | Alias collapse | C-2 |
| S-C-03 | `_DIRECTION_ALIASES` | hardcoded dict | HARD (`models.py:335`) | Direction surface normalization | experiment C-1 (direction conflict) |
| S-C-04 | `_RELATION_TYPE_ALIASES` | hardcoded dict | HARD (`models.py:293`) | Relation-type collapse | C-1 |

---

## 7. Summarization — RELATE

Source: `config.py:RelateConfig` + `current_stages/relate_stage.py` +
`helpers/corpus_relate.py`.

| ID | Knob | Default | Range | Wired via | Affects | Measurement |
|---|---|---|---|---|---|---|
| S-R-01 | `relate.entailment_threshold` | `0.50` | `[0, 1]` | YAML | NLI score above which SUPPORT / SCOPE_QUALIFY label fires | `support_edge_count`, `scope_qualify_edge_count` (proxy); experiment R-1 P/R |
| S-R-02 | `relate.contradiction_threshold` | `0.50` | `[0, 1]` | YAML | NLI score (both directions) for CONTRADICT | `contradiction_edge_count`; R-1 |
| S-R-03 | `_norm_subject` vs `_norm_outcome` | asymmetric (subject CUI-aware, outcome string-only) | HARD (`relate_stage.py:194-204`) | Gate that decides which pairs reach NLI | `outcome_incompatible` skip rate; B-051 |
| E-NLI-01 | `NLP_HISTO_NLI_MODEL` | unset → registry default | ENV (`nli_config.py:74`) | Cross-encoder NLI model swap | Re-run R-1 per model; affects both GROUNDING + RELATE |
| E-NLI-02 | `NLP_HISTO_NLI_BATCH_SIZE` | unset → registry default | ENV (`nli_config.py:84`) | NLI inference batch size | wall-time only |
| NLI-REG | `configs/nli_models.yaml` | `default: pritamdeka/PubMedBERT-MNLI-MedNLI` | YAML | NLI model registry + default | model swap experiments |

---

## 8. Summarization — RESOLVE

Source: `config.py:ResolveConfig`. Two-mode formula — relations-present vs.
relations-absent.

### 8.1 Relations-present mode

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| S-Rs-01 | `resolve.grounding_weight` | `0.60` | YAML | Grounding-score multiplier (max 0.60) | experiment Rs-2 weight sweep |
| S-Rs-02 | `resolve.grounding_default` | `0.50` | YAML | Fallback when `mean_grounding_score` is None | Rs-2 |
| S-Rs-03 | `resolve.finding_bonus_max` | `0.10` | YAML | Max finding-count bonus | Rs-2 |
| S-Rs-04 | `resolve.finding_bonus_scale` | `5` findings | YAML | N findings to reach `finding_bonus_max` | Rs-2 |
| S-Rs-05 | `resolve.support_boost_per_rel` | `0.08` | YAML | Score bump per SUPPORT relation | Rs-2 |
| S-Rs-06 | `resolve.support_boost_cap` | `0.20` | YAML | Max total support bonus | Rs-2 |
| S-Rs-07 | `resolve.single_study_pen` | `0.10` | YAML | Penalty when `study_coverage == 'single_study'` | Rs-2 |
| S-Rs-08 | `resolve.contradict_pen_per_rel` | `0.15` | YAML | Penalty per CONTRADICT | Rs-2 |
| S-Rs-09 | `resolve.contradict_pen_cap` | `0.30` | YAML | Max total contradiction penalty | Rs-2 |

### 8.2 Relations-absent mode

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| S-Rs-10 | `resolve.no_rel_grounding_weight` | `0.80` | YAML | Grounding multiplier when RELATE produced none | Rs-2 |
| S-Rs-11 | `resolve.no_rel_finding_bonus_max` | `0.15` | YAML | Finding bonus in this mode | Rs-2 |
| S-Rs-12 | `resolve.no_rel_single_study_pen` | `0.05` | YAML | Halved single-study penalty | Rs-2 |

**Hidden behavior:** Mode is auto-selected by `relations_present = len(relations) > 0`,
not logged on `FinalRule`, no `score_mode` field (B audit item §8 in static
verification). Fix scheduled — see `BUGS.md`.

---

## 9. Contradiction detector (helper)

Source: `helpers/contradiction_detector.py`.

| ID | Knob | Default | Wired via | Affects | Measurement |
|---|---|---|---|---|---|
| S-CD-01 | `contradiction_similarity_threshold` | `0.7` (None disables) | YAML (top-level `SummarizationConfig`) | Cosine similarity floor for candidate pairs | candidate-pair count; experiment N-2 / R-2 |
| H-CD-01 | `max_workers` for NLI batch | `min(len(pairs), 8)` | HARD (`contradiction_detector.py:139`) | Thread pool size | wall-time only |

---

## 10. Hardcoded constants to surface (Tier 2 backlog)

These are calibratable knobs that currently live as Python constants. They
appear in zany-pnueli Tier 2 + this audit. Surfacing → YAML is a precondition
for sweeping any of them.

**Re-scoped 2026-05-23.** On inspection only **H-EMB-01** was a genuine
summarisation sweep knob, and it is now surfaced (`Status` column). The
"surface temp / max_tokens / retries" items were dropped — temperature is
intentionally per-voter (`VoterBatchConfig`, L3=0.0 deterministic) not a
global constant; `max_tokens` is already a `VoterBatchConfig` field + every
factory param; `stop_after_attempt=2` lives only in the off-by-default
REDUCE/RULES/judge chains (`build_map_chain`, the active path, has none).
The masking / layout / evidence / table-reconstructor rows are
PDF-extraction constants, deferred with the PDF-extraction experiments.

| ID | Constant | Value | File:line | Why it matters | Status |
|---|---|---|---|---|---|
| H-LLM-01 | `DEFAULT_MAX_TOKENS` | `16384` | `summarization/config.py:23` | Voter / escalation truncation cap | Dropped — already a `VoterBatchConfig` field + factory param |
| H-LLM-02 | `temperature` (per provider factory) | `0.1` (6 factories) | `llm_providers.py:77, 125, 185, 241, 261, 285` | Global default; per-voter overrides in `VoterBatchConfig` | Dropped — intentionally per-voter, not a global knob |
| H-LLM-03 | `stop_after_attempt` | `2` | `prompts.py:430, 438, 445` | LLM retry budget per chain | Dropped — only in off-by-default REDUCE/RULES/judge chains |
| H-MASK-01 | `_SIDEBAR_MAX_W` | `150` pts | `region_masker.py:19` + `two_pass_extractor.py:109` | Annotation-column heuristic; **duplicated** | Deferred (PDF extraction) |
| H-MASK-02 | `_COLUMN_GAP_MIN` | `50` pts | same files :20 / :110 | Column-boundary heuristic; **duplicated** | Deferred (PDF extraction) |
| H-LAY-01 | `MIN_ANCHOR_H` | `15` pts | `parsers/layout_utils.py:251` | Min bbox height to qualify as substantial block | Deferred (PDF extraction) |
| H-EV-01 | `_INK_LUMINANCE_THRESHOLD` | `200` | `evidence_gatherer.py:52` | Pixel-darkness cutoff for "ink present" | Deferred (PDF extraction) |
| H-EV-02 | `_WHITE_COLOR_THRESHOLD` | `240` | `evidence_gatherer.py:57` | Per-channel cutoff for "near-white" | Deferred (PDF extraction) |
| H-TR-01 | `threshold_multiplier` | `1.2` | `table_reconstructor.py:38` | Table reconstruction grouping multiplier | Deferred (PDF extraction) |
| H-EMB-01 | soft-alignment weights | `tau=0.15`, `count_alpha=0.25`, `reuse_weight=0.15`, `contradiction_weight=0.20` | `config.py:AgreementConfig` (was `agreement/embedding.py:304-307`) | Soft-alignment weighting; tuned in the scorer-choice experiment | **Surfaced 2026-05-23** → `AgreementConfig` (YAML `summarization.agreement.*`); read by both `EmbeddingSimilarityStrategy` + `HybridStructuredSimilarity` via `_align()`. **Caveat:** in the per-paper result-cache hash but not the chunk-level `PipelineCache` key (same as `theta`) — sweep via replay. |
| H-CONF-01 | `_CONFIDENCE_VALID` enum | `("high", "medium", "low")` | `models.py:307` | Allowed confidence levels | Dropped — a `Literal` type alias, not a sweepable knob |

---

## 11. Environment variables

| Variable | Default | Consumed by | Effect |
|---|---|---|---|
| `NLP_HISTO_PROFILE` | unset (raises) | `batch/voter_configs.py:165`, `scripts/run_paper.py:411` | Cascade profile (`cheap`/`real`) |
| `NLP_HISTO_NLI_MODEL` | unset → registry default | `nli_config.py:74` | NLI model swap (GROUNDING + RELATE) |
| `NLP_HISTO_NLI_BATCH_SIZE` | unset → registry default | `nli_config.py:84` | NLI batch size |
| `NLP_HISTO_DISABLE_UMLS` | unset | `umls_resources.py:50` | Full UMLS bypass |
| `NLP_HISTO_SKIP_UMLS_ENRICHMENT` | unset | `helpers/entity_linker.py:15` | Skip CUI enrichment (keep linker) |
| `NLP_HISTO_LOG_DIR` | unset → repo-relative default | `enum_logging.py:19`, `runner.py:411` | Log output dir |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | localhost / 5432 / nlp_histo / postgres / postgres | `database/db_connection.py:28-32` | Postgres connection |
| `AZURE_FOUNDRY_ENDPOINT` / `_API_KEY` / `_API_VERSION` | required / required / `2024-05-01-preview` | `llm_providers.py:103-105`, `batch/azure_batch.py:23-25` | Azure OpenAI/Foundry creds |
| `VERTEX_PROJECT` / `VERTEX_LOCATION` / `VERTEX_BATCH_GCS_BUCKET` | required / `us-central1` / required | `llm_providers.py:155-156`, `batch/vertex_batch.py:26-28` | Vertex AI creds |
| `CLAUDE_VERTEX_LOCATION` | `us-east5` | `llm_providers.py:213` | Vertex-Claude location override |
| `GOOGLE_API_KEY` | required | `llm_providers.py:246`, `batch/gemini_batch.py:25` | Direct Gemini API |
| `ANTHROPIC_API_KEY` | required | `llm_providers.py:271`, `batch/claude_batch.py:39` | Direct Anthropic API |
| `OPENAI_API_KEY` | required | `llm_providers.py:290`, `batch/openai_batch.py:24` | Direct OpenAI API |

---

## 12. Cost / observability knobs

Source: `config.py:CostConfig`.

| ID | Knob | Default | Wired via | Affects |
|---|---|---|---|---|
| S-Co-01 | `cost.enable_cost_report` | `True` | YAML | Collect usage + write `cost_report.json` |
| S-Co-02 | `cost.write_usage_jsonl` | `True` | YAML | Persist canonical `llm_usage_records.jsonl` |
| S-Co-03 | `cost.model_prices_path` | `None` → `configs/model_prices.json` | YAML | Pricing table override |
| S-Co-04 | `cost.cost_report_output_dir` | `None` → run artifact dir | YAML | Cost output location |

---

## 13. Cross-references

- **Proxy metrics produced today:** `docs/CALIBRATION_EVAL.md` (`compute_proxy_metrics.py`).
- **Experiments designed:** `eval/EXPERIMENTS.md` (Opus-judge, 16 experiments
  M-1 … Rs-2) + `docs/STAGE_EVAL_EXPERIMENTS.md` (model-agnostic battery).
- **Bug-driven knobs:** `docs/BUGS.md` (especially B-052 per-voter persistence,
  blocker for θ sweep; B-051 RELATE outcome-gate asymmetry).
- **Config-plumbing cleanup separate from this doc:** plan
  `/Users/emir/.claude/plans/you-are-reviewing-the-zany-pnueli.md` (Tier 2
  surfacing list mirrored in §10 above).
- **Eval framework plan:** `/Users/emir/.claude/plans/i-want-you-to-reactive-quiche.md`.

## 14. Maintenance

When a new knob lands:

1. Add a row in the matching section above.
2. Specify default, range, wired-via, what it affects.
3. Pick a measurement target — a proxy column or an experiment ID. If none
   exists, add a TODO row in `docs/THESIS.md` referencing this inventory.
4. If the knob is currently HARD, add it to §10 and link to the surfacing PR.
