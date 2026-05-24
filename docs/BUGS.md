# Bug catalogue — nlp-histo

Per-bug write-ups with status, evidence, diagnosis, fix, and verification.
Carry-forward work items live in [`THESIS.md`](THESIS.md#todos); permanent
design calls live in [`THESIS.md`](THESIS.md#decisions-log).

> **How to use this file:**
> * [Bugs (catalogue)](#bugs-catalogue) — every substantive defect, with status.
> * Detailed write-ups follow as `## Bug N — …` and `## Topic — …` sections.
>
> Add a new entry the same day the issue is discovered. New bug → row in the
> catalogue + detail section. Bump the ID monotonically (`B-017`, `B-018`, …).
> Never delete entries — flip `Status` to `Fixed` / `Won't fix` / `Superseded`
> so the history survives.

---

## Bugs (catalogue)

| ID | Status | Severity | Surface | One-line summary | Detail |
|----|--------|----------|---------|------------------|--------|
| B-001 | Fixed (2026-05-13) | High | Summarisation, corpus relate | `canonical_id` collided across papers because `group_id` did not include `pmcid`; same-rule self-pairs appeared as bogus "intra_paper" SUPPORT relations. | [Bug 1](#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| B-002 | Mitigated (2026-05-13) | Medium | PDF extraction, Docling layout | Docling emits phantom layout elements — real text content at a bbox that does not render (often in the header zone of the wrong page). `ContextAwareStitcher` had been masking this; now caught upstream by `TwoPassConfig.enabled=True` + Rule R1/R3. | [Bug 2](#bug-2--docling-phantom-layout-elements) |
| B-003 | Mitigated (2026-05-13) | Low (latent) | PDF extraction, color signal | Rule R-color (`max_white_char_fraction < 1.0`) treats *any* near-white span as ghost text, ignoring the rendered background. Would have produced false positives on white-on-coloured headers if applied to TEXT/CAPTION types. Currently dormant because `SECTION_HEADER` is in `NodeScorer._ALWAYS_KEEP`. Disabled by default now (`max_white_char_fraction=1.0`). | [Bug 3](#bug-3--r-color-white-text-false-positive-latent) |
| B-004 | Observed | Low | PDF extraction, Docling glyph fallback | CID-only PDFs surface in Docling as `GLYPH<…>` / `/gid00001` text strings. R1 drops them because their bboxes have no ink (Docling didn't decode the font). No production impact, but worth keeping an eye on for corpora with subset-only fonts. | [Bug 4](#bug-4--cid-glyph-fallback-strings) |
| B-005 | Mitigated (2026-05-14) | High | Summarisation, batch runner | `BatchSummarizationRunner.finalize()` was missing six features the sync runner had: (1) `_replace_verbatim_from_db` — grounding NLI ran against LLM paraphrases instead of source text; (2) stable `compute_finding_id`; (3) DB persistence to `sum_*` tables; (4) `corpus_relate_incremental`; (5) `rejection_summary` build + persist; (6) NER + UMLS linking. Since `scripts/run_paper.py` defaults to batch mode, every batched production result between commit `5c59c3e` (2026-04-27) and the 05-14 backport was grounded against paraphrased text. | [Bug 5](#bug-5--batch-runner-missing-sync-parity-features) |
| B-006 | Fixed (2026-05-14) | Medium | Summarisation, RELATE / RESOLVE | `RelationTypeLabel.SCOPE_QUALIFY` plumbing (the enum, the RESOLVE filter, the RELATE info-log column) was wired end-to-end but no `_classify_pair` branch ever emitted it. Stripped: enum value removed, RESOLVE `scope_qualifies` list-comp dropped, RELATE log no longer prints the column. `FinalRule.scope_qualify_count` and the DB column retained as hard-zero fields so existing readers (HTML inspector, downstream consumers) don't break. | [Bug 6](#bug-6--scope_qualify-plumbing-is-dead) |
| B-007 | Fixed (2026-05-14) | Medium | Summarisation, sync runner result cache | `SummarizationRunner._load_result` returned cached `{pmcid}.json` unconditionally and `_save_result` never stamped a hash. Fixed in commit `b03d4f6`: a `_pipeline_config_hash()` helper composes cascade signature + thresholds + model identifiers + schema/prompt versions + `enable_router` state; `_load_result` recomputes current hash and returns `None` on mismatch (with a `cached result stale` log line); `_save_result` stamps the hash via `setdefault`. Manifest builder reuses the same helper to avoid drift. | [Bug 7](#bug-7--sync-runner-cached-result-load-ignores-pipeline_config_hash) |
| B-008 | Fixed (2026-05-14) | Low | Summarisation, sync runner batch reporting | `SummarizationRunner.process_batch` reported `n_skip = len(results) - n_ok - n_err` but `_load_result` returned cached dicts with `status="success"`, so cached papers counted in `n_ok` and `n_skip` was structurally 0. Fixed by tagging the in-memory cached dict with `status="skipped"` inside `_load_result` (on-disk JSON unchanged) and counting that key explicitly in `process_batch`. Three downstream call-sites (`scripts/summarize_paper.py`, `scripts/run_single_doc.py`, `scripts/run_paper_single_model.py`) updated to treat `success` and `skipped` interchangeably so cached papers still feed the corpus-relate gate. | [Bug 8](#bug-8--process_batch-skip-counter-is-structurally-zero) |
| B-009 | Fixed (2026-05-14) | Low | Summarisation, sync runner instance state | `SummarizationRunner` kept per-paper state in eight instance dicts (`_scored_map_findings`, `_normal_findings`, `_finding_groups`, `_canonical_rules`, `_relations`, `_relate_raw_pairs`, `_relate_skipped_pairs`, `_final_rules`). Inside `process_batch` they accumulated across papers and were never cleared. Memory grew O(papers × avg eligible pairs). Fixed by popping the per-paper entries from all eight dicts in `process()`'s `finally` block — runs after the result dict has been materialised but before the function returns, so external callers see the same payload they always did. Verified no external reader of these dicts exists (only `last_map_*` properties on `self` and the cache helpers are exposed). | [Bug 9](#bug-9--sync-runner-instance-dicts-leak-across-papers) |
| B-010 | Fixed (2026-05-14) | Medium | PDF extraction, artifact filter | `components/artifact_filter.py:59` rebuilt `List[LayoutElement]` after filtering via `[el for i, el in enumerate(elements) if element_dicts[i] in filtered_dicts]` — list-`__contains__` over dicts. O(N²); and the moment `filter_artifacts` ever mutated a kept dict (e.g. a future ligature normalisation), the post-filter dict no longer `==`'d the pre-filter dict and the corresponding `LayoutElement` was silently dropped. Replaced with an `id()`-keyed `dict[int, LayoutElement]` lookup built before the filter call — O(N), survives in-place mutation, and doesn't change `filter_artifacts`'s public contract (other callers in `scripts/` unaffected). | [Bug 10](#bug-10--artifact_filter-rebuild-uses-dict-equality-instead-of-identity) |
| B-011 | Fixed (2026-05-14) | Low | PDF extraction, `ModelRegistry` | `resources.py` `ModelRegistry.docling_converter` ignored `DoclingConfig.images_scale`, `accelerator_device`, `ocr_engine`, `force_full_page_ocr`; hard-coded `images_scale=2.0` and never built `AcceleratorOptions`. Was unused by `PipelineRunner` (each component constructs its own converter) but exported as public API — a caller who flipped a non-default `DoclingConfig` and used `ModelRegistry` silently got CPU + scale 2.0. Fixed by deleting the entire class — zero in-tree consumers existed; each component already lazy-loads its own model (Docling via `DoclingLayoutExtractor._get_converter`, TATR via `TATRTableDetector`'s process-wide singleton, scispaCy via `summarization/umls_resources.get_nlp()`). `resources.py` removed; `__init__.py` re-export and four docs files updated. | [Bug 11](#bug-11--modelregistrydocling_converter-ignores-doclingconfig) |
| B-012 | Observed | Low | PDF extraction, two-pass extractor | `components/two_pass_extractor.py:382-398` header/footer strip construction mixes Docling y-coords (`docling_y1=page_h`) and fitz coords (`fitz_header_bottom`) on adjacent lines. Today only a `docling_y1 > docling_y2` comparison guards against a sign-flip if those names ever get muddled. Clarity issue today, latent bug surface for the next refactor. | [Bug 12](#bug-12--two_pass_extractor-header-strip-mixes-coordinate-systems) |
| B-013 | Fixed (2026-05-14) | Low | Inspector batch index, sort handler | `scripts/templates/pipeline_batch_index.html.jinja2:276` read `dataset.nilBa` instead of `dataset.nliBa`. `parseFloat(undefined) → NaN → 0`, so clicking the "NLI B→A" column compared zeros and produced no reorder. Fixed by correcting the typo. | [Bug 13](#bug-13--inspector-nli-ba-sort-typo) |
| B-014 | Fixed (2026-05-14) | Low (latent) | Inspector batch index, badge style | `pipeline_batch_index.html.jinja2:194` renders SCOPE_QUALIFY relations with class `badge-blue`, but the stylesheet only defined `badge-green/red/orange/gray/cyan`. Badge rendered unstyled. Currently dormant because B-006 means SCOPE_QUALIFY is never emitted; would surface the moment B-006 is fixed. Added `.badge-blue` rule. | [Bug 14](#bug-14--inspector-badge-blue-class-missing) |
| B-015 | Fixed (2026-05-14) | Medium | Summarisation, MAP enum coercion | Raw LLM-emitted `relation_type` / `direction` / `category` values were coerced (or alias-repaired) to enum members and the originals were dropped from the row — only landed in `logs/enum_observations.jsonl` with no FK back to the finding. Downstream stages saw only `unclear` / coerced values. Fixed by capturing raw values in a `model_validator(mode="wrap")` on `Finding`, persisting them to new `sum_map_findings.raw_{relation_type,direction,category}` columns (Alembic `0011`). | [Bug 15](#bug-15--raw-llm-enum-values-lost-on-coercion) |
| B-016 | Fixed (2026-05-14) | Low | Summarisation, MAP prompt + schema | `category` enum was `"demographics"` (plural) while `relation_type` enum was `"demographic"` (singular) — same concept, two spellings, requiring an alias map and prompt warning. `Rule.confidence` Literal was `"High"|"Medium"|"Low"` while MAP `Finding.confidence` was lowercase. Aligned both to `"demographic"` (singular, consistent with sibling category labels) and lowercase confidence; inverted `_CATEGORY_ALIASES` to repair legacy `"demographics"`; bumped `MAP_PROMPT_VERSION` to `map_prompt_v2_singular_demographic`. | [Bug 16](#bug-16--demographic-spelling-and-confidence-casing-divergence) |
| B-017 | Fixed (2026-05-15) | High | Summarisation, batch entry-points in `scripts/run_paper.py` | Both batch entry-points (`_run_batch_multi` line 766, `_run_batch_single` line 863) called `build_batch_runner(...)` without passing `db=`, so `BatchSummarizationRunner.__init__` got `db=None`. Every `_persist_*` method and `_corpus_relate_incremental` short-circuits on `if self._db is None: return` — silently. Net effect: production batch runs since the B-005 backport (2026-05-14) wrote per-paper `out/summaries/summaries/*.json` artifacts but no `sum_*` rows and no `sum_corpus_relations` rows. The sync path at `build_runner` already opened a DB connection; the batch entry-points were left behind. Fixed by extracting a module-level `_open_db_connection(caller_label)` helper and passing its return value to both `build_batch_runner` call-sites (and using it from the sync path too, removing the duplicated try/except). | [Bug 17](#bug-17--batch-entry-points-pass-no-db-to-buildbatchrunner) |
| B-018 | Fixed (2026-05-15) | High | Summarisation, MAP enum coercion | `category` enum exposes the noun `"prognosis"` while `relation_type` enum exposes the adjective `"prognostic"` — same concept, two surface forms, both visible in the same MAP prompt. L1 voters bleed `"prognosis"` from category into relation_type; the pre-existing `_coerce_invalid_relation_type` validator fell through to the unknown-value branch and coerced to `"unclear"`. Downstream `is_groupable()` (`group_stage.py:39`) drops `unclear` findings, so every mislabelled prognostic relation silently disappeared before GROUP → CANONICALIZE → RELATE → RESOLVE. Symmetric symptom on the category field: LLMs emit `category="expression"` (the relation_type value) which fails Literal validation outright and the entire Finding is dropped by `AuditableSummary._drop_invalid_findings`. The calibration_set_v1 run on 2026-05-15 produced 10+ `prognosis` coercions on PMC9826086 alone and 4 fully-dropped findings with `category="expression"` — exactly the prognostic + IHC claims that should dominate a histopathology rule set. Fixed by adding `_RELATION_TYPE_ALIASES = {"prognosis": "prognostic"}` mirroring `_CATEGORY_ALIASES`, applied in `_coerce_invalid_relation_type` *before* the unknown-value branch, with `log_enum_observation(reason="alias_repair")` so the raw value still lands in `logs/enum_observations.jsonl`. Bumped `MAP_SCHEMA_VERSION` → `"map_v2_relation_type_alias_repair"` to invalidate the MAP cache. `category="expression"` deliberately left unmapped — collapsing to `IHC` would lose info because gene-expression claims could also belong in `molecular_genetics`; tracked as a follow-up. | [Bug 18](#bug-18--relation_type-prognosis-noun-form-coerced-to-unclear-instead-of-prognostic) |
| B-019 | Fixed (2026-05-15) | High | Summarisation, `MapOutputRouter` citation regex | `_CITATION_RE = r"^S\d+\|PMC\d+\|\d+$"` in both `routing/schema_validator.py:23` and `routing/provenance_validator.py:27` requires the PMC token to be `PMC` followed by *digits only*. The pipeline uses suffixed document IDs as opaque doc keys (`PMC10100421_HIS-82-393`, `PMC7150310_main`, `PMC4329418_his0066-0409`, …). Every citation emitted by every voter on such a paper contained an underscore-suffix, failed the regex, raised `ReasonCode.INVALID_SENTENCE_ID` for the voter, which sits in `_HARD_CODES` so the voter was classified UNUSABLE and dropped from the agreement matrix. Net effect on calibration_set_v1 (verified via `out/summaries/cascade_decisions/PMC6635746_HIS-73-68.jsonl`): every chunk ran with `voter_count=1` instead of 3 — the cheap-profile 3-voter consensus design was silently bypassed on every suffixed-pmcid paper since the router became default-on (2026-05-14). L1 acceptance rate looks artificially high because single-voter agreement gates pass trivially. Fixed by relaxing the PMC token to `[\w\-]+`: `_CITATION_RE = r"^S\d+\|PMC[\w\-]+\|\d+$"` (schema) and `r"^S(\d+)\|(PMC[\w\-]+)\|(\d+)$"` (provenance). Bumped `MAP_SCHEMA_VERSION` → `"map_v4_citation_regex_suffixed_pmcids"` to invalidate cached AuditableSummary results selected under single-voter routing. Cross-document equality check at `provenance_validator.py:116` unchanged — still exact comparison against `self._pmcid`, so the broader regex doesn't widen the safety net. | [Bug 19](#bug-19--mapoutputrouter-citation-regex-rejects-suffixed-pmcids-silently-strips-voters) |
| B-020 | Fixed (2026-05-15) | High | Summarisation, batch dispatch grouping | `dispatch.submit_level` groups requests by `provider` only (`dispatch.py:300-302`) before calling `provider.submit(reqs, …)`. OpenAI's Batch API requires **all requests in a single batch to use the same model** — any batch mixing models is rejected per-line with `BatchError(code='mismatched_model')`, leaving the batch status=`failed` with `output_file_id=None`. `openai_batch.OpenAIBatchProvider.check()` correctly detects this case and sets `job.status='failed'`, but the runner silently moves on. The cheap-profile L1 cascade has TWO OpenAI voters (`gpt-4o-mini`, `gpt-4.1-nano`) plus one Gemini voter; the OpenAI half-batch failed on every run since multi-model OpenAI L1 was introduced. Net effect: every batch run silently degraded to **single-voter L1 (Gemini only)**, the cheap-profile 3-voter consensus was bypassed, escalation reports under-stated cost by ~67% (no OpenAI tokens recorded), and OpenAI dashboard showed no charge because all requests were rejected pre-execution. Discovered when user noticed missing OpenAI cost; confirmed by inspecting `out/summaries/batch_handles/PMC4329418_his0066-0409.batch.json` (jobs: openai=failed with 30 requests, gemini=completed with 15) and pulling the OpenAI error file directly (`BatchError code='mismatched_model'`). Fixed by grouping requests by `(provider, model)` instead of `provider` in `dispatch.submit_level`. Conservative — works for every provider; adds at most one extra batch submission per profile. Bumped `MAP_SCHEMA_VERSION` → `"map_v5_batch_group_by_provider_model"` so cached batch handles with the broken job set re-submit cleanly. | [Bug 20](#bug-20--batch-dispatch-groups-by-provider-only-not-provider--model--openai-multi-model-batches-rejected-silently) |

| B-021 | Fixed (2026-05-15) | High | Summarisation, CANONICALIZE direction split | `_split_by_direction` and `_compute_scope_fields` in `canonicalize_stage.py` excluded the literal string `"None"` from the polarity-bearing set, but `DirectionEnum` has no such value — the real "doesn't apply" value is `"no_direction"`. MAP coerces missing direction to `no_direction` (`models.py:401`), so those findings got split into their own canonical bin, counted toward `is_conflicted`, and surfaced as rules with `direction=no_direction`. Fixed by replacing `"None"` with `"no_direction"` in both helpers + 9 new regression tests in `tests/summarization/test_canonicalize_direction_split.py`. | [Bug 21](#bug-21--canonicalize-no_direction-treated-as-real-polarity-due-to-none-string-typo) |
| B-022 | Fixed (2026-05-15) | High | Summarisation, GROUP bucket key | `group_stage._group_id` mixes namespaces: `subj_key = subject_cui if subject_cui else subject`. Two NormalFindings with identical normalized subject (e.g. `"CD30"`) where one has `subject_cui` populated and the other does not (intermittent UMLS link miss, or the synonym-dict path in `_resolve_entity` returning canonical name without CUI) land in different buckets — dedup defeated and downstream CanonicalRule count inflated. | [Bug 22](#bug-22--group_id-mixes-cui-and-string-keys-when-cui-population-is-partial) |
| B-023 | Fixed (2026-05-15) | Medium | Summarisation, NORMALIZE dedup | `_dedup_key` keys on `(text_element_id, subject, outcome, relation_type)` but not on `direction`. Findings extracted from the same sentence with opposing directions (positive vs. negative) collapse into one `NormalFinding`; `_merge` picks `rep.direction` (highest grounding wins) and the opposite-direction finding is silently absorbed. Docstring defends this as "contradictions surface at RELATE", but RELATE only ever sees the surviving direction so they cannot surface. Either include `direction` in the dedup key or carry per-direction histograms onto `NormalFinding` so CANONICALIZE can split. | [Bug 23](#bug-23--normalize-dedup-collapses-opposite-direction-findings-from-the-same-sentence) |
| B-024 | Mitigated (2026-05-15) | Low | Summarisation, RELATE → Relation schema | `Relation.nli_score_a_to_b` / `nli_score_b_to_a` field docstring says "entailment score from A→B direction", but `relate_stage._classify_pair` stores **contradiction** score for CONTRADICT and **entailment** for SUPPORT (`relate_stage.py:400-405`). DB columns and inspector scripts surface the field without label context (`scripts/run_paper_single_model.py:405` prints as `A→B={:.2f}`). Downstream readers cannot tell which score they're looking at. Either rename to a label-neutral name or update the schema doc — `RawNLIPair` already stores entailment and contradiction separately so no information loss either way. | [Bug 24](#bug-24--relationnli_score_-field-doc-disagrees-with-relate_stage-write-path) |
| B-025 | Observed | Low | Summarisation, RELATE polarity guard | `relate_stage._classify_pair` groups `DirectionEnum.partial` with `_POSITIVE_DIRECTIONS` for the same-polarity guard. `partial` sits between positive and unclear — bundling it with positive means a `partial`-vs-`positive` pair can never emit CONTRADICT even with high bidirectional contradiction scores (e.g. "focal positivity in a subset of cells" vs. "broadly positive expression"). `partial`-vs-`negative` is unaffected because the two sets are disjoint. Worth a calibration sweep against the gold set before flipping. | [Bug 25](#bug-25--relate-polarity-guard-treats-partial-as-positive-blocking-partial-vs-positive-contradictions) |
| B-026 | Superseded (2026-05-15) | Low | Summarisation, CANONICALIZE tie-break | Determinism hole in the unclear-folding policy. Superseded by [B-049](#bug-49--canonicalize-folds-unclear--no_direction-into-majority-polarity-bin) which removes the folding logic entirely, eliminating the `max(...)` tiebreak it depended on. | [Bug 26](#bug-26--canonicalize-split_by_direction-tie-break-is-member-order-dependent) |
| B-027 | Fixed (2026-05-15) | High | PDF extraction, `PipelineRunner` runtime knobs | All four `RuntimeConfig` knobs now consumed. `num_workers` + `log_level` wired earlier. `seed`: `PipelineRunner._seed_pipeline()` seeds `random` / `numpy` / `torch` (+ `torch.cuda` when available) at `__init__`; `seed` widened to `int \| None` so callers can opt out. Does **not** promise determinism for external libs (Docling/TATR/OCR/scispaCy). `skip_existing_outputs`: new `_StageCache` (`stage_cache.py`) caches stages 2 (table detection), 5 (artifact filtering), 6 (text assembly) to `out/stage_cache/<stage>/<pmcid>.json` with a config-hash sidecar. Sidecar / loader corruption falls through to recompute with WARNING; bugs propagate. Final writers (steps 7/8) still always run. Regression test in `tests/pdf_text_extraction/test_b027_seed_and_cache.py` (22 cases). | [Bug 27](#bug-27--runtimeconfig-knobs-num_workers-log_level-seed-skip_existing_outputs-not-consumed) |
| B-028 | Fixed (2026-05-15, deleted) | High | PDF extraction, DB ingester | `DatabaseConfig.{schema, create_tables_if_missing, batch_size, connect_timeout_sec}` had no consumers. Deleted, not wired — no current thesis demand for schema isolation, custom batching, or tunable connect timeouts; wiring four fake knobs would have multiplied DB-layer surface area for zero behaviour gain. `DatabaseConfig` now exposes only `enabled` + `db_url`. Loader's strict-unknown-key check rejects YAMLs referencing the removed fields (regression test in `tests/test_config_loader.py::test_deleted_database_keys_rejected`). | [Bug 28](#bug-28--databaseconfig-sub-fields-never-propagated-to-postgresdatabaseingester) |
| B-029 | Fixed (2026-05-15) | High | PDF extraction, scispaCy loader | `PipelineRunner._get_nlp` (`runner.py:199`) called `spacy.load("en_core_sci_sm")` directly, bypassing the documented `umls_resources.get_nlp()` singleton. If the PDF extraction and summarisation pipelines ran in the same process the small scispaCy model loaded twice (once here, once via `umls_resources` which loads `en_core_sci_lg`). Fixed by adding `umls_resources.get_small_nlp(model_name)` (process-wide per-model cache; honours `$NLP_HISTO_DISABLE_UMLS`) and routing `PipelineRunner._get_nlp` through it. Co-fixed with B-038. | [Bug 29](#bug-29--pipelinerunner_get_nlp-bypasses-umls_resources-singleton) |
| B-030 | Fixed (2026-05-15, deleted) | Medium | PDF extraction, filtering config | `FilteringConfig.{fix_ligatures, remove_reference_markers, min_paragraph_chars}` had no consumers. Dropped all three from the dataclass — current behaviour is "always fix ligatures, always strip citations, no minimum-length filter". Inverting any default would silently change extraction output across the corpus, so wiring was rejected in favour of deletion. `FilteringConfig` now exposes only the three knobs that actually drive code (`enabled`, `apply_ner_filtering`, `apply_paragraph_relevance_filtering`). | [Bug 30](#bug-30--filteringconfig-dead-knobs-fix_ligatures-remove_reference_markers-min_paragraph_chars) |
| B-031 | Fixed (2026-05-15, deleted) | Medium | PDF extraction, text assembly config | Six unread `TextAssemblyConfig` fields removed: `enabled`, `baseline_mode`, `use_hierarchical_extraction`, `use_context_aware_stitching`, `compare_combinations`, `save_combination_outputs`. Hierarchical extraction + stitching are hardcoded; the baseline-mode A/B harness is dead. `BaselineMode` enum and its package-level export also dropped. `TextAssemblyConfig` now exposes only `write_raw_text` and `pre_filter_relevance`. `tests/test_config_loader.py::test_enum_coerced_from_string` rewritten to use `LogLevel` instead of `BaselineMode`. | [Bug 31](#bug-31--textassemblyconfig-six-of-eight-fields-unread) |
| B-032 | Fixed (2026-05-15, deleted) | Low | PDF extraction, cropping config | `CroppingConfig.{include_captions_in_metadata, panel_counting_enabled}` removed — captions are unconditionally included in metadata, panel counting was wholly unimplemented. Remaining fields are all consumed by `PyMuPDFMediaCropper`. | [Bug 32](#bug-32--croppingconfig-dead-knobs-include_captions_in_metadata-panel_counting_enabled) |
| B-033 | Fixed (2026-05-15, deleted) | Low | PDF extraction, masking config | `MaskingConfig.merge_iou_threshold` was a leftover from a different algorithm — `merge_rects` (`parsers/layout_utils.py:103`) merges on any-intersection (`Rect.intersects`), not on IOU. Field deleted from `MaskingConfig` and `merge_rects` docstring tightened to clarify the semantics. | [Bug 33](#bug-33--maskingconfigmerge_iou_threshold-never-passed-to-merge_rects) |
| B-034 | Fixed (2026-05-15) | Medium | PDF extraction, TATR detector | Promoted hardcoded `_RENDER_DPI = 150` (`tatr_detector.py`) to a real `TATRConfig.render_dpi: int = 150` field consumed in `detect()`. Deleted four dead `TATRConfig` fields: `enabled` (redundant — `PipelineConfig.table_detector` already gates the detector), `max_detections_per_page`, `batch_size_pages`, `structure_model_name`. DPI is the load-bearing knob for the thesis recall sweep. | [Bug 34](#bug-34--tatrconfig-dead-knobs-render-dpi-hardcoded) |
| B-035 | Fixed (2026-05-15) | Medium | PDF extraction, Docling timeout | `DoclingConfig.timeout_sec=300` was documented but `DoclingLayoutExtractor` never wrapped the conversion. Pathological PDFs (very large / OCR-heavy / corrupt) hung the entire batch indefinitely. Fixed by routing the `converter.convert(...)` call through a new `_convert_with_timeout` helper that submits the work to a single-worker `ThreadPoolExecutor` and raises `TimeoutError` on `future.result(timeout=)`. The runner's per-paper try/except in `PipelineRunner.run_document` already blacklists the pmcid on exception, so a timeout naturally moves the batch on. `timeout_sec <= 0` disables the guard. The runaway thread is abandoned — acceptable trade-off for batch resilience. Regression test in `tests/pdf_text_extraction/test_docling_timeout.py`. | [Bug 35](#bug-35--doclingconfigtimeout_sec-never-enforced-by-doclinglayoutextractor) |
| B-036 | Fixed (2026-05-15) | Low | Summarisation, NLI helpers config surface | `GroundingFilter.__init__` (`helpers/grounding_filter.py:68`) accepts `model_name`, `batch_size`, `device`; `RelateStage.__init__` (`current_stages/relate_stage.py:268`) accepts the same trio. `GroundingConfig` exposes only `threshold`; `RelateConfig` exposes only the two thresholds. `runner.py:258` instantiates `GroundingFilter(cfg.grounding.threshold)` and `runner.py:251` instantiates `RelateStage(entailment_threshold=…, contradiction_threshold=…)` — model / batch / device always fall back to module defaults regardless of caller intent. No way to switch the NLI model or move it to GPU via `SummarizationConfig`. | [Bug 36](#bug-36--groundingfilter--relatestage-modelbatchdevice-not-exposed-via-summarizationconfig) |
| B-037 | Fixed (2026-05-15) | Low | Summarisation, normalize stage | Added `NormalizeConfig.extra_synonyms: dict[str, str] \| None` to `SummarizationConfig`; both sync and batch runners now pass `cfg.normalize.extra_synonyms` to `NormalizeStage(...)`. Side fix in the YAML loader: `_unwrap_optional` now also handles PEP-604 `X \| None` (was only matching `typing.Union`), and `_coerce` skips the nested-dataclass branch when the field type is a `dict[...]` mapping — without this, `extra_synonyms: {acme: ACME}` crashed with `Field type dict[str, str] \| None does not resolve to a dataclass`. New tests in `tests/test_config_loader.py`: `test_normalize_extra_synonyms_loaded_as_mapping`, `test_tatr_render_dpi_overridable`. | [Bug 37](#bug-37--normalizestageextra_synonyms-not-exposed-via-summarizationconfig) |
| B-038 | Fixed (2026-05-15) | Medium | Summarisation, sentence loader | `SummarizationRunner.load_paper_from_db` (`runner.py:905`) called `spacy.load("en_core_sci_sm")` on every invocation — bypassed the `umls_resources.get_nlp()` singleton. In batch mode (`process_batch([load_paper_from_db(p) for p in pmcids])`) the small model deserialised once per paper. Same class as B-029. Fixed by routing through `umls_resources.get_small_nlp("en_core_sci_sm")`; raises a clear error when the model isn't installed. Regression test `tests/summarization/test_scispacy_singleton.py` asserts no `spacy.load(...)` call sites exist outside `umls_resources.py` under `pipeline/stages/`. | [Bug 38](#bug-38--summarizationrunnerload_paper_from_db-bypasses-scispacy-singleton) |
| B-039 | Fixed (2026-05-15) | High | Summarisation, sentence ordering | `SummarizationRunner.load_paper_from_db` (`runner.py:912-916`) orders `TextElement` rows by `position_in_section` alone — but per `database/models.py:79` + the composite index `idx_document_path_position`, `position_in_section` is *local to each `path_string`*. Single-column sort interleaves sections: every section's position-0 paragraph emits first, then every section's position-1, etc. `MapStage._make_chunks` then packs adjacent sentences from unrelated sections into the same chunk, destroying topical locality and depressing voter agreement. Affects every paper on every sync + batch run today. Compounds with B-040 once those rows have already been written out-of-order to `TextElement`. | [Bug 39](#bug-39--load_paper_from_db-orders-by-position_in_section-only-interleaves-sections) |
| B-040 | Fixed (2026-05-15) | Medium | PDF extraction, text assembly | `parsers/layout_utils.extract_text` (`layout_utils.py:469-524`) accumulates paragraphs into `by_path = defaultdict(list)` keyed by `path_string`, then emits `rows` by iterating `by_path` in *insertion order*. Sections that get revisited after a sub-section (parent text → sub-section text → more parent text) have their later paragraphs appended at the parent's first-emit position; the sub-section's content ends up emitted *after* the entire parent block. Output `HierarchicalRow` order is "path-first-appearance" order, not document order, and the bug compounds with B-039 once those rows are written to `TextElement` and re-read by `load_paper_from_db`. | [Bug 40](#bug-40--extract_text-emits-paragraphs-in-path-first-appearance-order-not-document-order) |
| B-041 | Fixed (2026-05-15) | High | Summarisation, MAP cascade attribution | `MapStage._run_voters` (`current_stages/map_stage.py:1183`) returns the API-survivor list with `[r for r in results if r is not None]` — the original-voter-index → survivor-index mapping is dropped at return. `agreement.compute(voters)` then assigns `bundle.best_index` over the survivor list, and `producer_from_outcome` (`agreement/decision.py:210-211`) / `make_decision_record` (`decision.py:255-267`) use that index as if it referred to the original `voter_specs`. Router path is also affected: `_classify_voters` (`routing/router.py:271`) re-indexes from 0 over the survivor list, so its `valid_voter_indices` are survivor-list indices that `voter_specs[global_idx]` then treats as original indices. Whenever ≥1 voter fails an API call (or the router strips a voter as UNUSABLE before MAP sees the failure), MAP cache metadata, cost report, and cascade decision log all carry the wrong `(provider, model)` for the kept chunk. Dormant when zero voters fail. | [Bug 41](#bug-41--producer-attribution-mis-indexed-when-any-voter-fails) |
| B-042 | Fixed (2026-05-15) | Low | PDF extraction, text stitching | `ContextAwareStitcher._is_cut_off` (`parsers/text_processing.py:152-181`) returned `False` on any terminal `.`/`?`/`!`/`)`/`]`/`"`/`'`/`»` early-return at lines 152-154, *before* the `_MID_SENTENCE_ABBREVS` check at lines 175-181. Every abbreviation in that frozenset (`fig.`, `et al.`, `vs.`, `approx.`, `e.g.`, `i.e.`, `cf.`, `ref.`, `refs.`, `dept.`, `no.`, `nos.`) ends in a period, so the abbrev rule was dead code. Paragraphs ending in those abbreviations were treated as sentence-final and never stitched with the next narrative paragraph — sentences got fragmented at abbreviation boundaries, biasing both MAP input and downstream NLI grounding. Fix moved the abbreviation check ahead of the period early-return and added a multi-token form so "et al." (two tokens) also triggers. | [Bug 42](#bug-42--is_cut_off-mid-sentence-abbreviation-rule-is-dead-code) |
| B-043 | Fixed (2026-05-15) | Low | PDF extraction, citation removal | `parsers/text_processing.remove_citations` regex `(?<!\n)\.\s+\d+(?:[,–\-]\d+)*(?=\s|$)` stripped any `". <digits> "` pattern after a period — including 4-digit years. "Smith et al. 2020 reported …" became "Smith et al. reported …", losing claim context. Fixed by capping the citation-index run at 1–3 digits in all three after-period / after-comma / standalone branches: `\d+` → `\d{1,3}`. Citation indices in pathology papers are practically never ≥1000 (one bracket-style branch was left as `\d+` because brackets disambiguate years from indices). Regression tests in `tests/parsers/test_remove_citations.py` cover year preservation + citation stripping. | [Bug 43](#bug-43--remove_citations-strips-publication-years) |
| B-044 | Mitigated (2026-05-15) | Medium | Summarisation, MAP relation_type | MAP voters bleed `category` values (`morphology`, `IHC`, `molecular_genetics`, `prognosis`, `treatment`, `staging`) into the `relation_type` field. Prior coercion mapped only `prognosis → prognostic` and `treatment → treatment_response`; the rest fell through to `unclear` and got dropped at GROUP (relation_type is part of the grouping key and `unclear` is non-groupable). Net effect: 10+ findings silently lost per run on the calibration set, concentrated in `molecular_genetics` and `IHC` claims. Mitigation: prompt anti-pattern line + prognostic-crossover example + extended `_RELATION_TYPE_ALIASES` (`morphology→has_feature`, `ihc→expression`, `molecular_genetics→expression`) + new `reason="cross_field_bleed"` JSONL counter for measurement. `staging` left unaliased (descriptive vs prognostic crossover needs claim context). | [Bug 44](#bug-44--map-relation_type-bleeds-category-names-and-loses-findings-at-group) |
| B-045 | Fixed (2026-05-15) | Low | Summarisation, MAP `FindingScope.scope_parsed` | `scope_parsed` is trivially derivable (`any(sub_field is not None)`) but was being computed by the LLM. One more thing it could get wrong, and output tokens spent reasoning about it. Fixed by `@model_validator(mode="after")` on `FindingScope` (`models.py:148-159`) that overrides whatever the LLM emitted; prompt instruction updated to "always emit false — computed automatically" (`prompts.py:213`). Field stays in the schema because OpenAI strict mode requires every property to be present. Bumped `MAP_SCHEMA_VERSION` → `"map_v7_scope_parsed_autocompute"`. Regression test in `tests/summarization/test_scope_parsed_autocompute.py`. From [MAP_PROMPT_AUDIT Issue 5](MAP_PROMPT_AUDIT.md#issue-5--scopescope_parsed-is-llm-set-but-trivially-derivable-low). | [Bug 45](#bug-45--scope_parsed-is-llm-set-but-trivially-derivable) |

| B-046 | Fixed (2026-05-15) | Low | Summarisation, MAP direction enum | Hedging words (`maybe`, `possibly`, `none`, `n/a`) outside `DirectionEnum` fell through to the unknown-value branch and coerced to `unclear`, losing the polarity-vs-uncertainty distinction in the live enum (raw still on `_raw_direction` per B-015). Added `_DIRECTION_ALIASES` mapping hedging → `unclear` and `none`/`n/a` → `no_direction`; alias-repair branch in `_coerce_invalid_direction` logs `reason="alias_repair"`. Bumped `MAP_SCHEMA_VERSION` → `"map_v8_direction_alias_repair"`. Tests in `tests/summarization/test_enum_alias_repair.py`. From [MAP_PROMPT_AUDIT Issue 8](MAP_PROMPT_AUDIT.md#issue-8--directionmaybe-single-occurrence-low). | [Bug 46](#bug-46--direction-hedging-words-coerce-to-unclear-instead-of-alias-repair) |
| B-047 | Fixed (2026-05-15) | Low | Summarisation, MAP direction prompt | Prompt example mapped `"BCL2 was negative"` to `direction=absent` but both `negative` and `absent` plausibly applied for expression-context negation. Same FindingGroup could end up with both labels on opposite-polarity findings, blocking RELATE's CONTRADICT signal. Added a disambiguating rule under the `direction` definition (`expression`-only: `negative staining`/`no expression` → `absent`; `decreased`/`reduced` → `negative`; other relation_types: prefer `negative`, reserve `absent` for literal "absent"/"not present"/"lacking"). Bumped `MAP_PROMPT_VERSION` → `"map_prompt_v5_expression_absent_vs_negative"`. From [MAP_PROMPT_AUDIT Issue 6](MAP_PROMPT_AUDIT.md#issue-6--directionabsent-vs-directionnegative-ambiguity-in-expression-contexts-low). | [Bug 47](#bug-47--direction-absent-vs-negative-ambiguity-on-expression-claims) |
| B-048 | Fixed (2026-05-15) | Low | Summarisation, optional RULE block enums | `Rule.type` was `Literal["Diagnostic", "Prognostic", "Management"]` (Title-Case) and `RuleCounts` mirrored the casing in field names — inconsistent with the lowercase `Finding.confidence` / `Finding.category` convention. Lowered all three; added a `mode="before"` validator on `Rule.type` so legacy Title-Case payloads round-trip. Updated MAP RULE OutputFormat prompt + `_recompute_audit` helper. RULE block is off by default, no DB rows to backfill. Tests in `tests/summarization/test_enum_alias_repair.py`. From [MAP_PROMPT_AUDIT Issue 7](MAP_PROMPT_AUDIT.md#issue-7--ruletype-is-title-case-diagnosticprognosticmanagement-everything-else-lowercase-low). | [Bug 48](#bug-48--ruletype-title-case-inconsistent-with-lowercase-convention) |
| B-049 | Fixed (2026-05-15) | Medium | Summarisation, CANONICALIZE direction policy | `_split_by_direction` folded `unclear` and `no_direction` members into the largest polarity bin. Two holes: (a) **reproducibility** — `max(non_unclear, key=len)` returns the first dict key on ties, traceable to upstream member-arrival order, so the same paper produced different `member_normal_ids` / `finding_count` / `mean_grounding_score` across re-runs (supersedes B-026); (b) **honesty** — hedged findings got re-cast as votes for the majority direction, inflating downstream confidence and feeding RELATE pairs as if the model had really claimed that polarity. Fixed: every observed direction gets its own `CanonicalRule` bin (no folding); RELATE and corpus_relate skip pairs where either side is `unclear` / `no_direction`; `is_conflicted` repurposed to a **group-level** signal (True iff the group emits ≥2 polarity-bearing bins, stamped on every rule from the group). Added `direction_value`, `POLARITY_BEARING_DIRS`, `NON_POLARITY_DIRS` to `models.py` as the single source of truth; gates use the normalizer so `DirectionEnum` / raw string / `None` paths all behave the same. `partial` deliberately kept polarity-bearing for now — the semantic question of whether partial really conflicts with positive is owned by B-025. Bumped `CANONICALIZE_DIRECTION_POLICY_VERSION` (fed into `pipeline_config_hash`) to force cache invalidation. Tests: rewritten `tests/summarization/test_canonicalize_direction_split.py` (16 cases incl. S5 core invariant against unclear leakage into polarity bins), new `tests/summarization/test_corpus_relate_non_polarity.py`, extended `tests/summarization/test_relate_skipped_pairs.py`, `tests/summarization/test_pipeline_config_hash.py`. | [Bug 49](#bug-49--canonicalize-folds-unclear--no_direction-into-majority-polarity-bin) |
| B-050 | Fixed (2026-05-15) | Low | Scripts, batch poll interval | `scripts/run_paper.py` carried three diverging defaults for `--poll-interval`: argparse `60` (line 331), `_run_all_batch(poll_interval=20)` (line 787), `_run_batch(poll_interval=60)` (line 885). CLI flows passed `args.poll_interval` so the call-site defaults rarely fired — but a direct programmatic caller of either batch helper got 20s or 60s depending on which one they imported. Consolidated onto module-level `DEFAULT_POLL_INTERVAL_SEC = 60` referenced by argparse + both function signatures. Regression: `tests/test_poll_interval_defaults.py` introspects via `inspect.signature` (not `__defaults__` tuple indexing) and asserts all three resolve to 60. **Note**: if another agent's parallel work also claims B-050, renumber to B-051 at commit time. | [Bug 50](#bug-50--poll_interval-default-mismatch-across-cli-and-batch-helpers) |
| B-057 | Fixed (2026-05-17) | High | PDF extraction, committed merge-conflict markers | Three files on `eval-speedrun` HEAD shipped with unresolved git merge-conflict markers (`<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes`): `pipeline/stages/pdf_text_extraction/components/visualizer.py` (lines 113–121), `pipeline/stages/pdf_text_extraction/table_detectors/tatr_detector.py` (lines 68–78), and `eval/run.py` (lines 120–125). `components/__init__.py:7` and `table_detectors/__init__.py:3` re-export these modules eagerly, so any pipeline import (e.g. `DoclingLayoutExtractor`) crashed with `SyntaxError`. Discovered while running the Stage-1 observability-patch smoke test (2026-05-17). Resolved by picking the "Updated upstream" branch in each file: (1) visualizer — `setdefault(pg, []).append(...)`, semantically identical to the alternative; (2) tatr — preserves configurable `self._config.device` per B-034 (the alternative hardcoded `to("cpu")` plus `low_cpu_mem_usage=False, device_map=None` kwargs that were a transformers-loading workaround no longer needed); (3) eval/run.py — log format `"Eligible PDFs: %d / %d (%.1f–%.1f MB)"` matching the actual min+max byte filter applied earlier in the same function. All three resolutions are no-op behaviour changes relative to the documented intent of the surrounding code. | [Bug 57](#bug-57--committed-merge-conflict-markers-in-visualizerpy) |
| B-056 | Observed (2026-05-16) | Medium | Summarisation, batch runner → `sum_map_voter_outputs` | `BatchSummarizationRunner.finalize()` has no code path that buffers per-voter `AuditableSummary` rows or writes them to `sum_map_voter_outputs`; no `_persist_voter_outputs` method exists on the batch class. Discovered during the Phase 0 audit per [`CALIBRATION_EXECUTION_PLAN.md`](CALIBRATION_EXECUTION_PLAN.md) §10. Blocks θ / reject_θ sweeps over batch-processed papers; sync runs are unaffected. B-055's empirical claim that `sum_map_voter_outputs` was populated on batch runs 38–48 contradicts code-level inspection and requires runtime re-verification on a paper never previously processed via sync. | [Bug 56](#bug-56--batch-runner-omits-per-voter-map-persistence-code-path-absent) |
| B-055 | Mitigated (2026-05-23) | High | Summarisation, batch runner → `sum_map_findings` | `sum_map_findings` rows missing for 9 of the last 10 batch-mode pipeline runs (ids 38–48 across 5 papers), despite each paper's `rejection_summary.map_findings_total` recording 100–230 MAP findings produced. Other `sum_*` tables (`sum_normal_findings`, `sum_finding_groups`, `sum_canonical_rules`, `sum_final_rules`, `sum_rejection_summaries`, `sum_map_voter_outputs`) get rows on the same runs, so the DB connection + `pipeline_run_db_id` + persistence wiring are not at fault. The function itself works: a direct call to `persist_map_findings(db, 48, pmcid, chunk_summaries)` against the same paper's batch handle on disk wrote 154 rows successfully, with `verbatim_support` exactly matching `text_elements.text_content`. Bug pre-dates the 2026-05-16 B-005 dedup (same behaviour on HEAD `7ea254a`); the dedup rewrote the wrapper but the production failure mode was already present. Suspect call-path issue inside `BatchSummarizationRunner.finalize()` between L483 (`chunk_summaries = [AuditableSummary.model_validate(v) for v in handle.finalized.values()]`) and L522 (`self._persist_map_findings(...)`) — either `chunk_summaries` is empty at the call site (which would also break downstream `all_findings = [f for cs in chunk_summaries for f in cs.findings]` at L536, contradicted by 214 `sum_normal_findings` rows on run 48), or an exception in the bulk `INSERT` is being swallowed by `except Exception as exc: logger.warning(...)` and the run's stdout went unrecorded. Adjacent inconsistency: runs 47/46/44/43 wrote `sum_canonical_rules` (100+ rows) with **zero** `sum_normal_findings` — physically impossible from the in-process flow, suggests these were cache short-circuits whose `_load_result` path skipped MAP/NORMALIZE persistence but still wrote canonical/final from the cached JSON. | [Bug 55](#bug-55--sum_map_findings-not-populated-by-batch-runner) |
| B-054 | Fixed (2026-05-16) | High | Summarisation, NER stage scispaCy singleton bypass | `named_entity_recognition/ner.py`'s `load_ner_model()` and `load_linker_model()` issued direct `spacy.load("en_core_sci_lg", …)` calls — completely bypassing the `umls_resources.get_nlp()` singleton documented in CLAUDE.md and MEMORY.md. `SummarizationRunner._run_stages` calls `run_ner_on_db(pmcid, save_to_db=True, force=False)` per paper *without* passing `nlp=` / `linker_nlp=`, so the loaders fired with `None` defaults and freshly loaded ~2.6 GB of scispaCy + UMLS twice per paper. Concretely visible in production runs: `[pmcid] NER done [136.4s]` on a paper that bailed out (already had entities) — the time was pure model-load waste. Compounded by `umls_resources.get_nlp()` already holding its own copy for NORMALIZE / UMLS_ENRICH, so peak RSS hit ~3 copies of en_core_sci_lg in memory. Same class of bug as B-029 (PDF runner) / B-038 (summariser load_paper_from_db), missed because `named_entity_recognition/` lives outside `pipeline/stages/` and the existing singleton-guard test only scanned the stages tree. Fixed by routing both loaders through `umls_resources.get_nlp()`; the "fast NER" pass wraps the span-extraction loop in `nlp.select_pipes(disable=["scispacy_linker"])` so it stays cheap on the linker-attached singleton; the "Document already has entities" skip check moved *above* the model-load block so a skipped paper now costs ~0 s (was ~150 s). Regression test in `tests/summarization/test_scispacy_singleton.py::test_ner_module_routes_through_singleton` asserts neither loader contains a direct `spacy.load(` call. `batch_ner.py` inherits the fix automatically (it imports the same functions). | [Bug 54](#bug-54--ner-stage-scispacy-singleton-bypass) |
| B-053 | Fixed (2026-05-16) | Low | Tooling, percentiles cost estimator | `scripts/estimate_pipeline_cost_percentiles.py` hygiene cluster — dead `import json` + `import statistics`; two never-called helpers (`estimate_non_llm_stages`, `render_paper_table`) that duplicated the markdown rendering inline in `main()`; misleading inline comment claiming `est_chunks = ceil((n - overlap) / stride)` while the code (correctly, matching `MapStage._make_chunks`) did `ceil(n / stride)`; `pick_percentile` had no guard for an empty paper list (`idx = ceil(0.5*0) - 1 = -1` → silent off-end indexing) or out-of-range `p`; `CHUNK_SIZE`/`CHUNK_OVERLAP` duplicated as module constants instead of sourced from `MapConfig.chunk_size`/`chunk_overlap`, leaving a silent drift hazard if production config changes. Fixed: dead imports + helpers removed, comment rewritten to cite `_make_chunks`, `pick_percentile` raises `ValueError` on empty corpus / `p ∉ (0, 1]`, chunk constants now read off `MapConfig()` at module load. Numbers unchanged — none of this moved the printed cost (the percentiles report is an intentional upper-bound budget for the top-decile/P80–P90 papers by `n_te`). Regression test `tests/test_estimate_pipeline_cost_percentiles.py` (15 cases). | [Bug 53](#bug-53--percentiles-cost-estimator-hygiene-cluster) |
| B-052 | Fixed (2026-05-16) | Medium | Tooling, cost estimation script | `scripts/estimate_selection_cost.py:per_chunk_input_tokens` modelled the average sentences per MAP chunk as `min(chunk_size, n_sentences / n_chunks * (1 + 0))`. At production defaults (`chunk_size=10`, `chunk_overlap=2`, stride=8) `n_sentences / n_chunks ≈ stride = 8`, so the clamp returned ~8 sentences per chunk — but `MapStage._make_chunks` (`map_stage.py:1263-1267`) slices `sentences[i:i+chunk_size]` so each non-tail chunk actually sees 10 sentences. The trailing `* (1 + 0)` was a leftover from a removed overlap term. Net: every cost number in the projection table was ~15–20% low, exactly the headline figure a thesis budget review reads. Fixed by replacing the formula with a sum over `min(chunk_size, n_sentences - start) for start in range(0, n_sentences, stride)` divided by `n_chunks` — matches `_make_chunks` line-for-line, accounts for the truncated tail, and rounds with `ceil` for conservative budget estimates. Function signature gained `chunk_overlap` (caller updated); also validates `0 <= chunk_overlap < chunk_size`. Co-fixed in the same change: `.order_by(TextElement.position_in_section)` (the B-039 bug) flipped to `.order_by(TextElement.id)` to actually mirror `SummarizationRunner.load_paper_from_db`. Dead `import math` cleaned up. Regression test in `tests/test_estimate_selection_cost.py`. | [Bug 52](#bug-52--cost-estimation-script-underestimates-per-chunk-input-tokens) |
| B-051 | Fixed (2026-05-15) | High | Summarisation, MAP agreement gate | `EmbeddingScorer._polarity` applied only a 20% multiplicative penalty; opposite-polarity paraphrases with cos≈1.0 produced score=0.80, passing `theta=0.7` and accepting the chunk as KEEP despite a direct voter contradiction. Fixed: new pure helper `agreement/polarity_conflict.detect_polarity_conflict` invoked from `AgreementChecker.compute` after the scorer runs but before theta — when two **comparable** findings (same `subject_entity` / `outcome_entity` / `relation_type` / `category`, all four required, strings `.strip().casefold()`d) carry opposite `{positive, negative}` directions, decision is forced to `ChunkDecision.ESCALATE` with `score_details["hard_fail_reason"] = "polarity_conflict"`. `MapOutputRouter._agreement_gate` emits ONLY `ReasonCode.POLARITY_CONFLICT` (never co-emits low-agreement codes — the score was high; only the structural check failed); explanation makes the override explicit. v1 conservative: scope fields excluded from comparability (cross-cohort false-escalate cheaper than missed contradiction); `absent`/`partial`/`unclear`/`no_direction` excluded from the hard-polarity set pending B-025 calibration. Cache invalidation: bumped `MAP_SCHEMA_VERSION` → `"map_v9_polarity_hard_fail"` (invalidates `PipelineCache`); added `MAP_AGREEMENT_POLICY_VERSION = "polarity_hard_fail_v1"` routed into `compute_pipeline_config_hash` on both runners (invalidates per-paper result cache). 11 deterministic regression tests in `tests/summarization/agreement/test_b051_hard_fail_polarity.py` + 3 hash regression tests in `tests/summarization/test_pipeline_config_hash.py`. | [Bug 51](#bug-51--map-agreement-gate-treats-opposite-polarity-as-soft-disagreement) |

| B-058 | Fixed (2026-05-20) | Medium | PDF extraction, media cropper | `MaskingConfig.drop_tables_inside_figures` runs at Step 2 (post-detection filter in `runner.py::_drop_tables_inside_figures`) but the dropped table re-enters at Step 7 via the cropper's supplementary source (`media_cropper.py:245` — iterates layout TABLE/RECONSTRUCTED_TABLE elements and re-adds any that don't overlap an existing detection). When the Step-2 drop removes a `table_in_figure` FP, that table is no longer in `detection.regions` → no overlap match in cropper → re-added. Final `source` field is `"docling"` (one source) instead of `"docling+docling"` (two), confirming the bypass. Discovered while inspecting variant 18 (`drop=ON`) on PMC11791726/p9 — the FP was still emitted as Table_4_p9 despite `table_regions_dropped_inside_figures=1` in the run metadata. Fixed: `media_cropper.crop()` takes a new `drop_tables_inside_figures: bool = False` parameter; when True it skips layout TABLE elements ≥0.8 inside any FIGURE/PICTURE on the same page (same threshold as Step 2). `runner.py` plumbs `self._cfg.masking.drop_tables_inside_figures` into all three `cropper.crop()` call sites (main + two multi-source crops). No new config field — single `MaskingConfig.drop_tables_inside_figures` flag now governs both Step-2 detection filter and Step-7 supplementary-source filter. Pre-fix variant 18 had drop=ON behaving identically to drop=OFF on docling, invalidating its Stage 3 verdict (treated as "no effect" — actually the bug). | [Bug 58](#bug-58--drop_tables_inside_figures-bypassed-by-cropper-supplementary-source) |

| B-059 | Observed (2026-05-21) | Medium | PDF extraction, figure cropping | Decorative icons emitted as figure crops.  ~70% of figure-side error labels across all variants are `icon` (304 of 437 figure errors aggregated across 16 variants on the 28-PDF corpus).  These are small image-like layout elements that Docling correctly identifies as PICTURE/FIGURE elements geometrically but that are decorative graphics (publisher logos, small inline ornaments, watermark-style icons), not scientific figures.  Crops are emitted (FP for figure-output) but masking is correct (image content shouldn't appear in body text either way).  No current pipeline stage filters them.  Possible fix path: heuristic filter based on size (`min_figure_pts` already exists in `CroppingConfig` but doesn't address this — icons can be moderately sized), bbox aspect ratio, low text density inside the bbox, or appearance on every page of a multi-page paper (publisher logo case).  Need a figure-side analogue of `_drop_tables_inside_figures` or a stand-alone `drop_icon_figures` filter.  Decision/scope: outside the 2026-05-21 thesis-day budget; document as known limitation in `docs/THESIS.md` future-work section. | [Bug 59](#bug-59--decorative-icons-emitted-as-figure-crops) |
| B-060 | Observed (2026-05-21) | Medium | PDF extraction, caption parser | Cluster of `nearest_caption()` + `parse_caption_num()` defects in `parsers/layout_utils.py`.  Six recurring failure modes seen across all variants of the 28-PDF corpus (counts are aggregated label occurrences across variants):  (1) **Rotated-image footnote-as-caption** — 49 table cases (`wrong caption (footnotes matched to captions, rotated image)`); attacher pulls footnote text up because the image rotation messes up vertical proximity ordering.  (2) **Continuation-marker parsed as new table number** — 30 table cases; `(continued)` next to a table caption is parsed as the table's number by `parse_caption_num()` / `TAB_NUM_RE`.  (3) **Caption "Table N" prefix dropped** — 22 table cases; parser returns the descriptive caption body without the "Table N" identifier, breaking strict-match scoring on the caption dim.  (4) **Multi-caption merge across page boundaries** — 30 table cases; when a continued table caption sits adjacent to the next table's caption, the attacher concatenates them.  (5) **Side-mounted figure caption missed** — 19 figure cases (`no caption, caption is to the right of the figure`); spatial proximity heuristic in `nearest_caption()` doesn't handle 2-column layouts with side-mounted captions.  (6) **Page footer treated as caption** — 38 figure cases combined (bottom-left + bottom-right variants); attacher confuses page footers with figure captions when the real caption is on a different position of the page.  These bugs are interconnected via the shared caption-attacher logic — touching one risks regressing another, so they need a focused investigation rather than ad-hoc patches.  Affects ~18% of table errors and ~17% of figure errors; second-largest unaddressed bucket after `should be masked` for tables and `icon` for figures.  Decision/scope: outside the 2026-05-21 thesis-day budget; document as known limitation. | [Bug 60](#bug-60--caption-parser-bug-cluster) |
| B-061 | Observed (2026-05-21) | Low | PDF extraction, table cropping geometry | `crop too small minor` family of labels — 16 aggregated table cases across all variants (8 with caption issue + 6 stand-alone + 2 unmasked-letters variant).  Tables whose emitted crop bbox is smaller than the true table extent, missing some content.  Currently no config knob or filter tests this.  Symmetric to `crop too big` (which the Stage 5 `footnote_multiplier` sweep partially addresses).  Possible fix: dilate detection bboxes by a small margin before cropping, gated by a new `CroppingConfig.table_crop_dilation_pts` field, sweep values in {0, 2, 4, 8}.  Risk: dilation increases overlap with adjacent layout elements (captions, footnotes — though expand_tables_with_footnotes handles the latter). Decision/scope: outside the 2026-05-21 thesis-day budget; document as known limitation. | [Bug 61](#bug-61--crop-too-small-table-geometry-no-config-knob) |
| B-062 | Fixed (2026-05-23) | High | Summarisation, MAP cascade / config | `scripts/run_paper.py` never set `enable_router`, so both production entry points (`build_runner`, `build_batch_runner`) used the runner default `False` → **production ran the legacy `AgreementChecker` cascade, not `MapOutputRouter`**. Yet THESIS.md asserted router-on production in four places + `eval/silver/map_theta_sweep.py` in three comments. Documented production cascade ≠ actual behaviour. Found while wiring the config pin (calibration review item 3). Fix: path made config-governed (`summarization.map.enable_router`, default `false`) + logged at load; user decided production keeps the legacy L1→L2→L3 cascade (the router L1→L3-skip path is opt-in/experimental); stale docs corrected. | [Bug 62](#bug-62--documented-router-on-production-cascade-never-actually-enabled) |
| B-063 | Fixed (2026-05-24) | Low | Tooling, cost estimation script | `scripts/estimate_selection_cost.py` imports `from pipeline.stages.summarization.costing import PriceBook` inside `main()` but never bootstraps the repo root onto `sys.path`. Run as the documented bare `python scripts/estimate_selection_cost.py …` (HOW_TO_RUN §5) it dies with `ModuleNotFoundError: No module named 'pipeline'` — Python puts the script's own dir (`scripts/`) on `sys.path`, not the CWD. Only `PYTHONPATH=. python …` worked, which the script's own docstring documented. Every sibling script (`run_paper.py`, `check_apis.py`) self-bootstraps with `_REPO_ROOT = Path(__file__).resolve().parents[1]`; this one was the lone violator of the CLAUDE.md "scripts must bootstrap their own path" convention. Hit while running the cost estimate for `related15_full`. Distinct from [B-052](#bug-52--cost-estimation-script-underestimates-per-chunk-input-tokens) (same file, token-formula fix). Fixed by adding the standard 3-line bootstrap before the first repo import + dropping `PYTHONPATH=.` from the docstring. | [Bug 63](#bug-63--estimate_selection_costpy-missing-syspath-bootstrap) |

Add new rows here when you discover something. Bump the ID monotonically (`B-051`, `B-052`, …). Put the long write-up in a new `## Bug N — …` section below.

---

## Bug 61 — `crop too small` table geometry, no config knob

### Status / Severity / Surface
Observed (2026-05-21) · Low · PDF extraction, table cropping geometry.

### Symptom
16 aggregated table-error labels of the form `crop too small minor, …`
across all 16 variants on the 28-PDF corpus.  The emitted table crop is
smaller than the true table extent, missing some rows or columns.
Caption + footnote dims are correct; only the crop dim is hurt.

### Evidence
Label counts (aggregated across all variants):
* 8 × `crop too small minor, loses some info, caption missed "table" prefix`
* 6 × `crop too small minor, loses some info`
* 2 × `crop too small minor, might cause unmasked random letters in text`

Total: 16 instances (~2% of all table errors).

### Diagnosis
No config knob currently exists for tightening or loosening the table
detection bbox post-detection.  The crop bbox comes directly from the
table detector (Docling/TATR/Hybrid).  The pipeline's `expand_tables_with_footnotes`
extends the crop downward to absorb footnotes, but it doesn't dilate the
table bbox upward / sideways.

### Fix (planned, not implemented)
Add a new `CroppingConfig.table_crop_dilation_pts: float = 0.0` field;
when > 0, dilate each table bbox by that many points in all four
directions before cropping.  Sweep values in {0, 2, 4, 8} as a new
stage variant block (or fold into Stage 5 `merge_flags` as a single-flag
flip on `BEST_BASE`).

### Risk
Dilation could increase overlap with adjacent captions or footnotes.
The existing `_drop_tables_inside_figures` and
`expand_tables_with_footnotes` filters handle the figure / footnote
boundary, but a dilation on top could re-introduce overlap with the
caption itself (already attached separately).  Need to test on the 16
existing `crop too small` cases.

### Decision / Scope
Outside the 2026-05-21 thesis-day budget.  Document as known
limitation in `docs/THESIS.md` future-work section.

---

## Bug 60 — Caption-parser bug cluster

### Status / Severity / Surface
Observed (2026-05-21) · Medium · PDF extraction, caption parser
(`parsers/layout_utils.py::nearest_caption` and `::parse_caption_num`).

### Symptom
Six recurring failure modes in caption attachment / parsing.  Counts are
aggregated label occurrences across all 16 variants on the 28-PDF
corpus.

| Mode | Cases | Label |
|---|---|---|
| Rotated-image footnote→caption | 49 | `wrong caption (footnotes matched to captions, rotated image)` |
| `(continued)` parsed as table number | 30 | `wrong auto table number from continuation header` |
| Multi-caption merge across page boundary | 30 | `wrong caption (continued table, caption merged with prev table's caption)` |
| Page footer mistaken for figure caption (bottom-right) | 19 | `correct figure, wrong caption (real caption in bottom right, confused page footer with caption)` |
| Page footer mistaken for figure caption (bottom-left) | 19 | `correct figure, wrong caption (real caption in bottom left, confused page footer with caption)` |
| Detected caption too long (duplicated content) | 19 | `the detected caption is too long and has copies of the real one.` |
| Side-mounted figure caption missed | 19 | `no caption, caption is to the right of the figure` |
| Caption "Table N" prefix dropped | 22 | `caption missed "table" prefix, otherwise correct` |

Total: ~207 label instances ≈ 18% of all table errors + 17% of all
figure errors.  Second-largest unaddressed error bucket overall.

### Evidence
All six modes share root causes in two functions:
* `nearest_caption(element, captions)` (`parsers/layout_utils.py`)
  uses spatial proximity (vertical distance) to attach a caption to a
  figure/table element.  Fails when:
  - Element is rotated (rotated-image bug → vertical proximity is
    measured in unrotated coords, so footnote text below the table —
    which is "above" in the rotated frame — gets pulled as the caption).
  - Page footer text sits closer to the figure than the real caption
    (figure footer-as-caption bugs).
  - Caption is side-mounted (2-column layouts with caption in the
    adjacent column → spatial-proximity heuristic picks the wrong text).
  - Two captions sit adjacent across a page break (continued-table
    bug → attacher merges them).
* `parse_caption_num(caption, regex)` (`parsers/layout_utils.py`) uses
  `TAB_NUM_RE` / `FIG_NUM_RE` to extract the table/figure number.
  Fails when:
  - Caption starts with `(continued)` rather than `Table N` (parser
    grabs whatever number appears next, often the next figure's number).
  - Caption lacks the `Table N` prefix entirely (parser returns no
    number, "table" prefix label is dropped from caption text).

### Diagnosis
The two functions are tightly coupled: `nearest_caption` decides WHICH
text becomes the caption, then `parse_caption_num` extracts the
identifier.  A fix to either can regress the other:
* Tightening vertical-proximity heuristics in `nearest_caption` to
  handle rotation would shift the attachment for non-rotated cases.
* Adding `(continued)` handling to `parse_caption_num` would require
  the attacher to ALSO recognise continuation markers as belonging to
  a different (already-emitted) table.

The interconnection means a focused investigation is needed, not
ad-hoc patches.

### Fix (planned, not implemented)
* Add rotation-aware vertical-proximity logic to `nearest_caption` —
  detect rotated layout elements (Docling element has `rotation` or
  page-coord orientation differs) and rotate the bbox into a "caption
  reading frame" before measuring distances.
* Handle `(continued)` markers in `parse_caption_num` by returning
  `None` (signalling "continuation, defer to previous emit") instead
  of grabbing the next number on the line.
* Score caption candidates by spatial bracket (above / below /
  side / footer) and reject footer candidates when the figure bbox
  doesn't touch the page-footer band.
* Add a "caption deduplicator" pass that detects when two adjacent
  captions reference the same table number and merges them into a
  single attribution to the FIRST table.

Each of these fixes warrants its own bug entry once scoped.

### Risk
High — touching either function risks regressing the other 80+% of
captions that are currently being attached correctly.  Needs a
fixture-based regression suite (sample of correct captions across the
28-PDF corpus) before any code change.

### Decision / Scope
Outside the 2026-05-21 thesis-day budget.  Document as known
limitation in `docs/THESIS.md` future-work section.  When time permits,
spend a focused day on `nearest_caption` + `parse_caption_num` with a
golden-fixture regression suite built up front.

---

## Bug 59 — Decorative icons emitted as figure crops

### Status / Severity / Surface
Observed (2026-05-21) · Medium · PDF extraction, figure cropping.

### Symptom
Decorative icons (publisher logos, small inline ornaments,
watermark-style graphics) are emitted as separate figure crops.  Label
`icon` accounts for **304 of 437 figure errors (~70%)** aggregated
across 16 variants on the 28-PDF corpus — by far the largest figure-side
error class.

### Evidence
Across all 16 variants, every variant emits the same ~16 icon crops per
PDF (consistent because figure detection is detector-invariant — all
variants share the docling layout for the figure pass).  The icons are
correctly identified as PICTURE/FIGURE element geometry-wise by Docling,
so they get cropped + emitted.  But they're not scientific figures —
they're decorative.  Mask dim is fine (`mask=1.0` for the `icon` rubric
entry), only crop dim is hurt (`crop=0.0`).

### Diagnosis
No current pipeline stage filters them.  Existing
`CroppingConfig.min_figure_pts` (`config.py`) drops figures smaller than
a threshold, but icons can be moderately sized (e.g. publisher logos
that are 40-60pt tall) — raising `min_figure_pts` to catch them would
also drop legitimate small scientific figures.

Three signals that could distinguish icons from real figures:
1. **Repetition** — publisher logo / icon appears on EVERY page of a
   multi-page paper.  Real scientific figures appear once.
2. **Text density** — icons typically contain no text (or just the
   publisher name).  Real figures often have axis labels, callouts.
3. **No caption** — icons typically have no nearby CAPTION element.

### Fix (planned, not implemented)
Add a heuristic filter `drop_icon_figures` to
`MaskingConfig` (or `CroppingConfig`) that combines the above signals:
* Drop figure elements that appear in the same bbox position on N+ pages
  (publisher logo).
* Drop figure elements with zero overlapping CAPTION element and bbox
  area below some threshold (e.g. < 5% of page area).

Test as a new stage variant block similar to `drop_tables_inside_figures`.

### Risk
False-positives on real figures that happen to lack captions or repeat
across pages (rare in scientific papers but possible).  Needs corpus
tuning.

### Decision / Scope
Outside the 2026-05-21 thesis-day budget.  Document as known
limitation in `docs/THESIS.md` future-work section.  Worth pursuing in
a follow-up sprint — 70% of figure errors is a high-value target.

---

## Bug 58 — `drop_tables_inside_figures` bypassed by cropper supplementary source

### Status / Severity / Surface
Fixed (2026-05-20) · Medium · PDF extraction, media cropper Step-7 supplementary source.

### Symptom
Variant 18 (`MaskingConfig.drop_tables_inside_figures = True`) on
PMC11791726 page 9 emits `Table_4_p9.png` — a `table_in_figure` FP
(human-labelled) — despite the run metadata reporting
`table_regions_dropped_inside_figures = 1`. Crop-precision improvement
that the flag was supposed to deliver did not materialise in Stage 3
scoring (variant 18 was bbox-identical to variant 08 / drop=OFF on this
metric).

### Evidence
Same bbox `(49.8, 533.6, 520.6, 402.4)` on page 9 across variants:

```
01 (drop=OFF):  source = "docling+docling"
08 (drop=OFF):  source = "docling+docling"
18 (drop=ON):   source = "docling"          ← one source, not two
```

The shortened `source` field showed the table came in through only ONE
of the cropper's two sources after the drop filter ran. The Step-1
docling layout for that page:

```
TABLE   bbox=(49.77, 533.64) → (520.61, 402.35)
PICTURE bbox=(48.02, 686.61) → (524.11, 143.87)
```

The TABLE is 100% inside the PICTURE — both share the same page; the
0.8 threshold would trigger.  `_drop_tables_inside_figures` in
`runner.py:140` correctly removed it from `detection.regions`.

### Diagnosis
`PyMuPDFMediaCropper.crop` (`media_cropper.py:245-265`) has two
ordered sources for tables:

1. **Primary source**: `detection.regions` — already filtered by
   `_drop_tables_inside_figures` in Step 2. The page-9 FP is gone here.
2. **Supplementary source**: layout TABLE/RECONSTRUCTED_TABLE elements
   (line 246 onwards). For each, the code checks
   `_overlap_ratio(b, existing.bbox) > 0.5` against entries already in
   `merged_tables`. With the FP removed from source #1, there's no
   overlap match → the cropper adds the layout TABLE as a fresh entry.

So the Step-2 drop is reduced to a no-op for any TABLE that docling's
layout independently emits — the supplementary source rehydrates
exactly the bboxes that drop tried to suppress.

### Fix
`media_cropper.py::crop()` takes a new keyword-only parameter
`drop_tables_inside_figures: bool = False`. When True, the
"Supplementary source" loop skips any layout TABLE/RECONSTRUCTED_TABLE
element whose bbox area is ≥0.8 inside any FIGURE/PICTURE on the same
page (same threshold as `_drop_tables_inside_figures` in the runner;
coordinate-system-agnostic min/max normalization, matching
`_bbox_intersect_area`).

`runner.py` plumbs `self._cfg.masking.drop_tables_inside_figures`
into all three `cropper.crop()` call sites (Step 7 main pass + the
two multi-source `docling` / `docling_recon` passes inside
`if self._cfg.runtime.multi_source_crops`).

Single config flag still governs both filters — no new field added to
`MaskingConfig` or `CroppingConfig`.

### Verification
1. Manually inspected `out/sweeps/18_best_drop_tables_in_figures/json/PMC11791726_HIS-86-485_media.json`
   pre-fix: 4 tables emitted including the FP.  Re-run with the fix to
   verify the FP no longer appears in the emitted table list (drops to
   3 tables for that PMC).
2. Stage 3 verdict for variant 18 needs re-evaluation: pre-fix it was
   "no effect" (the bug); post-fix it should show a measurable Δ in
   table crop precision (~+2.7pp on this 28-PDF corpus, eliminating 1
   `table_in_figure` FP out of 38 emitted tables).
3. Re-run command:
   ```bash
   rm -rf out/sweeps/18_best_drop_tables_in_figures
   python3 scripts/eval/run_all_sweeps.py --stage merge_drop --only 18_best_drop_tables_in_figures
   python3 scripts/eval/build_share_map.py
   python3 scripts/eval/backfill_shared_labels.py
   python3 scripts/eval/score_pdf_variants.py --md-out reports/stage3_PR.md
   ```

### Downstream impact
- **Stage 3 verdict (`BEST_STAGE3["drop_tables_inside_figures"]`)** must
  be re-decided after re-running variant 18.  If drop=True now wins,
  flip the dict entry to True and re-run Stage 4 variants 19/20 (their
  base inherits `BEST_STAGE3` via `_apply_stage3_kept`).
- **Freeze defaults**: existing default `MaskingConfig.drop_tables_inside_figures = True`
  (`config.py:218`) is now load-bearing.  If the new verdict keeps
  drop=False, the freeze must change the default to `False`.

### Related labels
Only label-class affected: `table_in_figure` (2026-05-19 rubric entry).
1 underlying instance on this corpus (PMC11791726/p9), appearing under
different filenames depending on variant numbering:
* `Table_3_p9.png` — variants where docling numbering keeps the FP at
  position 3 (01, 05/06/07 hybrid, 08, 10 hybrid, 16, 17, 19, 20).
* `Table_4_p9.png` — variants where pre-fix drop shifted numbering
  (02/03/04 TATR, 09 TATR, 18).

Post-fix, the FP filename disappears from variant 18's emission; that
label becomes orphaned (harmless).  All other 37 table labels per
variant + 80 figure labels stay valid; `share_map.json` carries them
across re-runs.

---

## Bug 57 — Committed merge-conflict markers in `visualizer.py` (and two siblings)

### Status / Severity / Surface
Fixed (2026-05-17) · High · PDF extraction, committed merge-conflict markers.

### Symptom
Every end-to-end PDF run on the `eval-speedrun` branch died before reaching
any pipeline stage with:

```
SyntaxError: invalid syntax (visualizer.py, line 113)
```

After fixing visualizer.py the same error surfaced one import-level deeper:

```
SyntaxError: invalid syntax (tatr_detector.py, line 68)
```

Discovered while running the Stage-1 observability-patch smoke test
(2026-05-17).  The error fired even when `cfg.visualization.enabled = False`,
because `components/__init__.py:7` re-exports `DetectionVisualizer` and
`table_detectors/__init__.py:3` re-exports `TATRTableDetector` — Python must
parse both modules the first time *any* sibling in those packages is
imported, so disabling features at runtime did not help.

### Evidence
Three files shipped with conflict markers at HEAD on `eval-speedrun`:

1. `pipeline/stages/pdf_text_extraction/components/visualizer.py` — lines 113–121, in `_pre-compute detection rects` block.
2. `pipeline/stages/pdf_text_extraction/table_detectors/tatr_detector.py` — lines 68–78, in the lazy `_load()` path that constructs `_SHARED_MODEL`.
3. `eval/run.py` — lines 120–125, in the "Eligible PDFs" log line.

`git status` was clean, so the markers were part of a committed snapshot —
most likely a partial conflict resolution from commit `2d70119 … resolve
stash-pop conflicts` on this branch.

### Diagnosis
Each conflict had two branches.  In all three cases the "Updated upstream"
branch was the one that matched the surrounding non-conflicted code intent:

* **visualizer.py** — both branches append the same fitz rect to
  `detection_rects_by_page[pg]`.  Updated upstream uses
  `setdefault(pg, []).append(...)`; Stashed uses a separate
  `if pg not in dict` initialiser.  Functionally identical.
* **tatr_detector.py** — Updated upstream loads the model and moves it to
  `self._config.device` (the configurable knob promoted in B-034).  Stashed
  hardcodes `to("cpu")` plus a transformers-loading workaround
  (`low_cpu_mem_usage=False, device_map=None`) that's not needed today and
  silently regresses the GPU path.
* **eval/run.py** — Updated upstream logs
  `"Eligible PDFs: %d / %d (%.1f–%.1f MB)"` matching the min+max byte filter
  applied two lines earlier; Stashed shows only the upper bound.

### Fix
Picked the "Updated upstream" branch in all three files.  No behaviour
change relative to the documented intent of the surrounding code.

### Verification
1. `grep -rln "<<<<<<< " pipeline/ parsers/ database/ eval/` returns no results.
2. `python -m py_compile pipeline/stages/pdf_text_extraction/components/visualizer.py pipeline/stages/pdf_text_extraction/table_detectors/tatr_detector.py eval/run.py` succeeds.
3. `python -c "from pipeline.stages.pdf_text_extraction.components import DetectionVisualizer; from pipeline.stages.pdf_text_extraction.table_detectors import TATRTableDetector"` returns without error.
4. End-to-end smoke run on `PMC10047158_dermatopathology-10-00017.pdf` completes; per-document `out/run_metadata/{pmcid}_stats.json` is produced (verifies B-057 is no longer blocking the Stage-1 observability patch).

---

## Bug 1 — Duplicate "intra-paper" relations produced by `canonical_id` collisions

### Symptom

The cross-paper relations table contained rows whose `pmcid_a == pmcid_b` and
whose `predicate_a == predicate_b` (NLI scores 1.0 / 1.0). Example for
`PMC7150310_main`:

| Scope      | Type    | PMCID A           | PMCID B           | Predicate A                                              | Predicate B                                              |
|------------|---------|-------------------|-------------------|----------------------------------------------------------|----------------------------------------------------------|
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Rickets → calcitriol (vitamin D) deficiency             | Rickets → calcitriol (vitamin D) deficiency             |
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Beri-beri → thiamine (vitamin B1) deficiency            | Beri-beri → thiamine (vitamin B1) deficiency            |
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Scurvy → ascorbic acid (vitamin C) deficiency           | Scurvy → ascorbic acid (vitamin C) deficiency           |
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Pernicious anemia → cobalamin (vitamin B12) deficiency  | Pernicious anemia → cobalamin (vitamin B12) deficiency  |

The first hypothesis — that the PDF extractor had emitted the same sentence
twice (e.g. a publisher "ghost text" layer duplicating body content) — turned
out to be wrong.

### Diagnosis

1. `text_elements` for `PMC7150310_main` were verified clean:
   * 61 rows, 61 distinct `text_content` values
   * 0 near-duplicates after whitespace/case normalisation
   * The five vitamin-deficiency findings all trace back to **one** sentence
     in **one** text element under `Natural Product Chemistry and the Rise
     of Clinical Laboratories`.
2. The summarisation artifacts on disk were inspected:
   * `out/summaries/runs/runA_cheap_main/canonicalize/PMC7150310_main/canonical_rules.jsonl`
     contained 196 rules with 196 distinct `canonical_id`s.
   * `out/summaries/corpus_relations.json` contained 19 rows where
     `rule_id_a == rule_id_b` for this paper.
3. Cross-paper comparison of canonical IDs:
   * `PMC7150046_main` produced 171 rules; `PMC7150310_main` produced 196.
   * **22 `canonical_id`s collided across the two papers**, including
     `CR_3193d882_positive` ("Rickets → calcitriol").
4. Root cause traced to `pipeline/stages/summarization/current_stages/group_stage.py:_group_id`:

   ```python
   def _group_id(subject, outcome, relation_type, category="",
                 subject_cui=None, outcome_cui=None) -> str:
       subj_key = subject_cui if subject_cui else subject
       out_key  = outcome_cui if outcome_cui else outcome
       return f"GRP_{_sha8(subj_key)}_{_sha8(out_key)}_{relation_type}_{_sha8(category)}"
   ```

   Two different papers that produced the same `(subject, outcome,
   relation_type, category)` got the same `group_id` → same `canonical_id`
   (`canonical_id = CR_{sha8(group_id)}_{direction}`).
5. In `corpus_relate.py`, the rule pool concatenates rules from all papers,
   so the colliding IDs landed at two distinct list indices. `RelateStage`
   used `itertools.combinations(range(len(rules)), 2)`, which guarantees
   `i != j` but *not* `rules[i].canonical_id != rules[j].canonical_id`,
   so the collisions were paired against themselves. The final enrichment
   step mapped the canonical_id back to a PMCID through a single-writer-wins
   dict, so both sides of each pair displayed the same `pmcid`, producing
   the misleading "intra_paper" rows above.

### Fix

[`pipeline/stages/summarization/current_stages/group_stage.py:57`](../pipeline/stages/summarization/current_stages/group_stage.py#L57)
— added `pmcid` to the hash input:

```python
def _group_id(subject, outcome, relation_type, category="",
              subject_cui=None, outcome_cui=None,
              pmcid: str = "") -> str:
    subj_key = subject_cui if subject_cui else subject
    out_key  = outcome_cui if outcome_cui else outcome
    return (
        f"GRP_{_sha8(pmcid)}_{_sha8(subj_key)}_{_sha8(out_key)}"
        f"_{relation_type}_{_sha8(category)}"
    )
```

`canonical_id` inherits the change because it derives from `group_id`:
`CR_{_sha8(group_id)}_{direction}`. Cross-paper matching still works — it is
performed by the
[`_should_compare_cross_paper`](../pipeline/stages/summarization/helpers/corpus_relate.py)
gate (which keys on CUIs or normalised entity strings), not on `canonical_id`
equality.

### Verification

* Updated tests in `tests/summarization/test_phase3_group.py` (added
  `test_group_id_differs_across_pmcids`) and
  `tests/summarization/test_demographics.py` (updated existing `_group_id`
  call sites). 40/40 pass.
* Full summarisation test suite: **437 passed** in 174 s.
* On-disk artifacts under `out/summaries/` still hold the colliding IDs;
  regenerate them by re-running summarisation on the affected papers.

---

## Topic — Ghost-text detection: empirical verification and policy fix

The duplicate-relations investigation surfaced the question of how the
extractor handles "ghost text" — selectable but unrendered text layers
embedded in publisher PDFs (e.g. accessibility duplicates, watermarks, white
text on white background). Two policy fixes followed, plus three bug entries
in the catalogue ([B-002](#bug-2--docling-phantom-layout-elements),
[B-003](#bug-3--r-color-white-text-false-positive-latent),
[B-004](#bug-4--cid-glyph-fallback-strings)).

### Background

The pipeline already implements three ghost-text-detection rules in
`pipeline/stages/pdf_text_extraction/components/node_scorer.py`:

* **R1 (pixel-render)** — render the element's bbox at 150 dpi; if mean
  luminance ≥ 245 *and* dark-pixel fraction ≤ 0.02, mark `visually_blank=True`
  and drop. Operates on the actual ink that reaches the page.
* **R-color** — read `page.get_text("dict")` and tally span colors;
  drop elements whose near-white-character fraction exceeds
  `max_white_char_fraction`.
* **R3 (dense-text)** — drop elements where `len(text) / bbox_height` exceeds
  `max_chars_per_bbox_pt`. Hidden text layers tend to cram many characters
  into a sliver of vertical space.

The two-pass extractor that *invokes* these rules is gated by
`TwoPassConfig.enabled`.

### 2.1 — Synthetic verification (does R1 catch Tr=3 / opacity=0?)

Built a one-page PDF with three text rows:

| Row | Render style                              |
|-----|-------------------------------------------|
| A   | Visible baseline (Tr=0)                   |
| B   | PDF render-mode 3 (`3 Tr`)                |
| C   | ExtGState fill_opacity `ca=0`             |

Output from
[`scripts/verify_ghost_text_detection.py`](../scripts/verify_ghost_text_detection.py):

```
row            visually_blank  brightness  dark_frac  inv_char_frac  render_skipped
visible        False           246.53      0.0351     0.0            False
tr3            True            255.0       0.0        0.0            False
opacity_0      True            255.0       0.0        0.0            False

PASS: pixel-render path catches both Tr=3 and fill_opacity=0 text.
```

Both ghost variants reduce to "no ink in the rendered bbox" — exactly what
R1 measures. No content-stream parser (e.g. `pdfminer.six` render-mode
extraction) was needed; adding one would have been duplicated work.

### 2.2 — Corpus scan (12 papers, 3,131 text-bearing elements)

`scripts/scan_ghost_text_real_papers.py` evaluated every TEXT-typed Docling
element against the rules. Aggregate:

| Signal                                         | Count | Rate  |
|------------------------------------------------|------:|------:|
| `visually_blank = True` (R1 fires)             | 20    | 0.64% |
| `invisible_char_fraction > 0.5` (R-color fires) | 18    | 0.57% |

Two patterns dominated:

**Pattern A — Docling phantom layout elements (`PMC10047158`, 18 hits).**
Docling occasionally emits a layout element with real text content but a
bbox that points to an empty area of a *different* page. Example:

| Field            | Value                                                                                              |
|------------------|----------------------------------------------------------------------------------------------------|
| Reported page    | 2                                                                                                  |
| Reported bbox    | `(168.2, 809.4) – (563.6, 799.3)` — i.e. 10 pt tall, sitting in the page header zone (≈ y=32 from top) |
| Reported text    | `large, centrally-located, occasionally bilobated nuclei with conspicuous nucleoli...`             |
| Pixel evidence   | `pixel_brightness_mean=255.0`, `dark_pixel_fraction=0.0` → `visually_blank=True`                    |
| `fitz.get_text('words', clip=…)` in that rect | `0` words found                                                                          |
| Actual location of the text | Page 1, inside the main body paragraph                                                  |

The downstream `ContextAwareStitcher` happens to merge these phantoms back
into legitimate paragraphs in most cases — verified by querying
`text_elements`:

```
unique_path                                            | preview
PMC10047158_dermatopathology-10-00017/2. Case Report/1 | Histological examination of routinely-stained sections showed
                                                       | a well-circumscribed exo-endophytic de…
```

— only one DB row holds "large, centrally" rather than two. But the stitcher
is incidental cleanup; future Docling layout regressions could surface
extra rows.

**Pattern B — White text on coloured banners (`PMC7158325`, 18 hits).**
"Key Points" SECTION_HEADER elements report `span.color = 0xFFFFFF` (pure
white) → `invisible_char_fraction = 1.0`. But the rendered bbox is full
of ink: `dark_pixel_fraction ≈ 0.73`, `pixel_brightness_mean ≈ 136`.
These are legitimate headers printed in white type on a coloured banner.
R-color would label them ghost text; R1 correctly keeps them.

This established that **R1 is the trustworthy signal**; R-color produces
false positives whenever a publisher inverts header type colours.

### 2.3 — Policy decisions

Two defaults changed in
[`pipeline/stages/pdf_text_extraction/config.py`](../pipeline/stages/pdf_text_extraction/config.py):

| Setting                         | Before | After | Reason                                                                                                        |
|---------------------------------|--------|-------|---------------------------------------------------------------------------------------------------------------|
| `TwoPassConfig.enabled`         | `False` | `True` | R1 catches Docling phantoms upstream; the stitcher is no longer the only line of defence.                  |
| `TwoPassConfig.max_white_char_fraction` | `0.5` | `1.0` | Disables R-color by default. R1 is the source of truth; R-color produced false positives on inverted headers. |

### 2.4 — Before/after demonstrations

[`scripts/thesis_demo_ghost_text.py`](../scripts/thesis_demo_ghost_text.py)
runs both demos and writes JSON + PNG artifacts under
`out/thesis_demo/ghost_text/`.

#### Demo 1 — phantom passes through old policy, dropped by new policy

`PMC10047158_dermatopathology-10-00017`, page 2 phantom element.

Old policy (`TwoPassConfig.enabled=False`): NodeScorer never runs.

| Policy | Total Docling text elements | Kept | Dropped | Phantom kept/dropped |
|--------|-----------------------------:|-----:|---------:|----------------------|
| OLD    | 129                          | 129  | 0        | **kept** (it would survive Docling layout output) |
| NEW    | 129                          | 82   | 47       | **dropped** by R3 ("hidden text layer — 181 chars in 10.2pt bbox = 17.8 chars/pt, in header zone"); R1 would also fire because the bbox is `visually_blank=True`. |

Rendered crop of the phantom region (page 2, header strip outlined in red —
note that the visible text in that strip is the journal banner
"Dermatopathology", not the body sentence Docling reported):

![Phantom bbox on page 2 of PMC10047158](../out/thesis_demo/ghost_text/demo1_phantom_false_negative_p2.png)

#### Demo 2 — legitimate white-on-dark header (latent R-color false positive)

`PMC7158325_main`, 17 occurrences of "Key Points".

| Policy | "Key Points" headers kept | "Key Points" headers dropped |
|--------|---------------------------:|-----------------------------:|
| OLD    | 17                         | 0                            |
| NEW    | 17                         | 0                            |

**Caveat:** SECTION_HEADER appears in `NodeScorer._ALWAYS_KEEP`, so R-color
never actually fires on these headers in production. The demo therefore
documents the *evidence* (span color = `0xFFFFFF`, `invisible_char_fraction
= 1.0` on a bbox whose rendered pixels are 73% dark ink) rather than an
old-vs-new behaviour difference. The false-positive risk it captures is
latent — it would have surfaced if R-color were applied to a TEXT/CAPTION
element printed in the same white-on-coloured style (e.g. a callout box
or a figure caption banner). Disabling R-color by default closes that door
before such an element appears.

Rendered crops of three "Key Points" banners (red outline = bbox; the white
ink reads as ≈ 73 % dark pixels against the coloured background):

![Key Points banner on page 3 of PMC7158325](../out/thesis_demo/ghost_text/demo2_key_points_false_positive_p3.png)

![Key Points banner on page 4 of PMC7158325](../out/thesis_demo/ghost_text/demo2_key_points_false_positive_p4.png)

![Key Points banner on page 27 of PMC7158325](../out/thesis_demo/ghost_text/demo2_key_points_false_positive_p27.png)

### Reproducibility

```bash
python scripts/verify_ghost_text_detection.py       # synthetic Tr=3 / opacity=0 PDF
python scripts/scan_ghost_text_real_papers.py 12    # 12-paper corpus sample
python scripts/thesis_demo_ghost_text.py            # writes demos + PNGs
```

JSON artifacts in `../out/thesis_demo/ghost_text/` carry the full evidence
table for each demo and can be re-rendered into the thesis figures without
re-running the scorers.

---

## Bug 2 — Docling phantom layout elements

**Status:** Mitigated (2026-05-13) · **Severity:** Medium · **Surface:**
PDF extraction, Docling layout.

**Symptom.** Some Docling-emitted layout elements carry legitimate body
text but a bbox that points to an empty region of the wrong page (typically
the page header zone of the page *after* the page where the text actually
lives).

**Evidence.** On `PMC10047158_dermatopathology-10-00017`, 18 such phantom
elements were found across one paper. See the full data table and rendered
crop in [Topic — Ghost-text detection §2.2 Pattern A](#22--corpus-scan-12-papers-3131-text-bearing-elements)
and [§2.4 Demo 1](#24--beforeafter-demonstrations).

**Mitigation.** Flipping `TwoPassConfig.enabled` to `True` (May-2026) makes
NodeScorer R1 (`visually_blank=True`) and R3 (`chars/pt` ratio) reject the
phantoms upstream rather than relying on `ContextAwareStitcher` to absorb
them downstream. Demo confirms 47 / 129 phantom-like elements dropped on
this paper post-fix; one phantom would have entered the DB without the fix
unless the stitcher caught it.

**Why "mitigated" not "fixed".** The root cause is Docling's layout
extraction emitting incorrect bboxes — we cannot fix Docling. We just catch
the symptom downstream. Tracked in TODO list for follow-up audit.

---

## Bug 3 — R-color white-text false positive (latent)

**Status:** Mitigated (2026-05-13) · **Severity:** Low (latent) · **Surface:**
PDF extraction, color signal.

**Symptom.** Rule R-color in `NodeScorer` rejects any element whose text
spans report near-white colour. On `PMC7158325_main`, 18 SECTION_HEADER
elements (`"Key Points"`) report span colour `0xFFFFFF` and trigger the
rule — even though the bbox renders as 73 % dark ink (white type on a
coloured banner).

**Why "latent".** `SECTION_HEADER` is in `NodeScorer._ALWAYS_KEEP`, so the
rule never actually fires in production. But the same publisher style
applied to a TEXT or CAPTION-typed element would have produced data loss.

**Mitigation.** `TwoPassConfig.max_white_char_fraction` default flipped
from `0.5` → `1.0`, which disables R-color. Rule R1 (pixel render) is the
trustworthy signal because it measures actual rendered ink, not span
metadata. Detail and rendered crops in [Topic §2.4 Demo 2](#24--beforeafter-demonstrations).

---

## Bug 4 — CID-glyph fallback strings

**Status:** Observed · **Severity:** Low · **Surface:** PDF extraction,
Docling glyph fallback.

**Symptom.** When a PDF embeds a font subset whose glyphs cannot be mapped
to Unicode, Docling falls back to placeholder strings: `GLYPH<0>GLYPH<20>…`,
`/gid00001`, etc. Found in `PMC11863827_main` and
`PMC7583592_dermatopathology-07-00003` during the 12-paper scan.

**Pipeline behaviour.** R1 correctly drops these because their bboxes contain
no rendered ink (the unmapped glyphs were never decoded into visible glyphs
either). No further action needed at the extraction stage.

**Open question for the thesis.** Worth a brief note that the corpus
contains a small number of font-subset PDFs that lose text content
end-to-end. If the rate grows on a different corpus, an OCR fallback
becomes worth costing out.

---

## Bug 5 — Batch runner missing sync-parity features

**Status:** Mitigated (2026-05-14) · **Severity:** High · **Surface:**
Summarisation, `BatchSummarizationRunner`.

### Symptom

`scripts/run_paper.py` defaults to batch mode. Batched production runs since
late April 2026 were quietly skipping six runner-level features that were
present on `SummarizationRunner.process()`:

1. **`_replace_verbatim_from_db`** — sync replaces LLM-paraphrased
   `verbatim_support` with the actual `TextElement.text_content` from the DB
   *before* grounding so NLI entailment scores against the real source. Batch
   never did this — grounding was scoring paraphrases against paraphrases.
2. **Stable `compute_finding_id`** — sync stamps every finding with a
   deterministic id `(pmcid, chunk_id, position, claim)` before grounding so
   downstream stages share a lineage key. Batch findings had no stable id.
3. **DB persistence** — sync writes `sum_map_findings`,
   `sum_normal_findings`, `sum_finding_groups`, `sum_canonical_rules`,
   `sum_relations`, `sum_final_rules`, `sum_rejection_summaries` via
   `pipeline_run_db_id`. Batch wrote JSON files but never touched the DB.
4. **`corpus_relate_incremental`** — sync runs cross-paper RELATE after
   CANONICALIZE. Batch did not. Corpus-relate tables only saw sync papers.
5. **`rejection_summary`** — sync builds a `RejectionSummary` recording
   grounding + non-groupable drops and persists it. Batch had no equivalent.
6. **NER + UMLS** — sync optionally runs `run_ner_on_db` per paper. Batch
   did not.

### Diagnosis

`git log` shows the gap is purely temporal — sync existed first
(commit `b3eecf3`), batch was bolted on later (`2aa71f6`). Six
sync-only commits (`5c59c3e`, `fb1b9af`, `a64fa9a`, …) added features
that never got backported to `batch/runner.py`. The most recent
multi-file commit `d002eb2` (2026-05-13) touched both runners but only
added voter-dedup, not the parity features. Stage-level fixes
(`group_stage.py`, `map_stage.py`, `relate_stage.py`,
`grounding_filter.py`) *did* propagate, since both runners import them —
so the gap is runner-level only.

### Fix

Copied the missing helpers verbatim from `SummarizationRunner` into
`BatchSummarizationRunner` (`_replace_verbatim_from_db`,
`_create_pipeline_run`, `_finish_pipeline_run`, `_clear_normalized_run_data`,
`_persist_map_findings`, `_persist_normal_findings`, `_persist_finding_groups`,
`_persist_canonical_rules`, `_persist_relations`, `_persist_final_rules`,
`_persist_rejection_summary`, `_corpus_relate_incremental`). Wired into
`finalize()` between the existing filesystem-artifact persistence and
REDUCE/RULES. `__init__` now accepts `db`, `force_rerun`, `run_ner` (defaults
`None`, `False`, `False` to preserve existing call-sites).

Also added result caching with `pipeline_config_hash` invalidation
(`_load_result` + `_save_result`) checked at the top of `submit()` so a
stale cache cannot waste L1/L2/L3 batch dollars; `BatchHandle` now carries
`pipeline_run_db_id` and a `cached_result_only` marker so a resumed run
keeps its DB pointer and a cache-hit short-circuits `finalize()`.

### Verification

`tests/summarization/test_batch_persistence.py` (4 tests) and
`tests/summarization/test_batch_voter_dedup.py` (10 tests) pass on the
modified runner. End-to-end DB integration verification is tracked as a
follow-up TODO until a real-DB run can be done.

### Why not refactor `SummarizationRunner` to share code?

Tempting but high-blast-radius. The sync runner is working code touched by
multiple recent commits; reshuffling its private methods into a shared
module while also changing the batch runner doubles the surface area of
this change. A follow-up TODO captures the dedup once both runners have
been verified to behave identically.

### Dedup follow-up (shipped 2026-05-16)

The "dedup once verified" follow-up landed. 13 helpers lifted to
module-level functions in `pipeline/stages/summarization/persistence.py`
(taking `db` as the first parameter): `build_rejection_summary` +
`create_pipeline_run` / `finish_pipeline_run` / `clear_normalized_run_data` /
`replace_verbatim_from_db` + 7 `persist_*` per-table writers + the
already-extracted `non_groupable_reason`. Both `SummarizationRunner` and
`BatchSummarizationRunner` now thin-wrap each (1-3 line forwards).
Net −421 LOC across the three files. All `test_persistence.py` /
`test_batch_persistence.py` cases pass; full summarisation suite identical
pre/post (1 pre-existing flake unrelated to this change).

Four cache/result helpers (`_pipeline_config_hash`, `_result_path`,
`_load_result`, `_save_result`) were left as per-class methods because the
implementations diverge by design (sync `_load_result` tags the dict
`status="skipped"` per B-008; sync `_save_result` wraps hash compute in
try/except; sync `_pipeline_config_hash` reads `map_meta.run_metadata_summary()`
and includes `scorer`; sync uses `self._cfg` vs batch `self._cfg_full`).
Tracked as carry-forward in `docs/THESIS.md` *##TODOs* — either backport
the sync features to batch and unify, or formally split the contract.

---

## Bug 6 — `SCOPE_QUALIFY` plumbing is dead

**Status:** Fixed (2026-05-14) · **Severity:** Medium · **Surface:**
Summarisation, RELATE → RESOLVE.

**Symptom.** `RelationTypeLabel.SCOPE_QUALIFY` was declared in
`models.py`, `FinalRule.scope_qualify_count` existed, the RESOLVE filter
at `current_stages/resolve_stage.py:120-123, 183` populated it, and the
RELATE log line at `current_stages/relate_stage.py:412-421` included a
SCOPE_QUALIFY count — yet `_classify_pair` had no branch that returned
`SCOPE_QUALIFY`. Mutual entailment → `SUPPORT`; mutual contradiction with
opposite polarity → `CONTRADICT`; everything else fell through to
`None`/`UNRELATED`. Result: `scope_qualify_count` was always 0, the log
misreported class counts, and any downstream consumer expecting an
asymmetric-entailment signal silently got none.

**Diagnosis.** Likely the asymmetric-entailment branch
(`ent_ab >= threshold xor ent_ba >= threshold → SCOPE_QUALIFY`) was
deleted at some point but the scaffolding around it wasn't.

**Fix.** Option 2 chosen — torn out the unused branch.

* `models.py`: removed `SCOPE_QUALIFY` from `RelationTypeLabel`; updated
  `Relation` docstring; defaulted `FinalRule.scope_qualify_count` to 0
  with a comment marking it dormant.
* `current_stages/relate_stage.py`: dropped `SCOPE_QUALIFY` column from
  the info log; updated module + ctor docstrings.
* `current_stages/resolve_stage.py`: removed the `scope_qualifies`
  list-comp; `scope_qualify_count=0` written verbatim.
* `current_stages/group_stage.py`, `PIPELINE.md`, `database/models.py`
  docstrings updated to drop the SCOPE_QUALIFY callouts.
* `final_rules.scope_qualify_count` DB column and the inspector template
  filter option retained so existing rows / pages still render. Drop via
  Alembic only if the asymmetric branch isn't reinstated.

**Verification.** `python -c "from pipeline.stages.summarization.models
import RelationTypeLabel; assert 'SCOPE_QUALIFY' not in [m.value for m in
RelationTypeLabel]"` passes. RELATE log message no longer mentions the
column. RESOLVE writes `scope_qualify_count=0` for every FinalRule.

---

## Bug 7 — Sync runner cached-result load ignores `pipeline_config_hash`

**Status:** Fixed (2026-05-14) · **Severity:** Medium · **Surface:**
Summarisation, `SummarizationRunner`.

### Symptom

`SummarizationRunner._load_result` returned any on-disk
`out/summaries/summaries/{pmcid}.json` unconditionally (only `force_rerun`
bypassed it), and `_save_result` never stamped a hash. Run sync once with
cascade profile A, then again with profile B → the second run silently
returned profile A's cached result. The `runs/{run_id}/` artifact tree
(which had its own hash in `manifest.json`) told the right story; the
user-visible result JSON did not.

### Diagnosis

The cache code at the top of `process()` predated `pipeline_config_hash`;
it was never updated when the hash landed in `fb1b9af` / `a64fa9a`. The
batch runner picked up the hash check as part of [B-005](#bug-5--batch-runner-missing-sync-parity-features)
but the sync runner was left behind.

### Fix

Commit `b03d4f6` mirrored the batch implementation:

* Added [`SummarizationRunner._pipeline_config_hash()`](../pipeline/stages/summarization/runner.py)
  (runner.py:1640-1684) — composes cascade signature, grounding +
  entailment + contradiction + theta + reject-theta + similarity
  thresholds, voter / level-2 / escalation / scorer model identifiers,
  schema + prompt versions, and the `enable_router` /
  `router_single_voter_policy` state. Delegates to the existing
  `compute_pipeline_config_hash` helper in `persistence.py` so both
  runners stay in sync.
* `_load_result` (runner.py:1686-1710) now reads `pipeline_config_hash`
  from the cached dict, recomputes the current hash, logs `cached result
  stale (config hash X != Y) — re-running` on mismatch, and returns
  `None`. Unreadable JSON or hash-computation failures also return
  `None` (fail-closed for safety).
* `_save_result` (runner.py:1712-1721) stamps the hash via
  `result.setdefault("pipeline_config_hash", self._pipeline_config_hash())`
  so callers that already populated it (e.g. cache-hit short-circuits)
  aren't overwritten.
* Manifest builder updated to call `self._pipeline_config_hash()`
  instead of re-importing `compute_pipeline_config_hash` directly,
  eliminating drift between the two call-sites.

Stale files are left on disk (matches batch behaviour); audit-friendly
and the next run overwrites them.

### Verification

`grep -n "_load_result\|_save_result\|pipeline_config_hash"
pipeline/stages/summarization/runner.py` shows the helpers wired into
`process()` at line 367 (load) and line 729 (save), plus the manifest
extra at line 1158. Sync vs batch hash composition is identical by
construction (both delegate to `compute_pipeline_config_hash`).

---

## Bug 8 — `process_batch` skip counter is structurally zero

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:**
Summarisation, sync runner reporting.

### Symptom

`SummarizationRunner.process_batch` computed
`n_ok = sum(1 for r in results if r["status"] == "success")`,
`n_err = sum(... == "error")`, and
`n_skip = len(results) - n_ok - n_err`. But `_load_result` returned the
cached dict with `status="success"` (the value stamped by `_save_result`
at the *original* run's success path) — cached papers counted in `n_ok`
and `n_skip` was structurally 0. The summary log ("Batch complete: X
ok / 0 skipped (cached) / Y errors") therefore always undercounted the
real fresh-work figure and reported zero skips.

### Fix

* `_load_result` ([`runner.py:1717`](../pipeline/stages/summarization/runner.py))
  now mutates the in-memory dict it returns: `data["status"] = "skipped"`
  immediately before the `return`. The on-disk JSON is **not** rewritten
  — the file still says `"success"` because that run *did* succeed.
  `"skipped"` is purely an in-memory marker on the caller's copy
  describing how the value was obtained on this call.
* `process_batch` ([`runner.py:793`](../pipeline/stages/summarization/runner.py))
  counts the new key explicitly:
  `n_skip = sum(1 for r in results if r["status"] == "skipped")`.
* Three downstream consumers updated to treat `"skipped"` as equivalent
  to `"success"` for "this paper has a complete result on disk" gating:
  * [`scripts/summarize_paper.py:41`](../scripts/summarize_paper.py) —
    print branch.
  * [`scripts/run_single_doc.py:69, 90`](../scripts/run_single_doc.py) —
    write artifacts + print result.
  * [`scripts/run_paper_single_model.py:486, 523`](../scripts/run_paper_single_model.py)
    — `n_ok` count, which feeds the `n_ok >= 2` corpus-relate gate at
    line 532. Without this update, a corpus run resumed against fully
    cached papers would silently skip CORPUS_RELATE.

### Why not the alternative

The alternative was to count from inside `process()` at the cache-hit
branch and leave `_load_result` untouched. Rejected because:
1. Multiple callers want the "this came from cache" signal (run-paper
   scripts, observability/export, future tooling). Tagging at the load
   site puts the marker where the information lives.
2. `process()` returning the same dict with two possible status values
   (`success` for fresh, `skipped` for cached) is a clearer external
   contract than an internal counter the caller can't introspect.

The on-disk JSON keeps `"success"` so `corpus_relate.py` (which reads
files directly without going through `_load_result`) is unaffected.

### Verification

`tests/summarization/test_batch_persistence.py` (4 tests) and
`tests/summarization/test_demographics.py` (12 tests) — all 16 pass.
The batch tests cover the no-cache path; sync cache-hit path is covered
by the new tagging being a single line whose effect is observable in
`process_batch`'s log message on a re-run.

---

## Bug 9 — Sync runner instance dicts leak across papers

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:**
Summarisation, `SummarizationRunner`.

### Symptom

The runner stored every paper's intermediate state on `self`:
[`runner.py:288-302`](../pipeline/stages/summarization/runner.py)
defined `_scored_map_findings`, `_normal_findings`, `_finding_groups`,
`_canonical_rules`, `_relations`, `_relate_raw_pairs`,
`_relate_skipped_pairs`, `_final_rules` — all keyed by `pmcid`.
`process()` wrote to them but never deleted. Inside `process_batch` a
single runner processed N papers and these dicts grew without bound.
For a 100-paper sweep with thousands of RELATE pairs per paper, the
runner ended up carrying every raw NLI pair from every paper in memory.

### Diagnosis

The dicts exist only because `process()` builds its result dict from
them at the end. Local variables would also work but would require a
sweeping refactor of every stage call-site (the dicts are read at ~30
points across `process()`).

### Fix

Option (a) chosen — `pop` the per-paper entries from all eight dicts in
the `finally` block of `process()`
([`runner.py:775-794`](../pipeline/stages/summarization/runner.py)).
Runs after the result dict is materialised (line 684) and saved
(line 729), so external callers see the same payload. Survives error
paths and `KeyboardInterrupt` because it's in `finally`. `pop(pmcid,
None)` instead of `clear()` so a re-entrant `force_rerun` call for the
same pmcid starts from a clean slate without disturbing other in-flight
papers (relevant if a future contributor ever runs multiple `process()`
calls concurrently).

Option (b) — moving state to local variables — was rejected as too
high-blast-radius for a low-severity bug. Worth revisiting alongside a
broader runner refactor (also unlocks thread safety; right now the
runner is single-threaded by accident, not by design).

### Verification

`grep` confirms no external reader of any of the eight dicts exists —
only `last_map_escalation_counts` and `last_map_invocation_usage_records`
are exposed on the runner, and those read from `self._map`, not these
dicts. Full summarisation + PDF-extraction test suites pass post-fix.

The MAP cache (`self._cache`) and per-paper traces are unaffected — this
fix only touches the eight per-paper state dicts.

---

## Bug 10 — `artifact_filter` rebuild uses dict equality instead of identity

**Status:** Fixed (2026-05-14) · **Severity:** Medium (latent) ·
**Surface:** PDF extraction, artifact filter.

### Symptom

[`components/artifact_filter.py:59`](../pipeline/stages/pdf_text_extraction/components/artifact_filter.py)
rebuilt the filtered `List[LayoutElement]` via
`[el for i, el in enumerate(elements) if element_dicts[i] in filtered_dicts]`
— a list-`__contains__` membership check on dicts. Two issues. (1) The
scan was O(N²) in element count. (2) Critically, it relied on the dicts
being byte-equal across the call: if `filter_artifacts` ever mutated a
kept dict in place (a future Unicode normalisation, a `fix_ligatures`
pass on `text`, anything), the post-filter dict no longer `==`'d the
pre-filter dict and the corresponding `LayoutElement` was silently
dropped with no log or error.

### Diagnosis

Today
[`parsers/layout_utils.filter_artifacts`](../parsers/layout_utils.py)
returns the dicts unchanged (it appends references to `kept`, never
constructs new dicts), so the bug was dormant. But the abstraction
boundary didn't enforce that — a contributor adding any normalisation
would trigger silent data loss.

### Fix

Option 2 chosen — build an identity lookup before the filter call:

```python
dict_id_to_element = {id(d): el for d, el in zip(element_dicts, elements)}
filtered_dicts = filter_artifacts(element_dicts, nlp=nlp)
result = [dict_id_to_element[id(d)] for d in filtered_dicts]
```

O(N), survives in-place mutation of kept dicts, and (unlike option 1)
doesn't change `filter_artifacts`'s public contract. `filter_artifacts`
has two other callers (`scripts/combined_pipeline.py:663` and
`scripts/merged_pipeline.py:910`) that pass dicts in and consume dicts
out — keeping the contract as "filters by reference, returns dict list"
avoids touching them.

Option 1 (returning kept *indices*) was the cleaner contract on paper
but would have required updating both legacy script call-sites and
broadening the test surface. Option 2 fixes the bug at the only buggy
caller.

The new contract on `filter_artifacts` is implicit but narrow: it must
not construct new dicts to represent kept elements. If a future change
ever needs to (e.g. to wrap dicts), the lookup falls back via a
`KeyError` — fail-loud, not silent.

### Verification

[`tests/pdf_text_extraction/test_artifact_filter.py`](../tests/pdf_text_extraction/test_artifact_filter.py)
— two tests:

1. `test_filter_drops_empty_text` — sanity: an empty-text element is
   dropped, the surviving anchor is the original `LayoutElement`
   instance (`out[0] is elements[0]`).
2. `test_identity_preserved_under_dict_mutation` — patches
   `filter_artifacts` with a side-effect that rewrites `text` in place
   on every kept dict, then verifies both LayoutElements still come
   back by identity. This is the regression that the pre-fix code would
   have failed.

Both pass (`pytest tests/pdf_text_extraction/test_artifact_filter.py
-v` → `2 passed in 0.11s`).

---

## Bug 11 — `ModelRegistry.docling_converter` ignores `DoclingConfig`

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:**
PDF extraction, `ModelRegistry`.

### Symptom

`pipeline/stages/pdf_text_extraction/resources.py` exposed
`ModelRegistry.docling_converter` which built a
`PdfPipelineOptions(do_table_structure=..., do_ocr=...)` from
`DoclingConfig` but hard-coded `images_scale=2.0` and never constructed
`AcceleratorOptions`. So `DoclingConfig.images_scale`,
`accelerator_device`, `ocr_engine`, `force_full_page_ocr` were silently
ignored when a caller went through `ModelRegistry` — they got CPU at
scale 2.0 regardless of what they passed in.

### Diagnosis

`ModelRegistry` was an early "single shared converter" abstraction that
got sidelined when each pipeline component started lazy-loading its own
model (per the module docstring — "PipelineRunner does NOT route through
this class"). When `images_scale`, `accelerator_device`, `ocr_engine`,
`force_full_page_ocr` were added to `DoclingConfig` later,
`DoclingLayoutExtractor._get_converter` was updated; `ModelRegistry` was
forgotten because no in-tree code calls it.

`grep` across the repo confirmed zero consumers of any `ModelRegistry`
property (`docling_converter`, `tatr`, `spacy_nlp`) — only the
`__init__.py` re-export and four documentation references. Each
property already had a working alternative:

* **Docling converter** → `DoclingLayoutExtractor._get_converter`
  (full config fidelity, used by `PipelineRunner`).
* **TATR (proc, model)** → `TATRTableDetector` keeps its own
  process-wide singleton.
* **scispaCy nlp** → `pipeline/stages/summarization/umls_resources.get_nlp()`
  is the canonical singleton (CLAUDE.md explicitly warns against
  double-loading).

### Fix

Option (a) extended — deleted the entire `ModelRegistry` class instead
of just `docling_converter`. Removing one of three properties leaves a
class with two unused properties; removing all three is identical work
and removes the trap entirely.

**Removed:**

* `pipeline/stages/pdf_text_extraction/resources.py` — file deleted.

**Updated:**

* `pipeline/stages/pdf_text_extraction/__init__.py` — dropped the
  `from ... import ModelRegistry` line and the `"ModelRegistry"` entry
  from `__all__`.
* [`docs/REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md) — removed the
  `resources.py` row from the file table and the bullet underneath;
  appended a one-line note to the `__init__.py` bullet explaining
  where each model now loads.
* [`docs/STRUCTURE.md`](STRUCTURE.md) — removed the "Lazy model
  registry" row from the key-files table.
* [`.claude/CLAUDE.md`](../.claude/CLAUDE.md) — removed the
  `resources.py` line from the file-tree diagram.

External notebook/script callers who want a Docling converter should
instantiate `DoclingLayoutExtractor` directly — it already exposes the
full `DoclingConfig` plumbing.

### Verification

`grep -rn ModelRegistry .` returns only the `BUGS.md` and `THESIS.md`
historical entries. Pipeline import surface unchanged for everything
else (`PipelineRunner`, `PipelineConfig`, `BlacklistManager`,
`ParallelBatchRunner`, all sub-configs and enums).

---

## Bug 12 — `two_pass_extractor` header strip mixes coordinate systems

**Status:** Observed · **Severity:** Low (clarity / latent) · **Surface:**
PDF extraction, two-pass extractor.

**Symptom.**
[`components/two_pass_extractor.py:382-398`](../pipeline/stages/pdf_text_extraction/components/two_pass_extractor.py)
constructs the header strip with `docling_y1 = page_h` (Docling
coordinates, y=0 at bottom) and `docling_y2 = page_h - fitz_header_bottom`
(fitz coordinates, y=0 at top). The two are adjacent lines named with
different prefixes; a single `docling_y1 > docling_y2` comparison is the
only thing preventing a sign-flip if the names get muddled in a future
refactor.

**Fix.** Build the rect in fitz coordinates throughout
(`fitz.Rect(0, 0, page_w, fitz_header_bottom)`) and convert once with
`BoundingBox.from_fitz_rect`. Same numeric result, no mixed-coord lines.

---

## Bug 13 — Inspector "NLI B→A" sort typo

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:** Inspector
batch index template.

**Symptom.** Clicking the **NLI B→A** column header in
`out/inspector/**/index.html` did not reorder rows. Other numeric columns
sorted correctly.

**Diagnosis.**
[`scripts/templates/pipeline_batch_index.html.jinja2:276`](../scripts/templates/pipeline_batch_index.html.jinja2)
read `parseFloat(a.dataset.nilBa)` instead of `dataset.nliBa`. HTML5
`dataset` camel-cases dashed attributes (`data-nli-ba` → `nliBa`), so the
typed key resolved to `undefined` and `parseFloat(undefined) → NaN → 0` —
every row compared as 0, no movement.

**Fix.** One-character correction: `nilBa` → `nliBa`. Regenerate index
HTMLs with `scripts/inspect_pipeline_output.py --batch-dir <dir>`.

---

## Bug 14 — Inspector `badge-blue` class missing

**Status:** Fixed (2026-05-14) · **Severity:** Low (latent) · **Surface:**
Inspector batch index template.

**Symptom.** SCOPE_QUALIFY corpus relations would render with an unstyled
"badge-blue" tag (just text on the default body background).

**Diagnosis.**
[`pipeline_batch_index.html.jinja2:194`](../scripts/templates/pipeline_batch_index.html.jinja2)
applies class `badge-blue` for SCOPE_QUALIFY rows, but the stylesheet
(lines 44-48) only defined `badge-green/red/orange/gray/cyan`. CSS class
selector silently does nothing for an undefined class.

Currently dormant because [B-006](#bug-6--scope_qualify-plumbing-is-dead)
means `_classify_pair` never emits SCOPE_QUALIFY — no row ever takes the
blue branch. The moment B-006 is fixed and SCOPE_QUALIFY can be emitted,
this would have surfaced.

**Fix.** Added `.badge-blue { background: #1e3a8a; color: #93c5fd; }`
matching the neighbouring badge palette.

---

## Bug 15 — Raw LLM enum values lost on coercion

**Status:** Fixed (2026-05-14) · **Severity:** Medium · **Surface:**
Summarisation MAP — `Finding` Pydantic model + `sum_map_findings` table.

### Symptom

When an LLM voter emitted a `relation_type` not in
`RelationTypeEnum` (e.g. `"associates_with"`, `"correlation"`), the
`_coerce_invalid_relation_type` field-validator in
[`pipeline/stages/summarization/models.py`](../pipeline/stages/summarization/models.py)
silently rewrote it to `RelationTypeEnum.unclear`. Same story for
`direction` and (post-B-016) for `category` legacy `"demographics"`. The
row in `sum_map_findings` then carried only the coerced enum value;
downstream NORMALIZE / GROUP / CANONICALIZE / RELATE / RESOLVE stages
only ever saw the post-coercion value.

The raw string *was* logged to `logs/enum_observations.jsonl` via
`enum_logging.log_enum_observation`, but with empty `context` — no
`finding_id`, no `pmcid`, no `chunk_id`. There was no way to SQL-join a
row in `sum_map_findings` to the raw value the LLM had actually produced
for it.

### Diagnosis

Two failure modes:

1. **Hidden coercion.** If a strong model (Sonnet 4.6) systematically
   reaches for an out-of-enum label, the result table shows it "gave up"
   (`unclear`). That's wrong — it had an opinion, the enum was too
   narrow. We were losing the signal that would tell us so.
2. **No join key.** `enum_observations.jsonl` records were
   write-only telemetry: no PK back to the finding row, so even the raw
   strings we *did* capture were unattributable.

### Fix

1. **Pydantic capture.** Added a `@model_validator(mode="wrap")` to
   `Finding` (`models.py`) — `_capture_raw_then_validate` reads
   `relation_type`, `direction`, `category` from the raw input dict
   *before* any field validator runs, stashes them into three new
   `PrivateAttr`s (`_raw_relation_type`, `_raw_direction`,
   `_raw_category`), then calls `handler(data)` to run the existing
   validation. Exposed via `raw_relation_type` / `raw_direction` /
   `raw_category` read-only properties. PrivateAttrs don't appear in the
   OpenAI strict schema, so the prompt schema is unchanged.
2. **DB schema.** Added nullable `Text` columns
   `raw_relation_type`, `raw_direction`, `raw_category` on
   [`SumMapFinding`](../database/models.py) via Alembic
   [`0011_add_raw_llm_columns_to_sum_map_findings.py`](../alembic/versions/0011_add_raw_llm_columns_to_sum_map_findings.py).
3. **Plumbing.** `SummarizationRunner._persist_map_findings`
   ([`runner.py:1127`](../pipeline/stages/summarization/runner.py))
   passes `f.raw_relation_type` / `f.raw_direction` / `f.raw_category`
   into the row dict.

### Verification

`python -c "..."` smoke test exercised five cases — valid values, invalid
`relation_type` ("associates_with"), legacy `"demographics"`,
`direction=None`, invalid `direction="maybe"` — all coerce correctly and
preserve the raw string. `tests/summarization/test_demographics.py`
(12 tests) + `scripts/test_map_schema.py` (18/19, 1 unrelated
pre-existing failure) all still pass.

### Limitations

* Raw values only land on `sum_map_findings`. They do *not* propagate
  to `sum_normal_findings`, `sum_finding_groups`, `sum_canonical_rules`
  yet — see TODO under [B-015 propagation](THESIS.md#todos).
* `confidence` is a strict `Literal`; out-of-vocab values cause the whole
  finding to be dropped before this capture runs (see
  `_drop_invalid_findings` and `bad_findings.jsonl`). So there is no
  `raw_confidence` column — it would always equal `confidence` for
  successful rows.

---

## Bug 16 — `demographic` spelling and confidence casing divergence

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:**
Summarisation MAP prompt + Pydantic models.

### Symptom

Two prompt-schema inconsistencies that were live in production:

1. **`demographic` / `demographics` split.** `category` enum (Literal)
   used `"demographics"` (plural). `relation_type` enum
   (`RelationTypeEnum`) used `"demographic"` (singular). The MAP prompt
   had to carry an explicit warning about the divergence
   (`prompts.py:30, 73-75`) and a `_CATEGORY_ALIASES` map was needed to
   repair `"demographic" → "demographics"` on `category`. Every category
   sibling (`morphology, IHC, molecular_genetics, staging, treatment,
   prognosis`) is singular — `demographics` was the lone plural even
   within its own enum.
2. **`confidence` casing.** `Finding.confidence` =
   `Literal["high","medium","low"]`. `Rule.confidence` =
   `Literal["High","Medium","Low"]`. MAP prompt + RULE prompt
   instructed the LLM in matching cases. Different fields, but
   gratuitously inconsistent.

### Diagnosis

Both inconsistencies were accidents of incremental development, not
semantic. The first one even had a warning in the prompt explaining "do
not confuse them" — the right reaction is to remove the divergence, not
document it.

### Fix

1. Aligned `category` Literal → `"demographic"` across all 5 occurrences
   in `models.py` (`Finding`, `NormalFinding`, `FindingGroup`,
   `AtomicFinding`, `CanonicalRule`, `FinalRule`), plus the alias-repair
   `valid_values` list.
2. Inverted `_CATEGORY_ALIASES` to `{"demographics": "demographic"}` —
   legacy LLM output that still says `"demographics"` is now repaired
   the other way.
3. Rewrote MAP prompt category list (`prompts.py:26`); dropped the
   divergence warning at lines 30 and 73-75; updated the OutputFormat
   exemplar at line 175.
4. Aligned `Rule.confidence` Literal → lowercase (`models.py:368`).
   Updated RULE prompt exemplar at `prompts.py:298`.
5. Updated `routing/schema_validator.py:_VALID_CATEGORIES`.
6. Bumped `MAP_PROMPT_VERSION` → `map_prompt_v2_singular_demographic`
   so MAP caches built against the old prompt are invalidated.
7. Updated tests: `tests/summarization/test_demographics.py` and
   `scripts/test_map_schema.py` — assertions flipped, alias-repair
   direction inverted.

### Verification

12/12 `test_demographics.py` pass; 11/11 enum-related checks in
`scripts/test_map_schema.py` pass.

### Out of scope

* `eval/silver/prompts.py` still uses `"demographics"` — own
  `PROMPT_VERSION`, separate cache lineage. Main pipeline's alias-repair
  handles it. TODO entry filed for next silver regeneration.
* `README.md` example snippets show old casing — cosmetic only.
* Existing DB rows from pre-fix runs hold `"demographics"`. User plan is
  to re-run MAP rather than backfill.

---

## Bug 17 — Batch entry-points pass no `db=` to `build_batch_runner`

**Status:** Fixed (2026-05-15) · **Severity:** High · **Surface:**
Summarisation, `scripts/run_paper.py` batch entry-points.

### Symptom

Surfaced during a 5-paper batch run on 2026-05-15: results landed in
`out/summaries/summaries/*.json` and the final-rule counts printed
correctly, but no `CORPUS RELATE incremental` log line appeared and no
rows landed in `sum_corpus_relations`. Spot-checks confirmed the
per-paper `sum_*` tables were also empty.

### Diagnosis

`pipeline/stages/summarization/batch/runner.py:223` accepts
`db=None` and stores it on `self._db`. Every persistence helper
(`_persist_map_findings`, `_persist_normal_findings`,
`_persist_finding_groups`, `_persist_canonical_rules`,
`_persist_relations`, `_persist_final_rules`,
`_persist_rejection_summary`) and `_corpus_relate_incremental` opens
with the same guard:

```python
if self._db is None:
    return
```

— silent return, no log. So if the runner is built with `db=None`,
every `sum_*` table stays empty and cross-paper RELATE never runs.

`scripts/run_paper.py` had two batch entry-points and both built the
runner without a DB:

* `_run_batch_multi` (line 766) — `build_batch_runner(profile_name=...,
  artifact_root=..., artifact_run_id=...)`
* `_run_batch_single` (line 863) — same shape.

The sync path at the original `build_runner` already opened a DB
connection inline (lines 154-163) and threaded it into
`SummarizationRunner(db=db_conn)`. The batch path was simply left behind
when the connection-opening pattern landed.

This was the residual gap from [B-005](#bug-5--batch-runner-missing-sync-parity-features):
the batch runner was made *capable* of persisting and running
cross-paper RELATE (the `_persist_*` helpers and
`_corpus_relate_incremental` were copied verbatim from the sync runner),
but the *callers* of the batch runner were never updated to pass the
ingredient those helpers need (`db`).

### Fix

* New module-level helper `_open_db_connection(caller_label)` at
  `scripts/run_paper.py` — wraps `from database import get_db_connection`
  in a try/except, logs a warning with the caller label on failure,
  returns `None` (non-fatal — smoke tests on machines without Postgres
  still work).
* `build_runner` (sync path) now calls
  `_open_db_connection("build_runner")` instead of duplicating the
  try/except. Same behaviour, less drift surface.
* `_run_batch_multi` (`scripts/run_paper.py:770`) now passes
  `db=_open_db_connection("build_batch_runner (multi)")` to
  `build_batch_runner`.
* `_run_batch_single` (`scripts/run_paper.py:867`) — same.

The `caller_label` is threaded through to the warning string so logs
distinguish which entry-point failed to reach Postgres.

### Verification

`python -c "import ast; ast.parse(open('scripts/run_paper.py').read())"`
passes. End-to-end: re-running the same 5-paper batch should now produce
`[<pmcid>] CORPUS RELATE incremental: N cross-paper relations` log lines
during finalize() and populated `sum_*` tables. Verification deferred
until the next user-driven batch run since reproducing the original
symptom requires a real LLM cascade pass.

### Why "High" severity

Every batched production result between commits `5c59c3e` (2026-04-27)
and the 05-15 fix sat on disk only. No DB rows means: corpus-level
queries return empty, the inspector pages show "no relations found", the
thesis CORPUS_RELATE evidence pipeline has no input. The on-disk JSON
artifacts let you replay via the offline standalone
(`python -m pipeline.stages.summarization.helpers.corpus_relate`), but
that's a manual recovery step the user has to run for every historical
batch run.

## Bug 18 — `relation_type="prognosis"` (noun form) coerced to `unclear` instead of `prognostic`

### Status / Severity / Surface

Fixed (2026-05-15) · High · `pipeline/stages/summarization/models.py` MAP enum validators.

### Symptom

Running `scripts/run_paper.py --from-selection configs/paper_selection/calibration_set_v1.yaml --profile cheap` on 2026-05-15 produced waves of:

```
WARNING  pipeline.stages.summarization.models  Unknown relation_type 'prognosis' — coercing to 'unclear'
WARNING  pipeline.stages.summarization.models  Dropping malformed finding: 1 validation error for Finding
  category
    Input should be 'morphology', 'IHC', 'molecular_genetics', 'staging', 'treatment', 'prognosis' or 'demographic'
    [type=literal_error, input_value='expression', input_type=str]
```

The downstream effect was invisible in the warnings themselves: `is_groupable()` returns `False` for `relation_type=unclear` (`group_stage.py:39`), so every coerced finding silently disappeared at GROUP — never reaching CANONICALIZE / RELATE / RESOLVE. The category-failed findings were dropped one stage earlier by `AuditableSummary._drop_invalid_findings` (`models.py:337`).

### Evidence

`logs/bad_findings.jsonl` had 4 entries from this run, all with `"category": "expression"` and `"relation_type": "expression"`, e.g.:

```json
{"raw": {"category": "expression", "relation_type": "expression",
         "claim": "PhH3-positive cells counted in 2 mm^2 area",
         "verbatim_support": "PhH3-positive cells were counted in an area of 2 mm 2..."}}
```

`logs/enum_observations.jsonl` had 127 records, with `relation_type="prognosis" → "unclear"` (reason: `unknown_value`) the dominant pattern after the demographic-plural alias hits. The user's terminal showed ~10 `prognosis` warnings on PMC9826086 alone before the L1 batch returned.

### Diagnosis

Vocabulary asymmetry between two adjacent fields in the same prompt:

| Field | Valid values |
|---|---|
| `category` | morphology, IHC, molecular_genetics, staging, treatment, **prognosis**, demographic |
| `relation_type` | has_feature, expression, **prognostic**, comparative, demographic, treatment_response, unclear |

`category` uses the noun *prognosis*; `relation_type` uses the adjective *prognostic*. Both surface in `prompts.py` line 26 (category) and lines 67–69 / 140–144 (relation_type). L1 voters (gpt-4o-mini, gpt-4.1-nano, gemini-2.5-flash-lite) pattern-match on the lexeme rather than the field semantics and bleed `prognosis` into `relation_type`. The pre-existing `_coerce_invalid_relation_type` validator (`models.py:272`) had no alias map, so it fell straight through to the unknown-value branch and coerced to `unclear`. Mirror-image symptom on `category`: `expression` is a valid `relation_type` value that the model bled the other direction, which fails `Literal[...]` validation and drops the entire Finding.

`demographic` is the only label that's identical across the two fields and is — not coincidentally — the only one that never appears in `enum_observations.jsonl` under either field.

### Fix

`pipeline/stages/summarization/models.py`:

* Added `_RELATION_TYPE_ALIASES: dict[str, str] = {"prognosis": "prognostic"}` immediately below the existing `_CATEGORY_ALIASES` definition (`models.py:167-170`).
* Inserted an alias-repair branch in `_coerce_invalid_relation_type` *before* the unknown-value fallback, with `log_enum_observation(reason="alias_repair")` so the raw value still lands in `logs/enum_observations.jsonl` for future schema-expansion review.
* Bumped `MAP_SCHEMA_VERSION` → `"map_v2_relation_type_alias_repair"` (`models.py:23`) to invalidate the entire MAP cache (cache key includes `schema_version` per `cache.py:96`).

Two options were considered:

1. **Alias repair (chosen)** — one-line dict + validator branch; no schema/DB migration; raw value preserved on the `Finding._raw_relation_type` PrivateAttr (per B-015) so the original `"prognosis"` is still recoverable downstream.
2. **Vocabulary alignment** — rename `RelationTypeEnum.prognostic` → `prognosis` so both fields share the lexeme. Cleaner long-term but touches every persistence site (NormalFinding, FindingGroup, CanonicalRule, Relation, FinalRule) and would need an Alembic migration over existing `sum_*` rows. Deferred until alias-repair frequency is known from a few calibration runs.

`category="expression"` is **deliberately not aliased**. Collapsing it to `IHC` would silently lose `molecular_genetics`-class findings, and the LLM's choice of `"expression"` is signal that the rubric needs sharper category guidance — not an alias.

### Verification

Test plan for the next run:

* Bump-driven: every chunk re-calls the LLM (no cache hits) — verified by `cache.py:96` keying on `schema_version`.
* Tail `logs/enum_observations.jsonl` for `reason: alias_repair` events on `field_name: relation_type`.
* Confirm `runner.py:1837` `non_groupable_unclear_relation` counter drops compared to the 2026-05-15 baseline.
* Confirm `FinalRule.relation_type == "prognostic"` rows appear in `sum_final_rules` for prognostic claims that were missing from the prior run.

Smoke checked the patch by reading `_coerce_invalid_relation_type` after edit; full pipeline re-run pending user.

## Bug 19 — `MapOutputRouter` citation regex rejects suffixed PMCIDs, silently strips voters

### Status / Severity / Surface

Fixed (2026-05-15) · High · `pipeline/stages/summarization/routing/schema_validator.py:23` and `pipeline/stages/summarization/routing/provenance_validator.py:27`.

### Symptom

Cheap-profile MAP cascade on `calibration_set_v1` papers (suffixed pmcids like `PMC10100421_HIS-82-393`, `PMC6635746_HIS-73-68`) was running with `voter_count=1` at every L1 decision instead of the 3 voters defined in the profile. Surfaced as a question about why "real-profile" papers showed `voter_count=3` while suffixed papers showed `voter_count=1` in `out/summaries/cascade_decisions/<pmcid>.jsonl`.

Knock-on: every chunk passed the L1 agreement gate trivially (single-voter agreement = 1.0), making the thesis-level claim "cheap profile saves cost via L1 consensus" misleading — there was no consensus, just a single L1 voter being rubber-stamped.

### Evidence

Cascade decisions for `PMC6635746_HIS-73-68` (suffixed, cheap profile):

```
level=l1 voter_count=1 elig=null decision=keep ...   (× every chunk)
```

Cascade decisions for `PMC7150046_main` (same suffix shape, but already on a `real`-profile run):

```
level=l1 voter_count=3 elig=null decision=escalate ...
level=l2 voter_count=3 elig=null decision=escalate ...
level=l3 voter_count=1 elig=null decision=keep ...
```

The voter_count discrepancy is the smoking gun. The `cheap` profile registers three L1 voters (`make_cheap_profile`, `voter_configs.py`) so `voter_count=1` indicates router-level voter dropping rather than profile mismatch.

Regex check against the live citation form:

```
^S\d+\|PMC\d+\|\d+$
  S1|PMC123|456                           → match
  S1|PMC10100421_HIS-82-393|123           → no match (rejects at "_")
  S1|PMC7150310_main|456                  → no match
  S1|PMC4329418_his0066-0409|7            → no match
```

Every citation in every suffixed-pmcid paper failed validation. Two of three L1 voters always lost the lottery for which one parsed cleanly (probably the third "won" by chance of slightly different LLM citation rendering — needs follow-up to confirm).

### Diagnosis

`MapOutputRouter` has been default-on since 2026-05-14 (`STRUCTURE.md` changelog row "MapOutputRouter wired into both runners by default"). The router runs `SchemaValidator._validate_finding()` against each voter's `Finding.evidence` list. A malformed citation produces `ReasonCode.INVALID_SENTENCE_ID`, which sits in `_HARD_CODES` (`router.py:53`), classifying the voter UNUSABLE. UNUSABLE voters are dropped from the agreement matrix before the gate decision.

The bug originates from the regex predating the suffixed-pmcid doc-id convention. The pipeline uses suffixed document IDs as opaque keys (set by `file-selector/pdf_organizer.py` from the source filename) and threads them through `_format_sentences` (`map_stage.py:64`) verbatim, so every citation embeds the suffix. The router validator then rejects them.

`provenance_validator.py:27` has an identical regex (with capture groups) for the cross-document equality check. Same vulnerability; same fix.

### Fix

Relaxed PMC token in both regexes:

```python
# schema_validator.py:23
_CITATION_RE = re.compile(r"^S\d+\|PMC[\w\-]+\|\d+$")

# provenance_validator.py:27
_CITATION_RE = re.compile(r"^S(\d+)\|(PMC[\w\-]+)\|(\d+)$")
```

`[\w\-]+` accepts word characters (alphanumeric + underscore) and hyphens — covers every observed suffix shape (`_main`, `_HIS-82-393`, `_his0066-0409`) and plain digit-only pmcids.

The cross-document safety check at `provenance_validator.py:116` is exact equality (`if cited_pmcid != self._pmcid:`) so the broader regex does not loosen the cross-paper-citation rejection — captured pmcid still has to match the expected doc id exactly.

Bumped `MAP_SCHEMA_VERSION` → `"map_v4_citation_regex_suffixed_pmcids"`. Reason: cached `AuditableSummary` rows produced under the broken router were selected from a 1-of-3 voter pool — after the fix the same chunks should re-vote with 3-of-3 and may select a different "best" output. Cache invalidation forces the recompute.

### Verification

* Synthetic regex smoke test confirmed the four representative citation shapes parse correctly and the four malformed shapes still reject. Passed.
* End-to-end verification deferred until the next user-driven calibration_set_v1 run. After the run, check `out/summaries/cascade_decisions/PMC6635746_HIS-73-68.jsonl`: every L1 row should now report `voter_count=3`, and the L1 acceptance rate should drop materially as real disagreements surface and escalate.

### Why "High" severity

Two compounding consequences:

1. Every cheap-profile run on every suffixed-pmcid paper since 2026-05-14 was effectively a single-voter run. Voter-agreement evidence used in the thesis (acceptance rate, L1 cost savings, deferral score distributions) was computed on a 1-voter denominator. Re-running the affected papers is necessary to produce defensible numbers.
2. The bug pattern — regex too strict for an evolving doc-id convention — has the same shape as the enum-coercion bugs (B-016, B-018): silently drops content with no surfacing other than a non-obvious downstream count. Adding the suffixed shapes to a unit test (`tests/summarization/routing/test_schema_validator.py`) would catch any future tightening of the PMC token.

## Bug 20 — Batch dispatch groups by provider only (not provider × model); OpenAI multi-model batches rejected silently

### Status / Severity / Surface

Fixed (2026-05-15) · High · `pipeline/stages/summarization/batch/dispatch.py::submit_level`.

### Symptom

User noticed during the 2026-05-15 calibration_set_v1 batch runs that the OpenAI API dashboard showed no charge despite cost reports claiming cheap-profile cascade runs. Cost reports for the same runs showed token usage for `gemini-2.5-flash-lite` only at L1 — the two OpenAI L1 voters (`gpt-4o-mini`, `gpt-4.1-nano`) recorded zero tokens.

The cascade decision logs were superficially fine — `decision: keep` at L1 on every chunk — but `voter_count: 1` rather than 3 (the cheap profile defines three L1 voters).

### Evidence

Cheap-profile L1 token usage from `out/summaries/reports/escalation_report_20260515T143649.json`:

```
PMC4329418_his0066-0409 batch 15 chunks
  l1 gemini-2.5-flash-lite   in=  47558 out=  54454
PMC6635746_HIS-73-68 batch 12 chunks
  l1 gemini-2.5-flash-lite   in=  39345 out=  59349
# zero rows for gpt-4o-mini or gpt-4.1-nano
```

Compare against a sync-mode run from 2026-05-14 (`escalation_report_20260514T112408.json`) on the same profile:

```
PMC7150310_main sync 37 chunks
  l1 gpt-4.1-nano           in= 133438 out=  52180
  l1 gemini-2.5-flash-lite  in= 116943 out= 124721
  l1 gpt-4o-mini            in= 133438 out=  50362
```

Sync logs all three voters as expected; batch logs only Gemini.

Inspection of the batch handle (`out/summaries/batch_handles/PMC4329418_his0066-0409.batch.json`):

```json
jobs: [
  {"provider": "gemini",  "model": "gemini-2.5-flash-lite", "status": "completed",
   "request_count": 15, "job_id": "batches/cyzc..."},
  {"provider": "openai",  "model": "gpt-4o-mini",           "status": "failed",
   "request_count": 30,  "job_id": "batch_6a072daf..."}
]
l1_raw: 15 entries, ALL with custom_id ending in __0  (voter index 0 = gemini)
```

The OpenAI job request_count is 30 (= 15 chunks × 2 OpenAI voters), but it merged both models into one batch. Retrieving the OpenAI batch directly:

```
status='failed', output_file_id=None,
errors.data = [
  BatchError(code='mismatched_model', line=2,
             message='The model for this request does not match the rest of
                      the batch. Each batch must contain requests for a
                      single model.', param='body.model'),
  …  # one error per even-numbered line, 15 total
]
```

OpenAI accepted the first model that appeared on line 1, rejected every subsequent request whose model didn't match. The batch ended with status=`failed` and `output_file_id=None`. `OpenAIBatchProvider.check()` correctly translates this to `job.status='failed'`, but `BatchSummarizationRunner` simply skips failed jobs — no surfacing in the cost report, no exception, no warning beyond the in-method log line.

### Diagnosis

`dispatch.submit_level` (`dispatch.py:300-309`) groups by `req.provider` only:

```python
by_provider: dict[str, list[BatchRequest]] = {}
for req in all_requests:
    by_provider.setdefault(req.provider, []).append(req)
…
for pname, reqs in by_provider.items():
    job = providers[pname].submit(reqs, OPENAI_MAP_TOOL)
```

The OpenAI Batch API explicitly forbids mixed-model batches — this is a documented hard constraint, not a strict-mode interaction. Other providers do not share the constraint today (Anthropic's Message Batches API supports mixed models, Gemini batch is single-model-per-job by request shape so the issue doesn't arise), so the bug only surfaces when OpenAI L1 has ≥ 2 distinct models.

Compounding factor: cheap-profile L1 is the only place where two OpenAI voters appear. L2 (`gpt-4.1-mini`) and L3 (also `gpt-4.1-mini`) are single-model, so the L2/L3 escalation paths were always working — that's why `escalation_report_20260515T143649.json` does show one tiny L2 row.

### Fix

Group by `(provider, model)` so each unique model produces its own batch submission. Conservative; works for every provider; the only cost is one extra OpenAI HTTP submission per profile (negligible vs. the 5+ minute batch wait):

```python
by_provider_model: dict[tuple[str, str], list[BatchRequest]] = {}
for req in all_requests:
    by_provider_model.setdefault((req.provider, req.model), []).append(req)

providers = build_providers({pname for (pname, _model) in by_provider_model})
jobs: list[ProviderJob] = []
for (pname, _model), reqs in by_provider_model.items():
    job = providers[pname].submit(reqs, OPENAI_MAP_TOOL)
    jobs.append(job)
```

Bumped `MAP_SCHEMA_VERSION` → `"map_v5_batch_group_by_provider_model"`. The existing batch handles on disk hold a `jobs` list that omits the failed-but-real OpenAI work; the version bump invalidates them so the next run rebuilds the handle with the correct (provider, model) split.

### Verification

* Unit-style group-by smoke test (run inline): six requests across one Gemini + two OpenAI models produced 3 groups (Gemini × 1, gpt-4o-mini × 2, gpt-4.1-nano × 2) instead of 2 (Gemini × 1, OpenAI × 4). Confirmed correct splitting.
* End-to-end verification deferred until the next calibration_set_v1 batch run. Post-run, the cost report should show non-zero tokens for `gpt-4o-mini` AND `gpt-4.1-nano` at L1, and `out/summaries/cascade_decisions/<pmcid>.jsonl` should report `voter_count=3` on every L1 row (combined with B-019's citation regex fix).
* Pre-fix `request_count=30` failed OpenAI batch on `batch_6a072daf3600819098e020c285a84808` — kept on OpenAI as a historical reference.

### Why "High" severity (compounds with B-019)

Two separate bugs both reduced L1 to a single-voter run:

| Bug   | Mechanism                                              | Effective L1 voters |
|-------|--------------------------------------------------------|---------------------|
| B-019 | Citation regex rejects suffixed pmcids → router drops voters | 1 (Gemini)          |
| B-020 | OpenAI multi-model batch rejected wholesale → 2 of 3 voters never executed | 1 (Gemini)          |

Either alone would have produced `voter_count=1`. Together they made the symptom completely invisible — even after fixing B-019, the cascade would still have been single-voter on batch mode until B-020 was found. Every published cost figure or L1-acceptance figure from batch mode runs since the multi-model OpenAI L1 profile was introduced is invalid and must be regenerated for the thesis.

### Follow-up

* Add an end-to-end unit test in `tests/summarization/batch/test_dispatch_group.py` that constructs a multi-model voter list and asserts `len(jobs) == len(distinct models)` — guards against the same regression.
* Consider surfacing failed-job status as an exception (or at minimum a `WARNING`-level cost-report annotation) so a partial cascade run can't masquerade as a complete one.

---

## Bug 21 — CANONICALIZE `no_direction` treated as real polarity due to `"None"` string typo

### Status / Severity / Surface

* **Status:** Fixed (2026-05-15)
* **Severity:** High — silently corrupted every CanonicalRule whose underlying findings carry `direction=no_direction`, which is the value MAP assigns when the LLM emits `null`/`""`/`"null"` for direction (see `models.py:_coerce_invalid_direction`, line 401).
* **Surface:** `pipeline/stages/summarization/current_stages/canonicalize_stage.py`

### Symptom

`_split_by_direction` produces a `direction="no_direction"` bin and emits a CanonicalRule with `direction=DirectionEnum.no_direction` instead of folding the finding into the dominant direction's bin. `_compute_scope_fields` then counts `"no_direction"` as a polarity-bearing direction, falsely flipping `is_conflicted=True` whenever a mixed bin happens to contain both a real direction and a `no_direction` finding.

### Evidence

`canonicalize_stage.py:57` (in `_compute_scope_fields`):

```python
for nf in member_nfs:
    d = nf.direction.value if nf.direction is not None else "unclear"
    if d not in ("unclear", "None"):
        bin_directions.add(d)
is_conflicted = len(bin_directions) >= 2
```

`canonicalize_stage.py:96-98, 103-107` (in `_split_by_direction`):

```python
non_unclear = {
    d: [] for d, c in group.direction_counts.items()
    if d not in ("unclear", "None") and c > 0
}
unclear_nfs: list[NormalFinding] = []

for nf in member_nfs:
    d = nf.direction.value if nf.direction is not None else "unclear"
    if d in non_unclear:
        non_unclear[d].append(nf)
    else:
        unclear_nfs.append(nf)
```

`DirectionEnum` (`models.py:83-89`) values: `positive`, `negative`, `absent`, `partial`, `unclear`, `no_direction`. **There is no value `"None"`.** The string is dead code; the author intended `"no_direction"`.

### Diagnosis

The MAP enum-coercion path explicitly normalises missing direction to `DirectionEnum.no_direction` precisely because "direction does not apply" is semantically distinct from "the model could not decide" (`unclear`). CANONICALIZE was meant to treat both buckets the same way (both should bypass the polarity split and stay attached to the dominant direction), but the typo means only `unclear` is bypassed. `no_direction` reaches CANONICALIZE as a first-class polarity:

1. `direction_counts` from GROUP carries a `"no_direction": k` entry — see `group_stage._direction_counts` (lines 101–106), which calls `m.direction.value`.
2. `_split_by_direction` accepts `"no_direction"` into `non_unclear` because `"no_direction" not in ("unclear", "None")`.
3. The finding gets its own CanonicalRule with `direction=DirectionEnum.no_direction`, and `is_conflicted` is set true whenever any other non-unclear direction is also present in the original group.

### Fix

Replace the literal `"None"` with `"no_direction"` in both call sites:

```python
# _compute_scope_fields
if d not in ("unclear", "no_direction"):
    bin_directions.add(d)

# _split_by_direction
non_unclear = {
    d: [] for d, c in group.direction_counts.items()
    if d not in ("unclear", "no_direction") and c > 0
}
```

Both spots are self-contained — no schema or DB migration needed. Bump no version constant.

### Verification

Shipped 9 regression tests in `tests/summarization/test_canonicalize_direction_split.py`:

* `test_no_direction_alone_is_not_conflicted` — pure-`no_direction` bin keeps `is_conflicted=False`.
* `test_no_direction_plus_positive_is_not_conflicted` — the original bug case. Pre-fix returned `True` because `no_direction` was counted as a distinct polarity alongside `positive`; post-fix returns `False`.
* `test_unclear_plus_positive_is_not_conflicted` — guards that the pre-existing `unclear` handling is preserved.
* `test_positive_plus_negative_is_conflicted` — guards that genuine polarity contradictions still flip `is_conflicted=True`.
* `test_split_no_direction_only_collapses_to_unclear_bin` — all-`no_direction` group falls to the single `"unclear"` bin path instead of emitting a CanonicalRule with `direction="no_direction"`.
* `test_split_no_direction_joins_single_polarity_bin` — `no_direction` finding in a positive-dominant group joins the `positive` bin (no second CanonicalRule).
* `test_split_no_direction_attaches_to_largest_polarity_in_mixed_group` — in a mixed `(positive=2, negative=1, no_direction=1)` group the `no_direction` attaches to the largest polarity bin, not a third bin.
* `test_split_real_polarity_split_preserved` — guardrail against an over-aggressive fix that would also collapse genuine `positive`/`negative` contradictions.
* `test_study_coverage_multi_study` — guards that the `study_coverage` computation is independent of the `is_conflicted` change.

Full file: `pytest tests/summarization/test_canonicalize_direction_split.py` → 9 passed.

Docstrings for both helpers were updated to spell out the orthogonality (`unclear` = "model couldn't decide", `no_direction` = "polarity doesn't apply"; both bypass the polarity split).

---

## Bug 22 — `group_id` mixes CUI and string keys when CUI population is partial

### Status / Severity / Surface

* **Status:** Fixed (2026-05-15)
* **Severity:** High — defeated the per-paper dedup that GROUP exists to perform, and inflated CanonicalRule count for any paper where UMLS linkage was intermittent.
* **Surface:** `pipeline/stages/summarization/current_stages/group_stage.py`

### Symptom

Two `NormalFinding`s with identical normalized `subject_entity` (or `outcome_entity`) string land in different `FindingGroup` buckets, producing two `CanonicalRule`s where one was intended. Visible as inflated `len(canonical_rules)` and per-rule `finding_count=1` instead of the expected merged count.

### Evidence

`group_stage.py:57-71`:

```python
def _group_id(
    subject: str,
    outcome: str,
    relation_type: str,
    category: str = "",
    subject_cui: str | None = None,
    outcome_cui: str | None = None,
    pmcid: str = "",
) -> str:
    subj_key = subject_cui if subject_cui else subject
    out_key = outcome_cui if outcome_cui else outcome
    return (
        f"GRP_{_sha8(pmcid)}_{_sha8(subj_key)}_{_sha8(out_key)}"
        f"_{relation_type}_{_sha8(category)}"
    )
```

Two NFs with `subject_entity="CD30"`:
* NF₁: `subject_cui="C0054954"` → `subj_key = "C0054954"`
* NF₂: `subject_cui=None`         → `subj_key = "CD30"`

`_sha8("C0054954") != _sha8("CD30")` → different group_ids → different buckets.

### Diagnosis

`NormalizeStage._resolve_entity` (`normalize_stage.py:176-202`) returns `(canonical_name, cui | None)`. The synonym-dict path looks up the canonical name in `_SYNONYMS` then attempts a *follow-up* UMLS lookup on the canonical form:

```python
from_dict = synonyms.get(stripped.lower())
if from_dict is not None:
    _, cui = _umls_canonical_with_cui(from_dict)
    return from_dict, cui
```

That follow-up runs scispaCy NER on the bare canonical token (e.g. `"OS"`, `"CD30"`). For short acronyms, scispaCy frequently fails to entity-link or links inconsistently across runs (it depends on the in-document context, which is absent for a bare token). Net: identical canonical strings, drifting CUIs.

Same root cause for the UMLS-only path: `_umls_canonical_with_cui` returns `(canonical, None)` whenever `_best_cui` filters the link out for low score or junk-CUI status. Surface forms inside one paper that hit different scispaCy spans can therefore land with and without a CUI.

### Fix

**Shipped 2026-05-15** — Option A. Dropped CUI from `_group_id` entirely. GROUP is per-paper; `NormalizeStage._resolve_entity` already canonicalises `subject_entity` deterministically. The CUI rides along on `NormalFinding.{subject,outcome}_cui` for downstream use (`corpus_relate` cross-paper matching) but no longer participates in the per-paper bucket key. New signature:

```python
def _group_id(
    subject: str,
    outcome: str,
    relation_type: str,
    category: str = "",
    pmcid: str = "",
) -> str:
    return (
        f"GRP_{_sha8(pmcid)}_{_sha8(subject)}_{_sha8(outcome)}"
        f"_{relation_type}_{_sha8(category)}"
    )
```

Rejected Option B (require CUI on both sides → two-pass merge): heavier change, same end result given that `_resolve_entity` is already string-deterministic.

### Verification

* Regression test `tests/summarization/test_phase3_group.py::test_b022_partial_cui_population_still_same_bucket` constructs two NFs with identical `subject_entity="CD30"` where one has `subject_cui="C0XXXXXX"` and the other has `None`. Post-fix: one `FindingGroup` with `member_ids={"NF_a","NF_b"}`. Pre-fix would have produced two singleton groups.
* `tests/summarization/test_phase3_group.py::test_b022_group_id_independent_of_cui` asserts the helper-level invariant — same inputs return the same key.

---

## Bug 23 — NORMALIZE dedup collapses opposite-direction findings from the same sentence

### Status / Severity / Surface

* **Status:** Fixed (2026-05-15) — Option A applied (add `direction` to the dedup key).
* **Severity:** Medium — affected every paper that had voter disagreement on direction within the same sentence; bounded by the rarity of that case but unbounded in semantic damage when it fired (the contradiction signal disappeared).
* **Surface:** `pipeline/stages/summarization/current_stages/normalize_stage.py`

### Symptom

Two MAP findings extracted from the same `text_element_id` with the same `(subject, outcome, relation_type)` but **opposite directions** (e.g. one positive, one negative) merge into a single `NormalFinding`. The merged finding inherits the direction of whichever source had the higher `grounding_score`; the opposing direction is dropped from the `NormalFinding` payload and is not visible to GROUP / CANONICALIZE / RELATE / RESOLVE.

### Evidence

`normalize_stage._dedup_key` (lines 281–302):

```python
def _dedup_key(
    text_element_id: int | None,
    subject: str | None,
    outcome: str | None,
    relation_type: RelationTypeEnum,
) -> str | None:
    ...
    return f"{text_element_id}|{subject}|{outcome}|{relation_type.value}"
```

`direction` is not in the tuple. `NormalizeStage._merge` (lines 479–526) picks one representative for the merged record:

```python
rep = max(findings, key=lambda f: f.grounding_score or 0.0)
...
direction = (
    infer_direction(rep.claim)
    if (rep.direction is None or rep.direction == DirectionEnum.unclear)
    else rep.direction
)
return NormalFinding(
    ...
    direction=direction,
    ...
)
```

The docstring at the top of `normalize_stage.py` defends this:

> "Opposing directions on the same entity pair surface as CONTRADICT relations in Phase 5 RELATE, not as separate groups."

That defence holds only when the opposing-direction findings reach RELATE as distinct `CanonicalRule`s. When they share `te_id`, they cannot.

### Diagnosis

The dedup key was designed under the assumption that "same sentence + same entity pair = same claim", with direction inferable from the claim text. In practice the ABC voter cascade emits multiple findings per chunk and per sentence; voters often disagree on direction. When that disagreement happens in the same sentence:

* Pre-merge: two `Finding`s, one with `direction=positive`, one with `direction=negative`.
* Post-merge (one `NormalFinding`): single direction, single `predicate_text`.
* GROUP: one bucket, one `FindingGroup` with `direction_counts={"positive": 1}` (only the surviving direction).
* CANONICALIZE: one `CanonicalRule`, `is_conflicted=False`.
* RELATE: nothing to compare against — the opposing finding ceased to exist at NORMALIZE.

The intended contradiction signal is therefore unreachable from this code path. The only way it surfaces today is when the opposing finding lives in a *different* text element — but the moment two voters disagree on direction for the same sentence (which is exactly the case that should be most informative), the disagreement is laundered.

### Fix

**Shipped 2026-05-15** — Option A. Added `direction` to `_dedup_key`:

```python
def _dedup_key(
    text_element_id: int | None,
    subject: str | None,
    outcome: str | None,
    relation_type: RelationTypeEnum,
    direction: DirectionEnum | None,
) -> str | None:
    ...
    dir_key = direction.value if direction is not None else "none"
    return f"{text_element_id}|{subject}|{outcome}|{relation_type.value}|{dir_key}"
```

Caller (`normalize_stage.py:410`) passes `f.direction` from the post-`_normalize_entities` finding (where the direction-from-claim heuristic has already fired for null/unclear), so two findings that the heuristic resolves to the same direction still merge.

`_merge` semantics unchanged: a homogeneous-direction cluster trivially preserves the direction. Picked Option A over the direction-aware-merge variant (Option B from the old design notes) because the calibration risk is contained — the inflation hazard the original docstring flagged ("voter noise collapsed") only triggers when voters genuinely emit *different* concrete directions, which is exactly the disagreement we want to preserve.

### Verification

* `tests/summarization/test_phase2_normalize.py::test_b023_opposing_directions_same_sentence_kept_separate` — positive + negative on same `te_id` produce 2 NormalFindings (pre-fix: 1).
* `tests/summarization/test_phase2_normalize.py::test_b023_same_direction_same_sentence_still_merges` — guard that the fix narrows the dedup rather than disabling it.

---

## Bug 24 — `Relation.nli_score_*` field doc disagrees with relate_stage write path

### Status / Severity / Surface

* **Status:** Mitigated (2026-05-15) — Option C (docstring-only) applied. Doc/code contradiction resolved, but the user-facing symptom ("downstream readers cannot tell which score they're looking at") persists: field names, DB column names, and inspector labels are unchanged. Full fix (Option B — split into `nli_entailment_*` / `nli_contradiction_*` columns + Alembic migration + inspector relabel) is the carry-forward; tracked in `docs/THESIS.md ##TODOs`.
* **Severity:** Low — no algorithmic effect on RESOLVE (which only reads `relation_type`, not the score), but every downstream reader sees mis-labelled numbers.
* **Surface:** `pipeline/stages/summarization/models.py:740-741`, `pipeline/stages/summarization/current_stages/relate_stage.py:397-413`, `scripts/run_paper_single_model.py:405`, `scripts/inspect_pipeline_output.py:127-128, 409-410, 798`.

### Symptom

The DB columns `sum_relations.nli_score_a_to_b` and `sum_relations.nli_score_b_to_a` (and identical columns on `sum_corpus_relations`) hold *entailment* scores for SUPPORT rows and *contradiction* scores for CONTRADICT rows. The inspector script renders the field as `A→B={:.2f}` without surfacing which score it is.

### Evidence

`models.py:740-741`:

```python
nli_score_a_to_b: float   # entailment score from A→B direction
nli_score_b_to_a: float   # entailment score from B→A direction
```

`relate_stage.py:397-413`:

```python
if label == RelationTypeLabel.CONTRADICT:
    score_ab = s_ab.get("contradiction", 0.0)
    score_ba = s_ba.get("contradiction", 0.0)
else:
    score_ab = s_ab.get("entailment", 0.0)
    score_ba = s_ba.get("entailment", 0.0)

relations.append(Relation(
    rule_id_a=rules[i].canonical_id,
    rule_id_b=rules[j].canonical_id,
    relation_type=label,
    nli_score_a_to_b=score_ab,
    nli_score_b_to_a=score_ba,
))
```

`scripts/run_paper_single_model.py:405`:

```python
f"(A→B={rel['nli_score_a_to_b']:.2f}, B→A={rel['nli_score_b_to_a']:.2f})"
```

### Diagnosis

`RawNLIPair` already persists `ent_a_to_b`, `ent_b_to_a`, `con_a_to_b`, `con_b_to_a` separately and unambiguously, so this is not an information-loss bug. The `Relation` field's semantic intent was apparently "store whichever score fired the classification" (a label-conditional projection), which is convenient but contradicts the field's docstring.

### Fix options

**Option A — rename to label-neutral.** Rename `nli_score_a_to_b` → `nli_signal_a_to_b` (and the `b_to_a` companion) with a docstring that explicitly says "entailment if SUPPORT else contradiction". Requires Alembic migration for the DB column rename, plus a churn pass across persistence and inspector code.

**Option B — split into two fields.** Add `nli_entailment_a_to_b` and `nli_contradiction_a_to_b` (and the `b_to_a` pair); deprecate the polysemic field. Heavier but unambiguous; surfaces the data needed for offline threshold sweeps without forcing readers to join against `RawNLIPair`.

**Option C — update the docstring only.** Cheapest. Doc says "entailment for SUPPORT, contradiction for CONTRADICT". Acceptable if no consumer is treating the field as a pure entailment score for offline analysis.

Recommendation: Option C now, schedule Option B for a future schema bump.

### Mitigation (vs Fix)

This is a **mitigation**, not a fix, and the distinction matters for thesis accounting:

* **Mitigation applied:** `pipeline/stages/summarization/models.py:776-779` — replaced the inline `# entailment score from A→B direction` comments on `Relation.nli_score_a_to_b` / `nli_score_b_to_a` with a label-conditional docstring (entailment for SUPPORT, contradiction for CONTRADICT, pointing readers at `RawNLIPair` for unambiguous per-direction values). Resolves the doc/code contradiction so future maintainers don't write code against the wrong contract.
* **Symptom still live:** the bug's stated symptom — *"downstream readers cannot tell which score they're looking at"* — is unchanged. Field names (`nli_score_a_to_b` / `nli_score_b_to_a`), DB column names (`sum_relations.nli_score_*`, `sum_corpus_relations.nli_score_*`), and inspector labels (`scripts/run_paper_single_model.py:405` prints `A→B={:.2f}`; `scripts/inspect_pipeline_output.py:127-128, 409-410, 798` likewise) all still surface the value without label context.
* **Safety condition not verified:** Option C's recommendation note — *"acceptable if no consumer is treating the field as a pure entailment score for offline analysis"* — was not audited. A grep across `scripts/`, `eval/`, and notebook code is required before declaring this fully resolved.
* **Full fix (Option B), deferred:** split into `nli_entailment_a_to_b` / `nli_contradiction_a_to_b` (and the `b_to_a` pair); deprecate the polysemic field; Alembic migration to add columns and backfill; relabel inspector callers. Tracked as a TODO in `docs/THESIS.md ##TODOs`.

### Verification

Mitigation is doc-only and intentionally untested. The Option B follow-up's verification (per the original recommendation) remains: construct two CanonicalRules whose NLI scores force CONTRADICT (`con_ab=0.9, con_ba=0.9, ent_ab=0.2, ent_ba=0.2`); assert the new `nli_contradiction_a_to_b` column receives `0.9` and the new `nli_entailment_a_to_b` column receives `0.2`.

---

## Bug 25 — RELATE polarity guard treats `partial` as positive, blocking partial-vs-positive contradictions

### Status / Severity / Surface

* **Status:** Observed — calibration question, not a clean defect.
* **Severity:** Low — depends on how often the LLM emits `partial`; if rare, near-zero impact.
* **Surface:** `pipeline/stages/summarization/current_stages/relate_stage.py:162-174`

### Symptom

A `CanonicalRule` with `direction=DirectionEnum.partial` paired against another with `direction=DirectionEnum.positive` cannot be classified CONTRADICT, even when the NLI scores cross the threshold in both directions. The pair is downgraded to UNRELATED via the same-polarity guard. `partial`-vs-`negative` and `partial`-vs-`absent` are unaffected — the POS / NEG sets are disjoint, so those pairs reach the NLI score check normally.

### Evidence

`relate_stage.py:162-174`:

```python
_NEGATIVE_DIRECTIONS = {DirectionEnum.negative, DirectionEnum.absent}
_POSITIVE_DIRECTIONS = {DirectionEnum.positive, DirectionEnum.partial}

dir_a = rule_a.direction
dir_b = rule_b.direction

same_polarity = (
    (dir_a in _POSITIVE_DIRECTIONS and dir_b in _POSITIVE_DIRECTIONS)
    or (dir_a in _NEGATIVE_DIRECTIONS and dir_b in _NEGATIVE_DIRECTIONS)
)
contradict_allowed = not same_polarity
```

`DirectionEnum.partial` is in `_POSITIVE_DIRECTIONS`, so `partial`-vs-`positive` lands in the "same polarity" branch and never reaches the CONTRADICT classification — even when the structured claim text really is in tension (e.g. "focal positivity in a subset of tumour cells" against "broadly / diffusely positive").

### Diagnosis

`partial` is the LLM's escape hatch for "expressed in some but not all tumour cells" / "weak focal positivity" / "tumour cells with partial staining". Semantically it sits between `positive` and `unclear` — closer to positive than negative, but not so unambiguously positive that it always coexists with a flat positive claim. Bundling it with `positive` for the polarity guard codes the two as interchangeable and forecloses the CONTRADICT classification for the subset of the corpus where the partial / full-positive distinction is the load-bearing semantic conflict.

### Fix options

**Option A — drop `partial` from `_POSITIVE_DIRECTIONS`.** Treat it as neutral for the polarity guard so `partial`-vs-`positive` reaches the NLI score check. Risk: NLI alone may mis-label a `partial`-vs-`positive` lexical-overlap pair as CONTRADICT when the two findings are really coexisting observations.

**Option B — separate `_PARTIAL_DIRECTIONS` set.** Keep the same-polarity branch for `partial`-vs-`partial` (and possibly `partial`-vs-`positive`) but introduce an explicit "partial-vs-positive contradiction allowed when NLI scores are high enough in both directions" rule.

Recommendation: defer until calibration. Requires a sweep on the gold set first — verify how often `partial` actually appears and what the NLI model does on real partial-vs-positive pairs from the corpus.

### Verification

Sweep `out/summaries/runs/.../relate/<pmcid>/raw_pairs.jsonl` for pairs where one rule has `direction=partial` and the other has `direction=positive` and both `con_*` scores exceed `contradiction_threshold`. Count how many such pairs would have flipped to CONTRADICT under each option. Make the call.

---

## Bug 26 — CANONICALIZE `_split_by_direction` tie-break is member-order-dependent

### Status / Severity / Surface

* **Status:** Observed
* **Severity:** Low — affects only the edge case where two non-unclear directions tie on count *and* the group additionally has unclear/no_direction members that need to be parked on the "largest" direction. Same input + same member order is reproducible today.
* **Surface:** `pipeline/stages/summarization/current_stages/canonicalize_stage.py:117-120`

### Symptom

When `_split_by_direction` sees `direction_counts={"positive": 2, "negative": 2, "unclear": 1}` (the unclear nf needs to be parked on the "largest" bin), the assignment of the unclear member depends on member insertion order rather than on a stable tie-break rule. Re-runs that reshuffle GROUP member iteration order produce CanonicalRules with subtly different `member_normal_ids` and `finding_count` for the unclear member.

### Evidence

`canonicalize_stage.py:117-120`:

```python
# Mixed directions — assign unclear nfs to the largest direction bin
largest_dir = max(non_unclear, key=lambda d: len(non_unclear[d]))
non_unclear[largest_dir].extend(unclear_nfs)
return list(non_unclear.items())
```

`max(dict, key=...)` returns the first key encountered when keys tie on the comparator. Dict iteration order is insertion order, and `non_unclear` is built from `group.direction_counts.items()`, which inherits its order from `_direction_counts(members)` in `group_stage.py:101-106` — which iterates `members` in arrival order.

### Diagnosis

The pipeline aims to be deterministic across re-runs. The only thing keeping it stable today is that upstream stages happen to feed `member_nfs` in a consistent order (NORMALIZE → GROUP propagates list order). Any future change that sorts NFs at GROUP entry would silently flip the tie-break — non-reproducible runs without an obvious smoking gun.

### Fix

Tie-break explicitly on a stable secondary key, e.g. wrap the keyspace in `sorted()`:

```python
largest_dir = max(sorted(non_unclear), key=lambda d: len(non_unclear[d]))
```

The `sorted()` wrapper gives a deterministic iteration order; `max` over it is stable.

### Verification

Unit test: build a `FindingGroup` with `direction_counts={"positive": 2, "negative": 2}` plus one unclear NF, then reverse the member list and call `canonicalize()` again. Pre-fix: the unclear NF lands on whichever direction was inserted first in `direction_counts`. Post-fix: it lands on the lexicographically-first (or whatever rule the tie-break chooses), independent of input order.

---

## Topic — Config-wiring audit of pipeline runners (2026-05-15)

Bugs B-027 through B-037 came out of a single audit pass over
`pipeline/stages/pdf_text_extraction/runner.py`,
`pipeline/stages/pdf_text_extraction/config.py`,
`pipeline/stages/summarization/runner.py`, and
`pipeline/stages/summarization/config.py`, looking specifically for config
fields that are defined (sometimes validated) but never read by a downstream
consumer, and for constructor parameters on components that the runner could
plumb from config but doesn't.

Common pattern: a config field exists in the dataclass with a docstring
suggesting it tunes behaviour, but `grep` finds zero reads outside the
config module itself. Two side-effects:

1. Users (or future-us) edit the field expecting a result and get nothing.
2. The fields become load-bearing in name only — `PipelineConfig.validate`
   still range-checks `num_workers`, for example, lending false weight to
   a knob that controls nothing.

Each row below is filed separately because each represents a distinct
defect with its own fix surface. The "Why High vs Medium vs Low" calls
trace to whether mis-tuning the knob changes pipeline output (Medium+) or
purely operational characteristics (Low).

---

## Bug 27 — `RuntimeConfig` knobs (`num_workers`, `log_level`, `seed`, `skip_existing_outputs`) not consumed

### Status / Severity / Surface

Fixed (2026-05-15) · High · PDF extraction, `PipelineRunner` runtime knobs.

### Resolution

All four knobs now drive code:

* **`num_workers`** + **`log_level`** — wired earlier (`runner.py:564`,
  `runner.py main()`).
* **`seed`** — `PipelineRunner.__init__` calls `_seed_pipeline()` which
  seeds `random` / `numpy` / `torch` (+ `torch.cuda` when available) and
  best-effort sets `PYTHONHASHSEED`. `seed` field widened to
  `int | None = 42` so callers can opt out (`None` skips seeding
  entirely). Torch / numpy `ImportError` is swallowed so seed init works
  in slim envs.
* **`skip_existing_outputs`** — new `_StageCache`
  (`pipeline/stages/pdf_text_extraction/stage_cache.py`) wraps the three
  expensive non-Docling stages: table detection (Step 2), artifact
  filtering (Step 5), text assembly (Step 6). Each cached artifact is
  written to `out/stage_cache/<stage_name>/<pmcid>.json` with a
  `<pmcid>.hash` sidecar. Both writes go through atomic temp+rename;
  loader and sidecar-read failures inside a narrow expected-error set
  log WARNING and fall through to recompute (bugs outside that set
  propagate). Stages 1/4 stay on Docling's existing per-PDF JSON cache;
  stages 3/7/8 always run.

### Scope (what this fix does NOT cover)

* **External-library determinism.** Seeding `random` / `numpy` / `torch`
  does not make Docling, TATR, OCR engines, or scispaCy deterministic.
  The `_seed_pipeline()` log line spells this out.
* **Dependency-version invalidation.** scispaCy / spaCy package and
  model-weights versions are NOT in the cache hash. A weights upgrade
  can change `nlp(text)` output without changing the model name string.
  `docs/HOW_TO_RUN.md` documents the manual remediation:
  `rm -rf out/stage_cache/`.
* **Behaviour-version invalidation.** When a cached stage's algorithm
  or a module-level constant it reads changes, contributors must bump
  `STAGE_CACHE_VERSION[stage_name]` in `stage_cache.py`. The constant's
  comment enumerates the cases; each cached stage in `runner.py` carries
  a one-line reminder.
* **Stages 7/8.** Final user-facing writers (media JSON, text file, DB
  ingest) always run. Step 8 has its own short-circuits
  (`skip_existing_in_db`, `skip_existing_media_json`).

### Verification

`tests/pdf_text_extraction/test_b027_seed_and_cache.py` — 22 tests:

* `_StageCache` primitives: first-run write, second-run hit,
  enabled=False recompute, corrupted artifact → recompute with WARNING,
  hash mismatch → CACHE STALE + sidecar-and-artifact overwrite,
  invalid-UTF-8 sidecar → CACHE INVALID, unexpected loader exception
  propagates, write failure propagates, artifact-without-sidecar
  emits its own message.
* Round-trips: `TableDetectionResult` (asserts `Path` + `int` page-dim
  keys preserved), `LayoutElement[]` (all five fields), `HierarchicalRow[]`
  (preserves `list[str]` for `path_list` and `source_chunks`).
* Runner wiring: standard + two-pass first-run write all three caches;
  second run uses a **fresh** `PipelineRunner` instance with raise-on-call
  stage mocks (proves the cache truly short-circuits the expensive call
  site, not just the factory); two-pass→standard mode flip emits 3×
  CACHE STALE; `skip_existing_outputs=False` recomputes.
* Seed: random.seed called with config value, `seed=None` skips all
  seeding (with `PYTHONHASHSEED` left alone), missing torch / numpy
  swallowed.
* Pin test: `test_runner_pass1_layout_feeds_table_detection` captures
  the layout passed to `_run_table_detection` in two-pass mode so a
  future refactor that diverges the input from `pass1_layout` will fail
  loudly.

`python -m pytest tests/pdf_text_extraction/test_b027_seed_and_cache.py -q`
→ 22 passed. Wider sweep
(`tests/pdf_text_extraction tests/parsers tests/test_config_loader.py
tests/summarization/test_phase2_normalize.py`) → 92 passed.

### Pre-resolution context (kept for history)

### Symptom

Setting `cfg.runtime.num_workers = 16` produces no behavioural change. Same
for `log_level`, `seed`, and `skip_existing_outputs`. The first is
validated by `PipelineConfig.validate` (`config.py:355-356`) which lends
the appearance that it controls something.

### Evidence

* `pipeline/stages/pdf_text_extraction/runner.py:481-523` (`run_batch`) is a
  sequential `for` loop — no thread / process pool.
* `pipeline/stages/pdf_text_extraction/batch.py:64-71` — `ParallelBatchRunner.__init__`
  takes its own `max_workers` kwarg, defaulting to `cpu_count // 2` when
  omitted. It never inspects `self._cfg.runtime.num_workers`.
* `pipeline/stages/pdf_text_extraction/runner.py:556` — `ParallelBatchRunner(cfg, max_workers=4)`
  hardcodes 4 in the example `main()`.
* `runner.py:527` — `logging.basicConfig(level=logging.INFO, …)` ignores
  `cfg.runtime.log_level`.
* `grep -rn "seed\|skip_existing_outputs" pipeline/stages/pdf_text_extraction`
  finds only the dataclass declaration and one docstring; no consumers.

### Diagnosis

The knobs were added when the runner was being designed but the wiring
never followed. `ParallelBatchRunner` predates `RuntimeConfig` (its kwarg
defaulted to `cpu_count // 2` before the field existed). `log_level` was
likely meant to be applied in `main()` but `basicConfig` was left
hardcoded.

### Fix

**Shipped 2026-05-15** (partial):

* `runner.py:564` — `ParallelBatchRunner(cfg, max_workers=cfg.runtime.num_workers)` (was hardcoded `4`).
* `runner.py` `main()` — `logging.basicConfig(level=cfg.runtime.log_level.value, …)` after `cfg.prepare()` so the field actually drives the level.

**Still open**:

* `seed` — no RNG entry points seed yet. Would need `random.seed`, `np.random.seed`, and (for torch-backed table detectors) `torch.manual_seed` called once at `PipelineRunner.__init__`.
* `skip_existing_outputs` — per-stage output-cache check. Substantial: each component (layout extractor, masker, cropper, writer) needs to honour the flag. Defer behind a "config audit" cycle.

### Verification

* `python -c "from pipeline.stages.pdf_text_extraction.config import PipelineConfig; c = PipelineConfig(); c.prepare(); print(c.runtime.num_workers, c.runtime.log_level.value)"` runs clean.
* Bumping `cfg.runtime.num_workers` propagates through `runner.main()` → `ParallelBatchRunner.__init__` (which now uses the passed kwarg).

---

## Bug 28 — `DatabaseConfig` sub-fields never propagated to `PostgresDatabaseIngester`

### Status / Severity / Surface

Fixed (2026-05-15, deleted) · High · PDF extraction, DB ingester.

### Symptom

Setting `cfg.database.schema = "histo_v2"` or `cfg.database.batch_size = 500`
has no effect on ingest behaviour. Same for `create_tables_if_missing` and
`connect_timeout_sec`.

### Evidence

* `config.py:158-165` (pre-fix) — `DatabaseConfig` declared `schema`,
  `create_tables_if_missing`, `batch_size`, `connect_timeout_sec`.
* `runner.py:178` — `PostgresDatabaseIngester(db_url=self._cfg.database.db_url)`.
  Only `db_url` is forwarded.
* `outputs/db_ingester.py:35` — constructor signature is
  `__init__(self, db=None, db_url: str | None = None)`. No other config-derived
  parameters accepted.

### Diagnosis

The ingester was written before the config split was finalised. The four
unused fields were aspirational. There is no schema-aware `MetaData` or
batched-insert path in `PostgresDatabaseIngester.write` — it `session.add`s
each ORM object then commits via the session context manager.

### Fix — deleted, not wired

Considered both options:

1. **Wire them.** Pass the four fields into the ingester, route `batch_size`
   through `session.bulk_save_objects`, plumb `connect_timeout_sec` into
   `get_db_connection` (currently no such kwarg exists in
   `database/db_connection.py`), route `schema` into `__table_args__`
   overrides or a `search_path` set on the engine.
2. **Delete them.** Honest minimal config — removes the maintenance surface
   of four fake knobs that nothing reads.

**Chose (2)** in the 2026-05-15 Tier 1 config audit. No current thesis
demand for non-default schema isolation, configurable connect timeouts, or
custom insert batching. `create_tables_if_missing` and `batch_size` look
like simple wires but easily expand to deeper refactors (DB initialisation
ordering, transactional batching, ORM `MetaData` rebinding) — risk
disproportionate to a zero-user-payoff feature. If a future need surfaces
(multi-tenant schema isolation, ingest perf), reintroduce the relevant
field with a real consumer + a test that proves it changes behaviour.

`DatabaseConfig` now exposes only `enabled: bool` and `db_url: Optional[str]`.

### Verification

* `grep -rn 'database\.schema\|database\.create_tables\|database\.batch_size\|database\.connect_timeout' pipeline/ scripts/ database/ tests/ configs/` → zero matches (post-deletion sweep).
* `tests/test_config_loader.py::test_deleted_database_keys_rejected` (parametrised over the four fields) — YAML referencing any of them raises `ValueError` at load time, surfaced by the strict-unknown-key check in `pipeline/config_loader.py:48-52`.
* `python -c "from pipeline.config_loader import load_config; load_config('configs/run.yaml')"` — clean load.
* `configs/run.yaml` already did not reference the four deleted fields, so the YAML required no edits.

---

## Bug 29 — `PipelineRunner._get_nlp` bypasses `umls_resources` singleton

### Status / Severity / Surface

Fixed (2026-05-15) · High · PDF extraction, scispaCy loader.

### Symptom

When the PDF extraction pipeline and summarisation pipeline both run in
the same Python process, scispaCy gets loaded twice. The summarisation
pipeline loads `en_core_sci_lg` via `umls_resources.get_nlp()` (the
documented process-wide singleton); the PDF pipeline loads
`en_core_sci_sm` via a private `spacy.load` call. Two scispaCy contexts
in one process; RSS rises by the size of `en_core_sci_sm` plus duplicated
Python wrapper objects.

### Evidence

* `pipeline/stages/pdf_text_extraction/runner.py:198-204`:
  ```python
  def _get_nlp(self):
      …
      import spacy
      self._nlp = spacy.load("en_core_sci_sm")
  ```
* CLAUDE.md ("Critical Patterns" → "scispaCy / UMLS loading"): "Always go
  through `pipeline/stages/summarization/umls_resources.py`
  (`get_nlp()` / `get_linker()`) — process-wide singleton. Loading
  `en_core_sci_lg` + the UMLS KB twice OOM-kills the pipeline."
* MEMORY.md ("Mistakes to avoid"): identical warning.

### Diagnosis

The summarisation singleton was introduced after `PipelineRunner` was
written. The PDF runner uses the smaller `en_core_sci_sm` model and
doesn't need the linker, so the original direct-load looked harmless. It
becomes harmful once both pipelines run together, which is exactly the
production code path (`scripts/run_paper.py` → extract → summarise).

Note: `en_core_sci_sm` vs. `en_core_sci_lg` are different models, so this
is duplication of unrelated objects rather than the double-KB OOM the
docs warn about — but it would become a real OOM the moment anyone bumps
the PDF runner to `en_core_sci_lg`.

### Fix

Added `umls_resources.get_small_nlp(model_name: str)` — a per-model singleton cache that honours `$NLP_HISTO_DISABLE_UMLS`. `PipelineRunner._get_nlp` (`pipeline/stages/pdf_text_extraction/runner.py:202`) now delegates to it instead of calling `spacy.load` directly. Co-fixed B-038 (the summarisation-side site).

### Verification

* `grep -n "spacy.load" pipeline/` returns zero hits outside `pipeline/stages/summarization/umls_resources.py`.
* `runner.py:202` calls `get_small_nlp(...)` from the summarisation singleton module.

---

## Bug 30 — `FilteringConfig` dead knobs: `fix_ligatures`, `remove_reference_markers`, `min_paragraph_chars`

### Status / Severity / Surface

Fixed (2026-05-15, deleted) · Medium · PDF extraction, filtering config.

### Resolution

All three fields removed from `FilteringConfig`. Each has only one possible
behaviour today: `fix_ligatures(...)` is unconditional in
`parsers/layout_utils.py`, `remove_citations(...)` is unconditional in
`extract_text`, and there is no minimum-paragraph-length filter at all. Adding
the wiring would have made every default a behaviour change for any user who
had been treating the dataclass as authoritative — deletion keeps the surface
honest with what the code actually does. The remaining `FilteringConfig`
fields (`enabled`, `apply_ner_filtering`, `apply_paragraph_relevance_filtering`)
are all consumed by `ArtifactFilter` / `extract_text`.

### Symptom

Setting `cfg.filtering.fix_ligatures = False` does not stop ligature
normalisation. `remove_reference_markers` and `min_paragraph_chars` are
referenced nowhere outside their dataclass declaration.

### Evidence

* `config.py:130-136` — declarations.
* `parsers/layout_utils.py:206,423,481` — `fix_ligatures(...)` called
  unconditionally on every text path; no flag check.
* `grep -rn "remove_reference_markers\|min_paragraph_chars" pipeline parsers`
  returns only the config declaration.

### Diagnosis

Hardcoded paths in `layout_utils` predate the config fields. The fields
were probably added when planning to make the behaviour optional, but the
flag plumbing never landed.

### Fix (proposed)

Either wire (thread the flag into `extract_text` and gate each call) or
delete. Recommendation: wire `fix_ligatures` (genuinely useful to disable
during debugging when the raw glyph output matters); delete
`remove_reference_markers` (post-stitch boilerplate filter already covers
references) and `min_paragraph_chars` (relevance filter covers length).

### Verification

Pending.

---

## Bug 31 — `TextAssemblyConfig` six of eight fields unread

### Status / Severity / Surface

Fixed (2026-05-15, deleted) · Medium · PDF extraction, text assembly config.

### Resolution

Six fields removed: `enabled`, `baseline_mode`, `use_hierarchical_extraction`,
`use_context_aware_stitching`, `compare_combinations`,
`save_combination_outputs`. The runner no longer routes around any of these —
hierarchical extraction + stitching are the production behaviour, the
masked-vs-unmasked A/B harness is gone, and the `compare_combinations` /
`save_combination_outputs` knobs targeted a research workflow that doesn't
exist anymore. The `BaselineMode` enum and its `pipeline.stages.pdf_text_extraction`
re-export were dropped at the same time. `TextAssemblyConfig` now exposes
only the two fields that drive code: `write_raw_text` (toggles
`out/text_raw/` dump in `runner.py`) and `pre_filter_relevance` (passed
through to `extract_text`). `tests/test_config_loader.py::test_enum_coerced_from_string`
was rewritten to use `LogLevel` instead of `BaselineMode` for its
enum-coercion smoke test.

### Symptom

Setting `cfg.text.baseline_mode = BaselineMode.UNMASKED` does not change
behaviour. Same for `use_hierarchical_extraction`, `use_context_aware_stitching`,
`compare_combinations`, `save_combination_outputs`, and the top-level
`enabled` field.

### Evidence

* `config.py:158-166` — eight fields declared.
* `pipeline/stages/pdf_text_extraction/components/text_assembler.py:43-92`
  — `HierarchicalTextAssembler` reads only `self._config.pre_filter_relevance`.
* `pipeline/stages/pdf_text_extraction/runner.py:409` — only
  `cfg.text.write_raw_text` is checked.
* `grep -rn "compare_combinations\|use_hierarchical_extraction\|use_context_aware_stitching\|baseline_mode" pipeline parsers`
  returns only the config declaration.

### Diagnosis

These knobs trace back to an earlier "compare extraction modes" research
mode that was retired in favour of the standard masked-extract path.
`baseline_mode` predates the two-pass extractor and would be meaningful
again if anyone wanted to run unmasked extraction as a baseline.

### Fix (proposed)

Delete `compare_combinations`, `save_combination_outputs`,
`use_hierarchical_extraction`, `use_context_aware_stitching`. Either wire
`baseline_mode` (route through `runner._steps_1_3_4_standard` to skip
masking when `UNMASKED`) or delete. The `enabled` field is suspicious
because there is no obvious behaviour to disable — delete unless a use
case surfaces.

### Verification

Pending.

---

## Bug 32 — `CroppingConfig` dead knobs: `include_captions_in_metadata`, `panel_counting_enabled`

### Status / Severity / Surface

Fixed (2026-05-15, deleted) · Low · PDF extraction, cropping config.

### Resolution

Both fields removed from `CroppingConfig`. Captions are unconditionally
included in `CroppedMedia.caption` by `PyMuPDFMediaCropper`, and panel
counting was wholly unimplemented (no detector, no metadata field). All
remaining `CroppingConfig` fields are consumed by the cropper.

### Symptom

Toggling either knob has no effect.

### Evidence

* `config.py:140-154` — declarations.
* `pipeline/stages/pdf_text_extraction/components/media_cropper.py` reads
  `dpi`, `save_figure_crops`, `save_table_crops`, `image_format`,
  `min_figure_pts`, `subfigure_proximity_pts`, `merge_figures_by_caption`,
  `merge_tables_by_caption`, `expand_tables_with_footnotes`,
  `footnote_proximity_pts`, `text_footnote_proximity_pts`. The other two
  are never read.

### Diagnosis

Captions are always written into metadata; `panel_counting_enabled` was
designed for a panel-detection feature that was never implemented.

### Fix (proposed)

Delete both fields.

### Verification

Pending.

---

## Bug 33 — `MaskingConfig.merge_iou_threshold` never passed to `merge_rects`

### Status / Severity / Surface

Fixed (2026-05-15, deleted) · Low · PDF extraction, masking config.

### Symptom

Setting `cfg.masking.merge_iou_threshold = 0.1` does not change which
overlapping regions get merged.

### Evidence

* `config.py:125` (pre-fix) — declaration with default `0.3`.
* `pipeline/stages/pdf_text_extraction/components/region_masker.py:234`:
  ```python
  if self._config.merge_overlapping_boxes:
      rects = merge_rects(rects)
  ```
  No threshold forwarded.
* `parsers/layout_utils.merge_rects:103` — uses `Rect.intersects()` (boolean any-overlap), no IOU concept at all.

### Diagnosis

The field was misnamed-and-aspirational from the start. `merge_rects` was always an intersection-union algorithm (any non-zero overlap absorbs the smaller rect into the union). Adding IOU logic would change behaviour for *every* `merge_rects` caller — masking, table reconstruction, hybrid detection — not a configurable knob, a different algorithm. The field implied a threshold-based merge that the codebase doesn't implement.

### Fix

Deleted `MaskingConfig.merge_iou_threshold` from `pipeline/stages/pdf_text_extraction/config.py`. Tightened `merge_rects` docstring (`parsers/layout_utils.py:103`) to spell out the any-intersection semantics so future readers don't reach for the same field name.

Rejected alternative: adding IOU logic to `merge_rects`. Would be a behaviour change to multiple consumers under the guise of a config wire-up.

### Verification

* `grep -rn "merge_iou_threshold" pipeline parsers configs scripts` returns no hits.
* `python -c "from pipeline.stages.pdf_text_extraction.config import MaskingConfig; print(hasattr(MaskingConfig, 'merge_iou_threshold'))"` → `False`.

---

## Bug 34 — `TATRConfig` dead knobs; render DPI hardcoded

### Status / Severity / Surface

Fixed (2026-05-15) · Medium · PDF extraction, TATR detector.

### Resolution

Promoted the hardcoded `_RENDER_DPI = 150` (was at module level in
`tatr_detector.py`) to a real `TATRConfig.render_dpi: int = 150` field.
`TATRTableDetector.detect()` now reads `self._config.render_dpi` and computes
the `pixel → PDF points` scale per call, so a thesis-time DPI sweep is a
config change rather than a code edit.

Deleted four dead `TATRConfig` fields:
* `enabled` — redundant; `PipelineConfig.table_detector` already gates the
  detector via the `TableDetectorType` enum.
* `max_detections_per_page` — not used (TATR confidence threshold already
  keeps per-page counts in the single digits).
* `batch_size_pages` — TATR processes one page at a time; multi-page
  batching is unimplemented.
* `structure_model_name` — table structure recognition was never wired.

Regression coverage in `tests/test_config_loader.py::test_tatr_render_dpi_overridable`.

### Symptom

Setting `cfg.tatr.max_detections_per_page = 50` or `batch_size_pages = 4`
has no effect. `cfg.tatr.enabled = False` does not actually skip TATR (the
runner constructs the detector based on `cfg.table_detector` enum, not
the per-detector `enabled` flag). Render DPI cannot be tuned via config.

### Evidence

* `config.py:108-115` — declares `enabled`, `threshold`,
  `max_detections_per_page`, `device`, `model_name`,
  `structure_model_name`, `batch_size_pages`.
* `pipeline/stages/pdf_text_extraction/table_detectors/tatr_detector.py:42-152`
  reads `self._config.model_name`, `self._config.device`,
  `self._config.threshold` only.
* `tatr_detector.py:30` — `_RENDER_DPI = 150` is a module constant.

### Diagnosis

`enabled` is structurally redundant with `cfg.table_detector` (which
picks Docling / TATR / Hybrid). The other unused fields are speculative.
Render DPI was hardcoded for an experiment and never moved to config.

### Fix (proposed)

* Move `_RENDER_DPI` into `TATRConfig` (`render_dpi: int = 150`).
* Implement `max_detections_per_page` (sort `regions` by score, slice to
  the cap per page) and `batch_size_pages` (vectorised forward pass over
  N pages at a time).
* Delete `enabled` and `structure_model_name` unless there is a roadmap
  for them.

### Verification

Pending.

---

## Bug 35 — `DoclingConfig.timeout_sec` never enforced by `DoclingLayoutExtractor`

### Status / Severity / Surface

Fixed (2026-05-15) · Medium · PDF extraction, Docling timeout.

### Symptom

A pathological PDF (huge page count, malformed embedded fonts, OCR
fallback triggered on every page) could wedge `Docling.convert(...)` for
indefinite minutes. The documented `timeout_sec = 300` knob did nothing
to stop it — silent batch hang.

### Evidence

* `config.py:90` — `timeout_sec: int = 300`.
* Pre-fix `grep -rn "timeout_sec" pipeline parsers` returned only the
  declaration and one docstring; no consumers.
* `pipeline/stages/pdf_text_extraction/components/layout_extractor.py`
  wrapped the Docling call directly with no timeout.

### Diagnosis

Docling does not natively support a per-document timeout. Two options:

* **Subprocess isolation** — clean, but requires pickling the
  `LayoutResult` (and any nested Docling objects) across the boundary;
  the converter also has model state we'd lose per call.
* **Thread + `Future.result(timeout=)`** — leaks the worker thread on
  timeout, but the GIL holds the runaway in check and the pmcid lands on
  the blacklist for next-run skip. Acceptable trade-off for batch
  resilience; the alternative (hung run) costs more.

Picked the thread path because the runner already runs papers in parallel
via `ParallelBatchRunner` — each per-paper worker spawning its own
single-worker executor is bounded and the leaked thread dies with the
process when the batch finishes.

### Fix

* `pipeline/stages/pdf_text_extraction/components/layout_extractor.py`
  — new `_convert_with_timeout(converter, pdf_path)` helper. Submits
  `converter.convert(...)` to a `ThreadPoolExecutor(max_workers=1)` and
  calls `future.result(timeout=self._config.timeout_sec)`. Raises
  `TimeoutError` with a `Docling exceeded {N}s on {name}` message on
  expiry; logs an error line before re-raising. `timeout_sec <= 0`
  disables the guard (bypasses the executor entirely).
* `_run_docling` now calls `self._convert_with_timeout(converter, pdf_path)`
  instead of `converter.convert(...)` directly.
* `PipelineRunner.run_document`'s existing per-paper try/except
  (`runner.py:265`) catches the `TimeoutError`, logs `❌ {pmcid} — failed: …`,
  and adds the pmcid to the blacklist with the timeout message as reason.

### Verification

`pytest tests/pdf_text_extraction/test_docling_timeout.py` — 3 cases:
work-under-budget succeeds, sleep-past-budget raises with the expected
message, `timeout_sec=0` disables the guard and runs in the calling
thread. Uses a `_SleepConverter` stand-in (no real Docling) so the test
is hermetic and ~6 s.

---

## Bug 36 — `GroundingFilter` / `RelateStage` model/batch/device not exposed via `SummarizationConfig`

### Status / Severity / Surface

Fixed (2026-05-15) · Low · Summarisation, NLI helpers config surface.

### Symptom

Cannot switch the grounding or relate NLI model to a GPU build, a
different checkpoint, or a tuned batch size without editing the helper
module — `SummarizationConfig` has no field for it.

### Evidence

* `pipeline/stages/summarization/helpers/grounding_filter.py:68-78` —
  `GroundingFilter.__init__(threshold, model_name, batch_size, device)`.
* `pipeline/stages/summarization/current_stages/relate_stage.py:267-279`
  — `RelateStage.__init__(model_name, entailment_threshold, contradiction_threshold, batch_size, device)`.
* `pipeline/stages/summarization/config.py:48-63` — `GroundingConfig` has
  only `threshold`; `RelateConfig` has only the two thresholds.
* `runner.py:251-258` — instantiates both with only the threshold fields
  forwarded.

### Diagnosis

The model / batch / device knobs predate `SummarizationConfig`. When the
config dataclass was introduced, only the calibration-relevant thresholds
were lifted into it.

### Fix

Externalised NLI configuration to `configs/nli_models.yaml` with `pipeline/stages/summarization/nli_config.py:get_active_spec()` as the loader. `helpers/grounding_filter.py:72-78` and `current_stages/relate_stage.py:52-67` both consume `model_name`, `batch_size`, and `device` from the active spec rather than module defaults.

Trade-off vs. lifting fields into `SummarizationConfig`: the YAML path keeps NLI-specific knobs out of the calibration-thresholds dataclass and lets a corpus run pin a specific NLI build alongside model versions, without re-running Python config construction. If a future need to override per-run from Python emerges, layer it on top — pass `model_name=` through `runner.py:251/258` and let it take precedence over the YAML.

### Verification

* `grep -n "get_active_spec\|nli_models.yaml" pipeline/stages/summarization/` confirms both `grounding_filter.py` and `relate_stage.py` read from the YAML.
* Switching `configs/nli_models.yaml` `active:` key changes the model used by both stages at next run; no Python edits required.

---

## Bug 37 — `NormalizeStage.extra_synonyms` not exposed via `SummarizationConfig`

### Status / Severity / Surface

Fixed (2026-05-15) · Low · Summarisation, normalize stage.

### Resolution

Added `NormalizeConfig` dataclass to
`pipeline/stages/summarization/config.py` with one field,
`extra_synonyms: dict[str, str] | None = None`, and wired it through
`SummarizationConfig.normalize`. Both `SummarizationRunner` (sync) and
`BatchSummarizationRunner` now construct `NormalizeStage(extra_synonyms=cfg.normalize.extra_synonyms)`.

Side fix in `pipeline/config_loader.py` discovered while writing the
regression test:
* `_unwrap_optional` only matched `typing.Union[X, None]`; PEP-604
  `X | None` (origin `types.UnionType`) fell through unchanged. So the
  union wrapper around `dict[str, str] | None` was never stripped.
* `_coerce` then routed any dict-typed YAML value through the
  nested-dataclass branch, which crashed with
  `Field type dict[str, str] | None does not resolve to a dataclass`.

Both addressed in one pass: `_unwrap_optional` now also accepts
`types.UnionType`, and `_coerce` early-returns the raw dict when the
field type is a `dict[...]` mapping (new helper `_is_mapping_type`).

Regression tests in `tests/test_config_loader.py`:
* `test_normalize_extra_synonyms_loaded_as_mapping` — exercises the YAML
  → `dict[str, str]` path end-to-end.
* `test_tatr_render_dpi_overridable` — pairs nicely with the B-034 fix.

### Symptom

`NormalizeStage` accepts caller-supplied synonyms but there is no
`SummarizationConfig` field for them; overriding the curated
`synonyms.yaml` requires editing the file or subclassing the runner.

### Evidence

* `pipeline/stages/summarization/current_stages/normalize_stage.py:387` —
  `__init__(self, extra_synonyms: dict[str, str] | None = None)`.
* `runner.py:247` — `self._normalize = NormalizeStage()`.
* `config.py` — no `normalize` sub-config.

### Diagnosis

The `extra_synonyms` parameter was added for one-off experiments and was
never lifted into the public config surface.

### Fix (proposed)

Add a `NormalizeConfig` dataclass with `extra_synonyms: dict[str, str] |
None = None` and route it in the runner. Or, simpler, add a single
`SummarizationConfig.normalize_extra_synonyms` field.

### Verification

Pending.

---

## Bug 38 — `SummarizationRunner.load_paper_from_db` bypasses scispaCy singleton

### Status / Severity / Surface

Observed (2026-05-15) · Medium · `pipeline/stages/summarization/runner.py:905` (`load_paper_from_db`).

### Symptom

Every call to `SummarizationRunner.load_paper_from_db(pmcid)` instantiates a fresh `en_core_sci_sm` spaCy pipeline:

```python
import spacy  # type: ignore
…
nlp = spacy.load("en_core_sci_sm")
```

In batch mode (`process_batch([load_paper_from_db(p) for p in pmcids])`) this scales linearly with paper count. There is no module-level cache, and the same model is reloaded for every paper even within a single process.

### Evidence

`pipeline/stages/summarization/runner.py:902-906`:

```python
import spacy  # type: ignore
from database import get_db_connection, Document, TextElement  # type: ignore

nlp = spacy.load("en_core_sci_sm")
db = get_db_connection(database_url=db_url)
```

Same class of bug as [B-029](#bug-29--pipelinerunner_get_nlp-bypasses-umls_resources-singleton) — different file, same anti-pattern.

`.claude/CLAUDE.md` explicitly forbids this pattern: *"Always go through `pipeline/stages/summarization/umls_resources.py` (`get_nlp()` / `get_linker()`) — process-wide singleton. Loading `en_core_sci_lg` + the UMLS KB twice OOM-kills the pipeline."*

### Diagnosis

Two compounding issues:

1. **No singleton.** Each `load_paper_from_db` call hits `spacy.load(...)` which deserialises the model from disk and re-builds the pipeline. spaCy has no implicit cache.
2. **Model mismatch.** `umls_resources.get_nlp()` loads `en_core_sci_lg` (the large model with the UMLS linker attached). This site loads `en_core_sci_sm`. A process that does both — e.g. `scripts/run_paper.py` which loads paper data, then runs the summarisation pipeline through `runner.process()` which in turn triggers `umls_resources` — ends up with TWO scispaCy pipelines in RAM, neither shared.

Note: `load_paper_from_db` only uses `nlp` for sentence segmentation (`nlp(te.text_content).sents`), so the sm-model choice is *correct* for what's needed — it's the lack of a singleton that's the bug.

### Fix

Two options:

1. **Lean fix:** add a `get_small_nlp()` singleton in `pipeline/stages/summarization/umls_resources.py` for the sentence-splitting case, and route `load_paper_from_db` through it. Cheapest, preserves the small-model footprint for sentence splits.
2. **Consolidate:** route through `umls_resources.get_nlp()` (which loads `en_core_sci_lg`). RAM cost goes up but eliminates the two-model overlap.

Option 1 is the lower-risk fix. The summarisation pipeline (`umls_resources.get_nlp()`) is already paying the `_lg` cost for entity linking; adding a separate cached `_sm` instance for the sentence-splitter is bounded and explicit.

### Verification

After fix, a smoke run of `SummarizationRunner.process_batch([...])` with `tracemalloc` should show one scispaCy load event in process lifetime regardless of paper count, not one per paper. Easy to assert in a unit test by mocking `spacy.load` and confirming exactly one call.

### Follow-up

* B-029 covers the sibling site in `pipeline/stages/pdf_text_extraction/runner.py:199`. Both fixes should land together so an audit of `grep -n "spacy.load" pipeline/` returns zero hits outside `umls_resources.py`.

---

## Bug 39 — `load_paper_from_db` orders by `position_in_section` only, interleaves sections

### Status / Severity / Surface

Fixed (2026-05-15) · High · `pipeline/stages/summarization/runner.py:912-916` (`load_paper_from_db`).

### Symptom

`SummarizationRunner.load_paper_from_db` queries `TextElement` rows with only one ORDER BY column:

```python
rows = (
    session.query(TextElement)
    .filter_by(document_id=doc.id)
    .order_by(TextElement.position_in_section)
    .all()
)
```

`position_in_section` is local to each `path_string` (cf. `database/models.py:79`, composite unique index `idx_document_path_position = (document_id, path_string, position_in_section)` at line 102). With this single-column sort, every section's position-0 paragraph emits before any position-1 paragraph — sections are completely interleaved. The resulting `sentences_with_provenance` list mixes paragraphs from Methods, Results, Discussion, and References at each position level.

### Evidence

Schema check at `database/models.py:69-103`:

```python
position_in_section = Column(Integer, nullable=False)
…
Index('idx_document_path_position', 'document_id', 'path_string', 'position_in_section', unique=True),
```

No global doc-order column exists — `id` is autoincrement (so insert-order, not document-order) and `unique_path` is `{PMCID}/{path_string}/{position}` (string-sortable but only after parsing).

### Diagnosis

`MapStage._make_chunks` (`current_stages/map_stage.py:1193-1215`) packs adjacent sentences into fixed-size chunks. The whole MAP-cascade design assumes sentences within a chunk share topical context. Section-interleaved input destroys that assumption: a 10-sentence chunk built from this stream pulls one sentence from Methods, one from Results, one from Discussion, etc. — voters get less context to anchor on, agreement scores drop, and L3 escalations rise spuriously.

The interaction with B-041 (producer attribution mis-indexed) and B-019 (citation regex previously rejected suffixed pmcids) means production runs to date have been triple-handicapped: scrambled input, mis-attributed producer, and previously single-voter agreement.

### Fix

Switched the read site to autoincrement `id`:

```python
.order_by(TextElement.id)
```

This works because `db_ingester.write` iterates `rows` in order and `session.flush()` produces `id` in iteration order — so `id` reflects insertion order, and (once B-040 is fixed so `rows` is emitted in document order) `id` reflects document order. Whole-paper contiguous ingest plus the no-partial-reingest invariant (`PostgresDatabaseIngester.write` short-circuits if `Document` exists) make `id`-based ordering robust without a new schema column.

Rejected alternative: adding a `document_order: Integer` column + Alembic migration. The cheaper read-side flip closes the same gap given B-040 is fixed in the same change.

### Verification

* Pre-fix: dump `[s["text_element_id"] for s in file_data["sentences_with_provenance"]]` for a paper ingested pre-B-040-fix — te_id sequence cycles through path_strings instead of monotonically advancing within each path.
* Post-fix (read flip only, pre-B-040-fix data): improved but still scrambled since `id` reflects path-first-appearance order at ingest.
* Post-fix (read flip + B-040 fix + re-ingest): te_ids monotonically advance in document order; `MapStage._make_chunks` packs topically coherent sentences.
* Requires re-ingesting any papers whose `TextElement` rows were written pre-B-040-fix; existing rows have `id`s aligned with the (scrambled) ingest order, not document order.

### Follow-up

* Long-term: persist a `document_order` integer if any partial-reingest workflow ever lands (e.g. re-ingest just figures without re-ingesting text). Currently no such path; the column would be maintenance overhead without payoff.

---

## Bug 40 — `extract_text` emits paragraphs in path-first-appearance order, not document order

### Status / Severity / Surface

Fixed (2026-05-15) · Medium · `parsers/layout_utils.py:469-524` (`extract_text`).

### Symptom

`extract_text` walks Docling elements in document order but accumulates paragraphs into `by_path = defaultdict(list)` keyed by `path_string`, then emits its return value by iterating `by_path` in insertion order:

```python
by_path = defaultdict(list)
…
for idx, el in enumerate(elements):
    …
    path_parts = [hierarchy[k] for k in sorted(hierarchy) if hierarchy.get(k)]
    path_str   = ' > '.join(path_parts) or 'Root'
    …
    by_path[path_str].append(text)

stitcher = ContextAwareStitcher()
rows = []
for path_str, texts in by_path.items():
    …
```

Python 3.7+ dict iteration is insertion-ordered. When a section gets revisited after a sub-section — parent text → sub-section text → more parent text — the parent's later paragraphs get appended at the parent's *first-emit position*, after which the sub-section is iterated. The sub-section's content lands AFTER the entire parent block in `rows`.

### Evidence

`parsers/layout_utils.py:483-507`:

```python
if etype == 'SECTION_HEADER':
    level = el.get('level', 0)
    hierarchy[level] = text
    hierarchy = {k: v for k, v in hierarchy.items() if k <= level}
elif etype not in SKIP_TYPES:
    …
    path_parts = [hierarchy[k] for k in sorted(hierarchy) if hierarchy.get(k)]
    path_str   = ' > '.join(path_parts) or 'Root'
    if text in path_seen[path_str]:
        n_deduped += 1
        continue
    path_seen[path_str].add(text)
    by_path[path_str].append(text)
```

Emit loop at line 511 iterates `by_path.items()` in insertion order, not document position order.

### Diagnosis

Flat hierarchies (one parent → all sub-sections sequential without revisiting parent) emit correctly because each path is inserted exactly once. The bug surfaces when a parent section interleaves with its sub-sections — common in Discussion sections that mix narrative paragraphs with sub-headed analyses.

Output rows feed `HierarchicalTextAssembler.assemble`, which forwards them to `TextElement` DB rows. Since `position_in_section` is computed from emission order within each `path_string`, intra-section order is preserved; but cross-section order is determined by *first appearance*, not document position. Combined with B-039 (which sorts by `position_in_section` alone), the input to MAP is scrambled along two independent axes.

### Fix

Replaced the global `by_path` accumulator with arrival-order traversal. Elements stream into a flat `ordered: list[(path_str, text)]` in document walk order (dedup still applied per path via `path_seen`). The emit loop scans contiguous same-`path_str` runs and stitches each run independently:

```python
ordered: list = []
…
elif etype not in SKIP_TYPES:
    …
    path_seen[path_str].add(text)
    ordered.append((path_str, text))

…
i = 0
while i < len(ordered):
    path_str = ordered[i][0]
    j = i
    texts: list = []
    while j < len(ordered) and ordered[j][0] == path_str:
        texts.append(ordered[j][1])
        j += 1
    …
    i = j
```

Rows now emit in true document order. Side effect: when a parent section is interleaved with a sub-section, the parent's first half and second half stitch independently (they're no longer adjacent in `ordered`). That's the correct behavior — non-adjacent paragraphs should not stitch.

### Verification

* Unit test added at `tests/pdf_text_extraction/test_extract_text_ordering.py`:
  * `test_subsection_return_preserves_document_order` — Methods → Sub1 → Methods (parent repeated as level-1 header); asserts output paths are `["Methods", "Methods > Sub1", "Methods"]` and text contents arrive in document order.
  * `test_simple_sequential_sections_ordered` — two flat top-level sections emit in order.
  * `test_dedup_drops_duplicate_text_within_path` — same-text dedup still fires.
* All three pass against the new implementation.

### Follow-up

* Combined with the B-039 read-side flip (`ORDER BY id`), papers re-ingested post-fix have `TextElement.id` aligned with document order, restoring topical locality in `MapStage._make_chunks` input.

---

## Bug 41 — Producer attribution mis-indexed when any voter fails

### Status / Severity / Surface

Fixed (2026-05-15) · High · `pipeline/stages/summarization/current_stages/map_stage.py:1183` (`_run_voters`), `agreement/decision.py:186-212` (`producer_from_outcome`), `agreement/decision.py:255-267` (`make_decision_record`), `routing/router.py:271` (`_classify_voters`), `pipeline/stages/summarization/batch/runner.py:1414-1445` (batch parse loop).

### Symptom

When ≥1 voter fails its API call (transient 5xx, rate-limit retry exhaustion, parsing error after both retries), the MAP cache metadata, cost report, and cascade decision JSONL all carry the wrong `(provider, model)` for the kept chunk. Dormant on runs where every voter succeeds; activates the moment any single voter drops.

The router path is *also* affected — the router's `valid_voter_indices` are computed over the API-survivor list, not the original voter list, and the same `voter_specs[global_idx]` lookup misalignment occurs.

### Evidence

`_run_voters` builds a slot list indexed by original voter index then *filters Nones away* on return:

```python
# map_stage.py:1022, 1147-1183
results: list[AuditableSummary | None] = [None] * len(target)
…
future_to_idx = {pool.submit(_timed_invoke, chain, i): i for i, chain in enumerate(target)}
…
for future in as_completed(future_to_idx):
    idx = future_to_idx[future]
    try:
        results[idx] = future.result()
    except Exception as exc:
        …
        timings[idx] = None
…
return [r for r in results if r is not None], timings
```

The original-voter-index → survivor-index mapping is gone at return. Downstream `_cascade` receives `voters` as the filtered survivor list.

In the **no-router** path:

```python
# agreement/decision.py:106-114 (evaluate_chunk)
bundle = agreement.compute(voters, source_text=source_text)
…
return ChunkOutcome(keep=True, best=best, agreement_bundle=bundle, routing_decision=None, …)
```

`bundle.best_index` indexes the survivor list. Then `producer_from_outcome` falls into:

```python
# agreement/decision.py:210-211
if 0 <= best_idx < len(voter_specs):
    return voter_specs[best_idx]
```

`best_idx` (survivor-list index) is used to index `voter_specs` (original-list). If voter 0 failed, survivor index 0 corresponds to original voter 1, but `voter_specs[0]` is voter 0 — wrong attribution.

In the **router** path:

```python
# routing/router.py:271-307 (_classify_voters)
for i, output in enumerate(outputs):
    …
    classifications.append(VoterClassification(voter_index=i, …))
…
valid_voter_indices = [c.voter_index for c in eligible]
```

`i` enumerates the survivor list, so `voter_indices` are 0..M-1 (survivor-list indices). Then in `producer_from_outcome`:

```python
# agreement/decision.py:204-208
if rd is not None and rd.valid_voter_indices is not None:
    if 0 <= best_idx < len(rd.valid_voter_indices):
        global_idx = rd.valid_voter_indices[best_idx]
        if 0 <= global_idx < len(voter_specs):
            return voter_specs[global_idx]
```

`global_idx` is a survivor-list index treated as original-list. Same misalignment.

### Diagnosis

The root cause is `_run_voters` returning a filtered list while every downstream consumer (cache, cost report, decision log, router) assumes original-index alignment. The router was added later and inherits the bug — it does the right index-mapping for *its own* filtering logic (UNUSABLE strip) but starts from an already-filtered survivor list.

### Fix

**Shipped 2026-05-15** — Option 1 (None-padded list) + plumbing through to `evaluate_chunk`.

* `MapStage._run_voters` now returns `list[AuditableSummary | None]` of length N (original voter count). Failed voters leave `None` at their slot.
* `_cascade` builds `survivor_indices = [i for i, v in enumerate(voters_full) if v is not None]` and `voters = [voters_full[i] for i in survivor_indices]`, then passes both to `evaluate_chunk(..., voter_indices=survivor_indices)`. Same for the L2 escalation branch.
* `evaluate_chunk` (`agreement/decision.py`) gains a `voter_indices: list[int] | None` parameter. Router path: maps `decision.valid_voter_indices` (survivor-indexed) back through `voter_indices` to original-spec indices. No-router path: populates `ChunkOutcome.valid_voter_indices` directly from `voter_indices`. `ChunkOutcome.valid_voter_indices` is now the **single source of truth** for downstream producer attribution — always original-spec-indexed when populated.
* `producer_from_outcome` and `make_decision_record` rewritten to read from `outcome.valid_voter_indices` instead of `outcome.routing_decision.valid_voter_indices`. `CascadeDecisionRecord.best_voter_index` now records the original-spec index (was previously the bundle's survivor-indexed `best_index` — silently wrong per its own docstring).
* Batch runner (`batch/runner.py:_process_level`) builds `chunk_voters[chunk_id]` as a length-N None-padded list, slot `vi` populated from `custom_id`. At evaluate-time the same `survivor_indices` + `voters` split runs, with `voter_indices=survivor_indices` passed to `evaluate_chunk`.

Router internals unchanged: `_classify_voters` still enumerates the survivor list and returns survivor-indexed `valid_voter_indices`. The mapping back to original indices happens inside `evaluate_chunk` — keeps the router's `RoutingDecision` as a "router-internal view" and `ChunkOutcome` as the public "original-spec view".

Cache key shape unaffected — `PipelineCache._map_key` doesn't include producer metadata.

### Verification

Six regression tests in `tests/summarization/agreement/test_voter_index_alignment.py`:

* `test_b041_no_router_voter_0_failure_attributes_to_correct_survivor` — voter 0 fails, KEEP picks survivor[0] = original voter 1. Asserts `producer_from_outcome → ('google', 'gemini-flash-lite')` (pre-fix would have returned voter 0's spec).
* `test_b041_no_router_voter_1_failure_attributes_correctly` — voter 1 fails, survivors [v0, v2], KEEP picks survivor[0] = original voter 0.
* `test_b041_no_router_identity_when_voter_indices_none` — back-compat path with implicit identity mapping.
* `test_b041_no_router_all_voters_failed` — empty survivor list short-circuits cleanly.
* `test_b041_make_decision_record_records_original_index` — asserts `CascadeDecisionRecord.best_voter_index` matches the original voter spec index.
* `test_b041_router_path_maps_back_to_original_indices` — router strips a survivor as UNUSABLE; `ChunkOutcome.valid_voter_indices` contains the original-spec index of the kept voter.

Full summarization test suite (541 tests including the 6 above) passes.

### Follow-up

* Sweep `out/summaries/runs/*/cascade_decisions/*.jsonl` from pre-fix calibration runs for rows where `selected_provider/model` disagrees with the cost report's tokens-per-model breakdown — those rows are evidence of how often this fired in practice.
* `_record_chunk_trace` (`map_stage.py:816-846`) still keys `voter_timings.get(i)` by survivor-indexed `i` while `voter_timings` is original-indexed. Pre-existing bug, not regressed by this fix. Tracked separately.

---

## Bug 42 — `_is_cut_off` mid-sentence abbreviation rule is dead code

### Status / Severity / Surface

Fixed (2026-05-15) · Low · `parsers/text_processing.py:127-184` (`ContextAwareStitcher._is_cut_off`).

### Symptom

`ContextAwareStitcher._is_cut_off` is supposed to detect when a paragraph's last token is a known mid-sentence abbreviation (e.g. "see Fig.", "Smith et al.", "compared vs.") so the stitcher knows to merge with the next paragraph. The rule never fires.

### Evidence

`parsers/text_processing.py:148-183`:

```python
def _is_cut_off(self, text: str) -> bool:
    t = text.strip()
    if not t:
        return False

    # Sentence-final punctuation → definitely complete
    if t[-1] in '.?!)]"\'»':
        return False

    # Ends with a hyphen → word broken across a column/page boundary
    if t.endswith('-'):
        return True
    …
    last_word = t.split()[-1].lower().rstrip('.,;:')

    # … (connector check) …

    # Ends on a mid-sentence abbreviation that is never sentence-final
    _MID_SENTENCE_ABBREVS = frozenset({
        'fig', 'figs', 'et al', 'vs', 'approx', 'dept', 'no', 'nos',
        'e.g', 'i.e', 'cf', 'approx', 'ref', 'refs',
    })
    if last_word in _MID_SENTENCE_ABBREVS:
        return True

    return False
```

Every abbreviation in `_MID_SENTENCE_ABBREVS` ends in a period in actual text. The early-return at line 153-154 catches any text ending in `.` and returns `False` *before* the abbrev lookup at line 180 is reached. The intent (strip the trailing period via `rstrip('.,;:')` at line 164, then check the stem against the frozenset) is correct, but the flow never reaches that branch.

### Diagnosis

Lines 152-154 short-circuit on terminal `.` to avoid mis-classifying normal sentence-final periods. The abbrev rule was added later to recover the specific abbreviation cases. The author either missed that the early-return blocks the recovery, or the early-return was tightened later and the abbrev block was left behind.

### Fix

Reorder: check abbreviations first (after stripping the trailing period), then fall through to the sentence-final early-return.

```python
def _is_cut_off(self, text: str) -> bool:
    t = text.strip()
    if not t:
        return False

    last_word = t.split()[-1].lower().rstrip('.,;:')

    _MID_SENTENCE_ABBREVS = frozenset({…})
    if last_word in _MID_SENTENCE_ABBREVS:
        return True

    if t[-1] in '.?!)]"\'»':
        return False
    …
```

### Verification

Applied 2026-05-15. Reorder + multi-token `last_two` check landed in `parsers/text_processing.py:127-184`. Regression coverage in `tests/parsers/test_text_processing_cutoff.py` (21 cases): every entry in `_MID_SENTENCE_ABBREVS` now triggers stitching ("fig.", "et al.", "e.g.", "i.e.", "cf.", "ref.", "refs.", "vs.", "approx.", "dept.", "no.", "nos."), real sentence-final cases (`. ? ! ) ] " »`) still return `False`, and the existing connector / hyphen / comma branches are unchanged. `python -m pytest tests/parsers/test_text_processing_cutoff.py -q` → 21 passed.

Note: the multi-token check was added because `"Smith et al."` splits into two tokens (`"et"`, `"al."`), and only the final token (`"al"`) was being looked up against `_MID_SENTENCE_ABBREVS`. Joining the last two tokens (sans trailing punctuation) recovers the `"et al"` entry.

### Follow-up

* Quantify impact: walk `out/text/*.txt` from a recent batch and grep for paragraphs ending in any abbrev. Each such line was a stitching opportunity the pre-fix code missed. Likely small in absolute terms but compounds with the document-order bugs (B-039, B-040) for thesis-grade reproducibility.

---

## Bug 43 — `remove_citations` strips publication years

### Status / Severity / Surface

Fixed (2026-05-15) · Low · `parsers/text_processing.py:320` (`remove_citations`).

### Symptom

Narrative-style year mentions get stripped along with citation numbers. "Smith et al. 2020 reported …" becomes "Smith et al. reported …".

### Evidence

`parsers/text_processing.py:303-310`:

```python
# Bracket-style citations: [1], [1,2], [1-29], [3,11,21,22], [1–3]
cleaned = re.sub(r'\[\d+(?:[,–\-]\d+)*\]', '', cleaned)

# After period: ". 1 ", ". 19,20 ", ". 4,5"
cleaned = re.sub(r'(?<!\n)\.\s+\d+(?:[,–\-]\d+)*(?=\s|$)', '. ', cleaned)

# After comma: ", 5 ", ", 1,2 "
cleaned = re.sub(r',\s+\d+(?:[,–\-]\d+)*(?=\s|$)', ', ', cleaned)
```

The line-307 regex is *length-agnostic* — `\d+` matches a 4-digit year as readily as a single citation digit. So `". 2020 "` after "et al" is consumed as if it were `". 14 "`.

### Diagnosis

The citation-removal patterns date back to a simpler PDF era where bracketed citations dominated. The "after period" pattern is meant to catch reference-style numbers like "…shown in fig 4. 12 patients had…" where `12` is a paper citation, not a 1942 mention. The current heuristic can't tell them apart.

`is_reference_entry` filters reference-list paragraphs upstream, so the worst case (mangling actual references) is averted. The remaining surface is in-text year mentions — usually small numbers but material when present (e.g. "the 2018 WHO classification").

### Fix

Restrict the digit range to avoid 4-digit years:

```python
cleaned = re.sub(r'(?<!\n)\.\s+\d{1,3}(?:[,–\-]\d{1,3})*(?=\s|$)', '. ', cleaned)
```

Same fix applies to the line-310 "after comma" pattern. Three-digit cap permits citation numbers up to 999 — well above the largest citation count in modern papers — while explicitly excluding years.

A more aggressive fix: add a negative lookahead `(?!19\d{2}|20\d{2})` to the digit run. Both work; the digit-cap version is simpler and surfaces in `git diff` more clearly.

### Verification

`pytest tests/parsers/test_remove_citations.py` — 9/9 pass. Covers year-after-period, year-after-comma, year-range with en-dash, citation-after-period still stripped, citation-range still stripped, citation-list still stripped, bracket-style citations still stripped (unchanged `\d+` branch), standalone-citation-run still stripped, and bare-year survival.

### Follow-up

* Mine `enum_observations.jsonl` and `bad_findings.jsonl` for `verbatim_support` mismatches involving year strings — would surface real cases where the year-strip caused a downstream NLI grounding miss.

---

## Bug 44 — MAP `relation_type` bleeds category names and loses findings at GROUP

### Status / Severity / Surface

* **Status:** Mitigated (2026-05-15)
* **Severity:** Medium
* **Surface:** `pipeline/stages/summarization/prompts.py` (MAP system prompt), `pipeline/stages/summarization/models.py` (`Finding._coerce_invalid_relation_type`, `_RELATION_TYPE_ALIASES`)

### Symptom

10+ findings per calibration run silently dropped at GROUP. Concentration in `category=molecular_genetics` and `category=IHC` claims. Affected findings reached MAP with a valid `category` but a `relation_type` value taken from the category enum (`"molecular_genetics"`, `"IHC"`, `"morphology"`, `"staging"`) — `_coerce_invalid_relation_type` did not recognise them as aliases, fell through to `RelationTypeEnum.unclear`, and GROUP keys on `(subject, outcome, relation_type, category)` with `unclear` treated as non-groupable.

### Evidence

* Prior alias map (`models.py:178-181` pre-fix) only handled `prognosis` and `treatment`. Any other category-name leak hit the unknown-value path at `models.py:381-386` and was logged as `reason="unknown_value"` — indistinguishable from genuinely novel relation_type strings.
* The MAP prompt's "CRITICAL" block (`prompts.py:170-193`) already stated the orthogonality rule and called out `staging` / `molecular_genetics` by name, so the bleed is not a prompt-omission — it's a recall failure of voters under load and a missing safety net at the validator.
* Conceptual proximity makes the bleed predictable: `category=molecular_genetics` claims trigger `relation_type="molecular_genetics"` far more often than for unrelated buckets.

### Diagnosis

Two reinforcing problems:

1. **Prompt redundancy without sharpness.** The orthogonality discussion is correct but spread across lines 156-193, with no single-line anti-pattern enumeration at the field definition itself (`prompts.py:73`). Voters that read the field def and skim the examples never hit the anti-pattern list.
2. **No alias safety net.** When the bleed slipped through, the validator dropped the finding instead of recovering it. The mapping is unambiguous for `morphology` (always `has_feature`), `IHC` (always `expression`), and `molecular_genetics` (always `expression` per the existing variant-call convention in the prompt). Only `staging` stays ambiguous (`has_feature` for descriptive, `prognostic` for outcome-driven), so it remains unaliased.

`expression` and `has_feature` are partly synonymous in this codebase (both encode "entity has some readout") — a follow-up refactor could collapse them into one bucket, but that needs cache invalidation + downstream consumer audit and is out of scope here.

### Mitigation

Layered fix (kept narrow because the cleaner refactor — enum collapse — is deferred):

1. **Prompt anti-pattern line** at `prompts.py:74-81` directly under the `relation_type` enum listing:

   ```
   Invalid relation_type values (these are CATEGORY names, not predicates):
     "prognosis", "treatment", "staging", "molecular_genetics", "IHC", "morphology"
   ```

2. **Molecular-genetics prognostic-crossover example** added to `prompts.py:167-170` — same subject (`MYD88 L265P mutation`) with `relation_type=prognostic` when the predicate is survival, to break the assumption that "molecular subject ⇒ molecular relation".

3. **Extended alias map** in `models.py:178-194`:

   ```python
   _RELATION_TYPE_ALIASES: dict[str, str] = {
       "prognosis":          "prognostic",
       "treatment":          "treatment_response",
       "morphology":         "has_feature",
       "ihc":                "expression",
       "molecular_genetics": "expression",
   }
   ```

   Keys are lowercase; the case-folded branch of the validator already routes `"IHC"` / `"Morphology"` etc. through them.

4. **`cross_field_bleed` observability** in `_coerce_invalid_relation_type`. Whenever the raw value is a category name (`v.lower() in _CATEGORY_NAMES_LOWER`), the JSONL log writes `reason="cross_field_bleed"` instead of `alias_repair` / `unknown_value`. `coerced_value` distinguishes recovered (valid enum) from unrecovered (`unclear`). Lets us count exact loss per run.

### Verification

* `tests/summarization/test_enum_alias_repair.py` extended:
  * `test_relation_type_alias_repair` parametrised over the three new aliases (incl. both `"ihc"` and `"IHC"` cases).
  * `test_relation_type_staging_not_aliased_falls_to_unclear` documents the deliberate gap.
  * `test_cross_field_bleed_logged` monkeypatches `log_enum_observation` and asserts every category-name leak emits exactly one `cross_field_bleed` record with the correct `coerced_value`.
  * `test_non_bleed_unknown_still_tagged_unknown_value` guards against bleed detection over-firing.
* Full file passes: `pytest tests/summarization/test_enum_alias_repair.py` → 39 passed.

### Follow-up

* After the next calibration run, grep `enum_observations.jsonl` for `reason="cross_field_bleed"` and report counts split by `coerced_value`. If "staging" appears frequently in the `coerced_value="unclear"` bucket, escalate to a context-aware aliasing pass (or a small post-MAP repair stage that looks at the claim text).
* Bigger refactor on the table: collapse `expression` and `has_feature` into one `relation_type` bucket. `category` already carries the assay-type signal, so no information is lost — but cache invalidation + downstream GROUP/CANONICALIZE audit makes it a separate change. Track under `docs/THESIS.md ##TODOs`.

---

## Bug 45 — `scope_parsed` is LLM-set but trivially derivable

**Status / Severity / Surface:** Fixed (2026-05-15) / Low / Summarisation, MAP `FindingScope.scope_parsed`.

### Symptom

`scope_parsed: bool` was a required field on `FindingScope` that the prompt asked the LLM to compute as "true if at least one sub-field is non-null; false otherwise" (`prompts.py:213`). One more thing the model could get wrong, and output tokens spent reasoning about a one-line boolean. No downstream consumer cared what *the LLM* answered — only whether the boolean matched the actual sub-field state.

### Evidence

* Cataloged as Issue 5 in [MAP_PROMPT_AUDIT.md](MAP_PROMPT_AUDIT.md#issue-5--scopescope_parsed-is-llm-set-but-trivially-derivable-low).
* No bad-value telemetry hit (Pydantic `bool` parses `true`/`false` correctly) — so the bug was pure compute-waste, not data-loss.

### Diagnosis

Field is trivially derivable from the other sub-fields. The audit's recommended fix: validator overrides the LLM-emitted value after parse.

### Fix

* `pipeline/stages/summarization/models.py`
  * New `@model_validator(mode="after") _compute_scope_parsed` on `FindingScope`. Sets `scope_parsed = any(getattr(self, k) is not None for k in _SCOPE_FIELDS_DEFAULTS if k != "scope_parsed")`, overriding the LLM-emitted value.
  * Bumped `MAP_SCHEMA_VERSION` → `"map_v7_scope_parsed_autocompute"` so cached `AuditableSummary` payloads re-validate. In every existing case where the LLM was correct the stored boolean matches the new computation, but a fresh pass formalises the invariant.
* `pipeline/stages/summarization/prompts.py`
  * Line 213: changed instruction from "Set to true if at least one scope sub-field above is non-null; false otherwise" to "Always emit false. Computed automatically from sub-fields downstream; any value you provide is overridden." The field stays in the schema-as-instructions and `OutputFormat` JSON template because OpenAI strict mode requires every property to be present in the emitted object.
* `tests/summarization/test_scope_parsed_autocompute.py` — four cases covering empty scope, non-null sub-field with `scope_parsed=False` (LLM under-reports), all-null with `scope_parsed=True` (LLM over-reports), and `cohort_n=0` (numeric falsy ≠ null).

### Verification

`pytest tests/summarization/test_scope_parsed_autocompute.py` — 4/4 pass. Adjacent suites (`test_batch_persistence`, `test_canonicalize_direction_split`, `test_enum_alias_repair`, `test_demographics`, `test_phase1_schema`) — 83/83 pass; no existing call site constructed a `FindingScope` whose `scope_parsed` value disagreed with the derived boolean.

---

## Bug 46 — Direction hedging words coerce to `unclear` instead of alias-repair

**Status / Severity / Surface:** Fixed (2026-05-15) / Low / Summarisation, MAP `direction` enum coercion.

### Symptom

LLM voters occasionally emitted hedging strings outside `DirectionEnum` (`"maybe"`, `"possibly"`, `"perhaps"`, `"likely"`, `"unknown"`) or natural-language nulls (`"none"`, `"n/a"`, `"NA"`). Each fell through to the unknown-value branch of `_coerce_invalid_direction` and was silently coerced to `DirectionEnum.unclear` with `reason="unknown_value"` — the in-enum value still carried the right semantics, but the alias category was bucketed as "unknown" telemetry instead of "alias_repair", so the operator could not distinguish a model that was confidently wrong from a model that hedged.

### Evidence

* [MAP_PROMPT_AUDIT.md Issue 8](MAP_PROMPT_AUDIT.md#issue-8--directionmaybe-single-occurrence-low) — single observed occurrence of `direction="maybe"`. Audit said defer; we shipped anyway because the fix is a 5-line dict + branch matching the well-trusted `_RELATION_TYPE_ALIASES` pattern.

### Diagnosis

Pure observability gap: in-enum behaviour unchanged, but `enum_observations.jsonl` couldn't distinguish hedging from unknown. Same class as B-018 / B-044 (cross-field bleed) — the alias map turns silent telemetry into a labelled bucket.

### Fix

* `pipeline/stages/summarization/models.py`
  * New `_DIRECTION_ALIASES` dict: `maybe`/`possibly`/`perhaps`/`likely`/`unknown` → `unclear`; `none`/`n/a`/`na` → `no_direction`.
  * Alias-repair branch in `_coerce_invalid_direction` runs after case-fold (so the alias table is implicitly case-insensitive). Logs `reason="alias_repair"`.
  * Bumped `MAP_SCHEMA_VERSION` → `"map_v8_direction_alias_repair"`.
* `tests/summarization/test_enum_alias_repair.py` — parametrised `test_direction_alias_repair` covers all alias entries including case-fold variants; `test_direction_unknown_value_still_coerces_to_unclear` guards against the alias map silently over-firing.

### Verification

`pytest tests/summarization/test_enum_alias_repair.py` — 39 → 53 tests passing (14 new direction-alias cases). `_raw_direction` PrivateAttr (B-015 contract) still preserves the original hedging string for offline analysis.

---

## Bug 47 — Direction `absent` vs `negative` ambiguity on expression claims

**Status / Severity / Surface:** Fixed (2026-05-15) / Low / Summarisation, MAP prompt.

### Symptom

Prompt example mapped `"BCL2 was negative in GCB-DLBCL"` → `direction: absent` (`prompts.py:148`), but the `direction` definition allowed both `absent` ("explicitly not present / lacking / **negative staining**") and `negative` ("not expressed / decreased"). Two voters extracting the same expression claim could legitimately disagree on which label to emit, and the GROUP/RELATE polarity guard treats them as different polarities — blocking RELATE's CONTRADICT signal even on findings that downstream consumers would consider semantically identical.

### Evidence

* [MAP_PROMPT_AUDIT.md Issue 6](MAP_PROMPT_AUDIT.md#issue-6--directionabsent-vs-directionnegative-ambiguity-in-expression-contexts-low) — flagged as a latent split before downstream measurement could quantify it.

### Diagnosis

Two valid mappings, no disambiguator. The cleanest fix is to constrain the rubric by relation_type: for `expression` claims, "negative staining" / "no expression detected" is the standard pathology readout and is best captured by `absent`; "decreased / reduced expression" implies a continuous-axis comparison and is best captured by `negative`. Other relation_types don't have the staining-readout idiom, so they default to `negative` unless the text literally says "absent" / "not present" / "lacking".

### Fix

Prompt-only:

* `pipeline/stages/summarization/prompts.py` — added a "Disambiguating absent vs negative (relation_type=expression only)" block immediately under the `direction` definition, with concrete patterns for each label. Other relation_types pick up the explicit fallback.
* Bumped `MAP_PROMPT_VERSION` → `"map_prompt_v5_expression_absent_vs_negative"`.

### Verification

No regression test (prompt-only change to the rubric, no validator logic to assert against). Empirical verification deferred to the next calibration run: grep `enum_observations.jsonl` and `sum_canonical_rules` for `(relation_type=expression, direction ∈ {absent, negative})` co-occurrence within the same `FindingGroup`. If the post-fix count drops materially, the rubric tightening worked.

---

## Bug 48 — `Rule.type` Title-Case inconsistent with lowercase convention

**Status / Severity / Surface:** Fixed (2026-05-15) / Low / Summarisation, optional RULE block enums.

### Symptom

`Rule.type` declared `Literal["Diagnostic", "Prognostic", "Management"]` and `RuleCounts` mirrored the casing in field names (`Diagnostic: int`, `Prognostic: int`, `Management: int`). Every other enum in the summarisation pipeline is lowercase post-B-016 (`Finding.category`, `Finding.confidence`, `RelationTypeEnum`, `DirectionEnum`). The inconsistency invited any future refactor to (a) accidentally lowercase the Rule fields and break legacy payloads, or (b) re-introduce a casing alias map to paper over the lower-vs-Title-Case bleed.

### Evidence

* [MAP_PROMPT_AUDIT.md Issue 7](MAP_PROMPT_AUDIT.md#issue-7--ruletype-is-title-case-diagnosticprognosticmanagement-everything-else-lowercase-low) — flagged as a latent inconsistency, deferred until the optional RULE block was next exercised.

### Diagnosis

The audit deferred this because the optional REDUCE+RULES block is off by default — no production cost. We lifted the deferral once the rest of the audit was being addressed in the same pass; the casing migration is mechanical and the back-compat shim is a single `mode="before"` validator.

### Fix

* `pipeline/stages/summarization/models.py`
  * `Rule.type: Literal["diagnostic", "prognostic", "management"]`.
  * New `Rule._lowercase_type` `field_validator(mode="before")` so any legacy Title-Case payload (cached LLM output, hand-authored test fixtures) round-trips cleanly.
  * `RuleCounts` field names lowercased to `diagnostic`, `prognostic`, `management`.
* `pipeline/stages/summarization/helpers/grounding_filter.py` — `_recompute_audit` reads `counts["diagnostic"]` etc., matching the new `Rule.type` casing.
* `pipeline/stages/summarization/prompts.py` — RULE OutputFormat block now shows `"type": "diagnostic|prognostic|management"` and `"rules_by_type": {{"diagnostic": N, "prognostic": N, "management": N}}`.
* `tests/summarization/test_enum_alias_repair.py` — `test_rule_type_lowercase_validates`, `test_rule_type_case_repair` (parametrised over Title-Case / UPPERCASE legacy variants), `test_rule_counts_uses_lowercase_field_names`.
* `tests/test_inspector.py` fixture updated (the only in-tree consumer that hand-wrote a Title-Case `Rule.type` payload).

### Verification

`pytest tests/summarization/test_enum_alias_repair.py` — all 60 tests pass (5 new). RULE block is off by default; no production payloads to migrate. If the block is ever re-enabled, the back-compat validator on `Rule.type` ensures legacy cached `ExtractedRules` JSONs still parse.

---

## Bug 49 — CANONICALIZE folds `unclear` / `no_direction` into majority polarity bin

### Status / Severity / Surface

Fixed (2026-05-15) · Medium · Summarisation, CANONICALIZE direction policy. Supersedes [B-026](#bug-26--canonicalize-split_by_direction-tie-break-is-member-order-dependent).

### Symptom

`CanonicalizeStage._split_by_direction` folded findings with `direction=unclear` and `direction=no_direction` into the largest polarity bin in a mixed group. Two compounding problems:

1. **Reproducibility hole.** Tie-break in the folding logic was `max(non_unclear, key=lambda d: len(non_unclear[d]))`. On ties, Python's `max` returns the first key it iterates — and dict iteration order traces back to upstream member-arrival order via `_direction_counts(members)`. The unclear members then attached to a non-deterministically-chosen bin → same paper, same data, different `member_normal_ids` / `finding_count` / `mean_grounding_score` on the emitted `CanonicalRule` across re-runs. Bad for thesis reproducibility claims.
2. **Honesty hole.** Unclear / no_direction findings were re-cast as votes for the majority direction. RESOLVE then inflated `finding_count` using hedged findings; RELATE paired the inflated rule against other papers as if the model had really claimed the majority polarity. The "I don't know" signal vanished after CANONICALIZE.

### Evidence

* The tie-break case manifested whenever a group had ≥2 polarity bins of equal size *and* at least one unclear / no_direction member to distribute. Pre-fix the regression test `tests/summarization/test_canonicalize_direction_split.py::test_split_no_direction_attaches_to_largest_polarity_in_mixed_group` (B-021 era) encoded this folding explicitly.
* User-facing data: `sum_canonical_rules` rows where `finding_count` overcounted relative to the actual polarity-claiming source findings; `sum_relations` paired hedged rules against cross-paper polarity rules.

### Diagnosis

Two design holes, single mechanical cause: the folding logic. The fix is to *not* fold. Every observed direction in a group gets its own `CanonicalRule`. RELATE / corpus_relate skip pairs where either side carries `unclear` / `no_direction` (NLI on those is meaningless — there's nothing to SUPPORT or CONTRADICT). The data pipeline stays lossless; the inert rules carry traceability without polluting the relation graph.

`is_conflicted` cannot survive its old "within-bin polarity diversity" semantics — within a per-direction bin, polarity is uniform by construction. The flag is repurposed to **group-level** polarity disagreement (True iff the parent group produced ≥2 polarity-bearing direction bins, stamped on every rule emitted from that group). Documented in the `CanonicalRule.is_conflicted` field docstring (`models.py`) and at the assignment site in `canonicalize_stage.py`.

### Fix

* `pipeline/stages/summarization/models.py`
  * Added `direction_value(d) -> str` normalizer alongside `DirectionEnum`. Handles `DirectionEnum` members, raw strings (from JSON-roundtripped metadata), and `None` (legacy rows) uniformly so a naive `direction in NON_POLARITY_DIRS` check can't silently fail.
  * Added `POLARITY_BEARING_DIRS = frozenset({"positive", "negative", "absent", "partial"})` and `NON_POLARITY_DIRS = frozenset({"unclear", "no_direction"})` as the single source of truth used by canonicalize / relate / corpus_relate. `partial` is included in the polarity set so `positive + partial → is_conflicted=True`; this is not a final decision on partial's semantics — owned by [B-025](#bug-25--relate-polarity-guard-treats-partial-as-positive-blocking-partial-vs-negative-contradictions).
  * Added `CANONICALIZE_DIRECTION_POLICY_VERSION = "per_direction_no_folding_v2"` near `MAP_SCHEMA_VERSION`.
  * Updated `CanonicalRule.is_conflicted` field docstring to spell out the group-level semantics.
* `pipeline/stages/summarization/current_stages/canonicalize_stage.py`
  * Rewrote `_split_by_direction(member_nfs)` — one bin per observed direction via `direction_value()`; returns `sorted(bins.items())` so emit order is deterministic. Dropped the `group` argument and the `direction_counts` dependency.
  * Refactored `_compute_scope_fields` → `_study_coverage(member_nfs) -> str`; the dead within-bin `is_conflicted` computation is gone.
  * `canonicalize()` now computes group-level `is_conflicted` once per group using `POLARITY_BEARING_DIRS` (no local re-definition), applies it to every emitted rule, and inlines a comment on the new semantics.
* `pipeline/stages/summarization/current_stages/relate_stage.py`
  * Added an early-return at the top of `_should_compare`: `if direction_value(a.direction) in NON_POLARITY_DIRS or direction_value(b.direction) in NON_POLARITY_DIRS: return False, "non_polarity_direction"`. Runs before category / relation_type / subject / outcome gates so non-polarity pairs are never NLI-evaluated. `_POSITIVE_DIRECTIONS` / `_NEGATIVE_DIRECTIONS` / `same_polarity` logic for CONTRADICT eligibility unchanged — only operates on pairs that already passed the new gate.
* `pipeline/stages/summarization/helpers/corpus_relate.py`
  * Mirror non-polarity skip at the top of `_should_compare_cross_paper`. Cross-paper rules can arrive with `direction` as a raw string (metadata roundtrip) or `None` (legacy); the normalizer handles both.
* `pipeline/stages/summarization/runner.py` + `pipeline/stages/summarization/batch/runner.py`
  * Both `_pipeline_config_hash` implementations include `CANONICALIZE_DIRECTION_POLICY_VERSION` in their `thresholds` dict so the cache hash flips whenever canonicalization semantics change. Forces a clean re-run of `out/summaries/summaries/*.json` next invocation.

### Verification

* **Unit tests** — `pytest tests/summarization/test_canonicalize_direction_split.py` (16 cases including S5 core invariant `test_no_unclear_leakage_into_polarity_bins`); `pytest tests/summarization/test_relate_skipped_pairs.py` (14 cases, 4 new non-polarity); `pytest tests/summarization/test_corpus_relate_non_polarity.py` (6 cases); `pytest tests/summarization/test_pipeline_config_hash.py` (13 cases, 2 new for the constant). All green.
* **Cascade regression** — `pytest tests/summarization/ -q` (574 passed, 1 pre-existing unrelated UMLS-singleton test-order pollution in `test_phase_a_gate.py::test_normalize_stage_normalizes_entities` which passes standalone).
* **Determinism** — `tests/summarization/test_canonicalize_direction_split.py::test_b026_determinism_2pos_2neg_1unclear` re-runs the 2-positive-2-negative-1-unclear case with the member list reversed; the emitted rules are identical (sorted by direction, deterministic bin contents). Supersedes B-026.

### Follow-up

* **Presentation filtering** — `unclear` / `no_direction` `FinalRule` rows now appear in `sum_final_rules`. The data pipeline stays lossless; if a future inspector / report wants a polarity-only view, filter at the presentation layer rather than dropping rows in RESOLVE. Tracked in `docs/THESIS.md` TODO.
* **`partial` semantics** — B-049 keeps `partial` in `POLARITY_BEARING_DIRS` for the group-level flag and in `_POSITIVE_DIRECTIONS` for the CONTRADICT gate. Whether either is right is owned by [B-025](#bug-25--relate-polarity-guard-treats-partial-as-positive-blocking-partial-vs-negative-contradictions) and needs a calibration sweep before changing.

---

## Bug 50 — `poll_interval` default mismatch across CLI and batch helpers

### Status / Severity / Surface

Fixed (2026-05-15) · Low · Scripts, batch poll interval.

### Symptom

`scripts/run_paper.py` shipped three diverging defaults for the batch poll
interval. A programmatic caller importing `_run_all_batch` got a 20s default;
the same caller importing `_run_batch` got 60s; the argparse default was also
60s. CLI flows always passed `args.poll_interval` so the call-site defaults
rarely fired in production, but the duplicate defaults were a footgun for
anyone scripting against the helpers directly.

### Evidence

* `scripts/run_paper.py:331` (pre-fix) — `parser.add_argument("--poll-interval", type=int, default=60, ...)`.
* `scripts/run_paper.py:787` (pre-fix) — `def _run_all_batch(pmcids, poll_interval: int = 20, ...)`.
* `scripts/run_paper.py:885` (pre-fix) — `def _run_batch(pmcid, poll_interval: int = 60, ...)`.

### Diagnosis

Classic duplicate-defaults bug. The two batch helpers diverged historically
(one written for multi-paper polling, the other for single-paper) and the
argparse default was added later without re-aligning the function signatures.

### Fix

Module-level constant `DEFAULT_POLL_INTERVAL_SEC = 60` at
`scripts/run_paper.py:48`. Argparse `default=DEFAULT_POLL_INTERVAL_SEC`,
both function signatures `poll_interval: int = DEFAULT_POLL_INTERVAL_SEC`.
Single source of truth.

### Verification

* `tests/test_poll_interval_defaults.py::test_poll_interval_defaults_agree` — uses `inspect.signature(...).parameters["poll_interval"].default` on both helpers (deliberately not `func.__defaults__` tuple indexing, which breaks the moment a positional parameter is added before `poll_interval`) and asserts both equal `DEFAULT_POLL_INTERVAL_SEC == 60`.
* CLI flow unchanged — both helpers still receive `args.poll_interval` from `main()` (`run_paper.py:448, 458`), so behaviour in the default invocation path is identical to pre-fix.

---

## Bug 51 — MAP agreement gate treats opposite polarity as soft disagreement

### Status / Severity / Surface

Fixed (2026-05-15) · High · Summarisation, MAP agreement gate.

### Symptom

`EmbeddingScorer._polarity` applies a 20% multiplicative penalty when claim
text appears to contradict. Two voters with opposite-polarity paraphrases of
the same claim (embedding similarity ≈ 1.0) produce a final agreement score
of exactly 0.80 — passes `theta=0.7` (default in `AgreementChecker`), sits
on the boundary at `theta=0.8`. The cascade therefore accepts chunks where
voters directly contradict each other, contaminating any downstream theta
sweep that tries to claim "the cascade is safe."

### Evidence

* `pipeline/stages/summarization/agreement/embedding.py:34-73` (`_polarity` heuristic).
* `pipeline/stages/summarization/agreement/embedding.py:279` — `contradiction_factor = 1.0 - 0.20 * ratio`.
* `tests/summarization/agreement/test_embedding_scorer.py::test_contradiction_max_penalty_bounded` (pre-fix) asserts `score == pytest.approx(0.80, abs=1e-6)` for the worst-case opposite-polarity pair.
* `Finding.direction: DirectionEnum` is structured data already available at agreement-scoring time — `_polarity` ignores it and inspects `claim` text instead.

### Diagnosis

The scorer's polarity signal is a token-spotting heuristic on free text; it
participates as a multiplicative dampening rather than a veto. The cascade
needs a *structural* check on `Finding.direction` that runs as a hard-fail
regardless of embedding similarity. Adding this at scorer level would couple
agreement decisions to scoring details; adding it only in the router would
miss the no-router code path (and the batch runner shares
`AgreementChecker.compute` as the single chokepoint).

### Fix

**New helper** at `pipeline/stages/summarization/agreement/polarity_conflict.py`:
`detect_polarity_conflict(outputs: list[AuditableSummary]) -> dict | None`.
Pure function. Iterates voter pairs × finding pairs. A pair is a hard-fail
iff:

* **comparable** — same `subject_entity`, `outcome_entity`, `relation_type`,
  `category` (all four required, strings `.strip().casefold()`d; any `None`
  field disqualifies). Mirrors `group_stage._group_id` (post-B-022) — the
  same definition of "this is the same claim" the codebase already uses.
* **opposite hard polarity** — `{a.direction, b.direction} == {positive, negative}`.

`AgreementChecker.compute` (`agreement/checker.py`) calls the helper *after*
the scorer runs (so `pairwise_upper` / `embedding_agreement` stay available
for trace inspection) but *before* the theta fallback. On conflict it sets
`bundle.decision = ChunkDecision.ESCALATE` and stamps
`score_details["hard_fail_reason"] = "polarity_conflict"` plus
`score_details["polarity_conflict_details"]` with per-pair records.

`MapOutputRouter._agreement_gate` (`routing/router.py`) reads
`bundle.score_details["hard_fail_reason"]`. On conflict it returns a
`RoutingDecision` with `reason_codes=[ReasonCode.POLARITY_CONFLICT]` — never
co-emits `INSUFFICIENT_AGREEMENT` or `ESCALATED_DUE_TO_LOW_AGREEMENT` (the
score was high; only the structural check failed). Explanation reads
`"Polarity conflict on N comparable finding pair(s); embedding agreement=X.XX overridden."`

New reason code `ReasonCode.POLARITY_CONFLICT = "polarity_conflict"` in
`routing/models.py` (extensible enum; no schema migration; flows naturally
into `CascadeDecisionRecord.reason_codes` and the JSONL decision log).

**v1 conservative scope**:

* Polarity set is `{positive, negative}` only. `absent` / `partial` /
  `unclear` / `no_direction` deliberately excluded — `absent` overlaps
  semantically with `negative` in biomarker-expression contexts; `partial`
  is bundled with `positive` in `relate_stage._POSITIVE_DIRECTIONS` so MAP
  must not be stricter than RELATE. Broadening waits on B-025 calibration.
* Scope fields (`disease_subtype`, `tissue_site`, `treatment_context`, …)
  are NOT part of the comparability key. Same biomarker / different cohort /
  opposite direction will therefore hard-fail today — a conservative
  *false-escalation* trade-off (extra L3 call is cheap; a missed
  contradiction silently kept as L1 consensus is expensive). Scope-aware
  comparability is v2.
* Evidence-disjoint hard-fail is NOT implemented in this patch. Two correct
  paraphrases citing different but valid sentences (e.g. summary vs methods)
  would false-escalate without a proximity policy. TODO in the helper
  references `agreement/hybrid_structured.py:_evidence_jaccard`.

**Cache invalidation** — both layers covered:

* `MAP_SCHEMA_VERSION` bumped `"map_v8_direction_alias_repair"` →
  `"map_v9_polarity_hard_fail"`. Invalidates `PipelineCache.set_map` /
  `get_map` entries (key includes `schema_version`) so stale chunk-level
  KEEP winners cannot leak.
* `MAP_AGREEMENT_POLICY_VERSION = "polarity_hard_fail_v1"` added to
  `models.py`; included in `compute_pipeline_config_hash` on both
  `SummarizationRunner._pipeline_config_hash` and
  `BatchSummarizationRunner._pipeline_config_hash` (same pattern as B-049's
  `CANONICALIZE_DIRECTION_POLICY_VERSION`). Invalidates per-paper result
  cache (`out/summaries/summaries/*.json`).

### Verification

* `tests/summarization/agreement/test_b051_hard_fail_polarity.py` — 11
  deterministic regression tests (mocked scorer, synthetic `AuditableSummary`
  / `Finding`; no real embeddings, no LLM calls). Covers ESCALATE/KEEP
  paths, comparability gates, every polarity pair the v1 policy excludes
  (`unclear`, `no_direction`, `absent`, `partial`), trace metadata
  structure, and the router translation of the hard-fail signal into
  `ReasonCode.POLARITY_CONFLICT`.
* `tests/summarization/test_pipeline_config_hash.py` — 3 new tests:
  - `test_map_agreement_policy_version_changes_hash` — flipping the version
    string changes the hash.
  - `test_map_agreement_policy_constant_is_defined` — constant import path.
  - `test_both_runners_include_map_agreement_policy_version_in_thresholds`
    — greps both runner sources for the literal key so a future refactor
    that drops it gets caught.
* Full suites: `tests/summarization/agreement/` + `tests/summarization/routing/` — 207 passed.

### Follow-up (not in this patch)

* Theta sweep — runs *after* this lands; calibration numbers are now
  meaningful.
* Evidence-disjoint hard-fail (v2) — proposal sketched in the helper
  docstring.
* Scope-aware comparability (v2) — per-category equivalence rules over
  `FindingScope` fields.
* Broaden polarity set to include `absent` / `partial` once B-025
  calibration is in.

---

## Bug 52 — Cost-estimation script underestimates per-chunk input tokens

### Status / Severity / Surface

* **Status:** Fixed (2026-05-16)
* **Severity:** Medium — wrong headline number on every cost projection.
* **Surface:** `scripts/estimate_selection_cost.py:per_chunk_input_tokens`,
  `load_paper_stats`

### Symptom

Every row in the "MAP cost projection" table printed by
`scripts/estimate_selection_cost.py` was ~15–20% lower than the cost a
real run on the same selection would incur. The `in/chunk` column was
the load-bearing input: `n_chunks × in/chunk × $/MTok` is the L1 cost
formula, and `in/chunk` was systematically off.

### Diagnosis

The pre-fix `per_chunk_input_tokens` computed the average sentences
per chunk as:

```python
avg_sentences_per_chunk = min(chunk_size, n_sentences / n_chunks * (1 + 0))
```

At the production defaults (`chunk_size=10`, `chunk_overlap=2`,
`stride=8`), `n_sentences / n_chunks ≈ stride = 8`, so the clamp
returned ~8. But `MapStage._make_chunks`
(`pipeline/stages/summarization/current_stages/map_stage.py:1263-1267`)
slices:

```python
stride = self.chunk_size - self.chunk_overlap
chunks = [sentences[i : i + self.chunk_size]
          for i in range(0, len(sentences), stride)]
```

Each non-tail chunk holds exactly `chunk_size` (10) sentences. For
`n_sentences=100`, `range(0, 100, 8) = [0, 8, …, 96]` → 13 starts. The
first 12 chunks have 10 sentences each; the tail chunk has 4. Total
sentence-occurrences across chunks = 12·10 + 4 = 124. Average =
124 / 13 ≈ 9.54. The script reported ~7.69 — ~20% low.

The trailing `* (1 + 0)` in the original formula was a leftover from a
removed overlap term.

### Fix

[`scripts/estimate_selection_cost.py:per_chunk_input_tokens`](../scripts/estimate_selection_cost.py)
rewritten to compute total sentence-occurrences exactly the way
`_make_chunks` produces chunks, then divide by `n_chunks`:

```python
stride = chunk_size - chunk_overlap
total_sentence_occurrences = sum(
    min(chunk_size, n_sentences - start)
    for start in range(0, n_sentences, stride)
)
avg_sentences_per_chunk = total_sentence_occurrences / n_chunks
chunk_text_tokens = text_tokens * avg_sentences_per_chunk / n_sentences
return PROMPT_OVERHEAD_TOKENS + ceil(chunk_text_tokens)
```

Three associated changes in the same patch:

1. Function signature gained `chunk_overlap: int` (caller in
   `load_paper_stats` updated). Without it, the stride couldn't be
   recomputed to mirror `_make_chunks`.
2. Defensive validation: `0 <= chunk_overlap < chunk_size` — same
   precondition `MapStage.__init__` already enforces, surfaced at the
   helper boundary so a future caller that constructs the script
   programmatically gets a clear `ValueError` instead of a silent
   nonsense result.
3. Rounding switched from `round` to `ceil`. Budget estimates should
   not understate cost; the prior `round` could shave a half-token off
   every chunk. Imported `from math import ceil` (the dead `import
   math` was removed in the same edit).

Co-fixed: `load_paper_stats` was ordering DB rows by
`TextElement.position_in_section`, which interleaves sections (the
B-039 bug). Flipped to `.order_by(TextElement.id)` so the script
actually mirrors `SummarizationRunner.load_paper_from_db`
(`pipeline/stages/summarization/runner.py:940`) as its docstring claims.

### Verification

* Manual: `n_sentences=100`, `chunk_size=10`, `chunk_overlap=2`,
  `stride=8`. `range(0, 100, 8) = 13` starts. Per-chunk counts
  `[10, 10, …, 10, 4]`. Total = 124. Average = 124/13 ≈ 9.54.
  Pre-fix returned ~7.69; post-fix returns 9.54.
* Regression test:
  [`tests/test_estimate_selection_cost.py`](../tests/test_estimate_selection_cost.py)
  pins the formula against the worked example plus edge cases
  (empty input, exactly one chunk, no overlap, overlap-out-of-range
  validation).
* End-to-end smoke: rerun
  `python scripts/estimate_selection_cost.py --from-selection
  configs/paper_selection/calibration_set_v1.yaml --profile cheap` —
  `in/chunk` is ~15–20% higher than pre-fix on the same papers, per-tier
  totals scale by the same factor.

---

## Bug 53 — Percentiles cost estimator hygiene cluster

### Status / Severity / Surface

* **Status:** Fixed (2026-05-16)
* **Severity:** Low — none of these changed a printed cost number.
* **Surface:** `scripts/estimate_pipeline_cost_percentiles.py`

### Symptom

Pre-fix, the script projected MAP cost for the P80 / P90 papers (by
`text_elements` count — the long-tail/upper-bound budget the supervisor
quotes). Audit surfaced five small defects: two dead imports, two
never-called helpers, a misleading inline comment about `est_chunks`,
no validation in `pick_percentile`, and `CHUNK_SIZE` / `CHUNK_OVERLAP`
hardcoded as module constants instead of pulled from
`MapConfig` — a silent drift hazard if production chunking ever
changes.

### Fix

[`scripts/estimate_pipeline_cost_percentiles.py`](../scripts/estimate_pipeline_cost_percentiles.py):

* Removed `import json`, `import statistics` — neither was referenced.
* Removed `estimate_non_llm_stages` (line 278 pre-fix) and
  `render_paper_table` (line 325 pre-fix). Both were defined but never
  called; `main()` already covered the same output inline.
* Rewrote the `est_chunks` inline comment to cite
  `MapStage._make_chunks` so the next reader doesn't have to verify the
  formula against the (correctly different) code.
* `pick_percentile` now raises `ValueError` on empty corpus
  (previously `idx = -1` and silently returned the last paper) and on
  `p ∉ (0, 1]`.
* `CHUNK_SIZE` and `CHUNK_OVERLAP` are now read from `MapConfig()`
  defaults at module load — single source of truth with the
  production config.

### Why this is small

The script is a pre-run budget estimator for the *largest* papers in
the corpus (P80, P90 by `n_te`). It's deliberately a conservative
upper bound: per-chunk tokens come from a single observed L3 call,
applied uniformly to every voter at every tier; cascade escalation
rates are scenario inputs, not measurements. Nothing in this patch
changes the numbers it prints — purely hygiene + drift prevention.

### Verification

* `tests/test_estimate_pipeline_cost_percentiles.py` — 15 cases:
  `CHUNK_SIZE` / `CHUNK_OVERLAP` / `STRIDE` track `MapConfig` defaults;
  `est_chunks` matches `len(range(0, n, stride))` for n ∈ {1, 8, 9, 100,
  200}; `pick_percentile` nearest-rank for n ∈ {10, 100} and p ∈ {0.5,
  0.8, 0.9, 1.0}; empty-corpus + out-of-range `p` validation.
* Manual: running the script against the production DB produces the
  same per-paper rows and per-tier totals as before — verified by diff
  on `out/cost_percentile_report.md`.

---

## Bug 54 — NER stage scispaCy singleton bypass

### Status / Severity / Surface

* **Status:** Fixed (2026-05-16)
* **Severity:** High — ~150 s wasted per paper, peak RSS up to 3× the
  intended scispaCy footprint, OOM risk on low-RAM machines.
* **Surface:** `named_entity_recognition/ner.py` `load_ner_model()` /
  `load_linker_model()` / `run_ner_on_db()`. Triggered from
  `pipeline/stages/summarization/runner.py:660-671`.

### Symptom

A multi-paper sync run shows the UMLS / scispaCy linker load message
firing once for the first paper (via `umls_resources.get_nlp()`) as
expected — *and then again for every paper during the NER stage*:

```
02:51:31  [PMC10100421_HIS-82-393] NER — running entity extraction…
Loading fast NER model...
✓ Fast NER model loaded
Loading UMLS linker model...
✓ UMLS linker model loaded
⚠ Document PMC10100421_HIS-82-393 already has 2344 entities …
02:53:47  [PMC10100421_HIS-82-393] NER done [136.4s]
```

That 136-second NER wall-time on a *skipped* paper is pure model-load
overhead — both scispaCy loads completed before the existing-entities
check fired. Next paper repeats the loads from scratch.

### Diagnosis

`named_entity_recognition/ner.py` (pre-fix):

```python
def load_ner_model():
    nlp = spacy.load("en_core_sci_lg", disable=[…])
    return nlp

def load_linker_model():
    nlp = spacy.load("en_core_sci_lg", disable=[…])
    nlp.add_pipe("scispacy_linker", config={"threshold": 0.85, …})
    return nlp
```

Both call `spacy.load(...)` directly. No module-level cache, no
singleton routing. `run_ner_on_db(pmcid, …, nlp=None, linker_nlp=None)`
defaults both to `None`; the runner at
`pipeline/stages/summarization/runner.py:666` calls it without passing
either, so the loaders fire with `None` and load fresh every time.

Two-instance memory cost per paper because the codebase deliberately
keeps "fast NER" and "linker" as separate models — the fast pass
extracts entity spans without running the linker, then a second pass
batches the unique span strings through the linker model. Both
instances are full `en_core_sci_lg` loads (~2.6 GB each with the
linker), and `umls_resources.get_nlp()` already holds a third copy for
NORMALIZE / UMLS_ENRICH. Peak RSS in the user's run hit ~3.2 GB during
linker load on top of the existing 2.6 GB singleton.

Worse, the "Document already has entities" early-exit check at
`ner.py:186-199` (pre-fix) fired *after* both loads — so a paper with
cached entities still paid the full ~150 s load cost before bailing.

Same class as B-029 (`PipelineRunner._get_nlp` direct load) and B-038
(`SummarizationRunner.load_paper_from_db` direct load). Missed by the
existing singleton-guard test because that test only scanned files
under `pipeline/stages/`; `named_entity_recognition/` lives at repo
root and was outside the scan.

### Fix

1. **Route both loaders through the singleton.** `load_ner_model()`
   and `load_linker_model()` now call `umls_resources.get_nlp()` (raise
   a clear `RuntimeError` when it returns `None`, e.g. under
   `NLP_HISTO_DISABLE_UMLS=1` — the runner's `try/except` around the
   NER call logs and continues). Both functions return the same
   singleton object — kept as two names for API back-compat.

2. **Disable the linker pipe for the fast-NER pass.** The singleton has
   `scispacy_linker` attached for NORMALIZE / UMLS_ENRICH; running it
   on every doc during span extraction would slow pass 1 by ~10×.
   Wrap the pass-1 loop in `nlp.select_pipes(disable=["scispacy_linker"])`
   — temporary mute, restored on `with` exit, so the singleton stays
   intact for downstream stages.

3. **Move the "already has entities" check before model loading.** A
   paper that already has entities now bails in ~10 ms instead of
   ~150 s.

`batch_ner.py` benefits automatically — it imports the same loaders.

### Verification

* `tests/summarization/test_scispacy_singleton.py::test_ner_module_routes_through_singleton`
  — greps both loader sources for direct `spacy.load(` calls and
  asserts they route through `umls_resources` / `get_nlp`.
* `test_only_umls_resources_calls_spacy_load_in_pipeline_tree` (existing)
  continues to pass because `named_entity_recognition/` is outside its
  scope; the new test fills the same gap for the NER module.
* Manual check on a real run: paper 1 shows the one-time "UMLS:
  loading" log line; subsequent papers' NER stages no longer print
  "Loading fast NER model..." → "Loading UMLS linker model..." paired
  with ~75 s loads (they print as singleton cache hits and inference
  proceeds immediately).

### Follow-up (not in this patch)

* Collapse `run_ner_on_db`'s `nlp=` and `linker_nlp=` parameters into a
  single `nlp=` argument since both are now the same object. Cosmetic
  API cleanup, low priority.
* Quiet the "Loading fast NER model..." / "Loading UMLS linker model..."
  `print()` calls — they're misleading now that both are singleton
  hits after the first paper. Use `logger.debug` instead.
* Audit `named_entity_recognition/` for other direct `spacy.load(...)`
  call sites and consider extending the singleton-guard test's scope
  to include it.


---

## Bug 55 — `sum_map_findings` not populated by batch runner

**Status:** Mitigated (2026-05-23) · **Severity:** High · **Surface:**
Summarisation, `BatchSummarizationRunner.finalize()` → `sum_map_findings` DB persistence.

### Symptom

`sum_map_findings` shows 0 rows for 9 of the last 10 batch-mode pipeline
runs (pipeline_runs ids 38–48, all status='success', 5 distinct papers):

| run | pmcid                    | map | norm | canon | rel | final | voters |
|----:|--------------------------|----:|-----:|------:|----:|------:|-------:|
|  48 | PMC9826086_HIS-81-786    | **0** |  214 |   203 |   0 |   203 |    132 |
|  47 | PMC7539961_HIS-77-579    | **0** |    0 |   147 |   0 |   147 |    122 |
|  46 | PMC6635746_HIS-73-68     | **0** |    0 |   173 |   1 |   173 |      7 |
|  44 | PMC4329418_his0066-0409  | **0** |    0 |   146 |   1 |   146 |      0 |
|  43 | PMC10100421_HIS-82-393   | **0** |    0 |   185 |   2 |   185 |      0 |

Each paper's `sum_rejection_summaries.map_findings_total` records 100–230 MAP
findings produced. Discovered during the [B-005 end-to-end verification TODO](../docs/THESIS.md#todos)
(L36) on 2026-05-16.

### Evidence

* Direct call to `persist_map_findings(db, 48, 'PMC9826086_HIS-81-786',
  chunk_summaries)` against the saved batch handle at
  `out/summaries/batch_handles.prepatch/PMC9826086_HIS-81-786.batch.json`
  wrote **154 rows successfully** with `verbatim_support` exactly matching
  `text_elements.text_content`. So the persistence function and DB wiring
  are correct.
* Same code path on HEAD `7ea254a` (pre B-005 dedup) shows the same
  failure — not a regression from today's refactor.
* `sum_map_voter_outputs` and `sum_rejection_summaries` populate on the
  same failing runs → the runner reaches end-of-`finalize()` with a valid
  `pipeline_run_db_id`; the failure is specific to the MAP-findings table.
* `sum_normal_findings` is also zero on runs 47/46/44/43, yet
  `sum_canonical_rules` has 100+ rows on each — physically impossible
  from the in-process flow, which strongly suggests those runs hit the
  result-cache short-circuit at `BatchSummarizationRunner.finalize()` L472
  (`if handle.cached_result_only: return self._load_result(pmcid)`).
  The on-disk JSON for cached runs only stores `canonical_rules` /
  `final_rules` (not `chunk_summaries` / `normal_findings`), so a
  cache-hit replay would naturally produce the 0/0/100+ pattern observed.

### Diagnosis (hypotheses, not yet confirmed)

The function is good; the call path inside `finalize()` is the problem.
Three candidate root causes, in descending likelihood:

1. **Cache short-circuit at L472** masks MAP+NORMALIZE persistence
   entirely for cache-hit runs. The pipeline_runs row is still created
   upstream (by `submit()` or its caller), so a successful cache-hit
   leaves a pipeline_runs row with zero MAP rows. This matches runs
   47/46/44/43 (0 normal, 100+ canonical). For run 48 it would not match
   on its own (214 normal_findings present) — possibly run 48 was a
   mixed-mode partial cache hit.
2. **Silent exception in the bulk `INSERT`** caught by `except Exception
   as exc: logger.warning(...)`. Production stdout for these runs was
   not captured to `logs/runA*.log` (those files cover earlier runs only),
   so a warning may have been emitted and lost. Confirmable by adding
   `logger.warning("[%s] DB: persist_map_findings entering with %d
   chunk_summaries, db_id=%s", pmcid, len(chunk_summaries), db_id)`
   ahead of the `try:` and re-running one paper.
3. **`chunk_summaries` is empty at L522** for some structural reason
   (e.g., the `cached_result_only` branch already returned at L476 but
   somehow execution continued). Less likely — `all_findings = [f for cs
   in chunk_summaries for f in cs.findings]` at L536 would also collapse
   to `[]`, producing zero normal_findings, which is contradicted by
   run 48 (214 rows).

### Mitigation

Manually back-populated `sum_map_findings` for run 48 from the on-disk
batch handle (154 rows). Verbatim spot-check passes (5/5 spans exact
match against `text_elements.text_content`). Historical runs 38–47 are
similarly recoverable from `out/summaries/batch_handles.prepatch/*.batch.json`
via the same `persist_map_findings(db, run_id, pmcid, chunk_summaries)`
pattern — pending user authorisation.

### Fix (proposed, not yet shipped)

Order of work, cheapest first:

1. Add a one-line `logger.info("[%s] _persist_map_findings entering with
   %d chunks (%d findings) db_id=%s", pmcid, len(chunk_summaries),
   sum(len(cs.findings) for cs in chunk_summaries), db_id)` at
   `batch/runner.py:522` (and the equivalent sync runner site) so the
   next batch run leaves a paper trail.
2. Audit the cache short-circuit at `finalize()` L472–L482: when
   `handle.cached_result_only` is true and `_load_result` returns a
   cached dict, the function exits **before** any DB persistence runs.
   But the pipeline_runs row already exists. Either:
   (a) move `submit()`'s `_create_pipeline_run` call out of the always-on
   path so cache hits never create a row; or
   (b) on cache hit, replay persistence using the cached payload (would
   require persisting `chunk_summaries` / `normal_findings` to the JSON,
   which today are omitted by `_save_result`).
3. Re-run one paper end-to-end against a real DB after (1) and (2),
   confirm `sum_map_findings` populates, then back-populate runs 38–47.

### Verification (target state)

* After the fix, a fresh batch run for a single paper produces
  `sum_map_findings.COUNT(*) >= sum_normal_findings.COUNT(*)` for that
  `pipeline_run_id` (MAP is upstream of NORMALIZE; counts must be
  monotone non-increasing through the stage chain).
* A cache-hit re-run of the same paper either skips creating a new
  pipeline_runs row, or replays persistence consistently.
* Add a regression test that calls `BatchSummarizationRunner.finalize()`
  against a synthetic `BatchHandle` with `cached_result_only=False` and
  asserts that all `sum_*` tables (including `sum_map_findings`) get
  rows.

### Mitigation shipped (2026-05-23)

The *silent corruption* is the defect — a failed insert that leaves zero
rows while the run reports `success`. That is now closed regardless of the
still-unconfirmed underlying insert failure:

* `persist_map_findings` (`pipeline/stages/summarization/persistence.py:740`)
  no longer swallows. It logs an entry line
  (`persist_map_findings: entering with N chunks (M findings) run_id=…`) then
  **re-raises** on any exception. Both runners wrap the call in a
  finalize-level handler that flips the `pipeline_runs` row to `failed`
  (sync `runner.py:793`; batch `finalize()` re-raises at `runner.py:713`),
  so a bad insert now fails loudly with the real DB traceback instead of a
  lost WARNING. Sibling `persist_*` helpers still swallow by design — out of
  scope here; the canonical-without-normal pattern (runs 47/46/44/43) is a
  separate follow-up.
* The all-or-nothing `engine.begin()` bulk insert (one bad finding aborts
  the whole paper's batch) is retained intentionally — partial inserts would
  be a quieter version of the same corruption. Root cause must be fixed in
  the data, not papered over.
* `scripts/diagnose_b055.py` replays every on-disk handle in
  `out/summaries/batch_handles.prepatch/` through the now-loud persist path
  against a throwaway `pipeline_run` (deleted via FK CASCADE) — **zero LLM
  cost** since the bug is entirely post-MAP. It reports PASS/FAIL with the
  exact DB exception per paper, doubling as the back-population tool for
  Fix-step 3 (runs 38–47) once the root cause is known.

### Diagnostic replay (2026-05-23) — hypothesis #2 ruled out

`python scripts/diagnose_b055.py` replayed **all 26** on-disk handles through
`persist_map_findings`. **Every one persisted cleanly** — including the
production failures: PMC9826086 (run 48) 154/154 rows, PMC7539961 (run 47)
112/112, PMC6635746 (run 46) 119/119. Eight handles show `0/0 rows (0
chunks)` because their `finalized` dict is *empty* on disk (MAP never
finalised) — among them PMC10100421 (run 43) and PMC4329418 (run 44), which
nonetheless reported 100+ `sum_canonical_rules` in production.

**Conclusion:** the bad-data / column-constraint hypothesis (#2) is **ruled
out** — `persist_map_findings` is correct for every paper's data. The
failure is **upstream in `finalize()` control flow**, elevating hypothesis
#1 (cache short-circuit). Two distinct patterns now visible:

* **run 48** (full handle, `map=0` but `normal=214` in prod): both derive
  from the same `chunk_summaries` and `persist_map_findings` runs *before*
  NORMALIZE — the only way map=0/normal=214 happens in-process is the old
  swallow firing on a **transient** insert error (deadlock / connection
  blip), letting NORMALIZE proceed. The 2026-05-23 re-raise addresses
  exactly this: a transient failure now fails the run loudly instead of
  silently dropping MAP rows. *Not reproducible offline* (no transient
  condition in isolated replay), consistent with a non-deterministic cause.
* **runs 43/44/46/47** (`map=0` AND `normal=0`, yet 100+ canonical): these
  ran the *full* pipeline (4-min runtimes, `pipeline_run` rows created — not
  the `submit()` cache short-circuit, which sets no `db_id` and persists
  nothing). Canonical derives from in-*memory* normal findings, so the
  in-process flow had data; only the MAP + NORMALIZE *DB writes* produced
  zero rows while CANONICAL/FINAL wrote 100+. **This is the remaining open
  follow-up** (THESIS TODO B-055 step 3) — the re-raise does *not* fix it.

**DB ground truth (2026-05-23, current state).** `pipeline_runs` 38–48: ids
40/45 `failed`, 41/42/45 `interrupted`, rest `success`. Row counts by
`pipeline_run_id`: `sum_map_findings` = `{48:154}` (only the manual
back-fill); `sum_normal_findings` = `{48:214}`; `sum_canonical_rules` /
`sum_final_rules` = `{38:185, 39:146, 43:185, 44:146, 46:173, 47:147,
48:203}`. So **every** successful run wrote canonical/final, **none** wrote
map/normal (except run 48's manual fix).

**Cross-run deletion ruled out:** `clear_normalized_run_data` deletes on
`(pipeline_run_id == db_id) AND (pmcid)` — run-scoped, not pmcid-wide, so a
later same-paper run does not wipe an earlier run's rows.

**Lead for the next instrumented run.** The two writers that fail
(`persist_map_findings`, and `clear_normalized_run_data` invoked by
`persist_normal_findings`) both use `db.engine.begin()`; the writers that
succeed (`persist_canonical_rules`, `persist_final_rules`) use
`db.session_scope()`. Suspect the `engine.begin()` Core-transaction path —
not the row data — fails in the batch finalize context (pool/transaction
state), is swallowed pre-2026-05-23, and the in-memory pipeline continues to
CANONICAL. The failure is **production-only / non-deterministic** (clean in
isolated replay), so the fix path is: run one paper end-to-end now that
`persist_map_findings` re-raises — it will fail loudly at the exact call with
the real `engine.begin()` traceback, or succeed (confirming a transient
cause). Static analysis + frozen artifacts cannot go further.


## Bug 56 — Batch runner omits per-voter MAP persistence (code path absent)

**Status:** Observed (2026-05-16) · **Severity:** Medium · **Surface:**
Summarisation, `BatchSummarizationRunner.finalize()` → `sum_map_voter_outputs`
DB persistence.

### Symptom

`BatchSummarizationRunner.finalize()` does not buffer per-voter
`AuditableSummary` rows and does not write `sum_map_voter_outputs`. Any
θ / reject_θ replay over batch-processed papers has no per-voter rows to
read.

### Evidence

Code-level grep across `pipeline/stages/summarization/batch/`,
`routing/`, and `scripts/run_paper.py` for
`voter_output | VoterOutput | _persist_voter | sum_map_voter`:

* `runner.py` (sync) — 6 hits (`runner.py:346, 350, 521, 1109, 1127, 1129`).
* `current_stages/map_stage.py` — 8 hits (buffer + records helpers at
  L271, 457, 527, 641, 665, 701, 719, 1148–1212).
* `cache.py` — 4 hits (round-trips voter rows across cache hits at
  L180–215).
* `batch/runner.py` — **0 hits.**
* `batch/dispatch.py`, `batch/azure_batch.py`, etc. — **0 hits.**

`BatchSummarizationRunner.finalize()` reconstructs chunk-level
`AuditableSummary` objects from `handle.finalized.values()`
(`batch/runner.py:483-485`) and proceeds directly into grounding,
filesystem persistence, and `_persist_map_findings`. The per-level
collectors (`_collect_l1`, `_collect_l2`, `_collect_l3` at L1332+) hold
raw provider outputs in `handle.l*_raw` but discard them after the
winning `AuditableSummary` is materialised; `MapStage._buffer_voter_outputs`
is never called because the batch path never instantiates `MapStage`.
`BatchSummarizationRunner` exposes `_persist_map_findings` /
`_persist_normal_findings` / `_persist_finding_groups` /
`_persist_canonical_rules` / `_persist_relations` / `_persist_final_rules` /
`_persist_rejection_summary` but no `_persist_voter_outputs`.

Same root cause leaves the `PipelineCache` entries for batch runs
missing the `voter_outputs` field — a follow-up sync run that
cache-hits a batch entry has no per-voter rows to rehydrate.

### Contradiction with B-055 evidence

[B-055](#bug-55--sum_map_findings-not-populated-by-batch-runner) records
that `sum_map_voter_outputs` populated on the same batch runs (38–48)
where `sum_map_findings` did not. That observation cannot be reconciled
with the code-level search above and is treated as a data point
requiring runtime re-verification. Plausible explanations to rule out
before fixing:

1. Runs 38–48 were not pure batch runs — at least one of the papers
   was previously processed via sync (or a manual `_persist_voter_outputs`
   call), so its rows survived under the same `pmcid`.
2. The count was joined on `pmcid` rather than `pipeline_run_id` and
   picked up rows from an earlier sync run.
3. A recently-removed code path wrote the table during the period
   covered by the audit; current HEAD no longer has it.

A clean re-verification: pick a `pmcid` that has never appeared in
`pipeline_runs`, run it batch-only, then `SELECT COUNT(*) FROM
sum_map_voter_outputs WHERE pipeline_run_id = <new_id>`. If zero, the
code-level finding is confirmed and B-055's adjacent observation was a
misattribution. If non-zero, identify the missing write path before
shipping the fix below.

### Fix (proposed, not yet shipped)

Mirrors the sync wiring. Order of work, cheapest first:

1. Extend the per-level collectors (`_collect_l1`, `_collect_l2`,
   `_collect_l3` in `batch/runner.py`) to retain a structured per-voter
   record matching `MapStage._buffer_voter_outputs`'s row shape:
   `chunk_id, level, voter_index, provider, model, is_selected,
   failed, error_message, finding_count, latency_ms, raw_output`.
   `is_selected` is determined after the agreement gate runs in
   `_process_level`.
2. Add `_persist_voter_outputs(db_id, pmcid)` to `BatchSummarizationRunner`
   mirroring the sync implementation (`runner.py:1109-1153`). Call it
   immediately after `_persist_map_findings` in `finalize()` (after the
   eventual B-055 fix lands so both rows write together).
3. In the same call sites, pass the voter rows to `cache.set_map(...,
   voter_outputs=...)` so batch cache entries round-trip per-voter rows
   the same way sync entries do.
4. Leave `map_theta_sweep.py` untouched — it uses its own primer cache
   and is unaffected by this gap.

### Verification (target state)

* After the fix, a fresh batch-only run for a single paper produces
  `sum_map_voter_outputs.COUNT(*) == n_chunks × voters_per_level summed
  across levels actually invoked` for that `pipeline_run_id`.
* `is_selected=True` rows match the winning chunk summary at each
  level.
* Regression test asserts `SumMapVoterOutput` rows are written by
  the batch path (mirrors the sync regression test that ships with this
  audit, `tests/summarization/test_persist_voter_outputs.py`).

---

## Bug 62 — documented router-on production cascade never actually enabled

**Status / Severity / Surface:** Observed (2026-05-23) / High / Summarisation, MAP
cascade selection + config reproducibility.

**Symptom:** The thesis docs describe `MapOutputRouter` (grounding-first gating,
L1→L3 skip) as the production MAP cascade, but production has been running the
legacy `AgreementChecker` 3-tier cascade. No runtime error — the legacy path is
a valid, working cascade — so the divergence was invisible until audited.

**Evidence:**
* `pipeline/stages/summarization/runner.py:188` and
  `pipeline/stages/summarization/batch/runner.py:140` both default
  `enable_router: bool = False`.
* `scripts/run_paper.py` (as committed before 2026-05-23) contained **zero**
  `router` references — neither `build_runner` nor `build_batch_runner` passed
  `enable_router`, so both took the `False` default. Exhaustive grep:
  `grep -ni router scripts/run_paper.py` → no matches; repo-wide
  `grep -rn "enable_router=True"` → only comments/docstrings, never executable.
* Contradicting docs: `docs/THESIS.md` TODO lines 56, 57, 76 and Decisions-log
  row 2026-05-15 ("`MapOutputRouter` wired into both runners **by default**");
  `eval/silver/map_theta_sweep.py` lines 23, 112, 975/977 ("production enables
  the router via `enable_router=True` in `scripts/run_paper.py`").

**Diagnosis:** The router was built (ABC_IMPLEMENTATION_COMPARISON Gap 2/Gap 8)
and the *intent* recorded as "on by default", but the wiring that would flip it
on at the `run_paper` entry point was never present (or was lost in a refactor —
`run_paper.py` was not in the working tree's modified set this session, so the
omission predates it). The runner default `False` therefore won, silently. This
also means the in-flight MAP θ calibration (which replays the legacy
`AgreementChecker`, `CASCADE_PATH="legacy_agreement_checker"`) happens to match
*actual* production — but not *documented* production.

**Fix (2026-05-23):** Made the cascade path explicit and config-governed
instead of an implicit runner default. `MapConfig` gained `enable_router` /
`router_single_voter_policy`; `build_runner` and `build_batch_runner` now read
`sum_cfg.map.*` and pass them through; the load log prints `map.enable_router`.
Pinned `enable_router: false` in `configs/run.yaml` — codifying *actual* current
behaviour (and keeping the legacy θ calibration applicable).

*Constructor-site audit:* `grep -rn "SummarizationRunner(\|BatchSummarizationRunner("`
shows `scripts/run_paper.py` (the documented entry point) plus four siblings —
`scripts/summarize_paper.py`, `scripts/run_single_doc.py`,
`scripts/run_paper_single_model.py`, `eval/silver/pipeline_sweep.py`. Only
`run_paper.py` is wired to read `sum_cfg.map.*`; the siblings build runners
without passing `enable_router`, so they take the default `False` = legacy =
the decided production path. None set the router on, so all sites are
*consistent* with the decision; threading config into the siblings is deferred
(scope) and tracked as a follow-up only if the router experiment is adopted.

**Resolved decision (2026-05-23):** production keeps the **legacy L1→L2→L3
`AgreementChecker` cascade** (`enable_router: false`) — it matches the actual
prior behaviour and the legacy θ calibration applies directly. The
`MapOutputRouter` L1→L3-skip path stays opt-in/experimental; if it is adopted
later, the THESIS.md ABC-P1 "router-path experiment" TODO covers re-validating
the chosen `(theta, reject_theta, scorer)` on it. Stale "production is router-on"
claims corrected: THESIS.md (ABC-P1 sweep clause + the "default mismatch" TODO,
both now ticked) and `eval/silver/map_theta_sweep.py` (module docstring +
`CASCADE_PATH` comment + the collect-time log, downgraded from `warning` to
`info`). The Decisions-log 2026-05-15 row ("wired in by default") is left as
historical intent, cross-referenced here.

**Verification:** `python3 -c "from pipeline.config_loader import load_config;
print(load_config('configs/run.yaml')[1].map.enable_router)"` → `False`; flipping
the YAML to `true` and reloading returns `True` (round-trip confirmed the YAML
governs both builders). `grep -ni router scripts/run_paper.py` now shows the
config pass-through at both builders + the load-log line.

---

## Bug 63 — `estimate_selection_cost.py` missing sys.path bootstrap

**Status / Severity / Surface:** Fixed (2026-05-24) / Low / Tooling, cost
estimation script.

**Symptom.** Running the cost estimator the way HOW_TO_RUN §5 documents it —

```bash
python scripts/estimate_selection_cost.py \
    --from-selection configs/paper_selection/related15_full.yaml --profile cheap
```

dies immediately with:

```
File ".../scripts/estimate_selection_cost.py", line 380, in main
    from pipeline.stages.summarization.costing import PriceBook
ModuleNotFoundError: No module named 'pipeline'
```

**Evidence.** Hit 2026-05-24 while estimating MAP cost for the `related15_full`
selection (15 most-related papers). The script's own module docstring already
worked around it by documenting `PYTHONPATH=. python …` — so the bare form
copied into HOW_TO_RUN §5 had never actually run.

**Diagnosis.** The repo import (`from pipeline...`) is deferred into `main()`,
but the module never prepends the repo root to `sys.path`. Run as
`python scripts/foo.py`, Python puts the *script's own directory* (`scripts/`)
on `sys.path[0]`, not the CWD — so top-level packages (`pipeline`, `database`)
at the repo root are invisible. Every other script under `scripts/`
(`run_paper.py:34-36`, `check_apis.py:34-36`) bootstraps with
`_REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, …)`; this
one was the lone violator of the CLAUDE.md "scripts must bootstrap their own
path" rule. Distinct defect from [Bug 52](#bug-52--cost-estimation-script-underestimates-per-chunk-input-tokens)
(same file, per-chunk token-count formula).

**Fix.** Added a bare `sys.path.insert(0, <repo_root>)` bootstrap immediately
before the first repo import (after `from pathlib import Path`), matching
`run_paper.py`. The *bare* form is deliberate: ruff's E402 exempts `sys.path`
modifications before imports, but a guarded `_REPO_ROOT = …; if … not in
sys.path:` block does **not** (it trips E402 — which is why the sibling
`check_apis.py`, using that guarded form, still flags one). Dropped the
now-redundant `PYTHONPATH=.` prefix from both docstring usage examples so the
documented command matches the self-bootstrapping convention. HOW_TO_RUN §5's
bare invocation is correct as-is post-fix — no change needed there.

**Verification.** `python scripts/estimate_selection_cost.py --from-selection
configs/paper_selection/related15_full.yaml --profile cheap` (no `PYTHONPATH`)
now prints the per-paper token table + MAP cost projection (cheap-profile
middle = $0.71, real-profile middle = $1.60).

---
