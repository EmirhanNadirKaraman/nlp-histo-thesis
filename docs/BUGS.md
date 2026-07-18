# Bug catalogue — nlp-histo

Per-bug write-ups with status, evidence, diagnosis, fix, and verification.
Carry-forward work items live in [`THESIS.md`](THESIS.md#todos); permanent
design calls live in [`THESIS.md`](THESIS.md#decisions-log).

> **Two notes on links and paths in this file.** Some entries link into
> `docs/readmes/` (archived working notes), `out/` (generated artifacts), or
> `.claude/` — all **local-only and gitignored**, so those links resolve on the
> maintainer's machine but not in a fresh clone. That is expected; the write-up
> itself carries the evidence. Separately, historical entries quote the file
> layout **as it was when the bug was found** — many predate the `src/nlp_histo/`
> packaging migration. Those paths are evidence, not drift: do not "correct" them.

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
| B-005 | Mitigated (2026-05-14) | High | Summarisation, batch runner | `BatchKnowledgeExtractionRunner.finalize()` was missing six features the sync runner had: (1) `_replace_verbatim_from_db` — grounding NLI ran against LLM paraphrases instead of source text; (2) stable `compute_finding_id`; (3) DB persistence to `sum_*` tables; (4) `corpus_relate_incremental`; (5) `rejection_summary` build + persist; (6) NER + UMLS linking. Since `scripts/run_paper.py` defaults to batch mode, every batched production result between commit `5c59c3e` (2026-04-27) and the 05-14 backport was grounded against paraphrased text. | [Bug 5](#bug-5--batch-runner-missing-sync-parity-features) |
| B-006 | Fixed (2026-05-14) | Medium | Summarisation, RELATE / RESOLVE | `RelationTypeLabel.SCOPE_QUALIFY` plumbing (the enum, the RESOLVE filter, the RELATE info-log column) was wired end-to-end but no `_classify_pair` branch ever emitted it. Stripped: enum value removed, RESOLVE `scope_qualifies` list-comp dropped, RELATE log no longer prints the column. `FinalRule.scope_qualify_count` and the DB column retained as hard-zero fields so existing readers (HTML inspector, downstream consumers) don't break. | [Bug 6](#bug-6--scope_qualify-plumbing-is-dead) |
| B-007 | Fixed (2026-05-14) | Medium | Summarisation, sync runner result cache | `KnowledgeExtractionRunner._load_result` returned cached `{pmcid}.json` unconditionally and `_save_result` never stamped a hash. Fixed in commit `b03d4f6`: a `_pipeline_config_hash()` helper composes cascade signature + thresholds + model identifiers + schema/prompt versions + `enable_router` state; `_load_result` recomputes current hash and returns `None` on mismatch (with a `cached result stale` log line); `_save_result` stamps the hash via `setdefault`. Manifest builder reuses the same helper to avoid drift. | [Bug 7](#bug-7--sync-runner-cached-result-load-ignores-pipeline_config_hash) |
| B-008 | Fixed (2026-05-14) | Low | Summarisation, sync runner batch reporting | `KnowledgeExtractionRunner.process_batch` reported `n_skip = len(results) - n_ok - n_err` but `_load_result` returned cached dicts with `status="success"`, so cached papers counted in `n_ok` and `n_skip` was structurally 0. Fixed by tagging the in-memory cached dict with `status="skipped"` inside `_load_result` (on-disk JSON unchanged) and counting that key explicitly in `process_batch`. Three downstream call-sites (`scripts/summarize_paper.py`, `scripts/run_single_doc.py`, `scripts/run_paper_single_model.py`) updated to treat `success` and `skipped` interchangeably so cached papers still feed the corpus-relate gate. | [Bug 8](#bug-8--process_batch-skip-counter-is-structurally-zero) |
| B-009 | Fixed (2026-05-14) | Low | Summarisation, sync runner instance state | `KnowledgeExtractionRunner` kept per-paper state in eight instance dicts (`_scored_map_findings`, `_normal_findings`, `_finding_groups`, `_canonical_rules`, `_relations`, `_relate_raw_pairs`, `_relate_skipped_pairs`, `_final_rules`). Inside `process_batch` they accumulated across papers and were never cleared. Memory grew O(papers × avg eligible pairs). Fixed by popping the per-paper entries from all eight dicts in `process()`'s `finally` block — runs after the result dict has been materialised but before the function returns, so external callers see the same payload they always did. Verified no external reader of these dicts exists (only `last_map_*` properties on `self` and the cache helpers are exposed). | [Bug 9](#bug-9--sync-runner-instance-dicts-leak-across-papers) |
| B-010 | Fixed (2026-05-14) | Medium | PDF extraction, artifact filter | `components/artifact_filter.py:59` rebuilt `List[LayoutElement]` after filtering via `[el for i, el in enumerate(elements) if element_dicts[i] in filtered_dicts]` — list-`__contains__` over dicts. O(N²); and the moment `filter_artifacts` ever mutated a kept dict (e.g. a future ligature normalisation), the post-filter dict no longer `==`'d the pre-filter dict and the corresponding `LayoutElement` was silently dropped. Replaced with an `id()`-keyed `dict[int, LayoutElement]` lookup built before the filter call — O(N), survives in-place mutation, and doesn't change `filter_artifacts`'s public contract (other callers in `scripts/` unaffected). | [Bug 10](#bug-10--artifact_filter-rebuild-uses-dict-equality-instead-of-identity) |
| B-011 | Fixed (2026-05-14) | Low | PDF extraction, `ModelRegistry` | `resources.py` `ModelRegistry.docling_converter` ignored `DoclingConfig.images_scale`, `accelerator_device`, `ocr_engine`, `force_full_page_ocr`; hard-coded `images_scale=2.0` and never built `AcceleratorOptions`. Was unused by `PipelineRunner` (each component constructs its own converter) but exported as public API — a caller who flipped a non-default `DoclingConfig` and used `ModelRegistry` silently got CPU + scale 2.0. Fixed by deleting the entire class — zero in-tree consumers existed; each component already lazy-loads its own model (Docling via `DoclingLayoutExtractor._get_converter`, TATR via `TATRTableDetector`'s process-wide singleton, scispaCy via `summarization/umls_resources.get_nlp()`). `resources.py` removed; `__init__.py` re-export and four docs files updated. | [Bug 11](#bug-11--modelregistrydocling_converter-ignores-doclingconfig) |
| B-012 | Observed | Low | PDF extraction, two-pass extractor | `components/two_pass_extractor.py:382-398` header/footer strip construction mixes Docling y-coords (`docling_y1=page_h`) and fitz coords (`fitz_header_bottom`) on adjacent lines. Today only a `docling_y1 > docling_y2` comparison guards against a sign-flip if those names ever get muddled. Clarity issue today, latent bug surface for the next refactor. | [Bug 12](#bug-12--two_pass_extractor-header-strip-mixes-coordinate-systems) |
| B-013 | Fixed (2026-05-14) | Low | Inspector batch index, sort handler | `scripts/templates/pipeline_batch_index.html.jinja2:276` read `dataset.nilBa` instead of `dataset.nliBa`. `parseFloat(undefined) → NaN → 0`, so clicking the "NLI B→A" column compared zeros and produced no reorder. Fixed by correcting the typo. | [Bug 13](#bug-13--inspector-nli-ba-sort-typo) |
| B-014 | Fixed (2026-05-14) | Low (latent) | Inspector batch index, badge style | `pipeline_batch_index.html.jinja2:194` renders SCOPE_QUALIFY relations with class `badge-blue`, but the stylesheet only defined `badge-green/red/orange/gray/cyan`. Badge rendered unstyled. Currently dormant because B-006 means SCOPE_QUALIFY is never emitted; would surface the moment B-006 is fixed. Added `.badge-blue` rule. | [Bug 14](#bug-14--inspector-badge-blue-class-missing) |
| B-015 | Fixed (2026-05-14) | Medium | Summarisation, MAP enum coercion | Raw LLM-emitted `relation_type` / `direction` / `category` values were coerced (or alias-repaired) to enum members and the originals were dropped from the row — only landed in `logs/enum_observations.jsonl` with no FK back to the finding. Downstream stages saw only `unclear` / coerced values. Fixed by capturing raw values in a `model_validator(mode="wrap")` on `Finding`, persisting them to new `sum_map_findings.raw_{relation_type,direction,category}` columns (Alembic `0011`). | [Bug 15](#bug-15--raw-llm-enum-values-lost-on-coercion) |
| B-016 | Fixed (2026-05-14) | Low | Summarisation, MAP prompt + schema | `category` enum was `"demographics"` (plural) while `relation_type` enum was `"demographic"` (singular) — same concept, two spellings, requiring an alias map and prompt warning. `Rule.confidence` Literal was `"High"|"Medium"|"Low"` while MAP `Finding.confidence` was lowercase. Aligned both to `"demographic"` (singular, consistent with sibling category labels) and lowercase confidence; inverted `_CATEGORY_ALIASES` to repair legacy `"demographics"`; bumped `MAP_PROMPT_VERSION` to `map_prompt_v2_singular_demographic`. | [Bug 16](#bug-16--demographic-spelling-and-confidence-casing-divergence) |
| B-017 | Fixed (2026-05-15) | High | Summarisation, batch entry-points in `scripts/run_paper.py` | Both batch entry-points (`_run_batch_multi` line 766, `_run_batch_single` line 863) called `build_batch_runner(...)` without passing `db=`, so `BatchKnowledgeExtractionRunner.__init__` got `db=None`. Every `_persist_*` method and `_corpus_relate_incremental` short-circuits on `if self._db is None: return` — silently. Net effect: production batch runs since the B-005 backport (2026-05-14) wrote per-paper `out/summaries/summaries/*.json` artifacts but no `sum_*` rows and no `sum_corpus_relations` rows. The sync path at `build_runner` already opened a DB connection; the batch entry-points were left behind. Fixed by extracting a module-level `_open_db_connection(caller_label)` helper and passing its return value to both `build_batch_runner` call-sites (and using it from the sync path too, removing the duplicated try/except). | [Bug 17](#bug-17--batch-entry-points-pass-no-db-to-buildbatchrunner) |
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
| B-036 | Fixed (2026-05-15) | Low | Summarisation, NLI helpers config surface | `GroundingFilter.__init__` (`helpers/grounding_filter.py:68`) accepts `model_name`, `batch_size`, `device`; `RelateStage.__init__` (`current_stages/relate_stage.py:268`) accepts the same trio. `GroundingConfig` exposes only `threshold`; `RelateConfig` exposes only the two thresholds. `runner.py:258` instantiates `GroundingFilter(cfg.grounding.threshold)` and `runner.py:251` instantiates `RelateStage(entailment_threshold=…, contradiction_threshold=…)` — model / batch / device always fall back to module defaults regardless of caller intent. No way to switch the NLI model or move it to GPU via `KnowledgeExtractionConfig`. | [Bug 36](#bug-36--groundingfilter--relatestage-modelbatchdevice-not-exposed-via-summarizationconfig) |
| B-037 | Fixed (2026-05-15) | Low | Summarisation, normalize stage | Added `NormalizeConfig.extra_synonyms: dict[str, str] \| None` to `KnowledgeExtractionConfig`; both sync and batch runners now pass `cfg.normalize.extra_synonyms` to `NormalizeStage(...)`. Side fix in the YAML loader: `_unwrap_optional` now also handles PEP-604 `X \| None` (was only matching `typing.Union`), and `_coerce` skips the nested-dataclass branch when the field type is a `dict[...]` mapping — without this, `extra_synonyms: {acme: ACME}` crashed with `Field type dict[str, str] \| None does not resolve to a dataclass`. New tests in `tests/test_config_loader.py`: `test_normalize_extra_synonyms_loaded_as_mapping`, `test_tatr_render_dpi_overridable`. | [Bug 37](#bug-37--normalizestageextra_synonyms-not-exposed-via-summarizationconfig) |
| B-038 | Fixed (2026-05-15) | Medium | Summarisation, sentence loader | `KnowledgeExtractionRunner.load_paper_from_db` (`runner.py:905`) called `spacy.load("en_core_sci_sm")` on every invocation — bypassed the `umls_resources.get_nlp()` singleton. In batch mode (`process_batch([load_paper_from_db(p) for p in pmcids])`) the small model deserialised once per paper. Same class as B-029. Fixed by routing through `umls_resources.get_small_nlp("en_core_sci_sm")`; raises a clear error when the model isn't installed. Regression test `tests/summarization/test_scispacy_singleton.py` asserts no `spacy.load(...)` call sites exist outside `umls_resources.py` under `pipeline/stages/`. | [Bug 38](#bug-38--summarizationrunnerload_paper_from_db-bypasses-scispacy-singleton) |
| B-039 | Fixed (2026-05-15) | High | Summarisation, sentence ordering | `KnowledgeExtractionRunner.load_paper_from_db` (`runner.py:912-916`) orders `TextElement` rows by `position_in_section` alone — but per `database/models.py:79` + the composite index `idx_document_path_position`, `position_in_section` is *local to each `path_string`*. Single-column sort interleaves sections: every section's position-0 paragraph emits first, then every section's position-1, etc. `MapStage._make_chunks` then packs adjacent sentences from unrelated sections into the same chunk, destroying topical locality and depressing voter agreement. Affects every paper on every sync + batch run today. Compounds with B-040 once those rows have already been written out-of-order to `TextElement`. | [Bug 39](#bug-39--load_paper_from_db-orders-by-position_in_section-only-interleaves-sections) |
| B-040 | Fixed (2026-05-15) | Medium | PDF extraction, text assembly | `parsers/layout_utils.extract_text` (`layout_utils.py:469-524`) accumulates paragraphs into `by_path = defaultdict(list)` keyed by `path_string`, then emits `rows` by iterating `by_path` in *insertion order*. Sections that get revisited after a sub-section (parent text → sub-section text → more parent text) have their later paragraphs appended at the parent's first-emit position; the sub-section's content ends up emitted *after* the entire parent block. Output `HierarchicalRow` order is "path-first-appearance" order, not document order, and the bug compounds with B-039 once those rows are written to `TextElement` and re-read by `load_paper_from_db`. | [Bug 40](#bug-40--extract_text-emits-paragraphs-in-path-first-appearance-order-not-document-order) |
| B-041 | Fixed (2026-05-15) | High | Summarisation, MAP cascade attribution | `MapStage._run_voters` (`current_stages/map_stage.py:1183`) returns the API-survivor list with `[r for r in results if r is not None]` — the original-voter-index → survivor-index mapping is dropped at return. `agreement.compute(voters)` then assigns `bundle.best_index` over the survivor list, and `producer_from_outcome` (`agreement/decision.py:210-211`) / `make_decision_record` (`decision.py:255-267`) use that index as if it referred to the original `voter_specs`. Router path is also affected: `_classify_voters` (`routing/router.py:271`) re-indexes from 0 over the survivor list, so its `valid_voter_indices` are survivor-list indices that `voter_specs[global_idx]` then treats as original indices. Whenever ≥1 voter fails an API call (or the router strips a voter as UNUSABLE before MAP sees the failure), MAP cache metadata, cost report, and cascade decision log all carry the wrong `(provider, model)` for the kept chunk. Dormant when zero voters fail. | [Bug 41](#bug-41--producer-attribution-mis-indexed-when-any-voter-fails) |
| B-042 | Fixed (2026-05-15) | Low | PDF extraction, text stitching | `ContextAwareStitcher._is_cut_off` (`parsers/text_processing.py:152-181`) returned `False` on any terminal `.`/`?`/`!`/`)`/`]`/`"`/`'`/`»` early-return at lines 152-154, *before* the `_MID_SENTENCE_ABBREVS` check at lines 175-181. Every abbreviation in that frozenset (`fig.`, `et al.`, `vs.`, `approx.`, `e.g.`, `i.e.`, `cf.`, `ref.`, `refs.`, `dept.`, `no.`, `nos.`) ends in a period, so the abbrev rule was dead code. Paragraphs ending in those abbreviations were treated as sentence-final and never stitched with the next narrative paragraph — sentences got fragmented at abbreviation boundaries, biasing both MAP input and downstream NLI grounding. Fix moved the abbreviation check ahead of the period early-return and added a multi-token form so "et al." (two tokens) also triggers. | [Bug 42](#bug-42--is_cut_off-mid-sentence-abbreviation-rule-is-dead-code) |
| B-043 | Fixed (2026-05-15) | Low | PDF extraction, citation removal | `parsers/text_processing.remove_citations` regex `(?<!\n)\.\s+\d+(?:[,–\-]\d+)*(?=\s|$)` stripped any `". <digits> "` pattern after a period — including 4-digit years. "Smith et al. 2020 reported …" became "Smith et al. reported …", losing claim context. Fixed by capping the citation-index run at 1–3 digits in all three after-period / after-comma / standalone branches: `\d+` → `\d{1,3}`. Citation indices in pathology papers are practically never ≥1000 (one bracket-style branch was left as `\d+` because brackets disambiguate years from indices). Regression tests in `tests/parsers/test_remove_citations.py` cover year preservation + citation stripping. | [Bug 43](#bug-43--remove_citations-strips-publication-years) |
| B-044 | Mitigated (2026-05-15) | Medium | Summarisation, MAP relation_type | MAP voters bleed `category` values (`morphology`, `IHC`, `molecular_genetics`, `prognosis`, `treatment`, `staging`) into the `relation_type` field. Prior coercion mapped only `prognosis → prognostic` and `treatment → treatment_response`; the rest fell through to `unclear` and got dropped at GROUP (relation_type is part of the grouping key and `unclear` is non-groupable). Net effect: 10+ findings silently lost per run on the calibration set, concentrated in `molecular_genetics` and `IHC` claims. Mitigation: prompt anti-pattern line + prognostic-crossover example + extended `_RELATION_TYPE_ALIASES` (`morphology→has_feature`, `ihc→expression`, `molecular_genetics→expression`) + new `reason="cross_field_bleed"` JSONL counter for measurement. `staging` left unaliased (descriptive vs prognostic crossover needs claim context). | [Bug 44](#bug-44--map-relation_type-bleeds-category-names-and-loses-findings-at-group) |
| B-045 | Fixed (2026-05-15) | Low | Summarisation, MAP `FindingScope.scope_parsed` | `scope_parsed` is trivially derivable (`any(sub_field is not None)`) but was being computed by the LLM. One more thing it could get wrong, and output tokens spent reasoning about it. Fixed by `@model_validator(mode="after")` on `FindingScope` (`models.py:148-159`) that overrides whatever the LLM emitted; prompt instruction updated to "always emit false — computed automatically" (`prompts.py:213`). Field stays in the schema because OpenAI strict mode requires every property to be present. Bumped `MAP_SCHEMA_VERSION` → `"map_v7_scope_parsed_autocompute"`. Regression test in `tests/summarization/test_scope_parsed_autocompute.py`. From [MAP_PROMPT_AUDIT Issue 5](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-5--scopescope_parsed-is-llm-set-but-trivially-derivable-low). | [Bug 45](#bug-45--scope_parsed-is-llm-set-but-trivially-derivable) |

| B-046 | Fixed (2026-05-15) | Low | Summarisation, MAP direction enum | Hedging words (`maybe`, `possibly`, `none`, `n/a`) outside `DirectionEnum` fell through to the unknown-value branch and coerced to `unclear`, losing the polarity-vs-uncertainty distinction in the live enum (raw still on `_raw_direction` per B-015). Added `_DIRECTION_ALIASES` mapping hedging → `unclear` and `none`/`n/a` → `no_direction`; alias-repair branch in `_coerce_invalid_direction` logs `reason="alias_repair"`. Bumped `MAP_SCHEMA_VERSION` → `"map_v8_direction_alias_repair"`. Tests in `tests/summarization/test_enum_alias_repair.py`. From [MAP_PROMPT_AUDIT Issue 8](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-8--directionmaybe-single-occurrence-low). | [Bug 46](#bug-46--direction-hedging-words-coerce-to-unclear-instead-of-alias-repair) |
| B-047 | Fixed (2026-05-15) | Low | Summarisation, MAP direction prompt | Prompt example mapped `"BCL2 was negative"` to `direction=absent` but both `negative` and `absent` plausibly applied for expression-context negation. Same FindingGroup could end up with both labels on opposite-polarity findings, blocking RELATE's CONTRADICT signal. Added a disambiguating rule under the `direction` definition (`expression`-only: `negative staining`/`no expression` → `absent`; `decreased`/`reduced` → `negative`; other relation_types: prefer `negative`, reserve `absent` for literal "absent"/"not present"/"lacking"). Bumped `MAP_PROMPT_VERSION` → `"map_prompt_v5_expression_absent_vs_negative"`. From [MAP_PROMPT_AUDIT Issue 6](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-6--directionabsent-vs-directionnegative-ambiguity-in-expression-contexts-low). | [Bug 47](#bug-47--direction-absent-vs-negative-ambiguity-on-expression-claims) |
| B-048 | Fixed (2026-05-15) | Low | Summarisation, optional RULE block enums | `Rule.type` was `Literal["Diagnostic", "Prognostic", "Management"]` (Title-Case) and `RuleCounts` mirrored the casing in field names — inconsistent with the lowercase `Finding.confidence` / `Finding.category` convention. Lowered all three; added a `mode="before"` validator on `Rule.type` so legacy Title-Case payloads round-trip. Updated MAP RULE OutputFormat prompt + `_recompute_audit` helper. RULE block is off by default, no DB rows to backfill. Tests in `tests/summarization/test_enum_alias_repair.py`. From [MAP_PROMPT_AUDIT Issue 7](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-7--ruletype-is-title-case-diagnosticprognosticmanagement-everything-else-lowercase-low). | [Bug 48](#bug-48--ruletype-title-case-inconsistent-with-lowercase-convention) |
| B-049 | Fixed (2026-05-15) | Medium | Summarisation, CANONICALIZE direction policy | `_split_by_direction` folded `unclear` and `no_direction` members into the largest polarity bin. Two holes: (a) **reproducibility** — `max(non_unclear, key=len)` returns the first dict key on ties, traceable to upstream member-arrival order, so the same paper produced different `member_normal_ids` / `finding_count` / `mean_grounding_score` across re-runs (supersedes B-026); (b) **honesty** — hedged findings got re-cast as votes for the majority direction, inflating downstream confidence and feeding RELATE pairs as if the model had really claimed that polarity. Fixed: every observed direction gets its own `CanonicalRule` bin (no folding); RELATE and corpus_relate skip pairs where either side is `unclear` / `no_direction`; `is_conflicted` repurposed to a **group-level** signal (True iff the group emits ≥2 polarity-bearing bins, stamped on every rule from the group). Added `direction_value`, `POLARITY_BEARING_DIRS`, `NON_POLARITY_DIRS` to `models.py` as the single source of truth; gates use the normalizer so `DirectionEnum` / raw string / `None` paths all behave the same. `partial` deliberately kept polarity-bearing for now — the semantic question of whether partial really conflicts with positive is owned by B-025. Bumped `CANONICALIZE_DIRECTION_POLICY_VERSION` (fed into `pipeline_config_hash`) to force cache invalidation. Tests: rewritten `tests/summarization/test_canonicalize_direction_split.py` (16 cases incl. S5 core invariant against unclear leakage into polarity bins), new `tests/summarization/test_corpus_relate_non_polarity.py`, extended `tests/summarization/test_relate_skipped_pairs.py`, `tests/summarization/test_pipeline_config_hash.py`. | [Bug 49](#bug-49--canonicalize-folds-unclear--no_direction-into-majority-polarity-bin) |
| B-050 | Fixed (2026-05-15) | Low | Scripts, batch poll interval | `scripts/run_paper.py` carried three diverging defaults for `--poll-interval`: argparse `60` (line 331), `_run_all_batch(poll_interval=20)` (line 787), `_run_batch(poll_interval=60)` (line 885). CLI flows passed `args.poll_interval` so the call-site defaults rarely fired — but a direct programmatic caller of either batch helper got 20s or 60s depending on which one they imported. Consolidated onto module-level `DEFAULT_POLL_INTERVAL_SEC = 60` referenced by argparse + both function signatures. Regression: `tests/test_poll_interval_defaults.py` introspects via `inspect.signature` (not `__defaults__` tuple indexing) and asserts all three resolve to 60. **Note**: if another agent's parallel work also claims B-050, renumber to B-051 at commit time. | [Bug 50](#bug-50--poll_interval-default-mismatch-across-cli-and-batch-helpers) |
| B-057 | Fixed (2026-05-17) | High | PDF extraction, committed merge-conflict markers | Three files on `eval-speedrun` HEAD shipped with unresolved git merge-conflict markers (`<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes`): `pipeline/stages/pdf_text_extraction/components/visualizer.py` (lines 113–121), `pipeline/stages/pdf_text_extraction/table_detectors/tatr_detector.py` (lines 68–78), and `eval/run.py` (lines 120–125). `components/__init__.py:7` and `table_detectors/__init__.py:3` re-export these modules eagerly, so any pipeline import (e.g. `DoclingLayoutExtractor`) crashed with `SyntaxError`. Discovered while running the Stage-1 observability-patch smoke test (2026-05-17). Resolved by picking the "Updated upstream" branch in each file: (1) visualizer — `setdefault(pg, []).append(...)`, semantically identical to the alternative; (2) tatr — preserves configurable `self._config.device` per B-034 (the alternative hardcoded `to("cpu")` plus `low_cpu_mem_usage=False, device_map=None` kwargs that were a transformers-loading workaround no longer needed); (3) eval/run.py — log format `"Eligible PDFs: %d / %d (%.1f–%.1f MB)"` matching the actual min+max byte filter applied earlier in the same function. All three resolutions are no-op behaviour changes relative to the documented intent of the surrounding code. | [Bug 57](#bug-57--committed-merge-conflict-markers-in-visualizerpy) |
| B-056 | Observed (2026-05-16) | Medium | Summarisation, batch runner → `sum_map_voter_outputs` | `BatchKnowledgeExtractionRunner.finalize()` has no code path that buffers per-voter `AuditableSummary` rows or writes them to `sum_map_voter_outputs`; no `_persist_voter_outputs` method exists on the batch class. Discovered during the Phase 0 audit per [`CALIBRATION_EXECUTION_PLAN.md`](readmes/other_readmes/CALIBRATION_EXECUTION_PLAN.md) §10. Blocks θ / reject_θ sweeps over batch-processed papers; sync runs are unaffected. B-055's empirical claim that `sum_map_voter_outputs` was populated on batch runs 38–48 contradicts code-level inspection and requires runtime re-verification on a paper never previously processed via sync. | [Bug 56](#bug-56--batch-runner-omits-per-voter-map-persistence-code-path-absent) |
| B-055 | Mitigated (2026-05-23) | High | Summarisation, batch runner → `sum_map_findings` | `sum_map_findings` rows missing for 9 of the last 10 batch-mode pipeline runs (ids 38–48 across 5 papers), despite each paper's `rejection_summary.map_findings_total` recording 100–230 MAP findings produced. Other `sum_*` tables (`sum_normal_findings`, `sum_finding_groups`, `sum_canonical_rules`, `sum_final_rules`, `sum_rejection_summaries`, `sum_map_voter_outputs`) get rows on the same runs, so the DB connection + `pipeline_run_db_id` + persistence wiring are not at fault. The function itself works: a direct call to `persist_map_findings(db, 48, pmcid, chunk_summaries)` against the same paper's batch handle on disk wrote 154 rows successfully, with `verbatim_support` exactly matching `text_elements.text_content`. Bug pre-dates the 2026-05-16 B-005 dedup (same behaviour on HEAD `7ea254a`); the dedup rewrote the wrapper but the production failure mode was already present. Suspect call-path issue inside `BatchKnowledgeExtractionRunner.finalize()` between L483 (`chunk_summaries = [AuditableSummary.model_validate(v) for v in handle.finalized.values()]`) and L522 (`self._persist_map_findings(...)`) — either `chunk_summaries` is empty at the call site (which would also break downstream `all_findings = [f for cs in chunk_summaries for f in cs.findings]` at L536, contradicted by 214 `sum_normal_findings` rows on run 48), or an exception in the bulk `INSERT` is being swallowed by `except Exception as exc: logger.warning(...)` and the run's stdout went unrecorded. Adjacent inconsistency: runs 47/46/44/43 wrote `sum_canonical_rules` (100+ rows) with **zero** `sum_normal_findings` — physically impossible from the in-process flow, suggests these were cache short-circuits whose `_load_result` path skipped MAP/NORMALIZE persistence but still wrote canonical/final from the cached JSON. | [Bug 55](#bug-55--sum_map_findings-not-populated-by-batch-runner) |
| B-054 | Fixed (2026-05-16) | High | Summarisation, NER stage scispaCy singleton bypass | `named_entity_recognition/ner.py`'s `load_ner_model()` and `load_linker_model()` issued direct `spacy.load("en_core_sci_lg", …)` calls — completely bypassing the `umls_resources.get_nlp()` singleton documented in CLAUDE.md and MEMORY.md. `KnowledgeExtractionRunner._run_stages` calls `run_ner_on_db(pmcid, save_to_db=True, force=False)` per paper *without* passing `nlp=` / `linker_nlp=`, so the loaders fired with `None` defaults and freshly loaded ~2.6 GB of scispaCy + UMLS twice per paper. Concretely visible in production runs: `[pmcid] NER done [136.4s]` on a paper that bailed out (already had entities) — the time was pure model-load waste. Compounded by `umls_resources.get_nlp()` already holding its own copy for NORMALIZE / UMLS_ENRICH, so peak RSS hit ~3 copies of en_core_sci_lg in memory. Same class of bug as B-029 (PDF runner) / B-038 (summariser load_paper_from_db), missed because `named_entity_recognition/` lives outside `pipeline/stages/` and the existing singleton-guard test only scanned the stages tree. Fixed by routing both loaders through `umls_resources.get_nlp()`; the "fast NER" pass wraps the span-extraction loop in `nlp.select_pipes(disable=["scispacy_linker"])` so it stays cheap on the linker-attached singleton; the "Document already has entities" skip check moved *above* the model-load block so a skipped paper now costs ~0 s (was ~150 s). Regression test in `tests/summarization/test_scispacy_singleton.py::test_ner_module_routes_through_singleton` asserts neither loader contains a direct `spacy.load(` call. `batch_ner.py` inherits the fix automatically (it imports the same functions). | [Bug 54](#bug-54--ner-stage-scispacy-singleton-bypass) |
| B-053 | Fixed (2026-05-16) | Low | Tooling, percentiles cost estimator | `scripts/estimate_pipeline_cost_percentiles.py` hygiene cluster — dead `import json` + `import statistics`; two never-called helpers (`estimate_non_llm_stages`, `render_paper_table`) that duplicated the markdown rendering inline in `main()`; misleading inline comment claiming `est_chunks = ceil((n - overlap) / stride)` while the code (correctly, matching `MapStage._make_chunks`) did `ceil(n / stride)`; `pick_percentile` had no guard for an empty paper list (`idx = ceil(0.5*0) - 1 = -1` → silent off-end indexing) or out-of-range `p`; `CHUNK_SIZE`/`CHUNK_OVERLAP` duplicated as module constants instead of sourced from `MapConfig.chunk_size`/`chunk_overlap`, leaving a silent drift hazard if production config changes. Fixed: dead imports + helpers removed, comment rewritten to cite `_make_chunks`, `pick_percentile` raises `ValueError` on empty corpus / `p ∉ (0, 1]`, chunk constants now read off `MapConfig()` at module load. Numbers unchanged — none of this moved the printed cost (the percentiles report is an intentional upper-bound budget for the top-decile/P80–P90 papers by `n_te`). Regression test `tests/test_estimate_pipeline_cost_percentiles.py` (15 cases). | [Bug 53](#bug-53--percentiles-cost-estimator-hygiene-cluster) |
| B-052 | Fixed (2026-05-16) | Medium | Tooling, cost estimation script | `scripts/estimate_selection_cost.py:per_chunk_input_tokens` modelled the average sentences per MAP chunk as `min(chunk_size, n_sentences / n_chunks * (1 + 0))`. At production defaults (`chunk_size=10`, `chunk_overlap=2`, stride=8) `n_sentences / n_chunks ≈ stride = 8`, so the clamp returned ~8 sentences per chunk — but `MapStage._make_chunks` (`map_stage.py:1263-1267`) slices `sentences[i:i+chunk_size]` so each non-tail chunk actually sees 10 sentences. The trailing `* (1 + 0)` was a leftover from a removed overlap term. Net: every cost number in the projection table was ~15–20% low, exactly the headline figure a thesis budget review reads. Fixed by replacing the formula with a sum over `min(chunk_size, n_sentences - start) for start in range(0, n_sentences, stride)` divided by `n_chunks` — matches `_make_chunks` line-for-line, accounts for the truncated tail, and rounds with `ceil` for conservative budget estimates. Function signature gained `chunk_overlap` (caller updated); also validates `0 <= chunk_overlap < chunk_size`. Co-fixed in the same change: `.order_by(TextElement.position_in_section)` (the B-039 bug) flipped to `.order_by(TextElement.id)` to actually mirror `KnowledgeExtractionRunner.load_paper_from_db`. Dead `import math` cleaned up. Regression test in `tests/test_estimate_selection_cost.py`. | [Bug 52](#bug-52--cost-estimation-script-underestimates-per-chunk-input-tokens) |
| B-051 | Fixed (2026-05-15) | High | Summarisation, MAP agreement gate | `EmbeddingScorer._polarity` applied only a 20% multiplicative penalty; opposite-polarity paraphrases with cos≈1.0 produced score=0.80, passing `theta=0.7` and accepting the chunk as KEEP despite a direct voter contradiction. Fixed: new pure helper `agreement/polarity_conflict.detect_polarity_conflict` invoked from `AgreementChecker.compute` after the scorer runs but before theta — when two **comparable** findings (same `subject_entity` / `outcome_entity` / `relation_type` / `category`, all four required, strings `.strip().casefold()`d) carry opposite `{positive, negative}` directions, decision is forced to `ChunkDecision.ESCALATE` with `score_details["hard_fail_reason"] = "polarity_conflict"`. `MapOutputRouter._agreement_gate` emits ONLY `ReasonCode.POLARITY_CONFLICT` (never co-emits low-agreement codes — the score was high; only the structural check failed); explanation makes the override explicit. v1 conservative: scope fields excluded from comparability (cross-cohort false-escalate cheaper than missed contradiction); `absent`/`partial`/`unclear`/`no_direction` excluded from the hard-polarity set pending B-025 calibration. Cache invalidation: bumped `MAP_SCHEMA_VERSION` → `"map_v9_polarity_hard_fail"` (invalidates `PipelineCache`); added `MAP_AGREEMENT_POLICY_VERSION = "polarity_hard_fail_v1"` routed into `compute_pipeline_config_hash` on both runners (invalidates per-paper result cache). 11 deterministic regression tests in `tests/summarization/agreement/test_b051_hard_fail_polarity.py` + 3 hash regression tests in `tests/summarization/test_pipeline_config_hash.py`. | [Bug 51](#bug-51--map-agreement-gate-treats-opposite-polarity-as-soft-disagreement) |

| B-058 | Fixed (2026-05-20) | Medium | PDF extraction, media cropper | `MaskingConfig.drop_tables_inside_figures` runs at Step 2 (post-detection filter in `runner.py::_drop_tables_inside_figures`) but the dropped table re-enters at Step 7 via the cropper's supplementary source (`media_cropper.py:245` — iterates layout TABLE/RECONSTRUCTED_TABLE elements and re-adds any that don't overlap an existing detection). When the Step-2 drop removes a `table_in_figure` FP, that table is no longer in `detection.regions` → no overlap match in cropper → re-added. Final `source` field is `"docling"` (one source) instead of `"docling+docling"` (two), confirming the bypass. Discovered while inspecting variant 18 (`drop=ON`) on PMC11791726/p9 — the FP was still emitted as Table_4_p9 despite `table_regions_dropped_inside_figures=1` in the run metadata. Fixed: `media_cropper.crop()` takes a new `drop_tables_inside_figures: bool = False` parameter; when True it skips layout TABLE elements ≥0.8 inside any FIGURE/PICTURE on the same page (same threshold as Step 2). `runner.py` plumbs `self._cfg.masking.drop_tables_inside_figures` into all three `cropper.crop()` call sites (main + two multi-source crops). No new config field — single `MaskingConfig.drop_tables_inside_figures` flag now governs both Step-2 detection filter and Step-7 supplementary-source filter. Pre-fix variant 18 had drop=ON behaving identically to drop=OFF on docling, invalidating its Stage 3 verdict (treated as "no effect" — actually the bug). | [Bug 58](#bug-58--drop_tables_inside_figures-bypassed-by-cropper-supplementary-source) |

| B-059 | Observed (2026-05-21) | Medium | PDF extraction, figure cropping | Decorative icons emitted as figure crops.  ~70% of figure-side error labels across all variants are `icon` (304 of 437 figure errors aggregated across 16 variants on the 28-PDF corpus).  These are small image-like layout elements that Docling correctly identifies as PICTURE/FIGURE elements geometrically but that are decorative graphics (publisher logos, small inline ornaments, watermark-style icons), not scientific figures.  Crops are emitted (FP for figure-output) but masking is correct (image content shouldn't appear in body text either way).  No current pipeline stage filters them.  Possible fix path: heuristic filter based on size (`min_figure_pts` already exists in `CroppingConfig` but doesn't address this — icons can be moderately sized), bbox aspect ratio, low text density inside the bbox, or appearance on every page of a multi-page paper (publisher logo case).  Need a figure-side analogue of `_drop_tables_inside_figures` or a stand-alone `drop_icon_figures` filter.  Decision/scope: outside the 2026-05-21 thesis-day budget; document as known limitation in `docs/THESIS.md` future-work section. | [Bug 59](#bug-59--decorative-icons-emitted-as-figure-crops) |
| B-060 | Observed (2026-05-21) | Medium | PDF extraction, caption parser | Cluster of `nearest_caption()` + `parse_caption_num()` defects in `parsers/layout_utils.py`.  Six recurring failure modes seen across all variants of the 28-PDF corpus (counts are aggregated label occurrences across variants):  (1) **Rotated-image footnote-as-caption** — 49 table cases (`wrong caption (footnotes matched to captions, rotated image)`); attacher pulls footnote text up because the image rotation messes up vertical proximity ordering.  (2) **Continuation-marker parsed as new table number** — 30 table cases; `(continued)` next to a table caption is parsed as the table's number by `parse_caption_num()` / `TAB_NUM_RE`.  (3) **Caption "Table N" prefix dropped** — 22 table cases; parser returns the descriptive caption body without the "Table N" identifier, breaking strict-match scoring on the caption dim.  (4) **Multi-caption merge across page boundaries** — 30 table cases; when a continued table caption sits adjacent to the next table's caption, the attacher concatenates them.  (5) **Side-mounted figure caption missed** — 19 figure cases (`no caption, caption is to the right of the figure`); spatial proximity heuristic in `nearest_caption()` doesn't handle 2-column layouts with side-mounted captions.  (6) **Page footer treated as caption** — 38 figure cases combined (bottom-left + bottom-right variants); attacher confuses page footers with figure captions when the real caption is on a different position of the page.  These bugs are interconnected via the shared caption-attacher logic — touching one risks regressing another, so they need a focused investigation rather than ad-hoc patches.  Affects ~18% of table errors and ~17% of figure errors; second-largest unaddressed bucket after `should be masked` for tables and `icon` for figures.  Decision/scope: outside the 2026-05-21 thesis-day budget; document as known limitation. | [Bug 60](#bug-60--caption-parser-bug-cluster) |
| B-061 | Observed (2026-05-21) | Low | PDF extraction, table cropping geometry | `crop too small minor` family of labels — 16 aggregated table cases across all variants (8 with caption issue + 6 stand-alone + 2 unmasked-letters variant).  Tables whose emitted crop bbox is smaller than the true table extent, missing some content.  Currently no config knob or filter tests this.  Symmetric to `crop too big` (which the Stage 5 `footnote_multiplier` sweep partially addresses).  Possible fix: dilate detection bboxes by a small margin before cropping, gated by a new `CroppingConfig.table_crop_dilation_pts` field, sweep values in {0, 2, 4, 8}.  Risk: dilation increases overlap with adjacent layout elements (captions, footnotes — though expand_tables_with_footnotes handles the latter). Decision/scope: outside the 2026-05-21 thesis-day budget; document as known limitation. | [Bug 61](#bug-61--crop-too-small-table-geometry-no-config-knob) |
| B-062 | Fixed (2026-05-23) | High | Summarisation, MAP cascade / config | `scripts/run_paper.py` never set `enable_router`, so both production entry points (`build_runner`, `build_batch_runner`) used the runner default `False` → **production ran the legacy `AgreementChecker` cascade, not `MapOutputRouter`**. Yet THESIS.md asserted router-on production in four places + `eval/silver/map_theta_sweep.py` in three comments. Documented production cascade ≠ actual behaviour. Found while wiring the config pin (calibration review item 3). Fix: path made config-governed (`summarization.map.enable_router`, default `false`) + logged at load; user decided production keeps the legacy L1→L2→L3 cascade (the router L1→L3-skip path is opt-in/experimental); stale docs corrected. | [Bug 62](#bug-62--documented-router-on-production-cascade-never-actually-enabled) |
| B-063 | Fixed (2026-05-24) | Low | Tooling, cost estimation script | `scripts/estimate_selection_cost.py` imports `from pipeline.stages.summarization.costing import PriceBook` inside `main()` but never bootstraps the repo root onto `sys.path`. Run as the documented bare `python scripts/estimate_selection_cost.py …` (HOW_TO_RUN §5) it dies with `ModuleNotFoundError: No module named 'pipeline'` — Python puts the script's own dir (`scripts/`) on `sys.path`, not the CWD. Only `PYTHONPATH=. python …` worked, which the script's own docstring documented. Every sibling script (`run_paper.py`, `check_apis.py`) self-bootstraps with `_REPO_ROOT = Path(__file__).resolve().parents[1]`; this one was the lone violator of the CLAUDE.md "scripts must bootstrap their own path" convention. Hit while running the cost estimate for `related15_full`. Distinct from [B-052](#bug-52--cost-estimation-script-underestimates-per-chunk-input-tokens) (same file, token-formula fix). Fixed by adding the standard 3-line bootstrap before the first repo import + dropping `PYTHONPATH=.` from the docstring. | [Bug 63](#bug-63--estimate_selection_costpy-missing-syspath-bootstrap) |
| B-064 | Fixed (2026-05-24) | Medium | Eval, silver embedding cache | The JSON embedding cache (`eval/silver/matcher.py`) rewrote the **entire** file on every `save()`, and the MAP θ sweep calls `save()` after each ~100-text batch (`_prewarm_agreement_cache`, `get_embeddings`, `_make_cached_embed_fn`) → **O(N²)**: the 720 MB / 17,827-entry gemini cache was re-serialised hundreds of times, dominating sweep wall-time (~20 s gaps even for 4-vector fetches). The silver↔pipeline matcher also used a pure-Python 3072-dim cosine triple-loop (`compute_sim_matrix`), the per-cell compute bottleneck at high θ. Fix: interface-preserving `SQLiteEmbeddingCache` (`set`=INSERT-in-txn, `save`=commit, WAL, float32 BLOB → 720→219 MB) behind `make_embedding_cache` + `NLP_HISTO_EMBEDDING_CACHE_BACKEND` (default `sqlite`); idempotent JSON→SQLite import (`scripts/import_embedding_cache_sqlite.py`); `compute_sim_matrix` vectorised with numpy. No embedder / dimensionality / θ / scorer-semantics change. | [Bug 64](#bug-64--json-embedding-cache-rewrite--vectorised-cosine) |
| B-065 | Fixed (2026-05-27) | High | Summarisation, batch runner | `BatchKnowledgeExtractionRunner._process_level` Pass 2 (pre-embedding) iterates `chunk_voters` without filtering `None` slots; `_claims(None)` raises `AttributeError: 'NoneType' object has no attribute 'findings'` and kills the batch advance. Latent for `real` profile (3 L1 voters; rare for all to fail one chunk) but **certain** for the new `haiku_only` profile (N=1 voter → any parse failure → `voters_full = [None]` → crash). Hit while running batch on 15 ILP papers under `haiku_only` (this conversation). Fix: one-line filter `if v is not None` in the embed-collection comprehension; mirrors the survivor-index filter Pass 3 already does on line 1190 for the agreement-scoring path. | [Bug 65](#bug-65--batch-runner-_process_level-crashes-on-none-voter-output) |
| B-066 | Fixed (2026-05-31) | Low | Summarisation, NORMALIZE | NORMALIZE's curated synonym dictionary `synonyms.yaml` is never loaded: `normalize_stage.py` (in `current_stages/`) resolves the path as `Path(__file__).parent / "synonyms.yaml"` = `current_stages/synonyms.yaml`, but the file lives one level up at `pipeline/stages/summarization/synonyms.yaml`. `_load_synonyms()` hits `FileNotFoundError` and silently falls back to the hardcoded `_SYNONYMS_FALLBACK`. Zero functional impact today (YAML and fallback are in sync — 48/48 keys, 0 divergence), but edits to the YAML have no effect, defeating its purpose as a human-editable, clinician-curatable override layer. Surfaced 2026-05-31 while documenting the NORMALIZE rationale in THESIS.md. Fix: point the loader at the real location (`Path(__file__).resolve().parents[1] / "synonyms.yaml"`) or move the YAML into `current_stages/`. | [Bug 66](#bug-66--synonymsyaml-never-loaded-loader-looks-in-the-wrong-directory) |
| B-067 | Mitigated (2026-06-01) | Medium | Acquisition, PDF extraction, corpus accounting | No main-PDF selection at any stage: `pdf_organizer.py` copies every PDF per PMC package and `run_batch` globs all `*.pdf` with `pmcid = pdf_path.stem` (full filename), so supplementary PDFs that parse become separate `Document` rows. DB held 943 docs = 940 distinct papers — 3 double-counted (article + `mmc1`: PMC12272590 / PMC7508550 / PMC9239710) and PMC11863978 represented *only* by `mmc6` (its main was never ingested). Eval subsets unaffected. Mitigated 2026-06-01: deleted the 4 supplementary rows → 939 clean papers; root-cause code guard still TODO. Breaks §4.2 "943 ingested papers". | [Bug 67](#bug-67--supplementary-pdfs-ingested-as-separate-documents-no-main-pdf-selection) |
| B-068 | Observed | Medium | Corpus accounting, eval-set provenance | One of the 28 document-extraction eval PDFs, **PMC11863705_main**, is absent from the ingested `Document` corpus (27/28 present). **Root cause: it ships supplementary material (`PMC11863705_mmc1.pdf`), making it a multi-PDF package, and the 977-doc DB was built from the single-PDF candidate pool only** — all **23** multi-PDF packages are absent from the DB. Not a page-count exclusion (main is 9 pp, well under the 30-pp cap) and not blacklisted. The 116 downloaded-but-not-ingested ids decompose as 23 multi-PDF + 93 single-PDF (89 over the 30-pp cap + 4 other). Also surfaced: the live DB holds **977** distinct-PMC rows, not the 939/943/903 cited in `docs/thesis/05_corpus.md §4.2–4.3` — re-ingested since B-067's mitigation, all thesis counts stale. §4.3's "sampled from the ingested corpus" is violated by PMC11863705. | [Bug 68](#bug-68--eval-paper-pmc11863705-absent-from-ingested-corpus-stale-thesis-counts) |
| B-069 | Open (reopened 2026-06-15) | High | Summarisation eval, MAP/cascade calibration + EXP F | **The dev/test split is not enforced in the `run_sweep` selection path — almost all MAP/cascade calibration ran on the full 273-case corpus, not the dev split.** `run_sweep`'s `split` arg is metadata-only (written to the row, never used to filter), and the experiments orchestrator primes the voter cache with `--split all` and loads every case via `_load_map_context` (no split filter). Discriminator: `n_silver=1243` ⇒ dev-filtered, `n_silver=1596` ⇒ whole corpus. Recorded outputs: scorer comparison (exp_1/exp_4), agreement-weights (exp_2/exp_5), polarity-flag (exp_3/exp_6), the joint scorer×θ×reject_θ **cascade sweep** (map_cascade_sweep, `split=all`), the routing-policy sweep, **and** EXP F (§9.6) all show `n_silver=1596` ⇒ **whole corpus**, despite `split="dev"`/`"test"` labels. Only the cost-quality table **EXP B.2** (§9.4) and its family-bias diagnostic genuinely filtered to dev (`n_silver=1243`, via `case_filter=_case_in_split`) — but the config they evaluate was itself selected on the whole corpus, so even that is in-sample. EXP F proof: `n_matched=1533 > 353` (the test split's silver count). Net: the production config (scorer, θ, reject_θ, weights, polarity, embedder) was selected on all 273 cases; there is no out-of-sample held-out estimate. The split mechanism + canonical counts (221/52, 1243/353) are themselves correct; the defect is that selection never applies them. Related stale-thesis-counts issue: [B-068](#bug-68--eval-paper-pmc11863705-absent-from-ingested-corpus-stale-thesis-counts). **Reopened 2026-06-15** — the 2026-06-12 Won't-fix was reversed; the chosen fix abandons the within-corpus dev/test split in favour of two *physically disjoint* 15-paper clusters: `related15` (ILP calibration) and `heldout15` (random seed=19, 0 overlap), plus argparse guards (commit `2ea9188`) that force explicit `--silver`/`--pipeline`/`--source` so an eval can't silently run on the calibration set. Infra in place (both clusters + `source_cases_{related15,heldout15}.jsonl`); silver/pipeline-findings generation and the calibrate-on-`related15` → confirm-on-`heldout15` run are still pending. | [Bug 69](#bug-69--exp-f-held-out-confirmation-scored-the-full-corpus-not-the-52-case-test-split) |
| B-070 | Fixed (2026-06-10) | Medium | Summarisation, MAP cascade decision | `reject_theta` was computed by `AgreementChecker` (`s ≤ reject_theta → REJECT`) but `evaluate_chunk` collapsed REJECT into the generic `keep=False` outcome, and both sync (`map_stage`) and batch (`_process_level`) escalated every `keep=False` chunk — so no chunk was ever **dropped** on low agreement and the terminal L3 voter always emitted. The documented three-way accept/reject/escalate decision was behaviourally two-way (accept/escalate). Fixed by exposing `ChunkOutcome.rejected` and dropping on reject in the legacy path of both runners; `reject_theta` is now the live toggle (default lowered `0.2 → 0.0` = drop off, reproducing prior behaviour; `> 0` enables drop). Router path keeps escalate-on-reject by design. | [Bug 70](#bug-70--reject_theta-inert-reject-never-dropped-now-a-toggleable-drop-policy) |
| B-071 | Fixed (2026-06-12) | Low | Summarisation eval tests, corpus-relate | `test_corpus_relate_tuple_unpack_does_not_crash` mocks `RelateStage.relate` with a stale 2-tuple `([], [])`, but `relate` now returns a 3-tuple `(relations, raw_pairs, skipped_pairs)` (`relate_stage.py:412`); the unmocked `relate_from_dir` unpacks 3 → `ValueError: not enough values to unpack (expected 3, got 2)` at `corpus_relate.py:295`. Test-only — all five production callers (`runner.py:620`, `batch/runner.py:608`, `corpus_relate.py:295`/`:391`/`:401`) already unpack 3. The very tuple-unpacking regression test broke for the class of reason it guards: the tuple grew a third element and its mock wasn't resynced. Fix: mock → `([], [], [])` + docstring. | [Bug 71](#bug-71--stale-2-tuple-mock-for-relatestagerelate-crashes-its-own-regression-test) |
| B-072 | Fixed (2026-06-15) | High | Eval, silver generation | Silver batch: every request fails with `'temperature' is deprecated for this model` — `claude-opus-4-7` no longer accepts an explicit `temperature`, which `generator.py` sent (=0) on both the sync and batch paths. Worked 2026-05-24; deprecated server-side since. 0/454 cases succeeded; `silver_findings_related15.jsonl` never written. Fix: drop the `temperature` field from both silver requests (forced `extract_findings` tool keeps variance low; model uses default sampling). Claude voters unaffected — `claude_batch.py` never sent temperature. | [Bug 72](#bug-72--silver-generation-fails-claude-opus-4-7-rejects-deprecated-temperature) |
| B-073 | Fixed (2026-06-16) | Medium | Eval, summarization sweep harness | `run_summarization_sweeps.py` gated its `map_weights` stage on `BEST_SCORER == "hybrid"` (and `_weight_variant_specs` hardcoded `scorer="hybrid"`), but the soft-alignment weights (`tau`/`count_alpha`/`reuse_weight`/`contradiction_weight`) are consumed by the **embedding** scorer too (`EmbeddingScorer.__init__` takes all four; both scorers call `agreement.embedding._align`). So when the production-default embedding scorer wins, its soft-align weights were un-tunable through the harness. Fixed in the **new** harness `run_new_summarization_sweeps.py` (not patched in the old file): its `map_weights` stage tunes scorer-specific weights for whichever scorer won, and adds hybrid blend weights only when `BEST_SCORER == "hybrid"`. | [Bug 73](#bug-73--map_weights-gated-on-hybrid-blocks-embedding-scorer-weight-tuning) |
| B-074 | Fixed (2026-06-17) | High | Eval, silver scoring (`_finding_to_pipeline`) | `_finding_to_pipeline` did `str(...)` on `relation_type`/`direction`/`category`, but `AuditableSummary.model_dump()` yields Enum OBJECTS, so `str(RelationTypeEnum.demographic)` → `'RelationTypeEnum.demographic'` ≠ silver `'demographic'` → every matched finding takes a strict-field penalty (loose-F1 intact, strict-F1 ≈ halved). Corruption scales with early-accept rate: ~4 % at θ0.9 (frozen winner ≈ unaffected) but ~55 % at economy θ0.4 (strict-F1 heavily underestimated) — distorting the E09 cost-quality frontier and inflating the "escalate-everything" advantage; raw-string L3 outputs were unaffected, masking it. Surfaced by E10 single-model baselines (single-Sonnet 0.44, strict/loose ratio exactly 0.50). Fix: route enum fields through `_ev()` (enum→`.value`) + regression test `test_finding_enum_unwrap.py`. Re-run θ-sensitive experiments (E07 first to reconfirm the pin). | [Bug 74](#bug-74--enum-stringification-in-_finding_to_pipeline-halves-strict-f1) |
| B-075 | Fixed (2026-06-18) | Medium | Doc-extraction DB ingest (`db_ingester.py`) | The cropper computes each table's page+bbox onto `CroppedMedia.page`/`.bbox` and `out/json/*_media.json`, but `PostgresDatabaseIngester`'s `Table(...)` set only caption/image_path — silently dropping `page_number`/`bbox_*` (columns exist, data was computed; page is even in the crop filename). So **0/1960** corpus tables had coordinate-level provenance (surfaced by E02). Fix: pass `tbl.page`/`tbl.bbox` in the constructor (forward, Docling coords) + `backfill_media_provenance.py` to populate existing rows from cached media JSON (matched by image filename) → tables 1960/1960 + figures 4479/4479 now have page+bbox (figures via migration 0014). Residual: `table_content`/`section_context` (no source in crop path). | [Bug 75](#bug-75--db-ingester-drops-table-pagebbox-provenance) |
| B-076 | Fixed (2026-06-18) | Medium | Summarization DB persistence (`persistence.py`) | First DB-persisting summarization run (corpus had been file-only) failed the post-run row audit: `sum_normal_findings` + `sum_finding_groups` (+ children) = 0 rows despite 117 canonical/final rules. Cause: NORMALIZE emits ≥1 duplicate `normal_id` per paper — the in-memory `nf_by_id`/`nf_id_map` dicts silently collapse them (so canonical/final are correct), but `persist_normal_findings`/`persist_finding_groups` insert one row per item without deduping → `UniqueViolation` on `uq_sum_normal_finding(pipeline_run_id,normal_id)` / `uq_sum_group_member(finding_group_id,normal_id)` → `except` swallows → 0 rows (and canonical rules then persist with null group FKs). Surfaced by the primer→MAP bridge 1-paper validation. Fix: dedupe by `normal_id` in both persisters (last-wins, matching the in-memory dicts). Verified: 123 normal → 120 persisted, 112 groups, audit OK. | [Bug 76](#bug-76--persisters-dont-dedupe-duplicate-normal_id--0-rows-in-normalgroup-tables) |
| B-077 | Won't fix (2026-06-18) | Low | `rebuild_from_cached_map` idempotency (not grounding) | NOT a grounding defect (mis-filed as non-determinism). `rebuild_from_cached_map` overwrites its input JSON with its OUTPUT, whose `audit_trail.map_chunks` holds POST-grounding findings — so re-running it on the same paper grounds an already-grounded set: run4 (raw 149 → 22 rejected → 127 written), run5 (read 127 → 0 rejected). Grounding is deterministic: a fresh re-install of the raw 149 reproduced exactly 22 (run6). It's an idempotent keep-filter, so a 2nd pass removes *fewer* (0), never more. Mitigation: run rebuild ONCE per corpus build on a fresh `bridge_populate_corpus --install` (raw map_chunks); never re-run rebuild on its own output. | [Bug 77](#bug-77--grounding-rejection-count-differs-between-identical-rebuilds) |
| B-078 | Mitigated (2026-06-19) | Low | Thesis §9.1 + eval `RESULTS.md` (E01) | Stale "reconstruct/merge variants 28–32 reach **100 % figure strict-F1**" claim contradicts the cited E01 artifact. The 27-PDF rerun shows **every** variant's figures at **84.0 %** strict-F1 (incl. 28/29/31/32); the reconstruct/merge variants alter only *table* outputs and mostly regress them (28→79.5 %, 31→44.2 %, 32→80.5 %), with all 14 decorative-icon crop-FPs surviving in every row — so 100 % figure strict-F1 is unattainable in this sweep. The figure score is in fact invariant across all 32 variants (the sweep re-tunes only table detection/cropping). Number is not from this artifact; it propagated from a `RESULTS.md` E01 parenthetical into the thesis. Mitigation: corrected the `RESULTS.md` note; thesis §9.1 correction drafted (pending paste). | [Bug 78](#bug-78--stale-100-figure-strict-f1-claim-for-reconstructmerge-variants-2832) |
| B-079 | Fixed (2026-06-20) | Low | Eval harness, E03 grounding sweep (Thesis §9.5 / RQ2) | E03 reported grounding retention **87.8 %** @0.5 vs the production funnel's **83.8 %** over the same 2280 frozen-config findings. Not a model difference: production `grounding_filter` runs after `_replace_verbatim_from_db` swaps each finding's LLM-paraphrased `verbatim_support` for its real cited DB paragraph, whereas the offline E03 sweep grounded the **paraphrase** straight from `voter_cache` (the replay dropped finding-level `evidence`, so no DB lookup was possible). Paraphrases entail their own claim more readily → 91 extra survivors. Fixed: carry `evidence` through the replay (`PipelineFinding.evidence`, `_finding_to_pipeline`) + call `replace_verbatim_from_db` in E03 → retention 83.8 % (1911/2280), matching the funnel exactly. | [Bug 79](#bug-79--e03-grounding-sweep-grounds-paraphrases-not-db-paragraphs) |
| B-080 | Mitigated (2026-06-20) | Medium | Summarisation, MAP output (production legacy cascade) | Citation/provenance integrity was **unchecked** on the production path: the `ProvenanceValidator` + `SchemaValidator` gates only run inside `MapOutputRouter`, which is `enable_router=false` (B-062), so the legacy L1→L2→L3 cascade shipped every finding's `S{n}\|PMCID\|te_id` citation unvalidated; `replace_verbatim_from_db` silently falls back to the LLM paraphrase on a te_id miss, and the grounding NLI is then fooled by the self-paraphrase. Mitigation: added a finding-level citation filter (`provenance/citation_filter.py`) reusing `ProvenanceValidator`, wired into sync `_cascade`, the batch consumer, and the offline `_replay`, gated by `CitationConfig` (default on, structural checks only). Surfaced **B-081**. | [Bug 80](#bug-80--citation-integrity-unchecked-on-the-production-cascade) |
| B-081 | Mitigated (2026-06-20) | High | Summarisation, `voter_cache` Gemini L1 voter (related15) | The L1 lead voter (`gemini-2.5-flash-lite`, voter index 0) emits findings whose **content and citation both belong to a different corpus paper**. Example: case `PMC7540531` chunk C2 — voter0 produced 12 findings about Mib-1/PPH3 staining methods (verbatim *"sections were stained using mib-1 (dakocytomation… m7240)"*), all citing `PMC3564399_his0057-0212\|13512`; the DB confirms te_id 13512 is a **real** methods paragraph in `PMC3564399`, while this case's own text (te_id 22044) is about Ki-67 in melanoma. The other two L1 voters (GPT-4o-mini, GPT-4.1-nano) cite the case's own paper correctly. Systematic: ~6006 cross-document cites across 25 foreign PMCIDs (~400 each), 37 % of all voter findings. Valid foreign DB keys + coherent foreign verbatim ⇒ **cross-paper contamination / Gemini batch result-misalignment**, NOT model hallucination. Risk: contaminated findings may enter the selected MAP set and the silver-F1 (0.7135), and poison E10 (Gemini-alone baseline) / E12 (voter-LOO). Caught by the B-080 citation filter (correct symptom guard); root cause is upstream in the cache build. | [Bug 81](#bug-81--gemini-l1-voter-cross-paper-contamination-in-voter_cache) |

| B-082 | Fixed (2026-06-21) | Low | Summarisation, `ProvenanceValidator` / B-080 citation filter | The cross-document check compared the pmcid by **exact string**, so a finding citing the bare canonical accession `PMC4329418` against the corpus's suffixed document id `PMC4329418_his0066-0409` (the SAME paper) was false-flagged `CROSS_DOCUMENT_SOURCE_ERROR` → dropped by the citation filter (and the calibration replay shaved these legit findings off every config). On the clean related15/heldout caches this was the ENTIRE residual: 112/112 same base accession, present across **every** voter (gemini/gpt/claude), verbatim genuinely from the paper — not hallucination, not contamination. Tail of [B-019](#) (regex widened to accept suffixed pmcids; the comparison stayed exact). Fix: compare the base `PMC\d+` accession (`_base_accession`) — bare-vs-suffixed matches, a genuinely different paper still caught. Residual cross-doc 107→0 (related15) / 129→0 (heldout); only 8 genuine malformed-citation `invalid_sentence_id` remain. | [Bug 82](#bug-82--cross-document-citation-check-false-positives-on-bare-vs-suffixed-pmcid) |
| B-083 | Fixed (2026-06-22) | Low | Summarisation, `rebuild_from_cached_map` / batch `finalize()` (thesis funnel build) | `rebuild_from_cached_map` logged `citation filter: empty chunk for pmcid=… — skipping (kept N findings)` as a **WARNING for every chunk of every paper** (e.g. 17× for heldout `PMC10529628`). Cause: the rebuild handle is reconstructed by `_build_handle_from_cached_json`, which sets `finalized` but never `BatchHandle.chunk_map` — the cached `audit_trail.map_chunks` carry `chunk_id` + `findings` but **no per-chunk source index**. With an empty source index the B-080 filter hits its `if not chunk:` guard (a missing index must never read as "all citations invalid") and keeps everything, once per chunk → log spam. **Not heldout-specific** (identical on related15) and **cosmetic**: the B-080 structural filter drops **0 findings** under the shipped 5-voter/escalate config anyway — verified two ways: bridge replay (citation OFF) == sweep `_replay` (citation ON) for all 15 related15 papers (2294 == 2294), and a direct `_replay` citation ON/OFF diff (`drops=0`). So the rebuilt corpus is **not** desynced from the headline. (B-082's "8 remain" was the pre-5-voter measurement; the shipped selection contains none of them.) Fix: batch `finalize()` detects the whole-handle empty `chunk_map` (rebuild path), skips the per-chunk filter, and emits **one DEBUG line** instead of N warnings; the helper's per-chunk warning stays live for the real pipeline, where one empty chunk among populated ones is genuinely suspicious. | [Bug 83](#bug-83--rebuild_from_cached_map-citation-filter-empty-chunk-warning-is-cosmetic) |
| B-084 | Fixed (2026-06-22) | Low | Summarisation, `rebuild_from_cached_map --no-db` / batch `finalize()` (DB-coupled post-run steps) | Two DB-oriented post-run steps in batch `finalize()` gated only on `self._db is not None`, never on `pipeline_run_db_id`, so under `--no-db` (where the runner keeps a DB *read* connection for `replace_verbatim_from_db` but `pipeline_run_db_id=None`, so every persister no-ops) they fired against wrong/empty DB state. (1) The post-run **row-count audit** queried `sum_*` tables nothing was written to → `[FAIL] 0 rows` for every table of every paper. (2) The per-paper **incremental corpus-relate** (`_corpus_relate_incremental → relate_incremental → _load_rules_from_db`) pooled each new paper's rules against **every rule already in the DB** — for the held-out rebuild that's the *related15* corpus (1729 rules), so each paper enumerated `C(44+1729, 2) = 1 570 878` gate pairs (all rejected, 0 relations) and issued 0-row delete churn against the shared corpus tables. Both **cosmetic for the funnel** (E04 is per-paper, never reads corpus-relate or the audit) but wasteful and not isolated. Fix: gate both on `pipeline_run_db_id is not None`. The isolated held-out cross-paper relations still come from the final file-based `_run_corpus_relate → relate_from_dir` pass over the split's own dir (built with `db=None`, so it never persisted). | [Bug 84](#bug-84--no-db-rebuild-runs-db-coupled-post-run-steps-row-audit--incremental-corpus-relate) |
| B-085 | Observed | Low | Thesis §4.1 / eval E01 variant naming (`01_docling`) | The sweep's `01_docling` ("Docling baseline") is **not** vanilla Docling — it is the full document-extraction pipeline (two-pass, masking, artifact filter, `nearest_caption`, size/icon filter, sub-figure merge, cropping) with only the table-handling knobs disabled, so §4.1's "Docling baseline" label and "Docling behavior was not modified" overstate it. A true off-the-shelf baseline (`00_docling_offtheshelf`, `scripts/eval/baseline_offtheshelf_docling.py`, fully hand-labelled) shows stock-Docling **tables 36.6 % strict-F1** (vs `01_docling` 40.0 %, v18 83.8 %) and **figures 44.7 %** (vs pipeline 84.0 %). Tables: both stock and `01_docling` fail the footnote dimension → capped ~37–40 %, so the 40→83.8 gain is **footnote-driven**; `nearest_caption` lifts the table baseline ~3 pts (scaffolding inflates it *slightly*, not *not at all*). Figures: stock over-emits (139 vs 76; 73 icon-FPs vs 14) and under-captions (caption P 60.6 % vs 90.3 %) → the pipeline's icon filter + merge + `nearest_caption` are worth ~39 strict-F1 pts, so "Docling figure behaviour unmodified" is misleading. | [Bug 85](#bug-85--docling-baseline-is-not-off-the-shelf-docling) |
| B-086 | Observed | Low | `eval/annotate.py --variant` + `eval/annotations/share_map.json` propagation | Annotating a sweep variant with `--variant` propagates each label to *peer* variants that share the crop key (`p{page}_{type}_{x1}_{y1}`, `share_map.json`). For `00_docling_offtheshelf` the figure/table crops share keys with the pipeline variants but carry **different caption metadata** (stock attaches no caption where `nearest_caption` does), so the propagated rubric labels (e.g. `correct figure, no caption`) are **wrong for the peers** and silently corrupted 51 tracked annotation files (e.g. `01_docling` figures 84.0→80.3 strict-F1). Caught by a cross-variant score shift; reverted with `git checkout HEAD -- eval/annotations/` (the off-the-shelf dir is untracked → unaffected). The propagation assumes a shared crop ⇒ shared label, which breaks when caption/footnote attachment is variant-specific. **Mitigation:** when annotating a caption-divergent variant, revert tracked `eval/annotations/` afterward, or annotate with propagation disabled. | [Bug 86](#bug-86--annotate-py-share-map-propagation-corrupts-peer-variants-for-caption-divergent-crops) |
| B-087 | Fixed (2026-07-01) | Low | PDF extraction, `hybrid_detector.py` docstring | `HybridTableDetector`'s module docstring said it merges table boxes "using iterative **IoU-based** union", but the `merge_rects` helper it calls merges on **boolean overlap** (`Rect.intersects`, any non-zero intersection), explicitly "**not on IOU threshold**" (`parsers/layout_utils.py:103-106`). No IoU code path or config knob exists anywhere; `merge_rects` is a shared helper reused by masking, two-pass redaction, and the hybrid detector. Docstring-only inaccuracy — runtime behaviour and the thesis text ("overlapping bounding boxes are merged into one") were already correct. Fix: corrected the docstring to say boolean-overlap union and pointed to `merge_rects` for the exact rule. | [Bug 87](#bug-87--hybrid-detector-docstring-says-iou-but-merge-is-boolean-overlap) |
| B-088 | Fixed (2026-07-11) | Medium | PDF extraction, `outputs/stats_writer.py` doc-stats collector | `DocStatsCollector.stage()` caught the stage exception with a bare `except Exception:` (no `as exc`) but its recording lambda referenced `type(exc).__name__` — an undefined name. The resulting `NameError` was swallowed by `_safe()`, so the **failed-stage `_StageTiming` was never appended**: every crashed run wrote `stage_timings: []`, losing the timing + `ok=False`/`error` record for the very stage that failed (`failed_stage`/`error` on the top-level payload were still set, so the loss was silent). Surfaced by 3 tests asserting a 1-entry `stage_timings` on failure. Fix: `except Exception as exc:`. | [Bug 88](#bug-88--stats_writer-stage-failure-timing-dropped-by-undefined-exc) |
| B-089 | Fixed (2026-07-11) | Medium | Test infra, `tests/knowledge_extraction/` UMLS kill-switch | Two test modules (`test_persist_voter_outputs.py`, `test_batch_persistence.py`) set `NLP_HISTO_DISABLE_UMLS=1` via **module-level `os.environ.setdefault`**, which leaks process-wide and never resets. In a full-suite run this disabled UMLS for the whole session, so `test_phase_a_gate::test_normalize_stage_normalizes_entities` (which needs the linker to resolve "CD31 expression" → "CD31 Antigens") failed — passing in isolation, failing in-suite, an order-dependent flake. Fix: replaced both with an autouse `monkeypatch.setenv` fixture (auto-reset per test). Also `norecursedirs`-excluded the gitignored vendored `pdffigures2/` tree, whose own broken import-time tests pytest had been collecting. | [Bug 89](#bug-89--test-modules-leak-nlp_histo_disable_umls-process-wide) |
| B-090 | Fixed (2026-07-11) | Medium | Test infra, `test_b027_seed_and_cache.py` × `pytest-randomly` | The two "missing dependency" tests set `sys.modules["torch"]`/`["numpy"] = None` (via `monkeypatch.setitem`) to exercise the seed helper's graceful degradation. `pytest-randomly` reseeds RNGs at every test's **teardown**, calling `thinc.fix_random_seed → torch.manual_seed` and `numpy.random.seed` — which runs *before* monkeypatch's undo, hits the `None` sentinel, and raises `ModuleNotFoundError: import of torch halted; None in sys.modules`. The teardown abort skips monkeypatch's restore, so `None` leaks into every later test's reseed → a **353-error cascade** under seed=1. Only manifests with `pytest-randomly` installed. Fix: restore the nulled module inside the test body (`try/finally`) so it is real again before the teardown reseed runs. | [Bug 90](#bug-90--null-dep-tests-crash-pytest-randomly-teardown-reseed) |
| B-091 | Fixed (2026-07-11) | Medium | Summarisation / test infra, `umls_resources.get_nlp()` singleton cache | `get_nlp()` reads `NLP_HISTO_DISABLE_UMLS` only on its **first** call and caches the outcome in `_AVAILABLE`; once set it never re-reads the env. So the first test to probe the singleton while the kill-switch is on (a `DISABLE_UMLS` test) caches `_AVAILABLE=False` **process-wide**, and a later test needing real UMLS (`test_normalize_stage_normalizes_entities`) silently gets `None`. Invisible under deterministic file order (an un-disabled test loaded the singleton first); surfaced by `pytest-randomly`. Correct in production (env fixed at startup) — a test-isolation concern. Fix: autouse fixture in `tests/knowledge_extraction/conftest.py` that clears a *disabled/failed* cache (`_AVAILABLE is False`) after each test, never a successful load. | [Bug 91](#bug-91--get_nlp-caches-the-umls-disable-decision-poisoning-later-tests) |
| B-092 | Fixed (2026-07-11) | High | Test infra, `pytest-randomly` × thinc seed reseeding | Plain `python -m pytest` (no explicit `--randomly-seed`) failed with **1680 errors** — `ValueError: Seed must be between 0 and 2**32 - 1` at nearly every test's setup/teardown. `pytest-randomly` reseeds each test with `base_seed + crc32(nodeid)` (up to ~2**33); it clamps its *own* numpy reseed (`% 2**32`) but passes the **raw** seed to registered `pytest_randomly.random_seeder` entry points. The only one installed is thinc's `fix_random_seed` (via scispaCy), which forwards it straight to `numpy.random.seed` → overflow. With a large random default base seed most tests overflow → cascade; tiny explicit seeds (`--randomly-seed=1`) almost never do, which hid it during the B-089/B-090/B-091 work. Regression from adopting `pytest-randomly`. Fix: repo-root `conftest.py` `pytest_configure` pre-populates `pytest_randomly.entrypoint_reseeds` with clamped wrappers (`seed % 2**32`). | [Bug 92](#bug-92--pytest-randomly--thinc-seed-overflow-breaks-default-python--m-pytest) |
| B-093 | Observed | Low | Test infra / network, `test_normalize_stage_normalizes_entities` × scispaCy UMLS linker | A full-suite run (2026-07-12, `python -m pytest -q`) reported `1 failed, 1402 passed`; the single failure was `assert 'CD31 expression' == 'CD31 Antigens'`. Captured log shows the scispaCy **UMLS linker** failed to load — `NameResolutionError` resolving `s3-us-west-2.amazonaws.com` while fetching `umls_semantic_type_tree.tsv` — so NORMALIZE skipped CUI normalisation and left the raw surface form. **Isolation re-run: 1 passed.** A network-dependent flake (the linker downloads KB resources from S3 on first load if uncached), distinct from B-091's order-triggered `_AVAILABLE` cache-poisoning (already fixed). NOT caused by the concurrent `chore(scripts): remove superseded model-connection checker` deletion (an unreferenced script; the test passes in isolation). Tracked separately; no test/code change made in that commit. | [Bug 93](#bug-93--normalize-test-flakes-when-the-scispacy-umls-linker-s3-fetch-fails) |

| B-094 | Fixed (2026-07-13) | Medium | Scripts, `scripts/inspect/inspect_{map_normalize,normalize_group,phase123_pipeline}.py` — direct execution | The three stage-walker inspection CLIs cannot run via their **documented** commands (`python scripts/inspect/<script>.py PMC…` — their own docstrings and `HOW_TO_RUN.md` §"inspect", `REPOSITORY_GUIDE.md`). Two independent defects. **(1) sys.path:** the repo is *not* installed as a package, and these three scripts have **no repo-root bootstrap** (unlike `inspect_pipeline_output.py` / `viewer_server.py` / `run_paper.py`, which all do). Python puts the *script's* directory (`scripts/inspect/`) on `sys.path[0]`, **not** the cwd, so `from database …` / `from pipeline …` raise `ModuleNotFoundError` unless the user manually sets `PYTHONPATH=.`. **(2) stale module paths:** four lazily-imported modules moved in the `stages/` + `grounding/` refactors and were never updated here — `…knowledge_extraction.{map_stage,normalize_stage,group_stage}` are now under `…knowledge_extraction.stages.*`, and `…knowledge_extraction.grounding_filter` is now `…knowledge_extraction.grounding.grounding_filter`. These fail **even with** the repo root on `sys.path`. Both imports are inside `main()`/helpers, so nothing (py_compile, ruff, AST, module-level import-smoke) caught them. All imported **symbols still exist** and every call site matches current signatures (`MapStage(voter_llms, level2_voter_llms, escalation_llm, theta, chunk_size)` + `.process(sentences, pmcid, cache=)`, `GroundingFilter(threshold=)`, `._pipe`, `score_findings(nli_pipe=)`, `NormalizeStage()`, `GroupStage()`, `PipelineCache(Path)`, `is_groupable`) — so there is **no API drift**, only import breakage. Proposed fix: add the standard `_REPO_ROOT = Path(__file__).resolve().parents[2]` + `sys.path.insert` bootstrap **and** correct the four stale import paths; imports only — no CWD, CLI, query, prompt, output, or logic change. **Investigation was fully static** (AST + file-existence + a sys.path probe outside the repo): the scripts were **never executed**, and **no database, network, API, LLM/Vertex, model-download, cache, or output operation occurred** (note `inspect_map_normalize.py` and `inspect_phase123_pipeline.py` make **real paid Vertex/LLM calls** when run). A bounded sweep found 8 further self-documented, bootstrap-less scripts with defect (1) only (`compare_docling_options`, `compare_policies`, `compare_prefilter`, `copy_relevant_files`, `eval_policy`, `fit_routing_threshold`, `select_policy`, `two_pass_extract`) plus `create_tui_gin_index` (undocumented) — tracked as follow-ups, not in this fix. | [Bug 94](#bug-94--stage-walker-inspection-clis-are-not-directly-runnable) |

| B-095 | Observed | Medium | Scripts, nine direct-run scripts under `scripts/` — repository-root import resolution | Nine scripts live directly under `scripts/`, import repository-local packages **at module level**, and carry **no repo-root bootstrap**: `compare_docling_options`, `compare_policies`, `compare_prefilter`, `copy_relevant_files`, `eval_policy`, `fit_routing_threshold`, `select_policy`, `two_pass_extract`, `create_tui_gin_index`. Python puts `<repo>/scripts` — not `<repo>` — on `sys.path[0]` for `python scripts/<name>.py`, and the repo is not installed as a package, so the imports fail immediately at module import. Proven statically: with `<repo>/scripts` as `sys.path[0]`, `importlib.util.find_spec` returns **None** for all four of `pipeline`, `database`, `parsers`, `named_entity_recognition`; with `_REPO_ROOT = Path(__file__).resolve().parents[1]` inserted, all seven imported module paths resolve. Eight of the nine **self-document a bare `python scripts/<name>.py …` command** that therefore cannot work as written (`create_tui_gin_index` has no usage line). None appears in project documentation. **Unlike B-094 there are no stale imports** — every imported module path and symbol exists in the current tree, so the repair is bootstrap-only. **Side effects when run** (none was executed): `create_tui_gin_index` issues raw DDL — `CREATE INDEX IF NOT EXISTS idx_entity_semantic_types ON entities USING GIN (semantic_types)` — i.e. it **mutates the database schema**; `copy_relevant_files` reads the DB and **copies files** (`shutil.copy2`); `compare_docling_options`, `compare_prefilter`, `two_pass_extract` drive **Docling/two-pass extraction** and write outputs; the four routing-policy tools read the routing dataset/policy store and write reports. **No script makes paid LLM calls.** Note also that no Alembic migration creates `idx_entity_semantic_types` — that index is an **un-migrated, out-of-band schema change** in an otherwise Alembic-managed schema (flagged separately; not a bootstrap issue). `# noqa: E402` is **not** required: `pyproject.toml` already sets `[tool.ruff.lint.per-file-ignores] "scripts/**" = ["E402"]` for exactly this pattern. Proposed fix: add the standard `parents[1]` bootstrap immediately before the module-level repository-local imports; imports only — no CWD, CLI, default, query, extraction, copying, output or logging change. | [Bug 95](#bug-95--direct-run-scripts-under-scripts-lack-a-repository-root-bootstrap) |

| B-096 | Observed | Low | Test hygiene, `tests/pdf_text_extraction/test_b027_seed_and_cache.py` × `PathConfig.run_metadata_dir` | The pdf-extraction runner tests write a real artifact into the **repository's `out/` tree** instead of an isolated test directory. `test_runner_skip_false_does_not_use_cache` (and its `_build_runner_with_mocks` helper) pass `tmp_path` for the input PDF but leave `paths.run_metadata_dir` at its default, `Path("out/run_metadata")` (`pipeline/stages/pdf_text_extraction/config.py:101`). Running `python -m pytest` therefore creates/overwrites `out/run_metadata/PMC1_stats.json`, whose `pdf_path` points at a pytest tmpdir (`…/pytest-of-emir/pytest-NNN/test_runner_skip_false_does_no0/fake.pdf`) — proving it is test-generated. That directory otherwise holds ~1070 **genuine** pipeline run artifacts, so the test artifact pollutes real local output. Harmless in git terms (`out/` is gitignored, `.gitignore:246`, and the file is untracked), but the suite should not write outside its own tmp tree. Fix: have the test override `cfg.paths.run_metadata_dir` (and any sibling output dirs) to `tmp_path`, as the other runner tests do. **Unrelated to the Phase 8 bootstrap commit** (`3e8c819`) — no target script references `run_metadata`; the file was left in place, not deleted. | [Bug 96](#bug-96--runner-test-writes-stats-into-the-real-out-tree) |

| B-097 | Observed | **High** | Schema governance, `entities.semantic_types` × Alembic `0006` vs `database/models.py` vs `scripts/create_tui_gin_index.py` | **The `entities.semantic_types` column has two contradictory definitions, and a fresh Alembic-initialised database gets the wrong one.** The ORM declares `semantic_types = Column(ARRAY(Text))` (`database/models.py:294`), so `create_tables()` (`Base.metadata.create_all`) builds **`TEXT[]`**. But Alembic revision `0006_add_entity_semantic_types` — the **only** migration that touches the column, and the chain is linear `0001→0014` with head `0014` — executes `ALTER TABLE entities ADD COLUMN IF NOT EXISTS semantic_types VARCHAR;`, so `alembic upgrade head` builds **`VARCHAR`**. No later revision reconciles them. Because `0006` uses `IF NOT EXISTS`, it is a silent **no-op** on any database that already had the column (e.g. the developer's live DB, populated by the one-off `database/migrations/add_semantic_types.py` that `0006` says it "absorbs" — that script is no longer in the tree), which is why the divergence was never noticed. **Consequences on an Alembic-built DB:** (a) the array-overlap queries `Entity.semantic_types.op('&&')(array(...))` in `named_entity_recognition/export_disease_entities.py:126` and `scripts/copy_relevant_files.py:37` cannot work against a `varchar`; (b) `scripts/create_tui_gin_index.py` — which runs the **unmanaged DDL** `CREATE INDEX IF NOT EXISTS idx_entity_semantic_types ON entities USING GIN (semantic_types)` outside the migration history — would **error**, since `varchar` has no default GIN operator class; (c) the ORM and the physical schema disagree. **Related sub-issue (unmanaged schema mutation):** that GIN index exists in no migration and in no `__table_args__`; the script is the only way to create it, i.e. an out-of-band schema change in an otherwise Alembic-managed schema. Note also that the "GIN full-text index on `text_content`" claimed in `.claude/CLAUDE.md` exists in **neither** `database/models.py` **nor** any migration. Nothing was executed: no PostgreSQL connection, no Alembic run, no SQL. Fix direction: reconcile the column type (make `0006`'s type match the ORM's `TEXT[]`, or add a corrective revision `0015`), then decide index ownership; the index question cannot be settled independently of the column type. | [Bug 97](#bug-97--semantic_types-column-type-diverges-between-orm-and-alembic) |
| B-098 | Observed | **High** | Schema governance, Alembic head `0014` vs `Base.metadata` | **`alembic upgrade head` does not reproduce the ORM schema, so the two are not interchangeable and `alembic stamp head` is unsafe.** Proven statically by an AST replay of revisions `0001`–`0014` against `Base.metadata` (`tests/database/test_schema_drift.py`, commit `1ca7067`). Alembic **cannot initialise an empty database at all**: revision `0001` declares a foreign key to `documents.id`, and **no revision ever creates `documents`** — the core ORM tables are assumed to pre-exist. Nor can the two be combined: `create_tables()` followed by `alembic upgrade head` fails, because `0013`'s `op.create_table('pipeline_runs', …)` is unguarded and `Base.metadata` also defines that table. Characterised divergences: `pipeline_runs.narrative_summary` is **ORM-only** (in no migration); `entities.semantic_types` is `TEXT[]` in the ORM but `VARCHAR` under Alembic (see B-097); `sum_corpus_relations.scope_note` differs in **nullability** (ORM `NOT NULL`, Alembic nullable) and `server_default` (ORM `''`, Alembic none) — a **substantive pending defect**, not merely cosmetic; three columns are client-side `default=` in the ORM but `server_default=` in Alembic (`pipeline_runs.status`, `pipeline_runs.started_at`, `sum_corpus_relations.scope_check_result`); six indexes and one unique constraint are **renamed** between the two. Consequence: **the ORM (`create_tables()`, wrapped by `python -m database.init_db`) is the sole supported way to initialise a new database**; Alembic manages incremental change on an already-ORM-initialised database only. Do **not** run `alembic stamp head` — it would assert a parity that does not hold. Nothing was executed: no PostgreSQL connection, no Alembic run, no SQL. The divergence set is now **characterised (not approved)** by a fail-closed test, so any future drift is caught. | [Bug 98](#bug-98--alembic-head-does-not-reproduce-the-orm-schema-stamping-is-unsafe) |
| B-099 | Fixed (2026-07-13) | Medium | Docs / setup, `README.md` × PostgreSQL database ownership | **The documented fresh-database procedure omitted the ownership/privilege requirement, and following it verbatim can fail on PostgreSQL 15+.** Found empirically: a guarded clean-room verification (disposable database `nlp_histo_cleanroom_<ts>_<rand>`, real dev DB never touched) **halted at creation** with `createdb: error: database creation failed: ERROR: permission denied to create database`. The configured application role (`DB_USER` = `local_db_user`) has **`rolcreatedb = false`**; only `postgres` has `CREATEDB`/`SUPERUSER` on this server. Yet `pg_database` shows the real `nlp_histo` is **owned by `local_db_user`** — i.e. the development database was never created by any "`createdb` as `DB_USER`" path; an administrator created it and assigned ownership. Two distinct facts follow. **(1)** Creating the database through an administrative role is *normal*, and the application role does **not** need `CREATEDB` — only the admin role does. **(2)** The new database must be **owned by** the configured `DB_USER` (or that role must otherwise hold schema-creation rights), or `python -m database.init_db` cannot create the ORM tables. The README's original example (`createdb -U <postgres-user> <database-name>`) assigned **no owner**, so a supervisor connecting afterwards as a non-owner `DB_USER` would fail at table creation with `permission denied for schema public` on **PostgreSQL 15+**, where `PUBLIC` no longer holds `CREATE` on schema `public`. This server is **14.8**, where the permissive default masks the problem — so even a *successful* clean-room run here **would not have caught it**. Fix: document `createdb -U <admin-role> -O <db-user> <database-name>`; `-O` is version-independent and does not rely on default `public`-schema privileges. Note the connection-failure hint in `database/init_db.py:374` still suggests a bare `createdb <dbname>` — a diagnostic hint, not a setup example; left unchanged, tracked as a follow-up. No paid API, no extraction, no Alembic, no SQL beyond read-only `pg_roles` / `pg_database` inspection; the disposable database was **never created**, so the cleanup trap correctly dropped nothing. | [Bug 99](#bug-99--fresh-database-setup-omitted-the-ownership-requirement) |
| B-100 | Superseded (2026-07-13) | Low | Docs / thesis drafting, `docs/thesis/reference/appendix_reference.md` × silver-artifact locations | **The drafted thesis appendix lists silver-evaluation artifact paths that do not exist.** Rows at `:134-136` and `:225-228` place the artifacts under **`eval/silver/`** (`pipeline_findings.jsonl`, `silver_findings.jsonl`, `silver_findings_test.jsonl`, `embedding_cache_{openai,gemini}.sqlite`, `map_primer/`, `map_primer_test/`). **No `.jsonl` or `.sqlite` exists anywhere under `eval/silver/`.** The real location — and every code default — is **`eval/data/`**: `evaluate.py:48-49` (`silver_findings_related15.jsonl`, `pipeline_findings_related15.jsonl`), `matcher.py:50-51` (`embedding_cache_{openai,gemini}.sqlite`), `map_theta_sweep.py:95` (`map_primer/`). **A blind `eval/silver/` → `eval/data/` substitution would still be wrong**: two artifacts were also *renamed* (`silver_findings_test.jsonl` → `silver_findings_heldout15.jsonl`; `map_primer_test/` → `map_primer_heldout15/`), and `pipeline_findings_related15.jsonl` is an **ignored generated artifact that is not currently materialised** (regenerate with `python -m eval.silver.data.export_pipeline`). **Scope — the thesis itself is NOT affected:** `docs/histo_thesis/pages/appendix.tex` is still untouched LaTeX template boilerplate, and `main.tex:52,54` has `% \appendix{}` / `% \input{pages/appendix}` **commented out**, so the thesis contains no appendix and the bad table has never been compiled into it. The defect is confined to gitignored drafting material (`docs/thesis/reference/appendix_reference.md`, plus its generated derivative `docs/thesis/build/thesis_final.md`), which would carry the wrong paths into the appendix if it is ever written into `.tex`. Artifact classes: all six are **ignored generated artifacts/caches**, not tracked files — only `eval/data/source_cases_related15.jsonl` is tracked. Configuration-dependent: the `related15` / `heldout15` suffixes are dataset-specific and the primer/cache locations are overridable at the CLI (`--primer-dir`, cache-path overrides at `map_theta_sweep.py:1251,1261`), so the appendix should give the **exact canonical path used by the documented reproduction**, not a bare directory. No thesis source was edited; nothing was generated, no experiment or API call was run. | [Bug 100](#bug-100--drafted-appendix-lists-nonexistent-silver-artifact-paths) |
| B-101 | Fixed (2026-07-13) | Low | Eval CLI, `eval/silver/generate.py` × argparse help expansion | **`python -m eval.silver.generate --help` crashes with `TypeError: %c requires int or char`.** argparse expands help text via `self._get_help_string(action) % params`, so a literal `%` in a help string is read as a format spec. `generate.py:52` sets `help="Use Anthropic batch API (~50% cheaper). Re-run to check status."` — the `% c` in `50% cheaper` becomes `%c`. Reproduced in isolation with a 3-line argparse snippet, and confirmed **pre-existing at HEAD** (the crash is identical before and after the Phase-11A bootstrap removal; the only diff is the deleted `sys.path.insert`). Impact: the CLI itself runs fine — only `--help` is unusable, so it never surfaced. Fix: escape the percent as `~50%% cheaper` (argparse's documented escape). **Fixed** in `fix(eval): escape percent in generate help` — the help string now reads `~50%% cheaper`, which argparse renders as the intended `~50% cheaper`; `--help` exits 0. Deliberately kept out of `refactor(eval): normalize silver import and repository paths`, which was forbidden from altering CLI behaviour, arguments, or defaults. No other eval CLI is affected — the other six `--help` calls exit 0. | [Bug 101](#bug-101--generate---help-crashes-on-an-unescaped-percent) |
| B-102 | Observed | Medium | Reproducibility, `eval/silver/analysis/map_theta_sweep.py:94` × `scripts/thesis/run_chapter9_offline_replay.py` | **Chapter-9 offline replay silently requires the repository root as its working directory.** `map_theta_sweep.py:94` declares `PRIMER_DIR = Path("eval/data/map_primer")` — a bare **cwd-relative** path, not anchored to the repository root. Run the replay from any other directory (`cd /tmp && python3 /abs/path/scripts/thesis/run_chapter9_offline_replay.py`) and it imports fine, executes, and then dies with `voter cache not found: eval/data/map_primer/voter_cache.json`. From the repo root it completes normally, so the dependency is invisible in day-to-day use. Discovered during the Phase-11C `sys.path`-bootstrap validation (outside-repository execution test); **pre-existing and untouched** — the line is byte-identical at HEAD and was not modified by the eval/silver or tests anchor-centralization commits. Not a bootstrap bug: the import bootstrap resolves correctly from outside the repo; only the *data* path is cwd-relative. Impact: any clean-room reproduction that does not `cd` to the repository root first gets a confusing missing-cache error rather than a clear diagnostic. Deliberately **not fixed** in the Phase-11C rollback — reserved as an explicit item for the clean-room reproducibility pass, where the anchor (`REPO_ROOT / "eval" / "data" / "map_primer"`) and the other cwd-relative defaults in the same module (e.g. the `eval/reports/` sweep output path) should be audited together. Note the replay script has no `--help`/argparse handling, so `--help` executes the full offline replay rather than printing usage. | [Bug 102](#bug-102--chapter-9-offline-replay-requires-the-repo-root-as-cwd) |
| B-103 | Observed (2026-07-14) | Medium | Eval design, `eval/silver/experiments/E13_nli_ablation/evaluate.py:120-121` × `eval/data/relation_claim_pairs_300.jsonl` | **E13's `scope_aware` arm cannot measure scope-awareness — the mechanism is disabled by the dataset's construction.** Production RELATE builds the NLI text per rule via `_build_scope_prefix` (`relate_stage.py:291`), so each rule contributes **its own** `Scope` (8 fields: `disease_subtype`, `tissue_site`, `assay_method`, …) and the two sides of a pair **can differ** — the comparability gate matches on subject/outcome/category/relation type but deliberately **not** on scope, so scope divergence is exactly what the tag exists to surface (motivating case in THESIS.md: "TP53 absent in AciCCIS" vs "TP53 present in AciCC" scoring SUPPORT). E13 instead reads the synthetic set's **pair-level** `disease_or_entity` field and feeds the **same string to both claims** (`build_shim(r["claim_a"], r.get("disease_or_entity"))` / `build_shim(r["claim_b"], r.get("disease_or_entity"))`; the shim docstring at `:71` even says "the **shared** entity"). The dataset is not at fault — the generation prompt (`eval/prompts/relation_pairs/batch_01_prompt.txt:58`) defines the field as "a short disease/entity/context label **for the pair** … for UNRELATED pairs, use the closest common context" — it is a generation scaffold, correctly produced. Consequence: the scope prefix is **identical on both sides of every one of the 300 pairs**, carries no discriminating signal, and on the 63 `different_entity` UNRELATED pairs actively asserts a shared context over two claims whose gold label rests on them being about different entities (e.g. scope `"Lung neoplasia"` over "ALK in lung adenocarcinoma" vs "RB1/TP53 in small cell lung carcinoma"). **The numbers are correct; the interpretation was not.** `predicate_only` (0.927 accuracy — the headline result) never reads scope and is unaffected; the reported `scope_aware` 0.923 is a valid measurement of *an identical prefix prepended to both claims*, so the 0.004 gap is dilution, **not** evidence against scope-awareness. A claim-derived scope (scispaCy/UMLS over each claim) would **not** fix this: the synthetic claims are fluent sentences that already name their disease, so extracting and re-prepending it tests "does restating the disease help" — whereas in production the tag matters precisely because CANONICALIZE's abstracted `predicate_text` ("CD30 is expressed") has **dropped** the disease. **Not fixed** (thesis submission 2026-07-14): no code or data changed, no numbers re-run; §4.5 and §5.6 were reworded to state that the arm establishes only that the bracketed prefix does not degrade the classifier, and that the benefit of scope is untested rather than disconfirmed. Follow-ups in THESIS.md ##TODOs: (a) minimal-pair regeneration with per-claim scopes over disease-stripped predicates; (b) exhaustive audit of the gate-passing pairs on production rules. | [Bug 103](#bug-103--e13-scope_aware-arm-cannot-measure-scope-awareness) |
| B-104 | Observed (2026-07-14) | Low | Doc extraction, `parsers/text_processing.py:318` (`remove_citations`) | **Bracket-style citation markers written with spaces (`[1, 2, 3]`) are not stripped and ride into the database.** The regex is `re.sub(r'\[\d+(?:[,–\-]\d+)*\]', '', cleaned)` — digits separated by comma / hyphen / en-dash with **no `\s` anywhere** — so it removes `[1]`, `[1,2,3]`, `[1-29]`, `[1–3]` but not the spaced variant many publishers typeset. Measured on the production corpus (2026-07-14): **561 of 35 896 text elements (1.6 %) still contain a `[<digits>, <digits>` marker; 0 contain the unspaced form**, exactly as the regex predicts — papers that space their citation lists slip the filter wholesale, papers that don't are cleaned completely. Impact: leftover markers survive into `text_elements.text_content`, hence into MAP chunks and potentially into `verbatim_support` strings; low severity (cosmetic noise in the NLI/grounding inputs, no structural breakage). Fix (one line, untested): allow optional whitespace — `r'\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]'` — keeping the digits-only constraint so `[Table 1]` is still not matched. **Deliberately not fixed at submission (2026-07-14):** the fix requires a corpus re-ingest, which would invalidate every downstream number in the thesis (MAP findings, silver labels, all reported metrics). Thesis text is **not** affected — §3.3 says markers "such as `[1,2,3]`" are removed, which is accurate and claims no exhaustiveness. Note the project `CLAUDE.md` describes `remove_citations` as stripping "`[1, 2, 3]`-style" markers, with spaces — that doc line is wrong; the code is authoritative. | [Bug 104](#bug-104--spaced-bracket-citation-markers-1-2-3-are-not-stripped) |
| B-105 | Fixed (2026-07-16) | Medium | Docs, `docs/HOW_TO_RUN.md` §9 × `src/nlp_histo/workflows/knowledge.py:407,422,460` | **Neither documented knowledge-extraction command could run — §9 was wrong in three separate ways, and the section claimed "flag parsing" had been verified.** Command 1 (`nlp-histo knowledge --profile cheap --pmcid PMC1448691 --sync --health-check no`) passes `--pmcid`, **which does not exist** — the PMCID is a *positional* (`knowledge.py:407`, `parser.add_argument("pmcid", nargs="?")`) → `error: unrecognized arguments: --pmcid`. Command 2 (`… --profile real --all --source-cases …`) omitted **two** required arguments: `--health-check` (`:460`, `required=True`) and the required mutually-exclusive mode group `--sync`/`--batch` (`:422`, `required=True`). Both exit 2 at argparse. §9 nonetheless asserted "*Verified here:* `--help`, **flag parsing**, and the missing-file preflight" — the claim was false; only `--help` and the preflight had been checked, and the commands themselves never were. The **legacy** pre-packaging doc (gitignored `docs/readmes/HOW_TO_RUN.md:437-440`) had all of this right — it states "`--sync`/`--batch` AND `--profile` AND `--health-check` are all required" and shows the positional form (`run_paper.py PMC7150310_main --sync --profile cheap --health-check no`) — so this is a **regression introduced by the packaging-era doc rewrite**, not a long-standing gap. Impact: documentation only, and self-limiting — argparse rejects before any provider is constructed, so a supervisor loses time, not money. The required-ness is deliberate and correct: it is the property that stops a paid run starting by accident. **Fixed**: §9 now uses the positional PMCID, passes `--health-check no` and `--batch`/`--sync` on both lines, states that all three flags are required and have no defaults, documents `--dry-run` as the free way to check a paid invocation, and its verification note now records that both commands were executed as written under `--dry-run` (exit 0, zero paid hosts contacted) rather than inspected. | [Bug 105](#bug-105--neither-documented-knowledge-command-could-run) |
| B-106 | Fixed (2026-07-16) | Medium | Docs, `docs/HOW_TO_RUN.md` §10 × `nlp-histo replay chapter9` | **§10 claimed the chapter-9 replay is "Offline: no API key, no database, no model inference, no cost" — two of those four are false.** Verified 2026-07-16 by running the replay under a `getaddrinfo` guard that logs every resolved host: the replay contacts `s3-us-west-2.amazonaws.com` (scispaCy `en_core_sci_lg` + UMLS KB) and `huggingface.co` (`pritamdeka/PubMedBERT-MNLI-MedNLI`), then runs the NLI model locally on MPS — i.e. it is **not offline** and it **does run model inference**. "No API key" and "no cost" are both true and were re-confirmed: a denylist guard covering OpenAI/Anthropic/Gemini/Azure/Mistral/DeepSeek fired **zero** blocks, and both frozen embedding caches reported **0 cache misses**. Impact: a supervisor on a cold model cache faces several hundred MB of downloads on a command sold as offline, and one with no network at all hits B-107 instead of a clean failure. Once both models are in the local cache the replay genuinely needs no network, so the claim held for anyone who had already run the pipeline — which is why it survived. **Fixed**: §10 now separates "no paid call" (true, with the two free hosts named) from "offline" (false on a cold cache), and cross-links B-107. | [Bug 106](#bug-106--10-overstated-the-replay-as-offline-with-no-model-inference) |
| B-107 | Fixed (2026-07-16) — silent degradation; offline capability Won't fix (see detail) | High | Reproducibility, `nlp-histo replay chapter9` × `entities/umls_resources.py` (linker load) | **The chapter-9 replay silently produced different-but-plausible numbers when the UMLS linker could not be fetched — it warned once and exited 0.** **Fixed**: `require_umls()` gates the replay in `configure()`, before the output directory is created — an unusable linker now exits **3** with the real cause, the affected outputs named, and nothing written. `get_nlp()`'s return-`None` contract is unchanged, so the live pipeline's deliberate skip-CUI path still works. Discovered 2026-07-16 by running the replay with **all** network blocked: the scispaCy UMLS KB download failed, the run logged a single `WARNING UMLS: linker unavailable — downstream stages will skip CUI work`, and then **completed with exit 0** and wrote CSVs that differ from the frozen baseline on exactly the CUI-dependent analyses (`06_exp_f_test_split.csv`, `12_real_profile_grounding_polarity.csv`). Nothing in the exit code, the CSV contents, or `manifest.json` marks the run as degraded. **The normal networked path is unaffected and reproduces perfectly**: with only paid hosts blocked, the same replay regenerated **9/9 CSVs byte-identical** to `out/thesis_results/chapter9_offline_replay/`, with only the two documented non-regenerating files (`04`, `10`) absent — so this bites the clean-room reproduction and nothing else. **Two corrections to the initial filing, after re-testing with a faithful `socket.gaierror` instead of the first guard's custom `RuntimeError` (which library `except (ConnectionError, OSError)` fallbacks never catch):** (1) the contrasting "NLI model failed loudly" observation was an **artifact of that guard** — HuggingFace caches correctly and the NLI model loads offline from `~/.cache/huggingface` with or without `HF_HUB_OFFLINE=1`, so on a genuinely offline machine `11_nli_input_four_mode_ab.csv` regenerates normally and the one loud symptom that might warn a user **never fires**; (2) "cold cache" is wrong — cache warmth is **irrelevant**. scispaCy keys its cache on the live ETag (`file_cache.py:119` `requests.head(url)` → `:126` `ETag` → `:53-68` filename = `sha256(url).sha256(etag)`), so with no network the filename cannot be computed and a **byte-complete 2.1 GB cache already on disk is unfindable** (verified: all five linker artifacts present, each with a url+etag sidecar, and the offline load still dies fetching `tfidf_vectors_sparse.npz`). **Pre-fetching therefore cannot fix this.** Net: severity is *higher* than first filed — the trigger is any offline run, and the only symptom is one WARNING line. Not fixed; two separable decisions (stop the silence: fail hard or stamp `degraded=true` into `manifest.json`, likely in the replay's own preflight since `umls_resources` is shared with the live pipeline where skip-CUI-and-continue may be desired; and optionally make it genuinely offline-capable by falling back to the cache sidecars when `requests.head` fails). Mitigation until then: §10 tells the reader to confirm the warning is absent before trusting a run. | [Bug 107](#bug-107--replay-silently-degrades-when-the-umls-linker-is-unreachable) |
| B-108 | Observed (2026-07-16) | Medium | Eval reproducibility, `eval/sweeps/grounding.py` × `eval/results/grounding_sweep.md` (tracked) | **A command documented in HOW_TO_RUN §12 overwrites a tracked thesis artifact with different numbers.** `python eval/sweeps/grounding.py` writes `eval/results/grounding_sweep.md` (tracked) and `eval/results/grounding_sweep.csv` (untracked). The committed `.md` was generated 2026-05-16 from **5** papers / run `grounding_compare_calv1_runB_20260516T163007` / config hash `149023b87374cbc2`; re-running on 2026-07-16 read the current `out/summaries` (**15** papers, hash `cfb56a0289b557be`) and rewrote the file wholesale. This is **input drift, not non-determinism** — the script has no frozen input pin and simply consumes whatever is in `out/summaries`, so its output tracks the corpus rather than the thesis snapshot. Impact: a supervisor following §12 verbatim silently dirties a tracked results file and can no longer `git diff` cleanly; worse, the rewritten numbers do not correspond to any published table. Restored here with `git checkout eval/results/grounding_sweep.md`; worktree left clean. Not fixed: options are to pin the input set (a `--source`/`--run-id` filter matching the committed snapshot), write to an untracked path by default, or untrack the `.md` and treat it as regenerable output — a call worth making alongside the other frozen-artifact paths in the same module. Documented in §12 as a caveat in the meantime. | [Bug 108](#bug-108--groundingpy-overwrites-a-tracked-thesis-artifact) |
| B-109 | Fixed (2026-07-16, `ec11eec`) | High | Reproducibility / UX, `eval/silver/analysis/map_context.py:81` × frozen embedding caches | **The free, fully-cached §12 experiments refuse to start without `GOOGLE_API_KEY`, an API key they never use.** `_load_map_context` constructs the embedder before it touches the cache: `api_key = os.environ.get("GOOGLE_API_KEY")` → `raise SystemExit("GOOGLE_API_KEY not set")` (`map_context.py:81-83`; the OpenAI branch at `:94` is identical for `OPENAI_API_KEY`). But the frozen caches make the key dead weight — E14 on 2026-07-16 reported `Agreement embed pre-warm: 15273 unique claims, **0 cache misses**` and contacted no Google host under a denylist guard. Impact: a supervisor reproducing the thesis without a Google account cannot run `E14_heldout`/`E04` at all, and the error message points them at obtaining a paid API key rather than at the truth, which is that the run is free and complete offline. Any non-empty string satisfies the check today (the embedder is constructed but never called), which is exactly why this went unnoticed — every developer machine has the key in `.env`. **Severity raised from Medium on 2026-07-16**: a clean-clone test showed this is not merely UX — `replay chapter9` on a fresh clone with no `.env` produced **8 of 9** tables, analysis 05 failing with "OPENAI_API_KEY not set… constructor requires a non-empty key even though the cache is warm". Every prior run had `.env`, so it never surfaced while §10 claimed "no API key" throughout. The key was provably unused: `OPENAI_API_KEY=dummy-not-a-real-key` gave 9/9 byte-identical tables with 0 cache misses — pure constructor theatre. **Fixed in `ec11eec`**: `replay.py:1166,1777` (the two matcher sites B-112 did not cover) now use `NoLiveEmbedding` — no environment read, no client built, every embedding from the validated cache, `CacheOnlyViolation` on an unexpected miss. The guard moved into packaged `nlp_histo.evaluation.matching.embedders` so the replay needs no repository-only import and there is one definition of "must not reach a provider"; `map_context` re-exports it. The paid path is untouched — `_load_map_context` without `strict_cache_only` still demands a key and builds a real client. Verified: fresh clone, no `.env`, every provider credential unset → exit 0, **9/9 byte-identical**, 0 paid hosts, 0 misses, 0 violations, 0 Postgres connections, 0 key errors. §12's `GOOGLE_API_KEY` caveat is obsolete and removed — E04, `sweeps/grounding.py` and E14 each re-run credential-free, all exit 0. | [Bug 109](#bug-109--free-cached-experiments-hard-require-an-unused-google_api_key) |
| B-110 | Fixed (2026-07-16) | Low | Tests, `tests/test_config_loader.py` × `configs/run.yaml` | **Two tests asserted pre-calibration agreement defaults that the shipped config intentionally no longer uses — `python -m pytest` was 2 failed / 1552 passed.** `test_agreement_scorer_kind_loaded_from_run_yaml` asserts `sumcfg.agreement.scorer_kind == "embedding"` (actual: `hybrid`) and `test_hybrid_config_loaded_from_run_yaml` asserts `h.w_category == 0.25` (actual: `0.15`). Both read the tracked `configs/run.yaml`, which deliberately pins the calibrated E06/E08 winner — the file's own comments say so (`scorer_kind: hybrid   # E06 family_refine pin (was 'embedding')`, `alignment_strategy: greedy   # E06 family_refine pin (was 'soft_max')`). **The config is right and the tests are stale**, corroborated independently by `eval/silver/experiments/E14_heldout/heldout_eval.py:67-74`, whose `_frozen_spec()` — documented as "the E05–E08 winner scorer (= configs/run.yaml / E03 `_frozen_spec`)" — hard-codes exactly `w_category=0.15, w_embedding=0.30, w_entity=0.50, w_evidence=0.05`. So the tests encode a config that was superseded when the sweep winner was pinned, and were never updated. Impact: test-only; no runtime behaviour is wrong. But HOW_TO_RUN §11 instructs the supervisor to run `pytest`, so the suite greets a clean-room reproduction with two red tests over a config that is in fact correct — a credibility cost out of proportion to the defect. **Fixed**: both assertions updated to the calibrated values (`hybrid`; 0.15/0.30/0.50/0.05), with docstrings now stating that `run.yaml` pins the E06 winner rather than the dataclass defaults. The sibling `test_hybrid_config_defaults` (0.25/0.40/0.25/0.10) asserts the *dataclass* defaults, is a different assertion, and correctly still passes — left untouched. Suite is now **1554 passed, 0 failed**; `docs/HOW_TO_RUN.md` §11 updated to match. | [Bug 110](#bug-110--stale-config-loader-tests-assert-pre-calibration-defaults) |

| B-111 | Fixed (2026-07-16) | Medium | CLI, `src/nlp_histo/cli/main.py:186-192` (`_split_forwarded`) × `:128` (ingest help string) | **`nlp-histo ingest -- --help` — the invocation the CLI's own help tells you to use — did not work, and there was no other way to discover the forwarded options.** `ingest` and `ner extract|merge|export` take no options of their own; everything after the command path is sliced out of `argv` and forwarded to the underlying workflow parser, so `nlp-histo ingest --help` deliberately prints the CLI's short stub (which for `knowledge` carries the PAID warning — the reason the interception exists). To reach the *workflow's* help the stub says "Options are passed through to the extraction runner; see `nlp-histo ingest -- --help`" (`main.py:128`) — but that command exited **2** with `error: unrecognized arguments: -- --help`, as did `nlp-histo ner extract -- --help`. Cause: `_split_forwarded` forwarded the `--` verbatim, and argparse reads a bare `--` as its *positional separator*, so the following `--help` was demoted to a positional the runner has no slot for. Net effect: the workflow's real options (`--pdf-dir`, `--out-root`, `--entity-cache`, ~30 more) were **undiscoverable through the CLI** — the usage line was only visible as part of an error. Found 2026-07-16 by executing the documented hint rather than grepping for the flag names; the flags all existed, which is exactly why inspection missed it (same lesson as B-105). **Fixed**: `_split_forwarded` now treats a leading `--` as an explicit "everything after this is the workflow's" marker and **drops** it before forwarding, so `ingest -- --help` forwards `["--help"]` and the runner prints its own help (exit 0). Plain `--help` still resolves to this CLI, preserving the cost warning. Two regression tests added in `tests/cli/test_cli.py` (`test_explicit_passthrough_reaches_the_workflow_help`, `test_plain_help_after_command_is_the_clis_own`) — the existing CLI tests covered flag forwarding but never the `--` form. | [Bug 111](#bug-111--the-clis-own-documented-passthrough-help-did-not-work) |

| B-112 | Fixed (2026-07-16, `8d0c5c5`) | High | Cost / reproducibility, `workflows/replay.py:132,163` (`REQUIRED_ARTIFACTS`) × `eval/silver/analysis/map_context.py:36,90` | **The chapter-9 replay's "a missing cache can never cause paid calls" guarantee only covers the OpenAI cache — the Gemini cache is used but unvalidated, and a tree without it would issue ~15 000 paid embedding calls in a workflow documented as free.** The preflight requires `eval/data/embedding_cache_openai.sqlite` (`replay.py:132`, `FROZEN_EMBEDDING_CACHE` at `:163`) and §10 sells this as the reason an incomplete tree "refuses to start instead" of spending money. But the run opens **both** caches — verified from the 2026-07-16 run log, which reports `SQLite embedding cache: … embedding_cache_openai.sqlite` *and* `… embedding_cache_gemini.sqlite` (87 942 entries, 1.1 GB). The gemini cache reaches the run through `_load_map_context("gemini", embed_cache_path=None)` → `path = _FROZEN_GEMINI_CACHE` (`map_context.py:90`), and is **absent from `REQUIRED_ARTIFACTS`**. Consequence: an artifact tree carrying the openai cache but not the gemini one passes validation cleanly, then `_prewarm_agreement_cache` misses on every claim and `_make_cached_embed_fn` calls `embed_fn(miss_texts)` (`map_theta_sweep.py:679`) against a live `AgreementGeminiEmbedder` — 15 273 unique claims, all billable, in the command documented as costing nothing. **Second, compounding defect:** `_FROZEN_GEMINI_CACHE` is anchored to `eval.paths.REPO_ROOT` (`map_context.py:30,36`) — the *repository* root — not the replay's `--artifact-root`, so `--artifact-root` does not actually control where the cache is read from (same family as B-102's cwd-relative paths). A replay pointed at a copied artifact tree silently reads the gemini cache from the repository instead, or misses entirely when run outside one. Never bit in practice: both caches are complete here (**0 cache misses**, verified twice on 2026-07-16), so runs from a repository checkout were genuinely free — which is exactly why this went unnoticed. **Fixed in `8d0c5c5`**, both halves together: the gemini cache joins `REQUIRED_ARTIFACTS`, both caches resolve from `--artifact-root` (all four `_load_map_context` call sites now pass `embed_cache_path` explicitly instead of `None`), the preflight validates the required **entries** rather than mere file existence (an empty cache is a valid database), and the replay runs `strict_cache_only=True` so no provider is constructed and any unexpected miss raises instead of billing. Exit 4 on incompleteness, before the output directory exists. | [Bug 112](#bug-112--the-replays-paid-call-guard-misses-the-gemini-embedding-cache) |

| B-113 | Fixed (2026-07-16, `30755f8`) | High | Data integrity / safety, `src/nlp_histo/database/db_connection.py:22-25` × `ENV_LOADING.md:159-169` | **An explicitly selected `NLP_HISTO_ENV_FILE` is silently ignored for any `DB_*` variable already present in the environment — a run aimed at a test database can transparently target production.** Demonstrated 2026-07-16 during the §7 ingest verification: with `NLP_HISTO_ENV_FILE=/tmp/ingest_test.env` (`DB_NAME=new_local_db`), the resolved connection was **`nlp_histo` — the established 977-paper corpus**. The one-PDF ingest would have written into production. It was caught only because the verification harness asserted `url.database == 'new_local_db'` before connecting; nothing in the library says which database it is about to use. Mechanism: `db_connection.py:22-25` resolves *which file* to read (`_explicit or find_dotenv(usecwd=True)`) and calls `load_dotenv(_found)` — python-dotenv defaults to `override=False`, so file values never replace variables already exported in the shell. The shell had inherited `DB_NAME=nlp_histo` from an earlier `source .env`. **The precedence itself is deliberate and documented** — `ENV_LOADING.md:159-169` states "Actual environment variables (highest priority) → .env file → Default values" with the worked example `export DB_NAME=custom_db` "Uses 'custom_db', not the .env value" — so a blanket switch to `override=True` would break a documented contract for users who intentionally override file values from the environment, and is **not** the fix. The defect is narrower and is one of *silence*, not of ordering: `NLP_HISTO_ENV_FILE` reads as an explicit, intentional selection ("use *this* config"), and when it loses to ambient state the loss is invisible — no warning, no conflict detection, no echo of the resolved target. The existing test (`tests/test_runtime_paths.py:39-43` `test_env_file_override`) asserts only that `_env_path()` returns the explicit path — i.e. it covers **file selection**, never **value precedence** — so the gap is untested. `docs/HOW_TO_RUN.md` §4 lists the variable as "explicit path to the `.env` file", which reinforces the wrong mental model. Severity High on impact (silent writes to the production corpus by a command the operator believes is isolated), lower on likelihood (requires `DB_*` already exported — but `source .env`, direnv, CI secrets and IDE run-configs all do exactly that). **Fixed in `30755f8`**, without touching the documented precedence: (1) conflict detection for an *explicit* `NLP_HISTO_ENV_FILE` only — routing fields (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_SCHEMA`) declared by the file are compared against the inherited environment **before** `load_dotenv` runs, and a disagreement fails before any engine exists, naming the variables (never their values) and giving both resolutions; secrets are never conflicts; ordinary automatic `.env` discovery is untouched. (2) `db init`, `db check`, `ingest` and the NER commands now all echo `Target: user@host:port/database` through one formatter (`database/env_routing.py`), so a redirected write announces itself; the password and the credential-bearing URL are never printed. Verified: the reproduced near-miss exits 2 with an actionable error and no traceback; agreement and discovery still exit 0; `--help` unaffected. | [Bug 113](#bug-113--an-explicit-env-file-is-silently-overridden-by-inherited-db-variables) |
| B-114 | Observed (2026-07-16) | Low | Known limitation / unused fields, `database/models.py:32-34` × `pdf_text_extraction/outputs/db_ingester.py:84-89` | **`documents.title`, `documents.journal` and `documents.publication_year` are declared but never populated — 0/977 across the established corpus.** Measured 2026-07-16: `with_title=0 with_journal=0 with_year=0 with_text_source=977 total=977`, and reproduced on the verified one-PDF ingest (`title=<null>`), so it is a property of the pipeline rather than of any particular run. Cause: `db_ingester.py:84-89` constructs `Document(pmcid=…, filename=…, file_path=…, text_source="pdf")` and sets no bibliographic metadata; nothing else writes those columns (`scripts/seed_fake_papers.py:157` sets `title`, but only for synthetic papers). All three are nullable (`title = Column(Text)`), so nothing errors. The `text_source` default of `'xml'` suggests the columns date from an XML-ingest design that would have carried this metadata; the PDF path never acquired it. **Nothing depends on them being populated:** the only consumer is Pipeline C's paper selection, which carries `title` through `loaders.py:152` → `fingerprints.py:234` → `export.py:162,210` into an export field that is simply `null`; it is never used in scoring, similarity, or selection, and no doc promises it. Classified as an **unused-field / known-limitation cleanup item**, not a reproduction blocker — the thesis numbers do not read these columns. Options when someone picks it up: populate from the PMC XML (`files/organized_xmls/`, already downloaded) at ingest, populate from Docling's detected title, or drop the three columns and the dead export plumbing. **Deliberately not fixed during the verification pass** — adding speculative title extraction was out of scope and would touch the ingest path with no consumer demanding it. | [Bug 114](#bug-114--document-bibliographic-columns-are-never-populated) |

| B-115 | Fixed (2026-07-16, `8d6cc9c`) | Medium | NER, `ner/merge_entities_by_umls.py` + `ner/export_disease_entities.py` (`--model` default) × `ner/ner.py:215` | **`nlp-histo ner merge` and `ner export` filtered on a model name that is never stored, so both exited 0 having silently produced nothing — against a corpus of 1 792 440 entities.** `ner.py:215` persists `nlp.meta.get('name')`, and spaCy's `meta["name"]` for the `en_core_sci_lg` package is **`core_sci_lg`** (the `en` is `meta["lang"]`, not part of the name). Both consumers defaulted `--model` to the *package* name `en_core_sci_lg`, which matches zero rows: the established corpus reports `core_sci_lg | n=1792440` and nothing else, so §8's documented `ner merge` / `ner export` **have never emitted a file** in the corpus's lifetime. Discovered 2026-07-16 during the §8 verification: `ner extract` succeeded (865 entities, 749 with CUIs) and the next documented command reported "No entities with UMLS mappings found" against a table that plainly had them. `ner.py` already had it right — its skip-if-already-processed check hardcoded `"core_sci_lg"` and a comment at `:178` explained why — so the writer and the two readers had disagreed all along. **Fixed in `8d6cc9c`**: one shared `enums.DEFAULT_MODEL_NAME` now feeds `ner.py`, merge and export (explicit `--model` overrides unchanged), and `model_filter.check_model_filter()` raises `NoMatchingEntitiesError` → exit 1 when a filter matches nothing *while other model names exist*, naming what is available and why the identifier differs. A genuinely empty corpus, or no `--model`, remains a silent honest empty result — the fix distinguishes "no data" from "wrong question", the distinction the old code collapsed. Verified end-to-end on an isolated database (production neither re-run nor mutated): merge → 749 occurrences / 762 files, export → 89 disease CUIs / 178 files, `--model en_core_sci_lg` → exit 1. **Blast radius: none** — audited read-only, no thesis chapter, experiment or committed report consumes the output (`eval/paper_selection` reads the `entities` table directly and never filters `model_name`; its `PaperFingerprint.disease_entities` is a keyword-derived set, a name collision with the export directory, not a dependency; the only consumer of those directories is the quarantined `legacy/langchain-summarization/count_tokens.py`). No reported number changes — which is precisely why a silent no-op survived this long. | [Bug 115](#bug-115--ner-merge-and-export-filtered-on-a-model-name-that-is-never-stored) |
| B-116 | Fixed (2026-07-16) | Low | Docs / packaging, `docs/HOW_TO_RUN.md` §2 × `pyproject.toml` (no `dependencies`) | **§2's wheel block claims "no source tree needed" but omits the dependency step, so following it alone produces an install that imports and prints help and then dies on the first real command.** `pyproject.toml` deliberately declares no `[project] dependencies` — `requirements.txt` is the pinned, tested set (CLAUDE.md calls it "the tested source of truth") — so the built wheel's only metadata requirement is `Requires-Python`. Demonstrated 2026-07-16 in a throwaway venv outside the repository: `pip install --dry-run dist/nlp_histo-0.1.0-py3-none-any.whl` reports *"Would install nlp-histo-0.1.0"* and nothing else; after installing, `import nlp_histo` resolves from site-packages, all 11 command/subcommand `--help` invocations exit 0, and both packaged resources load (`model_prices.json` 1437 B, `nli_models.yaml` 1558 B) — but `nlp-histo db check` exits 1 with `ModuleNotFoundError: No module named 'sqlalchemy'`. The packaging is correct and the dep-less wheel is an intentional design; the *documentation* implies self-sufficiency it never had. Low severity: the failure is immediate, loud, and obvious to fix — but it lands on a supervisor who followed §2 verbatim and reasonably expected a working install. **Fixed**: §2's wheel block now installs `requirements.txt` alongside the wheel, states that the wheel carries no dependencies and why, and records the exact verification (including the ModuleNotFoundError as expected behaviour rather than a defect). Note the editable block above it was always correct — it installs `requirements.txt` first and then `-e . --no-deps`. | [Bug 116](#bug-116--the-wheel-install-block-omits-the-dependency-step) |
| B-117 | Fixed (2026-07-16) | Medium | Acquisition, `cli/main.py:40` (`_acquire`) × `acquisition/downloader.py:120` | **`nlp-histo acquire download` exited 0 after downloading nothing.** `download_papers()` returns the number of tarballs fetched; the CLI discarded it and returned 0 unconditionally, so a run in which *every* requested paper failed printed "Done — 0 tarball(s)" and reported success. Found 2026-07-16 while exercising §6 against NCBI with a single PMCID (`PMC10047158`): the OA API advertised `oa_package/fa/c8/PMC10047158.tar.gz`, the HTTPS GET returned **404**, and the command exited **0**. Same silent-success family as B-107 and B-115 — a failure that reads as a result. **Fixed**: `_acquire` compares the fetched count against the requested count and exits **1** when a non-empty request yields nothing, pointing at the per-PMCID reasons already printed. A *partial* result stays 0 on purpose — papers legitimately outside the OA subset are reported per-PMCID and are not errors — and an empty PMCID file stays 0 (nothing requested, nothing fetched). Verified against the live 404 (exit 1, was 0) and by tests for total failure, partial success, and the empty-file edge. Note the existing dispatch test's stub returned `None`, an artifact of the discarded-return contract; it now returns a count like the real function. **The 404 itself is an upstream blocker, not this bug** — a 5-PMCID diagnostic later showed 0/5 HTTPS successes and an identical FTP failure, exonerating the ftp→https rewrite; see *"Topic — NCBI OA packages are unreachable"*. | [Bug 117](#bug-117--acquire-download-reported-success-after-downloading-nothing) |
| B-118 | Fixed (2026-07-16, `27ed0f8`) — **expires August 2026** | High | Acquisition, `acquisition/downloader.py` (`candidate_urls`) × NCBI `oa.fcgi` | **NCBI moved its OA packages and its own API never followed, so `acquire download` failed for every paper — §6 was completely broken.** Every legacy FTP tree was relocated under `/pub/pmc/deprecated/` (NCBI readme, updated 2026-04-10) while `oa.fcgi` continues to advertise the pre-move paths. Measured 2026-07-16 on 5 PMCIDs stratified 2010→2025, all currently advertised as having packages: **0/5** returned an archive, every one a 404 with `content-type: text/html`. **The rewrite was exonerated before any change was made**: an FTP probe of the *original* advertised URL answers `550 … No such file or directory`, so both protocols fail identically, `downloader.py`'s `ftp://`→`https://` rewrite is sound, and the paths match NCBI's own committed index (`files/oa_file_list.csv`) byte-for-byte. Listing `/pub/pmc/` showed `oa_package/` gone and a `deprecated/` directory in its place, holding the same package at **7 556 375 bytes** over both FTP and HTTPS (200, `application/x-gzip`). **Fixed**: `candidate_urls()` tries the advertised URL first and the relocated one second — that order lets it self-heal if NCBI repairs its API, rather than pinning to a directory they intend to delete; only the last candidate reports failure, so an expected first miss across 1093 papers does not bury the signal. **⚠ Expires August 2026**: NCBI states the legacy files "will be removed in August 2026", after which both candidates 404 and acquisition fails loudly (B-117) — the signal to migrate to the AWS OA service (https://pmc.ncbi.nlm.nih.gov/tools/cloud/), tracked in THESIS.md. Verified end-to-end against live NCBI with the documented commands, isolated from the corpus: `download` → exit 0 + 7.2 MB archive; `unpack` → valid tar, 17 members, 1 PDF + 1 XML; `organize` → 1 PDF + 1 XML. | [Bug 118](#bug-118--ncbi-relocated-its-oa-packages-and-its-api-still-advertises-the-old-paths) |
| B-119 | Fixed (2026-07-16, `163cf91`) | High | Identity, `acquisition/downloader.py` (AWS naming) × `pdf_text_extraction/{runner.py:1009,1028, batch.py:201}` | **The AWS route would have minted a second document ID for a paper the corpus already holds.** AWS names its objects `PMC8395919.1.pdf` (`.1` = article version); taken as a filename stem that yields the document ID `PMC8395919.1`, while the same paper acquired over FTP yields `PMC8395919_dermatopathology-08-00036`. Ingest derives identity from the stem (`runner.py:1028`, `batch.py:201`), so the two routes would disagree and a resumed corpus would gain a **duplicate row, not an error**. With AWS made the default in `489e42e`, resuming an FTP-built corpus became a normal operation, so "don't mix sources" in the docs was not a fix. **Fixed** by reading the article's JATS `<self-uri content-type="pmc-pdf">` — the authoritative record of the publisher's filename, the same one the tarball carried — and reproducing `unpack`'s layout exactly (`corpus/<PMCID>/dermatopathology-08-00036.pdf` + `<PMCID>.nxml`), so `organize` yields the identical document ID from either route (verified live). The XML is fetched *before* the PDF name is finalised; an article with no authoritative mapping **fails** rather than acquiring a fabricated identifier. The `self-uri` is treated as untrusted input — URLs, absolute paths, `..` traversal, empty and non-PDF values are rejected, and a nested reference reduces to its filename only once known relative and traversal-free. Plus a shared safety net, `nlp_histo.document_id.canonical_document_id()`, at all three derivation sites: it strips a numeric version **only** immediately after `PMC<digits>` and only when the token ends there or a publisher component follows — `PMC8395919.1`→`PMC8395919`, `PMC8395919.1_derm-…`→`PMC8395919_derm-…`, while `PMC8395919_paper.v2.final` and `PMC8395919.1.2` are left alone. **The composite document ID is preserved deliberately**: 977 corpus rows, the frozen replay artifacts and the silver `case_id`s are keyed on it, so normalising to bare accessions would orphan all of it — filed as post-thesis migration debt, not a bug. Also fixed alongside: numeric version selection (v10 > v9, where a lexical `max` would serve a superseded article) and S3 pagination (a page caps at 1000 keys; the dropped object could be the PDF). Provenance — AWS keys, publisher filename, version, resulting document ID — is written to `corpus/<PMCID>/_source.json`, which `organize` ignores. **No cleanup needed**: 0 versioned IDs in either database (nlp_histo 0/977, new_local_db 0/1); the AWS work never reached ingest. | [Bug 119](#bug-119--aws-and-ftp-disagreed-about-document-identity) |

| B-120 | Fixed (2026-07-17) | Medium | Docs/UX, `workflows/knowledge.py:575` (`--dry-run`) | **`--dry-run` — the documented free way to check a paid invocation — misreported the required API keys for every profile, because the line was hardcoded.** It always printed `Env vars required: GOOGLE_API_KEY, ANTHROPIC_API_KEY` regardless of the resolved profile, while printing the true per-voter model/provider table directly above it. Actual needs: `cheap` → `GOOGLE_API_KEY, OPENAI_API_KEY` (the message **omits `OPENAI_API_KEY`, which every one of its four voters needs**, and demands an Anthropic key it never uses); `real`/`real_5` → all three (omits `OPENAI_API_KEY`); `haiku_only` → `ANTHROPIC_API_KEY` only (demands an unused Google key). So the output was wrong for **4 of 4** profiles. Impact: a supervisor following `--dry-run` provisions the wrong keys and the subsequent *paid* run dies at provider construction — time, not money, and self-limiting. Same class as [B-105](#bug-105--neither-documented-knowledge-command-could-run): an assertion on the paid-command surface that was never executed. **Fixed** by deriving the set from the resolved profile's voters (`{v.provider for v in (*l1, *l2, l3)}` → env var), with an unknown provider rendered as `<unknown provider 'x'>` rather than silently dropped. Verified against all four profiles vs. an independently computed expectation. Found while verifying README's key claim during the 2026-07-17 docs consolidation. | [Bug 120](#bug-120--dry-run-hardcoded-the-required-api-keys) |
| B-121 | Fixed (2026-07-17) | Medium | Database, `database/models.py:10,79` (`ARRAY`) × `.claude/CLAUDE.md` (Critical Patterns) | **The hierarchical query documented as a Critical Pattern raised `NotImplementedError` — the one query that makes the corpus's tree structure usable could not be run.** `path_list` was declared `Column(ARRAY(Text))` importing `ARRAY` from SQLAlchemy's **generic** namespace rather than the postgresql dialect; the generic type's `.contains()` deliberately raises `NotImplementedError: ARRAY.contains() not implemented for the base ARRAY type; please use the dialect-specific ARRAY type`. So `session.query(TextElement).filter(TextElement.path_list.contains(['Methods']))` — printed verbatim in CLAUDE.md, and the whole point of storing `path_list` as an array — failed for every caller. The column comment already said `# PostgreSQL array` and the module already imported `JSON, JSONB` from the dialect, so this was a one-word import slip, not a design choice. It survived because nothing in the pipeline queries by path: `path_list` is written by ingest and read back as a whole, so the array operators were only ever exercised by a human exploring the corpus — exactly the supervisor's use case, and untested. **Fixed** by importing `ARRAY` from `sqlalchemy.dialects.postgresql`. **DDL is identical** — both compile to `TEXT[]` (verified against the postgresql dialect), so no migration, no schema change, and the existing corpus dump restores unaffected; only the dialect type compiles containment to `@>`. Verified live against the 977-paper corpus: the documented query now returns 86 elements. The generic type does support `.any('Methods')` (`= ANY(...)`), which is why no workaround was needed for anything that already worked. Found while checking whether REPRODUCE.md's promise of "a working corpus database you can query" was true — the file showed only `count(*)`, and the first real query attempted was the documented one. | [Bug 121](#bug-121--the-documented-hierarchical-query-could-not-run) |

| B-122 | Fixed (2026-07-17) | Medium | Packaging, `requirements.txt` (was a `pip freeze` of a shared interpreter) | **`requirements.txt` was a `pip freeze` of a system-wide Python shared with unrelated projects: 404 pins, of which 186 belonged to no dependency of this project.** `pyproject.toml` declares **zero** dependencies (the wheel ships none — B-104), so this file is the *only* dependency source and every reader installs all 404. The freeze carried other projects' packages (`bcrypt`, `cleo`, `configobj`, `fastapi`, `gurobipy`, `easyocr`, `appnope`), dev tooling nothing runs (`black`, `flake8`, `mypy`, `ipython`, `debugpy`, seven unused pytest plugins), and — the largest item — **13 spaCy language models where the project loads 2**: German ×2, Spanish, French, Italian, Japanese, Korean, Polish, Portuguese, Russian, Swedish, all for an English-only corpus. `en_core_web_sm` appears only in a *code comment*; `en_core_sci_md` only as a mocked string in a test. **Fixed** by computing the dependency closure from installed metadata over the 32 top-level imports across `src`/`eval`/`scripts`/`tests`, then filtering the original file — **preserving every version pin**, because the tested versions are the thesis's provenance. Two entries static analysis could not see were added back by hand: `psycopg2-binary` (never imported — SQLAlchemy resolves it from the bare `postgresql://` URL) and `build` (§12 documents `python -m build`); `asyncpg` was correctly dropped, as the URL names no async driver. A first attempt with an *unpinned* direct list was **rejected**: pip resolved `transformers` 4.57.3 → **5.8.1**, `pandas` 2.1.4 → **3.0.3**, `torch` 2.9.1 → 2.13.0 and dropped `sentencepiece` — the suite still passed, which is precisely why "tests pass" is not sufficient evidence for a thesis whose replay must be byte-identical: an NLI major-version bump could move the grounding decisions the numbers rest on. **Verified** by installing the pruned file into three successive clean venvs until zero unpinned transitives remained (the closure initially missed docling's chunking stack — `semchunk`, `mpire`, `tree-sitter*`, `dill`, `multiprocess` — which pip then resolved at drifted versions): final state 229 pins, 0 extras, `1697 passed`, `ruff` clean, `torch 2.9.1`/`transformers 4.57.3` intact. `requirements.in` now records the direct set and why each entry exists. | [Bug 122](#bug-122--requirementstxt-was-a-freeze-of-an-unrelated-interpreter) |

| B-123 | Fixed (2026-07-17) | Medium | Reproducibility, `scripts/make_reproduction_bundle.py` (bundle scope) × `docs/REPRODUCE.md` Step 8 (E14) | **E14 — the headline generalization result REPRODUCE.md tells the reader to reproduce — could not run from the downloaded bundle: its two heldout15 inputs were never shipped.** `nlp-histo replay chapter9` needs the *related15* primer, and the bundle's file list mirrored exactly the replay's `REQUIRED_ARTIFACTS`; but REPRODUCE.md Step 8's free track also runs `E14_heldout`, which reads a *different* primer — `eval/data/map_primer_heldout15/voter_cache.json` (16 MB) and `eval/data/silver_findings_heldout15.jsonl` (1.1 MB). Neither was in the bundle nor in a clean clone, so a fresh checkout died with `voter cache not found: …/map_primer_heldout15/voter_cache.json`. Found by the user running Step 8 verbatim in a clean clone. E04 (the other Step 8 experiment) was unaffected — its inputs are shipped. **Not a cost bug:** E14 sets `strict_cache_only=True`, so a missing embedding *raises* rather than issuing a paid call (same guarantee as [B-112](#bug-112--the-replay-embedding-cache-preflight)); verified by running E14 against the shipped gemini cache — **0 cache misses, exit 0, `strict_f1_optimal = 0.7128`, gap `-0.0032`**, matching the documented values. So the caches were complete; only the two primer/silver *inputs* were absent. **Fixed** by committing the two files to git (they gzip to ~1.0 MB combined) rather than re-cutting and re-uploading the 1.2 GB bundle — the same home as the already-tracked `source_cases_related15.jsonl`, so a clone now carries them and E14 runs with the *existing* uploaded bundle unchanged. The gitignore re-includes **only** the clean `voter_cache.json`; the sibling `voter_cache.contaminated.json` (a leakage variant that must never ship) and the unused `primer.json` stay ignored. The build-script comment records that these live in git by design, so a future bundle does not redundantly carry them. Accepted trade-off: one primer (heldout15) is in git while its related15 twin remains in the bundle — inconsistent, but it avoids a 1.2 GB re-upload for 1 MB of data. | [Bug 123](#bug-123--e14s-heldout15-inputs-were-missing-from-the-reproduction-bundle) |

| B-124 | Fixed (2026-07-17) | Medium | Reproducibility, `E03/E10/E11/E12` experiments × `eval/silver/analysis/map_context.py` (`_load_map_context`) | **Four offline-replay experiments demanded an API key they never used, so they died in a keyless clean clone — the exact condition REPRODUCE.md Track A promises works ("no API key").** E03, E10, E11 and E12 call `_load_map_context("gemini", …)` without `strict_cache_only=True`, so the helper constructs a live `GeminiEmbedder(api_key)` and aborts with `GOOGLE_API_KEY not set` — even though every embedding is served from the frozen gemini cache (verified **0 cache misses** in each). Pure constructor theatre, identical in shape to [B-109](#bug-109--the-replay-needed-a-credential-it-never-used) and [B-112](#bug-112--the-replay-embedding-cache-preflight): a provider built and never called. It escaped my own Step-8 verification because I ran the experiments on the author's machine with `.env` keys present; the failure only appears when the keys are absent. Found by re-running E12 with every provider key blanked — the clean-clone condition. **E14 was already immune** (it passes `strict_cache_only=True`, which builds a keyless `NoLiveEmbedding` that raises on a miss instead of billing); E04/E13 construct no embedder. **Fixed** by adding `strict_cache_only=True` at all four call sites — the mechanism already existed, mirroring E14. The numbers are unchanged (strict-cache-only affects only cache *misses*, of which there are none): all four re-verified keyless with 0 misses and identical values (E03 retention 0.838/best 0.7160; E10 cascade 0.7160 vs Sonnet 0.7129; E11 cascade 0.7160; E12 CSV). The fix also **unblocked E10 and E11**, which were free but keyless-broken — both are now added to REPRODUCE.md Step 8. Suite 1697 green, ruff clean. | [Bug 124](#bug-124--offline-experiments-demanded-a-key-they-never-used) |

| B-125 | Fixed (2026-07-17) | Low | Reproducibility, `docs/REPRODUCE.md` (Step 8, Step 11) × shipped inputs | **E02c and E09 were reproducible for free but their inputs were not shipped, so the runbook listed them as "cannot reproduce" when in fact ~0.7 MB of data was all that was missing.** Following [B-123](#bug-123--e14s-heldout15-inputs-were-missing-from-the-reproduction-bundle) (which shipped E14's heldout15 primer), the same audit unlocked two more: **E09** (cost–quality frontier) is a pure keyless CSV re-analysis (`0` `_load_map_context`), but reads the frozen E06c/E07/E08b calibration-sweep CSVs a reader never re-runs; **E02c** (held-out final-rule provenance) is a keyless DB walk that reads the held-out per-paper summary JSONs. Both were verified free — E09 keyless → quality `0.7160@23.66` / knee `0.7067` / economy `0.5433`; E02c keyless → **100%** carry-rate across all 15 held-out papers — matching the registry. **Fixed** by committing the missing inputs to git (same choice as B-123 — no bundle re-upload): the 15 held-out summary JSONs (7.2 MB → ~0.6 MB packed) and the three frozen sweep CSVs (~52 KB). The gitignore re-includes *only* those: not the bundle's own `out/summaries/summaries` (still shipped in the archive), not the held-out `batch_handles`/`corpus_relations` intermediates, and not any other `eval/reports/` artifact. REPRODUCE.md Step 8 now lists **eight** free bundle experiments (was seven) and Step 11 **three** DB provenance experiments (was two). E01 remains separate — it needs the 27 rubric PDFs, a licensing question, tracked in the Decisions log. | [Bug 125](#bug-125--e02c-and-e09-were-free-but-their-inputs-were-unshipped) |

| B-126 | Fixed (2026-07-17) | Low | Eval, `E01_doc_extraction/flatten_to_csv.py:120` | **E01's `flatten_to_csv` wrote its CSV correctly and then crashed on the success message — a `ValueError` traceback on exit 1 despite a complete, correct output file.** The final line does `csv_path.relative_to(_REPO_ROOT)`, but `_REPO_ROOT` is absolute while `--csv-out` (or the default derived from a relative `--json`) is a **relative** path — and `Path("eval/reports/…").relative_to("/abs/repo")` raises. So the natural invocation a reader copies from the docs (`flatten_to_csv --json eval/reports/…_PR.json`) printed a stack trace and returned non-zero, even though the reshaped CSV was already on disk and byte-identical to the frozen one. Surfaced while wiring E01 into REPRODUCE.md Step 8 as the PDF-free, report-level reproduction of the doc-extraction rubric. **Fixed** by resolving the path before `relative_to` and falling back to the path as-given when it is outside the repo (`try/except ValueError`). Verified: both the relative `--csv-out` and default invocations now exit 0, and the regenerated CSV is byte-identical to the committed `figtable_extraction_sweep_rerun_27pdf_20260604_PR.csv` (winner var18: tables 40→83.8 %, figures 84 %). ruff clean. | [Bug 126](#bug-126--flatten_to_csv-crashed-on-its-own-success-message) |

Add new rows here when you discover something. Bump the ID monotonically (`B-051`, `B-052`, …). Put the long write-up in a new `## Bug N — …` section below.

---

## Bug 65 — Batch runner `_process_level` crashes on None voter output

### Status / Severity / Surface
Fixed (2026-05-27) · High · Summarisation, batch runner
(`pipeline/stages/summarization/batch/runner.py::_process_level`).

### Symptom
Batch advance crashes mid-paper with:

```
File "pipeline/stages/summarization/batch/runner.py", line 1117, in _process_level
    for c in _extract_claims(v)
File "pipeline/stages/summarization/agreement/embedding.py", line 328, in _claims
    return [f.claim for f in output.findings]
AttributeError: 'NoneType' object has no attribute 'findings'
```

Hit while running batch mode on the 15 ILP papers under the new `haiku_only`
profile (single L1 voter; this conversation, 2026-05-27).

### Evidence
- `_process_level` builds `chunk_voters[chunk_id]` as a length-N list aligned
  to the original voter spec (`voters_full: list[AuditableSummary | None] =
  [None] * len(current_voters)`, runner.py:1080). Voters whose batch result
  was missing or unparseable stay as `None`.
- Pass 2 (pre-embedding, runner.py:1113-1118 pre-fix) iterates the full
  voters list to collect unique claim texts:
  ```python
  all_texts = list({
      c
      for voters in chunk_voters.values()
      for v in voters
      for c in _extract_claims(v)   # ← _extract_claims(None) crashes
  })
  ```
  `agreement/embedding.py::_claims` is `[f.claim for f in output.findings]` —
  no None-guard.
- Pass 3 (agreement scoring, runner.py:1190) correctly handles Nones via
  `survivor_indices = [i for i, v in enumerate(voters_full) if v is not None]`.
  Pass 2 and Pass 3 are inconsistent — Pass 2 was missing the same guard.
- **Latent for `real` profile** (3 L1 voters; for the crash to fire all three
  would need to return None on the same chunk — rare in production). **Certain
  for `haiku_only` profile** (N=1 voter; any single parse failure → entire
  `voters_full = [None]` → Pass 2 crashes).

### Fix
One-line None filter in the Pass 2 comprehension to mirror Pass 3's
`survivor_indices` semantics:

```python
all_texts = list({
    c
    for voters in chunk_voters.values()
    for v in voters
    if v is not None                # ← added
    for c in _extract_claims(v)
})
```

All-None chunks then fall through to Pass 3's existing `if not voters:
escalated.append(chunk_id)` path (runner.py:1193-1196), which routes the
chunk up the cascade. For `haiku_only` the "escalation" lands at L2/L3 of the
same Haiku model — the dead-code path becomes the lifeline when a single L1
batch result fails to parse.

### Verification
Two regression tests in
`tests/summarization/test_b065_none_voter_handling.py`:

1. `test_advance_does_not_crash_when_lone_l1_voter_returns_none` — builds
   a `haiku_only`-shape runner (N=1 L1 voter), injects an empty-content
   synthetic batch result, calls `advance()`. Pre-fix this raises
   `AttributeError`. Post-fix the chunk routes to L2 via Pass 3's all-None
   branch; `advanced.l2_chunk_ids == ["C0"]`, `advanced.phase ==
   BatchPhase.L2_SUBMITTED`.
2. `test_advance_handles_mixed_none_and_valid_voters_in_real_profile_shape`
   — `real`-shape runner (3 L1 voters), one voter returns empty content,
   other two return identical valid `AuditableSummary` JSON. Pre-fix Pass 2
   would crash on the None slot before reaching agreement; post-fix Pass 2
   collects claims from the 2 survivors, Pass 3 emits the existing
   voter-count-regression warning (B-019/B-020 guard) and KEEPs the chunk
   since the surviving voters agree. `advanced.finalized["C0"]` confirmed.

Both tests use `monkeypatch.setattr(runner_module, "submit_level", ...)`
to stub out the provider call after escalation, isolating the regression
to `_process_level`'s None handling.

---

## Bug 64 — JSON embedding cache rewrite + vectorised cosine

### Status / Severity / Surface
Fixed (2026-05-24) · Medium · Eval, silver embedding cache
(`eval/silver/matcher.py`, `eval/silver/map_theta_sweep.py`).

### Symptom
The MAP θ sweep (`map_theta_sweep sweep`) spent most of its wall-time **not** on
API calls: ~20 s gaps between embedding fetches even when only 4–14 new vectors
were fetched, and ~14 s/cell of pure compute at high θ. A 56-cell sweep dragged
for hours.

### Evidence
- `embedding_cache_gemini.json` reached **720 MB** (17,827 × 3072-dim gemini
  vectors as JSON text). `EmbeddingCache.save()` does
  `path.write_text(json.dumps(...))` — a full rewrite — and the sweep calls
  `save()` after every ~100-text batch (`_prewarm_agreement_cache:622`), after
  each `get_embeddings` miss-fill (`matcher.py:175`), and in
  `_make_cached_embed_fn:645`. Hundreds of full 720 MB rewrites ⇒ O(N²).
- `compute_sim_matrix` built the silver↔pipeline similarity with a Python triple
  loop over `_cosine` (3072-dim) × 273 cases × 2 evals × 56 cells. The agreement
  scorer was already numpy; only the matcher was pure-Python.

### Fix
- New `SQLiteEmbeddingCache` with the **same** `get`/`set`/`save`/`__len__`
  interface: `set()` = `INSERT OR REPLACE` in an open transaction, `save()` =
  `commit()` (so the existing per-batch `save()` calls become cheap commits, not
  rewrites); `PRAGMA journal_mode=WAL` + `synchronous=NORMAL`; vectors stored as
  float32 BLOBs (720 MB → **219 MB**). Same
  `sha256(model\0 text.lower().strip())` key, so JSON and SQLite are
  key-compatible — no caller-loop changes.
- `make_embedding_cache()` factory + `NLP_HISTO_EMBEDDING_CACHE_BACKEND`
  (`sqlite` default, `json` fallback); all eight eval cache-construction sites
  routed through it.
- `import_json_cache_to_sqlite()` + `scripts/import_embedding_cache_sqlite.py`:
  idempotent, non-destructive JSON→SQLite import (copies hash keys verbatim,
  prints rows imported / skipped / dim distribution).
- `compute_sim_matrix` vectorised with numpy (one BLAS matmul; float64 to mirror
  `agreement/embedding.py`).

### Verification
`eval/silver/tests/test_embedding_cache_sqlite.py` (11 cases) + existing
`test_matcher.py` (12 cases incl. cosine/match) pass. Real import: 17,827 rows,
all 3072-dim, 720 MB JSON → 219 MB SQLite, idempotent on re-run.
**Caveat:** float32 storage changes cosine by ~1e-6 vs the float64 JSON values —
immaterial for matching, but SQLite results are not bit-identical to JSON.

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
`media_cropper.py::crop()` takes a new parameter
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

[`pipeline/stages/summarization/current_stages/group_stage.py:57`](../src/nlp_histo/pipeline/stages/knowledge_extraction/stages/group_stage.py#L57)
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
[`_should_compare_cross_paper`](../src/nlp_histo/pipeline/stages/knowledge_extraction/stages/corpus_relate.py)
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
[`pipeline/stages/pdf_text_extraction/config.py`](../src/nlp_histo/pipeline/stages/pdf_text_extraction/config.py):

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
Summarisation, `BatchKnowledgeExtractionRunner`.

### Symptom

`scripts/run_paper.py` defaults to batch mode. Batched production runs since
late April 2026 were quietly skipping six runner-level features that were
present on `KnowledgeExtractionRunner.process()`:

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

Copied the missing helpers verbatim from `KnowledgeExtractionRunner` into
`BatchKnowledgeExtractionRunner` (`_replace_verbatim_from_db`,
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

### Why not refactor `KnowledgeExtractionRunner` to share code?

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
already-extracted `non_groupable_reason`. Both `KnowledgeExtractionRunner` and
`BatchKnowledgeExtractionRunner` now thin-wrap each (1-3 line forwards).
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
Summarisation, `KnowledgeExtractionRunner`.

### Symptom

`KnowledgeExtractionRunner._load_result` returned any on-disk
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

* Added [`KnowledgeExtractionRunner._pipeline_config_hash()`](../src/nlp_histo/pipeline/stages/knowledge_extraction/runner.py)
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

`KnowledgeExtractionRunner.process_batch` computed
`n_ok = sum(1 for r in results if r["status"] == "success")`,
`n_err = sum(... == "error")`, and
`n_skip = len(results) - n_ok - n_err`. But `_load_result` returned the
cached dict with `status="success"` (the value stamped by `_save_result`
at the *original* run's success path) — cached papers counted in `n_ok`
and `n_skip` was structurally 0. The summary log ("Batch complete: X
ok / 0 skipped (cached) / Y errors") therefore always undercounted the
real fresh-work figure and reported zero skips.

### Fix

* `_load_result` ([`runner.py:1717`](../src/nlp_histo/pipeline/stages/knowledge_extraction/runner.py))
  now mutates the in-memory dict it returns: `data["status"] = "skipped"`
  immediately before the `return`. The on-disk JSON is **not** rewritten
  — the file still says `"success"` because that run *did* succeed.
  `"skipped"` is purely an in-memory marker on the caller's copy
  describing how the value was obtained on this call.
* `process_batch` ([`runner.py:793`](../src/nlp_histo/pipeline/stages/knowledge_extraction/runner.py))
  counts the new key explicitly:
  `n_skip = sum(1 for r in results if r["status"] == "skipped")`.
* Three downstream consumers updated to treat `"skipped"` as equivalent
  to `"success"` for "this paper has a complete result on disk" gating:
  * ``scripts/summarize_paper.py:41`` (removed) —
    print branch.
  * ``scripts/run_single_doc.py:69, 90`` (removed) —
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
Summarisation, `KnowledgeExtractionRunner`.

### Symptom

The runner stored every paper's intermediate state on `self`:
[`runner.py:288-302`](../src/nlp_histo/pipeline/stages/knowledge_extraction/runner.py)
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
([`runner.py:775-794`](../src/nlp_histo/pipeline/stages/knowledge_extraction/runner.py)).
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

[`components/artifact_filter.py:59`](../src/nlp_histo/pipeline/stages/pdf_text_extraction/components/artifact_filter.py)
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
[`parsers/layout_utils.filter_artifacts`](../src/nlp_histo/parsers/layout_utils.py)
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
* [`docs/REPOSITORY_GUIDE.md`](readmes/other_readmes/REPOSITORY_GUIDE.md) — removed the
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
[`components/two_pass_extractor.py:382-398`](../src/nlp_histo/pipeline/stages/pdf_text_extraction/components/two_pass_extractor.py)
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
[`scripts/templates/pipeline_batch_index.html.jinja2:276`](../scripts/inspect/templates/pipeline_batch_index.html.jinja2)
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
[`pipeline_batch_index.html.jinja2:194`](../scripts/inspect/templates/pipeline_batch_index.html.jinja2)
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
[`pipeline/stages/summarization/models.py`](../src/nlp_histo/pipeline/stages/knowledge_extraction/models.py)
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
   [`SumMapFinding`](../src/nlp_histo/database/models.py) via Alembic
   [`0011_add_raw_llm_columns_to_sum_map_findings.py`](../alembic/versions/0011_add_raw_llm_columns_to_sum_map_findings.py).
3. **Plumbing.** `KnowledgeExtractionRunner._persist_map_findings`
   ([`runner.py:1127`](../src/nlp_histo/pipeline/stages/knowledge_extraction/runner.py))
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
`KnowledgeExtractionRunner(db=db_conn)`. The batch path was simply left behind
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

`provenance/validator.py:27` has an identical regex (with capture groups) for the cross-document equality check. Same vulnerability; same fix.

### Fix

Relaxed PMC token in both regexes:

```python
# schema_validator.py:23
_CITATION_RE = re.compile(r"^S\d+\|PMC[\w\-]+\|\d+$")

# provenance/validator.py:27
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

OpenAI accepted the first model that appeared on line 1, rejected every subsequent request whose model didn't match. The batch ended with status=`failed` and `output_file_id=None`. `OpenAIBatchProvider.check()` correctly translates this to `job.status='failed'`, but `BatchKnowledgeExtractionRunner` simply skips failed jobs — no surfacing in the cost report, no exception, no warning beyond the in-method log line.

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

* **Status:** Superseded (2026-05-15) by B-049 — `_split_by_direction` no longer folds into the largest bin, so the `:117-120` tie-break no longer exists.
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

## Bug 36 — `GroundingFilter` / `RelateStage` model/batch/device not exposed via `KnowledgeExtractionConfig`

### Status / Severity / Surface

Fixed (2026-05-15) · Low · Summarisation, NLI helpers config surface.

### Symptom

Cannot switch the grounding or relate NLI model to a GPU build, a
different checkpoint, or a tuned batch size without editing the helper
module — `KnowledgeExtractionConfig` has no field for it.

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

The model / batch / device knobs predate `KnowledgeExtractionConfig`. When the
config dataclass was introduced, only the calibration-relevant thresholds
were lifted into it.

### Fix

Externalised NLI configuration to `configs/nli_models.yaml` with `pipeline/stages/summarization/nli_config.py:get_active_spec()` as the loader. `helpers/grounding_filter.py:72-78` and `current_stages/relate_stage.py:52-67` both consume `model_name`, `batch_size`, and `device` from the active spec rather than module defaults.

Trade-off vs. lifting fields into `KnowledgeExtractionConfig`: the YAML path keeps NLI-specific knobs out of the calibration-thresholds dataclass and lets a corpus run pin a specific NLI build alongside model versions, without re-running Python config construction. If a future need to override per-run from Python emerges, layer it on top — pass `model_name=` through `runner.py:251/258` and let it take precedence over the YAML.

### Verification

* `grep -n "get_active_spec\|nli_models.yaml" pipeline/stages/summarization/` confirms both `grounding_filter.py` and `relate_stage.py` read from the YAML.
* Switching `configs/nli_models.yaml` `active:` key changes the model used by both stages at next run; no Python edits required.

---

## Bug 37 — `NormalizeStage.extra_synonyms` not exposed via `KnowledgeExtractionConfig`

### Status / Severity / Surface

Fixed (2026-05-15) · Low · Summarisation, normalize stage.

### Resolution

Added `NormalizeConfig` dataclass to
`pipeline/stages/summarization/config.py` with one field,
`extra_synonyms: dict[str, str] | None = None`, and wired it through
`KnowledgeExtractionConfig.normalize`. Both `KnowledgeExtractionRunner` (sync) and
`BatchKnowledgeExtractionRunner` now construct `NormalizeStage(extra_synonyms=cfg.normalize.extra_synonyms)`.

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
`KnowledgeExtractionConfig` field for them; overriding the curated
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
`KnowledgeExtractionConfig.normalize_extra_synonyms` field.

### Verification

Pending.

---

## Bug 38 — `KnowledgeExtractionRunner.load_paper_from_db` bypasses scispaCy singleton

### Status / Severity / Surface

Fixed (2026-05-15) · Medium · `pipeline/stages/summarization/runner.py` (`load_paper_from_db`) — now routes through the process-wide `get_small_nlp('en_core_sci_sm')` singleton (`umls_resources.py`), so the model loads once. The symptom below describes the original (pre-fix) behavior.

### Symptom

Every call to `KnowledgeExtractionRunner.load_paper_from_db(pmcid)` instantiates a fresh `en_core_sci_sm` spaCy pipeline:

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

After fix, a smoke run of `KnowledgeExtractionRunner.process_batch([...])` with `tracemalloc` should show one scispaCy load event in process lifetime regardless of paper count, not one per paper. Easy to assert in a unit test by mocking `spacy.load` and confirming exactly one call.

### Follow-up

* B-029 covers the sibling site in `pipeline/stages/pdf_text_extraction/runner.py:199`. Both fixes should land together so an audit of `grep -n "spacy.load" pipeline/` returns zero hits outside `umls_resources.py`.

---

## Bug 39 — `load_paper_from_db` orders by `position_in_section` only, interleaves sections

### Status / Severity / Surface

Fixed (2026-05-15) · High · `pipeline/stages/summarization/runner.py:912-916` (`load_paper_from_db`).

### Symptom

`KnowledgeExtractionRunnerr.load_paper_from_db` queries `TextElement` rows with only one ORDER BY column:

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

* Cataloged as Issue 5 in [MAP_PROMPT_AUDIT.md](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-5--scopescope_parsed-is-llm-set-but-trivially-derivable-low).
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

* [MAP_PROMPT_AUDIT.md Issue 8](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-8--directionmaybe-single-occurrence-low) — single observed occurrence of `direction="maybe"`. Audit said defer; we shipped anyway because the fix is a 5-line dict + branch matching the well-trusted `_RELATION_TYPE_ALIASES` pattern.

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

* [MAP_PROMPT_AUDIT.md Issue 6](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-6--directionabsent-vs-directionnegative-ambiguity-in-expression-contexts-low) — flagged as a latent split before downstream measurement could quantify it.

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

* [MAP_PROMPT_AUDIT.md Issue 7](readmes/other_readmes/MAP_PROMPT_AUDIT.md#issue-7--ruletype-is-title-case-diagnosticprognosticmanagement-everything-else-lowercase-low) — flagged as a latent inconsistency, deferred until the optional RULE block was next exercised.

### Diagnosis

The audit deferred this because the optional REDUCE+RULES block is off by default — no production cost. We lifted the deferral once the rest of the audit was being addressed in the same pass; the casing migration is mechanical and the back-compat shim is a single `mode="before"` validator.

### Fix

* `pipeline/stages/summarization/models.py`
  * `Rule.type: Literal["diagnostic", "prognostic", "management"]`.
  * New `Rule._lowercase_type` `field_validator(mode="before")` so any legacy Title-Case payload (cached LLM output, hand-authored test fixtures) round-trips cleanly.
  * `RuleCounts` field names lowercased to `diagnostic`, `prognostic`, `management`.
* `pipeline/stages/summarization/helpers/grounding_filter.py` — `_recompute_audit` reads `counts["diagnostic"]` etc., matching the new `Rule.type` casing.

*Later note: the RULE/REDUCE block was retired wholesale in `d98a310`, so `Rule.type`, `RuleCounts`, and `_recompute_audit` no longer exist in the tree — this fix was not reverted; the feature it touched was deleted.*
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
  `KnowledgeExtractionRunnerr._pipeline_config_hash` and
  `BatchKnowledgeExtractionRunner._pipeline_config_hash` (same pattern as B-049's
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
actually mirrors `KnowledgeExtractionRunner.load_paper_from_db`
(`pipeline/stages/summarization/runner.py:940`) as its docstring claims.

### Verification

* Manual: `n_sentences=100`, `chunk_size=10`, `chunk_overlap=2`,
  `stride=8`. `range(0, 100, 8) = 13` starts. Per-chunk counts
  `[10, 10, …, 10, 4]`. Total = 124. Average = 124/13 ≈ 9.54.
  Pre-fix returned ~7.69; post-fix returns 9.54.
* Regression test:
  [`tests/test_estimate_selection_cost.py`](../tests/scripts/test_estimate_selection_cost.py)
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
(`KnowledgeExtractionRunner.load_paper_from_db` direct load). Missed by the
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
Summarisation, `BatchKnowledgeExtractionRunner.finalize()` → `sum_map_findings` DB persistence.

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
findings produced. Discovered during the [B-005 end-to-end verification TODO](THESIS.md#todos)
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
  result-cache short-circuit at `BatchKnowledgeExtractionRunner.finalize()` L472
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
* Add a regression test that calls `BatchKnowledgeExtractionRunner.finalize()`
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
* `legacy/scripts/probes/diagnose_b055.py` replays every on-disk handle in
  `out/summaries/batch_handles.prepatch/` through the now-loud persist path
  against a throwaway `pipeline_run` (deleted via FK CASCADE) — **zero LLM
  cost** since the bug is entirely post-MAP. It reports PASS/FAIL with the
  exact DB exception per paper, doubling as the back-population tool for
  Fix-step 3 (runs 38–47) once the root cause is known.

### Diagnostic replay (2026-05-23) — hypothesis #2 ruled out

`python legacy/scripts/probes/diagnose_b055.py` replayed **all 26** on-disk handles through
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
Summarisation, `BatchKnowledgeExtractionRunner.finalize()` → `sum_map_voter_outputs`
DB persistence.

### Symptom

`BatchKnowledgeExtractionRunner.finalize()` does not buffer per-voter
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

`BatchKnowledgeExtractionRunner.finalize()` reconstructs chunk-level
`AuditableSummary` objects from `handle.finalized.values()`
(`batch/runner.py:483-485`) and proceeds directly into grounding,
filesystem persistence, and `_persist_map_findings`. The per-level
collectors (`_collect_l1`, `_collect_l2`, `_collect_l3` at L1332+) hold
raw provider outputs in `handle.l*_raw` but discard them after the
winning `AuditableSummary` is materialised; `MapStage._buffer_voter_outputs`
is never called because the batch path never instantiates `MapStage`.
`BatchKnowledgeExtractionRunner` exposes `_persist_map_findings` /
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
2. Add `_persist_voter_outputs(db_id, pmcid)` to `BatchKnowledgeExtractionRunner`
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

**Status / Severity / Surface:** Fixed (2026-05-23) / High / Summarisation, MAP
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

*Constructor-site audit:* `grep -rn "KnowledgeExtractionRunner(\|BatchKnowledgeExtractionRunner("`
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

## Bug 66 — `synonyms.yaml` never loaded (loader looks in the wrong directory)

### Status / Severity / Surface
Fixed (2026-05-31) / Low — latent, no functional impact today / Summarisation, NORMALIZE entity normalization.

### Symptom
The curated synonym dictionary at `pipeline/stages/summarization/synonyms.yaml` has no effect at runtime. Entity normalization behaves as if only the hardcoded `_SYNONYMS_FALLBACK` dict exists, so editing the YAML — the intended clinician-facing override knob — changes nothing.

### Evidence
`normalize_stage.py` lives in the `current_stages/` subpackage and computes the YAML path relative to its own file:

```python
_SYNONYMS_YAML = Path(__file__).parent / "synonyms.yaml"
```

`__file__` is `.../summarization/current_stages/normalize_stage.py`, so the path resolves to `.../summarization/current_stages/synonyms.yaml` — which does not exist. The actual file is one level up at `.../summarization/synonyms.yaml`. Verified 2026-05-31:

* `current_stages/synonyms.yaml` → does not exist (`find` returns only the package-root copy).
* `_load_synonyms()` catches the `FileNotFoundError` and returns `dict(_SYNONYMS_FALLBACK)` (the hardcoded dict), logging only at debug level.

*Later note: a subsequent refactor moved `synonyms.yaml` into `.../knowledge_extraction/entities/`, loads it via `importlib.resources`, and removed the hardcoded fallback (the YAML is now the single source of truth) — so the "chose fixing the path over moving the YAML" decision and the `_SYNONYMS_FALLBACK` symbol are both historical.*
* Key comparison: YAML has 48 keys, the fallback has 48 keys, and 0 keys are in the YAML but not the fallback — they are currently in sync, which is why there is no behavioural symptom.

### Diagnosis
Introduced by the `current_stages/` reorganisation: `normalize_stage.py` moved into the subpackage but the `Path(__file__).parent` reference was not updated and the YAML stayed at the package root (the CLAUDE.md file map still lists it there). `_load_synonyms()`'s silent `FileNotFoundError` → fallback branch masks the failure as long as the two copies stay in sync; they will diverge the first time someone edits the YAML expecting it to take effect.

### Fix
Applied 2026-05-31 — changed the path to the package root:

```python
_SYNONYMS_YAML = Path(__file__).resolve().parents[1] / "synonyms.yaml"
```

(Chose fixing the path over moving the YAML — the package root is the documented home per the CLAUDE.md file map.) Follow-up worth doing: log at INFO which source actually loaded (YAML vs hardcoded fallback) so a future silent fallback is visible rather than masked.

### Verification
`_load_synonyms()` now resolves to `pipeline/stages/summarization/synonyms.yaml` (exists) and returns the YAML's 48 entries instead of hitting the `FileNotFoundError` fallback; spot-checked `ber-h2 → CD30`, `r-chop → R-CHOP`, `mib-1 → Ki-67`. The loaded dict equals `_SYNONYMS_FALLBACK` today (the two were in sync, 48/48 keys), so behaviour is unchanged — but YAML edits now take effect, restoring the editable-override design. Full divergence test (add a YAML-only key, assert `normalize_entity()` resolves it) deferred until the first real curation edit.

## Bug 67 — Supplementary PDFs ingested as separate documents (no main-PDF selection)

### Status / Severity / Surface
Mitigated (2026-06-01) — data cleaned, root-cause code unchanged / Medium — corpus-accounting error + one paper misrepresented; knowledge-extraction eval subsets unaffected / Acquisition (`file-selector/`), PDF extraction (`PipelineRunner.run_batch`), corpus accounting (§4.2).

### Symptom
The document database held 943 rows described in §4.2 as "ingested papers", but the rows are PDF *files*, not papers. PMC packages that ship more than one PDF (article main text + supplementary material) contributed multiple rows: three papers were double-counted (article + supplement) and one paper (PMC11863978) was represented *only* by a supplementary file, its main text absent.

### Evidence
* All 943 `documents.pmcid` values were full filename stems (`PMC10082646_main`, `PMC12272590_mmc1`), never clean `PMC\d+` — 0/943 matched `^PMC\d+$`.
* The 1132 organized PDFs span 1093 distinct PMC ids; 23 packages ship >1 PDF (62 files total).
* Exhaustive check of all 23 multi-PDF packages: every `-s00N`-named supplement failed extraction and dropped; four text-heavy `mmc` supplements survived into the DB —
  * `PMC12272590_mmc1` (7 text elements) alongside `PMC12272590_main` (105) — double-count
  * `PMC7508550_mmc1` (17) alongside `PMC7508550_main` (34) — double-count
  * `PMC9239710_mmc1` (85) alongside `PMC9239710_main` (45) — double-count
  * `PMC11863978_mmc6` (15) — the **only** row for this paper; `PMC11863978_main` was not in the run set.
* None of the four contaminated ids appear in any `configs/paper_selection/*` or `eval/` selection, so the Chapters 6–10 knowledge-extraction evaluation is unaffected.

### Diagnosis
There is no main-PDF selection at any stage. `file-selector/tarball_extractor.py` extracts every `.pdf` member of a package; `file-selector/pdf_organizer.py` copies each as `{PMCID}_{origname}.pdf`; `PipelineRunner.run_batch` (`runner.py:1029`) and `ParallelBatchRunner.run` (`batch.py:204`) glob all `*.pdf` and default `pmcid = pdf_path.stem` — the full filename. The DB idempotency guard (`db_ingester.py:79`, skip if `pmcid` exists) only collapses *identical* stems, so differently-named PDFs of the same paper dodge it and each become a separate `Document` row. (The full-stem pmcid convention is the same one noted in B-019.) The production DB was built from `files/_reextract/all/` — a curated 943-symlink set that itself contained `mmc` files and, for PMC11863978, only `mmc6`.

### Mitigation
Applied 2026-06-01 (data only — root cause not fixed). Deleted the four supplementary `Document` rows. The ORM delete path raises `psycopg2.errors.NotNullViolation` (the `TextElement → entities` backref has no ORM cascade, so SQLAlchemy nullifies the NOT-NULL `entities.text_element_id` instead of letting the DB cascade), so the delete was issued as raw SQL and Postgres' `ON DELETE CASCADE` removed the children:

```sql
DELETE FROM documents WHERE pmcid IN
  ('PMC11863978_mmc6','PMC12272590_mmc1','PMC7508550_mmc1','PMC9239710_mmc1');
```

Result: 943 → **939 documents = 939 distinct papers**, all article main text, zero supplementary/duplicate rows. PMC11863978 drops from the corpus (its main was never ingested); its `_main.pdf` parses and is cached (`out/docling_full/PMC11863978_main_*_layout.json`), so it can be re-ingested to reach 940 if wanted (deferred — THESIS.md TODO).

This is a *mitigation*, not a *fix*: the acquisition/extraction code still has no main-PDF selection, so re-running the pipeline over `files/organized_pdfs/` would reintroduce the contamination. The code-level guard (extract clean `PMC\d+`, select one main PDF per paper) is tracked in THESIS.md TODOs.

### Verification
Post-delete queries: `TOTAL_DOCS=939`, `DISTINCT_PAPERS=939`, `DUP_BASE_ROWS=0`, `SUPP_REMAINING=[]`. Orphan checks (text_elements / entities / figures / tables with a dangling parent) all return 0; `documents.id IN (210,267,662,870)` returns 0. §4.2 corpus accounting to be updated to report 939 distinct papers.

## Bug 68 — Eval paper PMC11863705 absent from ingested corpus; stale thesis counts

### Status / Severity / Surface
Observed (2026-06-03) / Medium — one document-extraction eval paper is not in the corpus it is documented as sampled from, and every thesis corpus count is stale / Corpus accounting (§4.2–4.3), eval-set provenance (`eval/pdfs/`), DB ingestion scope.

### Symptom
Checking whether the 28-PDF document-extraction eval set (`eval/pdfs/`) is contained in the ingested `Document` corpus: **27 of 28 are present; PMC11863705 is not.** The thesis (`docs/thesis/05_corpus.md §4.3`) states the 28 were a "random stratified sample from the … ingested corpus" — an invariant this violates.

### Evidence
* Live DB: `SELECT count(*) FROM documents` → **977** rows, 977 distinct `PMC\d+` prefixes (the `pmcid` column stores full filename stems, e.g. `PMC10047213_dermatopathology-10-00018`, per B-019/B-067).
* Matching the 28 eval stems (filename without `.pdf`) against `documents.pmcid`: 27 exact hits; only `PMC11863705_main` misses, and no other DB row shares its `PMC11863705` prefix.
* PMC11863705 **is** downloaded and organized: `files/organized_pdfs/PMC11863705_main.pdf` (+ `_mmc1.pdf`), `files/organized_xmls/PMC11863705.nxml`, and it is in `files/target_pmc_ids.txt`.
* It was processed by the **eval harness** (`eval/out/text/PMC11863705_main_text.txt`, `eval/pdfs/PMC11863705_main.pdf`) — but has no artifacts in the production `out/` (`out/text/`, `out/docling_full/` have nothing for it) and is **not** present in `files/_reextract/all/` (the 943-symlink set that B-067 documents as the DB's source).
* A stale `out/run_metadata/PMC11863705_main_stats.json` from the 2026-05-23 943-doc run (`run_20260523T222439Z_f817c5fc`, status `ok`, DB enabled) lists `pdf_path: files/_reextract/all/PMC11863705_main.pdf`, but that symlink no longer exists and the row is not in the current DB — i.e. the DB was rebuilt from a later set that excludes it.
* PMC11863705 is 1 of **116** distinct PMC ids that are present in `organized_pdfs/` (1093 distinct) but absent from the DB (977 distinct). No DB row lacks a corresponding download (`db_pmc − organized = 0`).
* Count drift: the live DB is **977**, whereas `docs/thesis/05_corpus.md` cites **943** ("successfully ingested"), **903** ("ingested non-calibration"), and B-067's mitigation landed on **939**. None match 977 — the corpus has been re-ingested since B-067 (2026-06-01) and the thesis figures are stale.

### Why PMC11863705 was excluded — confirmed root cause
**It ships supplementary material.** `files/organized_pdfs/` holds two PDFs for it — `PMC11863705_main.pdf` (9 pp) and `PMC11863705_mmc1.pdf` (1 pp) — so it is a *multi-PDF package*. The 977-doc DB was built from the **single-PDF candidate pool** (the `--single-pdf-only` pool of 1070 papers used in `scripts/eval/pdf_page_counts.py` → `reports/page_caps.tex`, "$n=1070$ papers"). Verified: **all 23 multi-PDF packages are absent from the DB (0/23 ingested)**; PMC11863705 is one of them.

The other two candidate-explanations are ruled out:
* **Not a page-count exclusion.** Main is 9 pp — far under the 30-pp cap (`reports/page_caps.tex` selects cap $\le30$ → 981 papers). Page count did not exclude it.
* **Not blacklisted.** Neither `out/failed_pdfs_blacklist.json` nor `eval/blacklist.json` lists it; the eval-harness run parsed it cleanly (39 text rows).

Full decomposition of the **116** downloaded-but-not-ingested ids (1093 organized − 977 DB):
* **23** — multi-PDF packages (have supplementary PDFs), excluded by the single-PDF-only pool. ← PMC11863705 is here.
* **89** — single-PDF papers whose main exceeds the 30-pp cap.
* **4** — single-PDF, under the cap, excluded for another reason (e.g. extraction failure; `1070 single − 89 over-cap = 981; 981 − 977 DB = 4`). PMC2680278/PMC2675007/PMC2941727/PMC2916223 are the under-cap four.

### Diagnosis
Two independent issues surfaced by the same check:
1. **Eval-set / corpus mismatch.** The 28-PDF eval set was assembled (and re-run) under the eval harness's own `eval/out/` pipeline, whose input list still contains multi-PDF packages, whereas the production DB ingestion deliberately used the single-PDF-only candidate pool. PMC11863705 is in the eval list but, as a multi-PDF package, was filtered out of the ingestion set — so §4.3's "sampled from the ingested corpus" does not hold for it. This is *intended* corpus-selection behaviour leaking into the eval set, not an extraction failure.
2. **Stale corpus counts.** The production DB has been rebuilt to 977 distinct papers since B-067's 939 mitigation (the rebuild is what cleanly removed all 23 multi-PDF packages, superseding B-067's manual 4-row delete), but no thesis count was updated. 116 downloaded papers remain un-ingested by design + cap; the 977-vs-1093 gap is undocumented.

### Fix
Not yet applied (Observed). Options, to decide deliberately:
* **Reconcile counts first.** Re-derive §4.2–4.3 from the live DB: report 977 (or the intended ingestion-set size), the downloaded-but-not-ingested count (116), and the exact selection basis for the 28-PDF set.
* **For PMC11863705 specifically**, pick one and state it: (a) re-ingest it so the 28 are genuinely ⊆ corpus (its main PDF parses; would make the DB 978), or (b) keep the thesis wording honest — describe the 28 as drawn from the downloaded/eval-processed corpus rather than the DB-ingested corpus, and footnote that one eval paper is not in the production DB.
* Root-cause (shared with B-067): there is no single canonical "ingested corpus" manifest; the eval set and the DB set are defined by different inputs (`eval/pdfs/` list vs `files/_reextract/all/`). A single source-of-truth PMC-id manifest would prevent the divergence.

### Verification
Membership re-check after reconciliation:
```bash
ls eval/pdfs/*.pdf | sed -E 's#.*/(PMC[0-9]+_[^/]*)\.pdf#\1#' | sort > /tmp/eval28_stems.txt
python3 -c "from database import get_db_connection,Document; db=get_db_connection();
import sys
with db.session_scope() as s: p={x for (x,) in s.query(Document.pmcid).all()}
miss=[l.strip() for l in open('/tmp/eval28_stems.txt') if l.strip() not in p]
print('eval stems missing from DB:', miss)"
```
Target: empty `miss` list (if option (a)), or a documented, footnoted exception (if option (b)).

**Does dropping PMC11863705 change the document-extraction experiment winner? No.**
Checked 2026-06-03 with `scripts/eval/recheck_drop_pmcid.py PMC11863705` — recomputes
every variant's strict F1 from the surviving labels (`eval/annotations/<variant>/*.json`)
+ `eval/ground_truth.csv`, since the sweep crops (`out/sweeps/`) were deleted. The metric is
a pure function of labels + GT (the scorer skips unlabelled emitted crops), so `emitted =
set(labels)`; the table-label file is picked from the variant name (docling vs full).

* **Reproduction gate:** the full-28 reconstruction reproduces `reports/stage7_PR.md`
  table strict-F1 for **all 32 variants with 0 mismatches** (variant 18 = 74.4%, 01_docling
  = 34.6%, …) — proving both reconstruction assumptions exact.
* **Impact of removing PMC11863705:** the **winner is unchanged** — `18_hybrid_best_family_
  fixes_footnote_expand_1_2` (HYBRID@0.99 + footnote expand) still tops the table on strict
  F1, rising 74.4% → 83.8%. Every variant's table strict-F1 rises ~+3–9 pp (PMC11863705
  contributed 6 hard tables that were dragging all variants down), and the family ordering
  (hybrid-best > docling > tatr > raw detectors) is preserved. Only trivial tie-order
  reshuffles occur in the low (<45%) cluster, none affecting config selection.
* **Conclusion:** the 28→27 deletion does **not** require re-running the sweep and does not
  change the frozen config. Re-running `run_all_sweeps.py` is unnecessary (and would conflate
  the deletion with code drift, since it regenerates crops with today's code).

**Executed 2026-06-03 (option b — exclude PMC11863705, do not re-ingest):**
* PDF removed from `eval/pdfs/`; `eval/pdfs_manifest.txt` already lists the 27-PDF set.
* `scripts/eval/drop_pmcid_from_annotations.py PMC11863705` stripped **410 annotation keys
  across 83 files** (82 per-variant label files + `share_map.json`'s 14 keys) and the
  `ground_truth.csv` row `PMC11863705_main,0,0,6`. The `eval/annotations/_archive_2026-05-18/`
  snapshot was deliberately left intact. Verified: 0 active references remain.
* Post-deletion recompute (`recheck_drop_pmcid.py`) confirms on-disk labels now yield the
  validated 27-PDF numbers (winner `18_hybrid…` at 83.8%, unchanged), and `full == drop`
  (nothing left to filter).
* **Reports refreshed (2026-06-03):** `score_pdf_variants.py` gained an emitted-from-labels
  fallback (derive `emitted = set(labels)` when `<variant>/json/` is absent) plus a name-based
  detector fallback when the run manifest is gone, and skips `_`-prefixed archive dirs. Both
  fire only when `out/sweeps/` is absent, so the normal path is unchanged. Regenerated the full
  27-PDF table into a **separate folder** for review: `reports/post_exclusion_PMC11863705/`
  (`variants_PR.md`, `variants_PR.json`, and `DELTA_vs_28pdf.md` — the old-vs-new diff). Winner
  unchanged (`18_hybrid…` 74.4% → 83.8%); every variant +3–9 pp; `unlabelled=0` on all 62 rows.
  The old `reports/stage*_PR.md` are left untouched (28-PDF baseline for comparison).
* **End-to-end rerun confirms reproduction (2026-06-04):** the full sweep was re-run *from
  scratch* on the 27-PDF set (`run_all_sweeps.py --stage all`, all 7 stages, today's extraction
  code — not the label-fallback) and re-scored. Result: **all 31 variants reproduce the baseline
  with 0 strict-F1 mismatches and `unlabelled=0` on every row** — i.e. today's code emits exactly
  the crops the human labels cover (no drift), and every staged `BEST_*`/`STAGE1_BASE_*` constant
  verifies against the fresh data with zero changes (S1 `01_docling`/`04_tatr_099`/`07_hybrid_099`
  on crop F1; S2 `drop`; S3 header-clip 50pt +3.4 pp; S4 `BEST_BASE=07_hybrid_099`, hybrid wins;
  S5 multipliers all tie; S6 both merge flags rejected; S7 recon OFF / expand ON). Winner
  `variant 18` at 83.8% strict F1 confirmed. Fresh report:
  `reports/figtable_extraction_sweep_rerun_27pdf_20260604_PR.{md,json}`; per-stage analysis:
  `reports/stage_analysis.md` + `reports/stage_analysis/` (regenerated by the two new helpers
  `scripts/eval/stage_analysis_tables.py` and `recheck_drop_pmcid.py`). This upgrades the result
  from "re-aggregated from old labels" to "reproducible from a fresh extraction".
* **Still stale (TODO):** `docs/readmes/other_readmes/PDF_EXTRACTION_RESULTS.md` narrative still cites the
  28-PDF numbers (e.g. "strict F1 74.4%"); update to the 27-PDF values when next editing the
  extraction chapter.

---

## Bug 69 — EXP F held-out confirmation scored the full corpus, not the 52-case test split

### Status / Severity / Surface
* **Status:** Open (reopened 2026-06-15). History: Observed 2026-06-03 → Won't fix 2026-06-12 →
  reopened 2026-06-15 when the proper fix (physically disjoint calibration / held-out clusters)
  was chosen over the original within-corpus split-filter patch. See *Fix* below.
* **Severity:** High — the dev/test split is not enforced in the `run_sweep` selection path, so
  the production MAP/cascade configuration was *selected* on the whole corpus (not the dev split)
  and the §9.6 confirmation was *evaluated* on the whole corpus (not the test split). There is no
  out-of-sample held-out estimate; the headline §9.4 result is in-sample.
* **Surface:** Summarisation eval harness — `scripts/eval/run_summarization_experiments.py`
  (`_run_exp_f`), `eval/silver/run_summarization_sweeps.py` (`_load_map_context`),
  `eval/silver/map_theta_sweep.py` (`run_sweep`).

### Symptom
The held-out "test-split" confirmation (EXP F, reported in thesis §9.6) records a strict-F1 of
**0.7361** with a dev-reference of **0.7434** and Δ = **−0.0073**, presented as evidence that the
dev-tuned configuration transfers to the 52-case / 353-finding held-out test split. The recorded
denominators are corpus-wide, not test-split-wide.

**Broader scope (the root cause is not specific to EXP F).** The same metadata-only `split`
defect means *every* `run_sweep`-based selection experiment ran on the full corpus, not the dev
split. Classifying the recorded CSVs in `eval/reports/` by `n_silver` (1243 ⇒ dev-filtered, 1596
⇒ whole corpus): scorer comparison (`exp_1`/`exp_4`), agreement-weights (`exp_2`/`exp_5`),
polarity-flag (`exp_3`/`exp_6`), the joint scorer×θ×reject_θ cascade sweep (`map_cascade_sweep`,
`split=all`), and the routing-policy sweep all show `n_silver=1596` ⇒ **whole corpus** (despite a
`split="dev"` label). Only `exp_b2_cost_quality` and `exp_b2_family_bias_diagnostic` show
`n_silver=1243` ⇒ genuinely dev. So the production config (scorer, θ, reject_θ, agreement weights,
polarity flag, embedder) was *selected* on all 273 cases; the dev-only cost-quality table (§9.4)
then re-scores that whole-corpus-selected config on the 221 dev cases — still in-sample, because
those 221 cases were in the selection set. (The stale `map_theta_sweep`/`grounding_sweep` CSVs
at `n_silver=27/53/145/165` are pre-corpus-freeze and irrelevant.)

### Evidence
* `out/thesis_results/chapter9_offline_replay/06_exp_f_test_split.json`:
  `split="test"`, `test_strict_f1=0.7361`, `test_n_silver=1596`, `test_n_pipeline=1920`,
  `test_n_matched=1533`, `test_recall=0.9605`, `test_precision=0.7984`,
  `dev_strict_f1_for_reference=0.7434`, `test_minus_dev_strict_f1_delta=-0.0073`.
* `eval/reports/exp_f_test_confirmation_20260527T000956.csv`: `split=test`, `n_silver=1596`.
* Canonical split (manifest.json::corpus_counts_canonical, independently reproduced via
  `eval/silver/split.py`, seed=42, dev_fraction=0.8): 273 cases → **221 dev / 52 test**; 1596
  silver findings → **1243 dev / 353 test**.
* **Arithmetic impossibility:** `n_matched=1533` ≫ 353 (the test split's silver count). Matched
  findings cannot exceed the silver pool being scored. `recall = 1533/1596 = 0.9605` and
  `precision = 1533/1920 = 0.7984` both use corpus-wide denominators. ⇒ EXP F scored all 273
  cases.
* **Contrast (correct path):** the dev cost-quality run EXP B.2
  (`eval/reports/exp_b2_cost_quality_20260527T005112.csv`) recorded `n_silver=1243` across all 9
  rows — i.e. it *was* dev-filtered. So `0.7434` is a genuine dev-split number while `0.7361`
  is a full-corpus number; the Δ compares two different scopes (1243 vs 1596).

### Diagnosis
The `split` argument is plumbed through but never reaches the metric in the EXP F path:
1. `_run_exp_f` (`run_summarization_experiments.py:684`) calls `_load_map_context(embedder)`
   with **no split argument**.
2. `_load_map_context` (`run_summarization_sweeps.py:467–511`) loads the entire voter cache
   (line 479) and **all** silver cases (`line 480`, no `filter_by_split`). It has no split
   parameter.
3. `_run_exp_f` (`:685–700`) calls `run_sweep(..., split="test")`.
4. `run_sweep` (`map_theta_sweep.py:799–908`) writes `"split": split` as a row label only
   (`:902`). Its sole case filter (`:843`) keeps cases *present in silver* — and all 273 are
   present. The intended filter ("dev split filtered at load time", comment `:842`) is never
   applied: priming uses `--split all` (`run_summarization_sweeps.py:692`; prime instruction
   `:473`) and `run_sweep` does not filter.

The dev baselines path differs: `_run_exp_b2` passes `case_filter=_case_in_split`
(`run_summarization_experiments.py:871–883`), where `_case_in_split` applies
`assign_split(case_id, dev_fraction, seed) == ctx.split` (`:872–874`). That is why EXP B.2 is
correctly dev-only (1243) but EXP F is not test-only.

The tuning-time contamination guard works as designed (calibration sweeps default to `--split
dev` and refuse `--split test`); the defect is purely in the *confirmation* metric, which silently
fell back to the full corpus.

### Fix (chosen 2026-06-15 — disjoint clusters; in progress)
**Superseded approach (within-corpus split filter).** The original proposal was to filter to the
test split before scoring in the EXP F path — either subset `map_ctx.silver_by_case` (+ the voter
cache feeding `run_sweep`) to `case_id`s with `assign_split(case_id, ctx.dev_fraction, ctx.seed)
== "test"`, or give `run_sweep` a real split filter mirroring `filter_by_split`. This was the basis
for the THESIS.md line-34 options (a)/(b) and the 2026-06-12 Won't-fix. It is **superseded**: a
within-corpus seed-split is still drawn from the same 273-case selection pool, so the failure mode
(silently scoring on the calibration set) stays one missing argument away.

**Chosen fix — two physically disjoint clusters.** Replace the metadata-only dev/test split with two
non-overlapping 15-paper clusters:
* `related15` (`configs/paper_selection/related15.yaml`) — the ILP relatedness-selected calibration
  cluster; everything is *tuned* here.
* `heldout15` (`configs/paper_selection/heldout15.yaml`) — a random, reproducible sample (seed=19) of
  15 eligible papers, disjoint from `related15` **and** from the 27 document-extraction papers;
  the held-out confirmation is *only ever* scored here.

Verified 0 PMC overlap between the two clusters. Plus a structural guard: `eval/silver/`
(`generate.py`, `evaluate.py`, `sweep.py`, `map_theta_sweep.py`) now **require** explicit
`--silver` / `--pipeline` / `--source` paths — no defaults — so a sweep can't silently run on the
calibration set when the held-out set was intended (commit `2ea9188`).

**Status of the rebuild:** infra in place — both cluster YAMLs + `source_cases_related15.jsonl`
(454 cases) and `source_cases_heldout15.jsonl` are generated. **Pending:** (1) generate
`silver_findings_{related15,heldout15}.jsonl` (Opus, ~$12 batch); (2) re-prime the voter cache over
the 454-case `related15` set (the old 273-case cache is stale); (3) run the pipeline over both
clusters → `pipeline_findings_{related15,heldout15}.jsonl`; (4) calibrate `(theta, reject_theta,
scorer)` on `related15`; (5) confirm on `heldout15` for a true out-of-sample strict-F1. Until step
5 lands, §9.6 must not be reported as a held-out result.

### Verification
Pending fix + re-run. Pre-fix verification of the diagnosis: the canonical split recomputed
independently (`eval/silver/split.py`, seed=42, dev_fraction=0.8) yields 221/52 and 1243/353,
while the recorded EXP F denominators (silver 1596, pipeline 1920, matched 1533) match the full
273-case corpus exactly — proving no test filter reached the metric. Tracked in
[`THESIS.md`](THESIS.md#todos).

### Resolution (2026-06-12) — Won't fix; held-out split abandoned by design — **SUPERSEDED 2026-06-15**
> **Superseded 2026-06-15.** This Won't-fix resolution was reversed: the held-out split is back,
> but implemented as a *physically disjoint* `heldout15` cluster rather than a within-corpus seed
> split (see *Fix (chosen 2026-06-15)* above). The thesis will report a genuine out-of-sample
> confirmation on `heldout15`, not in-sample. Text below is retained for history.

Design decision (THESIS.md Decisions log, 2026-06-12): the thesis will **not** use a dev/test
split. The knowledge-extraction evaluation is reported in-sample on the full ILP-cluster silver
set by design, so the split-non-enforcement is no longer a defect to repair. All experiments will
be re-run after the silver cases/findings are regenerated on the 15-paper ILP dataset; the
mislabeled "test" artifact (`06_exp_f_test_split.json` — full-corpus `n_silver=1596` under a
`split=test` label) will be removed on re-run. Thesis text (§8.8 limitations, §8.4, the corpus
"relationship" section, and `07_relationship…tex:35`'s "Development and held-out evaluation" row)
must describe the results as **in-sample** and must not cite `0.7361` / `0.7434` as a held-out
confirmation. The `split.py` machinery, the `--split` flag, and the `*_test.jsonl` files become
vestigial.

---

## Bug 70 — reject_theta inert: REJECT never dropped, now a toggleable drop policy

### Status / Severity / Surface
Fixed (2026-06-10) / Medium / Summarisation, MAP cascade decision
(`agreement/checker.py`, `agreement/decision.py`,
`current_stages/map_stage.py`, `batch/runner.py`, `config.py`).

### Symptom
The MAP cascade is documented as a three-way decision — accept (`s ≥ θ`),
reject (`s ≤ reject_theta`), escalate (otherwise) — with `MapConfig.reject_theta`
described as "hard REJECT (drop) the chunk." In practice **no chunk was ever
dropped on low agreement**: every sub-θ chunk escalated, and the terminal L3
voter always emitted.

### Evidence
`AgreementChecker.compute` correctly returns `ChunkDecision.REJECT` for
`primary <= reject_theta`. But `agreement/decision.py:evaluate_chunk` only
distinguished `KEEP` from not-`KEEP`: `ChunkOutcome.keep` was `True` only for
`KEEP`, so REJECT and ESCALATE produced identical `keep=False` outcomes. Both
orchestrators escalated on `keep=False` (`map_stage._process_chunk` → L2/L3;
`batch._process_level` → `escalated`), and `map_stage` always took `invoke_l3`
at the terminal tier (`l3_kept`), with the `dropped` counter incrementing only
when the L3 call itself returned `None`. The router path made the collapse
explicit in its own log: *"rejected by router — escalating to strong model."*
Net: `reject_theta` was a computed-but-inert knob, and the values swept for it
in MAP calibration had no behavioural effect.

### Diagnosis
The three-way decision existed at the scorer/checker layer but was flattened to
two-way at the orchestration layer; the drop action was never implemented.

### Fix
`ChunkOutcome` gained a `rejected: bool` field, set by `evaluate_chunk` from
`bundle.decision == REJECT` (legacy) / `decision.decision == REJECT` (router).
Both runners now **drop** on `rejected` in the *legacy* path (no escalation, no
finding emitted; chunk counted as dropped). The router path keeps
escalate-on-reject by design (a router REJECT is a schema/provenance failure
better repaired by the strong model than discarded). `reject_theta` is the
single toggle: its default was lowered `0.2 → 0.0`, which disables dropping for
all non-degenerate chunks (deferral scores are in [0, 1]) and reproduces the
prior escalate-everything behaviour; `reject_theta > 0` makes the drop live. No
separate enable flag — the threshold value is the switch.

### Verification
`/tmp/test_reject.py`: `evaluate_chunk` maps KEEP→keep, REJECT→rejected,
ESCALATE→neither; `AgreementChecker` with `reject_theta=0.0` returns ESCALATE
for a 0.1 score and with `reject_theta=0.2` returns REJECT; `MapConfig().reject_theta
== 0.0`; all five modified files parse. The default (`0.0`) leaves production
behaviour and existing calibration results unchanged.

### Note
Pre-existing MAP calibration swept `reject_theta` while it was inert (see also
[B-069](#bug-69--exp-f-held-out-confirmation-scored-the-full-corpus-not-the-52-case-test-split));
any future use of `reject_theta > 0` needs its own calibration, since the prior
sweep could not have measured the drop's effect.


## Bug 71 — stale 2-tuple mock for RelateStage.relate crashes its own regression test

### Status / Severity / Surface
Fixed (2026-06-12) / Low / Summarisation eval tests
(`eval/silver/tests/test_relate_sweep.py`; production code in
`pipeline/stages/summarization/helpers/corpus_relate.py` is unaffected).

### Symptom
`test_corpus_relate_tuple_unpack_does_not_crash` fails at call time with
`ValueError: not enough values to unpack (expected 3, got 2)` raised from
`corpus_relate.py:295`. Reproduces on a clean checkout independent of any other
work in progress (confirmed by stashing unrelated edits and re-running).

### Evidence
`RelateStage.relate` is annotated and documented to return a **3-tuple**
`tuple[list[Relation], list[RawNLIPair], list[SkippedPair]]` =
`(relations, raw_pairs, skipped_pairs)` (`relate_stage.py:412` / `:430`). Every
production caller already unpacks three: `runner.py:620`, `batch/runner.py:608`,
and `corpus_relate.py:295` / `:391` / `:401`. The regression test, however, mocks
it with `patch.object(stage._relate, "relate", return_value=([], []))` — a
**2-tuple** — and leaves the real `relate_from_dir` unmocked, so that method
unpacks three values from the two-element mock and raises inside its
`except Exception` path.

### Diagnosis
`relate` was extended from a 2-tuple to a 3-tuple (adding `skipped_pairs`, one
`SkippedPair` per pre-NLI gate rejection for offline debugging). The production
caller and the four other callsites were resynced; this test's mock and its
docstring were not. The irony is that the test exists *specifically* to guard
`relate`'s tuple-unpacking, yet it broke for the same class of reason it was
written to catch — the arity changed and a consumer (here, the mock) was not
updated. Test-only defect: no production path is affected.

### Fix
`eval/silver/tests/test_relate_sweep.py` — mock `return_value` changed from
`([], [])` to `([], [], [])`, and the stale docstring updated from
`(relations, raw_pairs)` / `len(raw_relations) == 2` to the current 3-tuple
`(relations, raw_pairs, skipped_pairs)` / `len(...) == 3`. No production file
touched.

### Verification
`python3 -m pytest eval/silver/tests/` → **45 passed** (previously 44 passed, 1
failed). The fix changes only the mock arity and a docstring.

## Bug 72 — silver generation fails: claude-opus-4-7 rejects deprecated temperature

### Status / Severity / Surface
Fixed (2026-06-15) / High / Eval, silver-label generation (`eval/silver/generator.py`).

### Symptom
`python -m eval.silver.generate --batch --source eval/data/source_cases_related15.jsonl`
submits the 454-case batch, but on collection **every** request fails:

```
BetaInvalidRequestError(message='`temperature` is deprecated for this model.',
                        type='invalid_request_error')
```

0/454 cases succeeded; `silver_findings_related15.jsonl` was never written.

### Evidence
`generator.py` sent `temperature=DEFAULT_TEMPERATURE` (=0) on both paths — sync
`_call_opus` and the batch request params. The same model+temperature worked on
2026-05-24 (silver regenerated under v3, 1596 findings), so Anthropic applied the
deprecation server-side between then and 2026-06-15. The Claude voter path
(`claude_batch.py`) omits `temperature` entirely, so the cascade voters (Haiku L2,
Sonnet L3) are unaffected; OpenAI/Gemini voters still send it and those providers
accept it.

### Diagnosis
`claude-opus-4-7` (reasoning-class) no longer accepts an explicit `temperature`;
the API rejects the whole request. Silver output is tool-forced
(`tool_choice` → `extract_findings`), so the sampling temperature had limited
effect on the structured result anyway.

### Fix
Removed the `temperature` field from both silver requests in `generator.py`;
`DEFAULT_TEMPERATURE` deleted (referenced nowhere else). The model now uses its
default sampling. Trade-off: silver is no longer generated at an explicit
`temperature=0`, so regeneration is not bit-deterministic — acceptable for a
frozen, generate-once reference set, and the forced `extract_findings` tool keeps
extraction variance low.

### Verification
`py_compile` clean; the silver request now carries only
`model / max_tokens / system / tools / tool_choice / messages`. Re-run
`generate --batch` to resubmit the 454 uncached cases.

## Bug 73 — map_weights gated on hybrid blocks embedding-scorer weight tuning

### Status / Severity / Surface
Fixed (2026-06-16) / Medium / Eval, summarisation sweep harness
(`eval/silver/run_summarization_sweeps.py` → fixed in the new
`eval/silver/run_new_summarization_sweeps.py`).

### Symptom
The greedy staged calibration harness refuses to tune the H-EMB-01
soft-alignment weights unless the hybrid scorer won: `run_summarization_sweeps.py`
`map_weights` raises `SystemExit("Stage 3 (map_weights) is only swept when
BEST_SCORER='hybrid' …")` when `BEST_SCORER == "embedding"`, and
`_weight_variant_specs` hardcodes `scorer="hybrid"` in every weight variant.
Since the embedding scorer is the production default
(`agreement.scorer_kind: embedding`), its soft-alignment weights could not be
calibrated through the harness at all.

### Evidence
`AgreementConfig.{tau, count_alpha, reuse_weight, contradiction_weight}` are the
soft-alignment weights for *embedding-based* agreement scoring (H-EMB-01).
`EmbeddingScorer.__init__` and `EmbeddingSimilarityStrategy` both accept all four
and feed them to `agreement.embedding._align`; the hybrid scorer consumes them
too via its `w_embedding` sub-signal. So the weights drive the embedding scorer —
the very scorer the stage refused to tune. The stage's error text mislabels them
as "the hybrid scorer's structural-blend knobs"; that description belongs to the
*blend* weights (`w_category`/`w_embedding`/`w_entity`/`w_evidence`), which are a
separate stage (`map_hybrid_blend`).

### Diagnosis
The stage conflated two distinct weight sets — the scorer-agnostic soft-alignment
weights and the hybrid-only blend weights — and gated both on `hybrid`.

### Fix
Fixed in the new harness `run_new_summarization_sweeps.py` (the old file is left
untouched so scorer semantics and harness orchestration remain separately
bisectable). Its `map_weights` stage builds soft-alignment weight variants with
`scorer=BEST_SCORER` (no hybrid gate) and only adds hybrid blend variants when
`BEST_SCORER == "hybrid"`.

### Verification
`python -m eval.silver.run_new_summarization_sweeps --stage map_weights
--list-variants` with `BEST_SCORER="embedding"` lists soft-align variants for the
embedding scorer (`embedding_tau_0.1`, `embedding_count_alpha_0.5`, …) — the
weights are reachable for the production-default scorer.

## Bug 74 — enum stringification in _finding_to_pipeline halves strict-F1

### Status / Severity / Surface
Fixed (2026-06-17) / High / Eval, silver scoring
(`eval/silver/map_theta_sweep.py` → `_finding_to_pipeline`).

### Symptom
The E10 single-model baselines scored every voter alone vs silver and showed a
strict/loose-F1 ratio of **exactly 0.50** for *every* model (e.g. single-Sonnet
loose-F1 0.885 but strict-F1 0.442), with single-Sonnet sitting 0.27 *below* the
cascade (0.7133) — impossible, since the frozen cascade escalates ~96 % of chunks
to Sonnet, so single-Sonnet should ≈ the cascade.

### Evidence
`AuditableSummary.model_validate(raw).model_dump()` returns `relation_type` /
`direction` as Enum OBJECTS (`<RelationTypeEnum.demographic: 'demographic'>`),
whereas the raw cached dict holds the value string. `_finding_to_pipeline` did
`str(f.get("relation_type"))` → `str(<RelationTypeEnum.demographic>)` =
`'RelationTypeEnum.demographic'`, which never equals the silver `'demographic'`.
STRICT_FIELDS = {category, relation_type, direction}; `category` was already a
plain string, but `relation_type` + `direction` (enums) mismatched on every
finding → each matched pair downgraded to a 0.5 partial TP → strict-F1 ≈ halved,
while loose-F1 (entity match) was untouched (the 0.50-ratio signature).

### Diagnosis
Two paths reach `_finding_to_pipeline` with different field types: the **L3
escalation** path in `_replay` passes the RAW cached dict (value strings →
correct), while the **L1/L2 early-accept** path (`checker.best(...).model_dump()`)
and the E10 single-model path (`AuditableSummary.model_validate(...).model_dump()`)
pass Enum objects → corrupted. Corruption is therefore proportional to the
fraction of findings from a `model_dump`'d path — i.e. the **early-accept rate**:
~4 % at θ0.9 (frozen winner barely affected, strict-F1 0.7133 ≈ correct) but ~55 %
at economy θ0.4 (strict-F1 heavily underestimated). This (a) distorted the E09
cost-quality frontier (economy understated) and inflated the apparent
"escalate-everything" advantage, and (b) inverted the E10 single-model headline.

### Fix
New helper `_ev(x) = getattr(x, "value", x)` (enum→value string; no-op for raw
strings / None), applied to `category`/`relation_type`/`direction`/`confidence` in
`_finding_to_pipeline`. Single chokepoint — fixes the calibration engine
(`_replay` early-accepts), the E10 single-model baselines, and every other caller
at once. Regression test `eval/silver/tests/test_finding_enum_unwrap.py`.

### Verification
102 eval/silver tests pass (3 new). Re-run plan: **E07 (map_theta) first** to
confirm the argmax (the frozen `run.yaml` pin) still holds; then re-run the
θ-sensitive analyses (E09/E10/E11/E12) and spot-check the fixed-θ comparisons
(E06b/E06c/E08/E03 — both arms equally corrupted, so relative deltas survive).
Expected post-fix: single-Sonnet ≈ 0.71 (≈ cascade), restoring the "cascade does
not beat Sonnet-only" reading the bug had inverted. No corrected result is
committed and `run.yaml` is not touched until E07 reconfirms the pin.

## Bug 75 — DB ingester drops table page/bbox provenance

### Status / Severity / Surface
Fixed (2026-06-18) / Medium / Doc-extraction DB ingest
(`pipeline/stages/pdf_text_extraction/outputs/db_ingester.py`).

### Symptom
E02's corpus provenance metric showed `Table.page_number` and `Table.bbox_*` at
**0 % across all 1 960 tables**, though the schema has those columns and the crop
filenames encode the page (e.g. `…_Table_1_p4.png`) — tables were not
coordinate-localizable in the DB.

### Evidence
`CroppedMedia` (`models/dto.py:139`) carries non-optional `page: int` and
`bbox: BoundingBox`; the cropper populates them and the media-JSON writer persists
them to `out/json/<pmcid>_media.json`. But the ingester's `Table(...)`
(`db_ingester.py:112`) set only `caption_text`/`image_filename`/`image_path` — never
`page_number`/`bbox_*`. The data reached the ingester and was dropped at the write.

### Diagnosis
A missing field-mapping in one constructor — not an upstream loss. `table_content`
(crop path emits an image, not structured table text) and `section_context` (no
section field on `CroppedMedia`) are *missing-source* gaps, separate from this drop.
`Figure` originally had no page/bbox columns (added in migration 0014).

### Fix
(1) Forward: add `page_number=tbl.page, bbox_x1=tbl.bbox.x1, …` to `Table(...)`
(Docling coords, y=0 at page bottom ⇒ y1>y2). (2) Backfill the existing corpus
without re-extraction or a DB drop: `eval/silver/experiments/E02_provenance/
backfill_media_provenance.py` reads each cached media JSON, matches its tables to DB
`Table` rows by image filename, sets `page_number`/`bbox_*` where NULL (dry-run
default; `--apply` commits). Validated on one paper (page 4, bbox match), then full
corpus: 1 960/1 960 matched, 1 959 updated.

### Verification
E02 re-run: tables `page_number` 0 % → **100 %**, valid bbox 0 % → **100 %** (after
also fixing the metric's own bbox-validity check, which assumed screen coords y2>y1
while Docling is y1>y2). Figures fixed identically — migration 0014 adds the columns, the ingester
persists them, and the backfill filled 4479/4479 → 100 % page+bbox.
`table_content`/`section_context` remain unpopulated (no source).

## Bug 76 — persisters don't dedupe duplicate normal_id → 0 rows in normal/group tables

### Status / Severity / Surface
Fixed (2026-06-18) / Medium / Summarization DB persistence (`persistence.py`:
`persist_normal_findings`, `persist_finding_groups`).

### Symptom
The first summarization run to persist to the DB (the corpus had only ever been run
file-only) finished cleanly but the post-run row-count audit FAILED: `sum_normal_findings`
and `sum_finding_groups` (and their children `sum_normal_finding_spans`,
`sum_group_members`) had **0 rows for the paper**, even though `sum_canonical_rules`
and `sum_final_rules` each had 117 — and canonical rules *derive* from normal findings →
groups. Canonical rows also persisted with **null group FKs**.

### Evidence
Surfaced by the primer→MAP bridge 1-paper validation (`PMC7540531_HIS-77-460`). Log:
`DB: failed to persist finding groups: (psycopg2.errors.UniqueViolation) duplicate key
value violates unique constraint "uq_sum_group_member"`. DB after the run: map 127,
normal 0, groups 0, group_members 0, canonical 117, final 117, relations 2.
`uq_sum_normal_finding = UNIQUE(pipeline_run_id, normal_id)`;
`uq_sum_group_member = UNIQUE(finding_group_id, normal_id)`.

### Diagnosis
NORMALIZE emits ≥1 duplicate `normal_id` per paper (3 in this paper: 123 → 120). The
in-memory pipeline already treats `normal_id` as unique — `nf_id_map[nf.normal_id]` and
`nf_by_id = {nf.normal_id: nf}` silently collapse duplicates — so CANONICALIZE/RESOLVE
operate on the collapsed set and are *correct*. But both persisters insert one row per
list item without deduping, so the second row with a repeated `normal_id` trips the
unique constraint; the `try/except` logs a warning and returns `{}` (0 rows). The empty
`nf_db_id_map`/`fg_db_id_map` then leave canonical rows with null group FKs.

### Fix
Dedupe by `normal_id` in both persisters, last-wins to match the in-memory dicts:
`persist_normal_findings` iterates `{nf.normal_id: nf for nf in normal_findings}.values()`;
`persist_finding_groups` skips already-seen `normal_id`s per group before inserting
`SumGroupMember`. (The upstream NORMALIZE duplicate-`normal_id` is the deeper cause —
tracked as a follow-up — but deduping on persist makes the DB consistent with the
already-correct in-memory canonical/final outputs.)

### Verification
Re-ran the 1-paper rebuild: `NORMALIZE 123 → persisted 120 normal findings + spans`,
`persisted 112 finding groups`, canonical/relations/final unchanged, and
`Post-run row-count audit OK (5 table(s) checked)`. DB: normal 120, groups 112.

## Bug 77 — grounding rejection count differs between identical rebuilds

### Status / Severity / Surface
Won't fix (2026-06-18) / Low / `rebuild_from_cached_map` idempotency — **not** a
grounding defect. (Originally mis-filed as grounding non-determinism.)

### Symptom
Two back-to-back no-API rebuilds of the same paper logged different grounding-rejection
counts: run_id=4 → `rejection summary (22 grounding, …)`, run_id=5 → `(0 grounding, …)`.
Both produced 117 canonical/final rules.

### Diagnosis (resolved — not a defect)
The two runs saw **different inputs**, not the same one. `rebuild_from_cached_map`
**overwrites its input JSON with its output**, and the runner serializes the
*post-grounding* findings into `audit_trail.map_chunks` (149 raw − 22 rejected = 127):
* run4 read the raw bridge JSON (149) → grounding rejected 22 → wrote 127 back over the file.
* run5 read run4's output (the already-grounded 127) → grounding rejected 0.
I re-ran rebuild without re-installing the raw bridge JSON. Grounding is a deterministic,
idempotent keep-filter (`entailment(verbatim_support → claim) ≥ 0.5`): a second pass over
the survivors re-confirms them all → removes 0. It can remove *fewer*, never more.

### Evidence
Pristine bridge JSON: 149 findings, `_bridge` marker. Installed JSON after run5:
**127** findings, rebuild's output schema (`status`/`run_id`/`canonical_rules`/…), no
`_bridge`. Definitive check: re-installed the raw 149 and rebuilt once (run_id=6) →
`rejection summary (22 grounding, …)` again, byte-for-byte the run4 result → grounding is
deterministic.

### Mitigation
Run `rebuild_from_cached_map` **once** per corpus build, on a fresh
`bridge_populate_corpus --install` (which writes raw, pre-grounding `map_chunks`); never
re-run rebuild on its own output. The documented full flow (`bridge --all --install` →
`rebuild` once) is correct. No code change — re-running on already-processed output is an
inherent property of the rebuild tool, not a defect.

---

## Bug 78 — stale "100 % figure strict-F1" claim for reconstruct/merge variants 28–32

### Status / Severity / Surface
Mitigated (2026-06-19) · Low · Thesis §9.1 (`docs/thesis/10_results.md`) + eval
`eval/reports/RESULTS.md` (E01 section).

### Symptom
§9.1 of the Results chapter ended with: *"A later branch of the sweep (the
reconstruct/merge variants 28–32) reaches 100 % figure strict F1 and is noted as a
possible future improvement."* The same number appears as a parenthetical in the
`RESULTS.md` E01 write-up. Both contradict the cited primary artifact.

### Evidence
The cited artifact is
`eval/reports/E01_doc_extraction/figtable_extraction_sweep_rerun_27pdf_20260604_PR.md`
(byte-identical copy at `…/post_exclusion_PMC11863705/variants_PR.md`). In it **every
`figures` row across all 32 variants is identical**: crop F1 89.9 %, **strict F1
84.0 %**, `strict tp/fp/fn = 55/21/0`, `icons = 14`. The reconstruct/merge variants
touch only the `tables` rows, and mostly regress them:

* `28_best_merge_tables_by_caption` → tables strict-F1 **79.5 %** (down from 83.8 %)
* `29_best_merge_figures_by_caption` → tables 83.8 %, figures **84.0 %** (no change)
* `31_best_reconstruct_only` → tables **44.2 %**
* `32_best_reconstruct_plus_selected_expand` → tables **80.5 %**

No row anywhere reports 100 % figure strict-F1. (Independently corroborated by
`docs/readmes/other_readmes/PDF_EXTRACTION_RESULTS.md`: "merge_figures_by_caption — zero impact".)

### Diagnosis
Figure strict-F1 is **invariant** across the whole sweep: the 32 variants re-tune table
detection and cropping only, and leave figure handling untouched. The 84.0 % figure
ceiling is bounded by 21 strict false positives, of which **14 are decorative icons**
emitted as figure crops (mask-benign, crop-incorrect); none of the swept knobs removes
them, so 100 % figure strict-F1 is unattainable in this sweep without an icon-suppression
step that was never run. The "100 %" value does not originate from the cited 27-PDF
artifact; it is a stale number that propagated from the `RESULTS.md` E01 parenthetical
into the thesis prose. Severity is Low: the claim sits in a non-headline "possible future
improvement" aside and affects no RQ verdict, no pinned configuration, and no number in
Table 9.1.

### Mitigation
* `RESULTS.md` E01 note corrected to state that figures are invariant at 84.0 % across the
  sweep and that the reconstruct/merge variants only move (and mostly regress) table
  scores — the route to a higher figure score is icon suppression, not any swept variant.
* Thesis §9.1 final paragraph rewritten to the same effect (drafted in chat; pending the
  author's paste into `docs/thesis/10_results.md`, per the thesis-editing workflow).

Mitigation, not Fixed, because the thesis-source correction lands by author paste rather
than direct edit; once §9.1 is updated this flips to Fixed.

### Verification
`grep` for any `figures … 100.0% … strict` row in either E01 artifact returns nothing;
all 32 figures rows read `89.9 % / 84.0 % / 55/21/0 / icons 14`. The corrected §9.1
asserts only the artifact-supported facts (84.0 % invariant figure ceiling; icon-bound
cap; table-only, regressive reconstruct/merge branch).

---

## Bug 79 — E03 grounding sweep grounds paraphrases, not DB paragraphs

### Status / Severity / Surface
Fixed (2026-06-20) / Low / Eval harness — `eval/silver/experiments/E03_grounding`
grounding-threshold sweep (Thesis §9.5 / RQ2). **Not** a production defect; production
grounding was correct throughout.

### Symptom
At grounding threshold 0.5 over the same 2280 frozen-config MAP findings, two retention
numbers disagreed by 91 findings:
* production funnel (E04, from persisted `map_grounding_rejected`): **1911 / 83.8 %**
* E03 sweep harness: **2002 / 87.8 %**

Same NLI model (PubMedBERT-MNLI-MedNLI), same threshold, same input count — yet different
retention. The thesis carried a paragraph attributing the gap vaguely to "the evaluation
harness."

### Diagnosis
The two paths grounded **different premise text**:
* Production (`runner.py` step 1a-pre) calls `_replace_verbatim_from_db` **before**
  grounding, overwriting each finding's `verbatim_support` with the real cited paragraph
  (`TextElement.text_content`, keyed on `evidence[0]`'s `te_id`). NLI then scores
  (real paragraph → claim).
* E03 runs offline from `voter_cache` with no DB, so it grounded the **LLM-paraphrased**
  `verbatim_support` straight from the replay. The paraphrase is the model's own
  justification for its claim, so it entails the claim more readily → higher scores →
  91 extra findings clear 0.5.

Why the production helper couldn't simply be called: the replay's `_finding_to_pipeline`
dropped finding-level `evidence`, so `PipelineFinding` carried no `te_id` to look up — the
first reconciliation attempt raised `AttributeError: 'PipelineFinding' object has no
attribute 'evidence'`.

### Evidence
* After the fix, `replace_verbatim_from_db` logged `replaced 2280/2280` — zero DB-miss
  fallbacks; every winning finding's `evidence[0]` te_id resolved in the DB.
* Reconciled sweep @0.5: `n_kept=1911, retention=0.838`, landing exactly on the funnel;
  `validation: n_findings=2280 [OK]`.
* The whole 91-finding gap closed — confirming premise text, not finding-set identity or
  model behaviour, was the entire cause (also settling the 2280-set-identity caveat).

### Fix
1. `eval/silver/schemas.py` — add `evidence: list[str] = []` to `PipelineFinding`
   (default-valued → backward-compatible with the DB-row / exporter construction sites).
2. `eval/silver/map_theta_sweep.py` `_finding_to_pipeline` — populate
   `evidence=f.get("evidence") or []`.
3. `eval/silver/experiments/E03_grounding/grounding_sweep_related15.py` — call
   `replace_verbatim_from_db(get_db_connection(), case_outputs)` between replay and NLI
   scoring; docstring updated to note the new DB dependency (still no API calls).

New artifact: `E03_grounding/sweep_20260620T155248.csv`.

### Verification
Reconciled E03 @0.5 = 1911 / 83.8 %, identical to the E04 funnel's grounded count. Best
silver strict-F1 stays at filter-off (0.7135, was 0.7133) and the curve shape is preserved,
so the pinned 0.5 groundedness choice is unaffected — **no production rerun or downstream
experiment rerun is triggered**. `RESULTS.md` E03/E04 and `EXPERIMENTS.md` E03 row updated;
`_E03_FROZEN["grounded_at_0_5"]` 2002→1911. The §9.5 reconciliation paragraph can now be
retired (author paste pending). One honest interpretive shift surfaced by the real-paragraph
curve: precision is near-flat across the sweep (0.875→0.879), so the dropped tail is **not**
silver-false — grounding enforces source-faithfulness, an axis silver-F1 does not reward
(captured in the `RESULTS.md` E03 interpretation).

### Related
[Bug 5](#bug-5--batch-runner-missing-sync-parity-features) — the same
paraphrase-vs-source grounding distinction, there as a production batch-runner parity gap
(`BatchKnowledgeExtractionRunner` missing `_replace_verbatim_from_db`); here as an eval-harness
measurement artifact.

---

## Bug 80 — Citation integrity unchecked on the production cascade

### Status / Severity / Surface
Mitigated (2026-06-20) · Medium · Summarisation, MAP output
(`current_stages/map_stage.py::_cascade`, `batch/runner.py`, offline
`eval/silver/map_theta_sweep.py::_replay`).

### Symptom
A MAP finding can cite a sentence position / `text_element_id` / PMCID that does
not exist in its chunk and still ship to NORMALIZE → … → the final rule set. No
stage on the production path rejected it.

### Diagnosis
The repo *has* a correct citation validator — `routing/provenance_validator.py`
(`ProvenanceValidator`) — which checks citation parse, sentence-position
existence, cross-document PMCID, te_id match, and verbatim fabrication. But it is
invoked **only** by `MapOutputRouter`, and the router is `enable_router=false`
(pinned in `run.yaml`, default in `config.py`, decided in B-062). The legacy
L1→L2→L3 `AgreementChecker` cascade — the production path and the one the
related15/E03 replay drives — calls `agreement.compute(voters)` directly with no
citation validation. Downstream, `replace_verbatim_from_db` silently keeps the
LLM paraphrase when a te_id has no DB row (`te_map.get(te_id) is None`), and the
grounding NLI then scores the claim against its own paraphrase (near-tautology),
so a bad citation is *not* caught there either. Net: citation integrity was
effectively unenforced in every shipped result.

### Mitigation
Added a finding-level citation filter, `provenance/citation_filter.py`
(`filter_summary_by_citation` / `citation_drop_indices`), that reuses
`ProvenanceValidator` and drops findings with hard **structural** citation
failures (`INVALID_SENTENCE_ID`, `NONEXISTENT_SOURCE`,
`CROSS_DOCUMENT_SOURCE_ERROR`, `INVALID_TEXT_ELEMENT_ID`). Wired in at every site
where a selected chunk summary is committed — sync `MapStage._cascade` (covers
L1/L2/L3), the batch `finalized` consumer, and the offline `_replay` — all
*before* `replace_verbatim_from_db`, so the optional fuzzy verbatim-fabrication
check (`check_verbatim`, off by default) sees the cited sentence not the DB
paragraph. Gated by `CitationConfig` (`config.py`, default `enabled=true`),
hash-stamped in both runners' `_pipeline_config_hash`.

This is a **mitigation, not a root-cause fix**: it removes bad-citation findings
from the output but does not explain *why* they are produced. Applying it
immediately surfaced [Bug 81](#bug-81--gemini-l1-voter-cross-paper-contamination-in-voter_cache):
37 % of cached MAP voter findings fail the structural citation check, almost all
from one voter.

### Verification
`tests/summarization/helpers/test_citation_filter.py` (12 tests): valid citation
kept; nonexistent position / te_id mismatch / cross-document / unparseable each
dropped; mixed batch drops only the invalid finding; fabricated verbatim opt-in;
empty-chunk and empty-findings no-ops.

### Default resolved
`enabled=true` is kept as the shipped default. The selected-set measurement
(see [Bug 81](#bug-81--gemini-l1-voter-cross-paper-contamination-in-voter_cache))
shows the filter drops **0/2280** findings on the production path today — the
agreement gate already excludes the contaminated voter — so the filter is a
**0-cost defense-in-depth guard**: it changes no current number but blocks any
future config (or fixed-but-still-imperfect voter) from leaking a bad-citation
finding into the output.

---

## Bug 81 — Gemini L1 voter cross-paper contamination in voter_cache

### Status / Severity / Surface
Observed (2026-06-20) · High · Summarisation, `eval/data/map_primer/voter_cache.json`
(related15); L1 voter index 0 = `gemini-2.5-flash-lite`.

### Symptom
The Gemini L1 voter emits findings whose **content and citation both belong to a
different corpus paper** than the case being processed. The cited
`PMCID|te_id` is a *valid, real* foreign key — not a fabricated one.

### Evidence
Case `PMC7540531_HIS-77-460` (te_id 22044), chunk C2 — the text shown to the
voters is about Ki-67 prognosis in melanoma:

```
[S1|PMC7540531_HIS-77-460|22044] Clinically, increased Ki-67 expression is a well-known marker of poor prognosis…
[S2|PMC7540531_HIS-77-460|22044] In melanoma, its prognostic role was most clearly shown in thick primary tumours…
```

Voter0 (`gemini-2.5-flash-lite`) returned **12** findings (voters 1 & 2:
3 each), e.g.:

| claim | verbatim_support | evidence |
|-------|------------------|----------|
| Mib-1 antibody used for staining | "sections were stained using mib-1 (dakocytomation, carpinteria, ca, usa; m7240)" | `S2\|PMC3564399_his0057-0212\|13512` |
| Mib-1 dilution 1:50 for SES | "for both ses and ncb 1:50 mib-1 dilution was used; 1:400 pph3 dilution was used" | `S2\|PMC3564399_his0057-0212\|13512` |

The verbatim is **not** in the shown chunk (fuzzy ratio 0.03–0.05). DB lookup:

* `te_id 13512` → real paragraph in **`PMC3564399_his0057-0212`**: *"Tissue
  sections (4 µm) were dewaxed in xylene and rehydrated before microwave antigen
  retrieval…"* — a methods section matching the voter's verbatim.
* `te_id 22044` → real paragraph in **`PMC7540531`** (this case's own paper) —
  the Ki-67/melanoma text actually shown.

So the voter's content and its citation are *internally consistent with each
other* but belong to the **wrong paper**. Prevalence (probe over the whole
related15 voter_cache, all L1/L2/L3 outputs): **6418 / 16273 findings (39 %)**
fail the structural citation check — 6006 `cross_document_source_error` across
**25** distinct foreign PMCIDs (~350–512 cites each, near-uniform), 296
`invalid_text_element_id`, 158 `nonexistent_source`, 1 `invalid_sentence_id`.
Per-voter: the failures are concentrated in voter0; voters 1 & 2 cite the case's
own paper. Probe: `legacy/scripts/probes/_citation_probe.py` (throwaway).

### Diagnosis (root cause located)
This is **not** model hallucination: a hallucinating model does not reliably
produce valid DB primary keys paired with coherent real text from the matching
paper. The pattern — one provider's voter, coherent foreign content + matching
foreign citation, systematic across the corpus — is **cross-paper contamination
from a Gemini batch custom_id misalignment**.

Root cause: **`batch/gemini_batch.py::retrieve()` reconstructs each result's
`custom_id` positionally** — `custom_id = custom_ids[i]` (line ~131), where
`custom_ids` is a comma-joined list stashed in `output_location` at submit time
in request order, and `i` indexes `batch_job.dest.inlined_responses`. This is
correct *only* if Gemini returns `inlined_responses` in the exact `src` order
with no drops. Every other provider (`openai_batch`, `azure_batch`,
`claude_batch`, `vertex_batch`) reads `custom_id` **back from the response
payload** (`data["custom_id"]` / `item.custom_id`), so they are immune. When
Gemini reorders or omits any response (e.g. an errored/filtered request not
emitted in place), each subsequent response is stamped with another request's
`{pmcid}__{chunk_id}__{level}__{vi}` → paper A's Gemini findings filed under
paper B. The `custom_ids[i] if i < len(custom_ids) else f"unknown_{i}"` fallback
silently masks the length mismatch instead of failing. A *partial* shift matches
the observed ~37 % (not 100 %) contamination. Same failure class as
[Bug 19](#) / [Bug 20](#) (both Gemini/batch citation defects).

Proper fix requires the google-genai inline-batch ordering/completeness contract
(SDK docs / a live probe): if responses can be key-matched, match on the key;
otherwise embed a recoverable marker per request and **fail loudly** on any
mismatch instead of the silent positional fallback.

### Impact — MEASURED: selected set is clean (0 %)
`legacy/scripts/probes/_replay_contamination.py` replayed the frozen cascade
(θ0.9/r0.1) over related15 and checked every **selected** finding's citation:

```
SELECTED findings (citation OFF, baseline): 2280
SELECTED findings (citation ON,  filtered): 2280
dropped by filter:                          0 (0.00%)
selected findings FAILING citation:         0 / 2280 (0.00%)
```

So although **37 % of all voter findings** are contaminated, **0 % survive into
the selected set**: the AgreementChecker treats the contaminated Gemini voter as
an outlier (its wrong-paper findings disagree with the two clean OpenAI voters),
so the chunk keeps a clean voter or escalates to L2/L3 (no Gemini). The strict-F1
**0.7135**, the E03 grounding curve, and the entire selected-set funnel are
**unaffected** — no rerun is required for the headline numbers, and the B-080
citation filter is a **0-drop no-op on the production path** (pure
defense-in-depth).

Residual exposure (do NOT read the voter directly without de-contaminating):
* **E10** (Gemini-alone baseline) and **E12** (voter-LOO dropping Gemini) consume
  the Gemini voter *bypassing* the agreement gate → their Gemini-attributed rows
  are contaminated and must not be reported as-is.
* **E07/E08/E09** (escalation / cost): the outlier Gemini voter can force *excess
  escalation* (chunks that would KEEP at L1 with 3 clean voters instead escalate),
  so cost/escalation-rate metrics may be mildly inflated — finding *quality* is
  not. Not yet quantified.

### Fix
**Root cause fixed in code (2026-06-20)** — `batch/gemini_batch.py`:
* `submit()` now tags every request with `metadata={"custom_id": <id>}`
  (`InlinedRequest.metadata`).
* `retrieve()` matches each response by its round-tripped
  `InlinedResponse.metadata["custom_id"]` instead of list position; falls back
  to positional **only** when metadata is absent and then logs a loud WARNING
  (never the silent `unknown_{i}` of before); and **raises** on a
  response/request count mismatch instead of grafting mislabelled output.

Two deliberate non-actions:
1. **Existing related15 `voter_cache` NOT regenerated.** The selected-set impact
   is 0 % (see Impact above), so the headline numbers don't need it; regenerating
   costs a live Gemini batch run. The cache remains contaminated for any path
   that reads the Gemini voter directly — see the E10/E12 residual.
2. **Symptom guard retained** — the B-080 citation filter stays on as
   defense-in-depth (0-drop on the production path today).

### Verification
* Logic: `tests/summarization/batch/test_gemini_custom_id_mapping.py` — reordered
  responses are matched by key (not position); a dropped response raises; absent
  metadata falls back to position **and warns**.
* **Live-verified (2026-06-21):** `legacy/scripts/probes/_gemini_roundtrip_probe.py` submitted
  4 marker-tagged requests to **each** Gemini voter — L1 `gemini-2.5-flash-lite`
  and L2 `gemini-2.5-flash` — and confirmed every response came back bound to the
  correct `custom_id` (marker match) with **no** "matched by POSITION" warning.
  So the google-genai 1.70 backend does echo `InlinedRequest.metadata` onto
  `InlinedResponse.metadata`, and the key-match is active for both Gemini models.
  The code defect is resolved; the only residual is the **already-contaminated
  related15 cache** (selected-set impact 0 %; E10/E12 still read it — see below).
* Reproduce the original contamination: `python3 legacy/scripts/probes/_citation_probe.py`
  (all-voter prevalence) and `legacy/scripts/probes/_replay_contamination.py` (selected-set
  0 %).

---

## Bug 82 — Cross-document citation check false-positives on bare-vs-suffixed pmcid

### Status / Severity / Surface
Fixed (2026-06-21) · Low · Summarisation,
`routing/provenance_validator.py` (and therefore the B-080 citation filter).

### Symptom
After the B-081 Gemini regeneration, the clean related15 / heldout caches still
showed a small residual of `CROSS_DOCUMENT_SOURCE_ERROR` (related15 107, heldout
129). These were **not** hallucinations or contamination — every one was the
document's *own* paper.

### Diagnosis
The corpus stores **suffixed document ids** (`PMC4329418_his0066-0409`), but the
models frequently cite the **bare canonical accession** (`PMC4329418`). The
cross-document guard compared the pmcid by exact string
(`cited_pmcid != self._pmcid`), so bare-vs-suffixed mismatched and was flagged as
a different document. On the clean caches **112/112** residual cross-doc fails
had the *same* base `PMC\d+` accession, spread across **every** voter
(gpt-4o-mini 36, gemini-flash 28, gpt-4.1-mini 27, claude-haiku 14,
gpt-4.1-nano 7 — not Gemini-specific), verbatim genuinely from the cited paper.
Because the B-080 filter drops on this code, it was false-positive-dropping
legitimate same-paper findings — and the re-calibration replay applies that
filter, so every config's F1 was slightly understated. Tail of [Bug 19](#) (the
regex was widened to *accept* suffixed pmcids, but the *comparison* stayed exact).

### Fix
`_base_accession(pmcid)` returns the bare `PMC<digits>` prefix; the
cross-document check now compares base accessions
(`_base_accession(cited) != _base_accession(doc)`). Bare-vs-suffixed match; a
genuinely different paper still has a different base accession and is caught.

### Verification
`legacy/scripts/probes/_citation_probe.py` on the clean caches after the fix: residual
cross-document **107 → 0** (related15) and **129 → 0** (heldout); no
`invalid_text_element_id` appeared (te_ids were always correct), leaving only 8
genuine malformed-citation `invalid_sentence_id` on related15 (0.05%). Tests:
`tests/summarization/helpers/test_citation_filter.py` —
`test_bare_accession_matches_suffixed_document` (kept) /
`test_different_base_accession_still_dropped` (still caught); full routing +
citation suite green (79 passed).

## Bug 83 — rebuild_from_cached_map citation-filter empty-chunk warning is cosmetic

### Status / Severity / Surface
Fixed (2026-06-22) · Low (cosmetic log spam, no behaviour/number change) ·
`scripts/rebuild_from_cached_map.py` → batch `finalize()` citation chokepoint
(`pipeline/stages/summarization/batch/runner.py`). Surfaced while building the
thesis funnel from the bridged 5-voter MAP.

### Symptom
Every `rebuild_from_cached_map` run logs, for *every* chunk of *every* paper:

```
WARNING  …  citation filter: empty chunk for pmcid=PMC10529628_dermatopathology-10-00035 — skipping (kept N findings)
```

17 such lines for held-out `PMC10529628` alone. The user asked why this fires on
every rebuild. It is **not** held-out-specific — the same warning fires on
related15; the held-out log was simply the one inspected.

### Evidence
* The warning originates in `provenance/citation_filter.py::citation_drop_indices`
  (`if not chunk: logger.warning("citation filter: empty chunk …"); return set()`)
  — a deliberate guard so a *missing* source index is never read as "every
  citation is invalid"; it keeps all findings.
* The batch consumer calls it with the per-chunk source index
  `handle.chunk_map.get(chunk_id)` (`batch/runner.py`).
* `BatchHandle.chunk_map` defaults to `{}` (`batch/models.py`). The rebuild
  handle is built by `rebuild_from_cached_map._build_handle_from_cached_json`,
  which sets `handle.finalized` from `audit_trail.map_chunks` but **never sets
  `chunk_map`** — the cached JSON stores `chunk_id` + `findings`, not the source
  sentence list. So `chunk_map.get(chunk_id)` is `None → or [] →` empty for
  every chunk → the guard fires once per chunk.

### Diagnosis
The cached MAP artifact carries no per-chunk source index, so the citation
filter has nothing to validate against in the rebuild path and short-circuits
(keeps everything) with a warning per chunk. The filter is therefore a **no-op**
in `rebuild_from_cached_map`. Crucially it is *also* a no-op in the headline
sweep on this corpus, so the rebuilt knowledge base is **not** desynced from the
reported numbers:

* `_replay` defaults `citation_enabled=True`, so the headline strict-F1 **0.7160**
  and E03's **2294** retention base are *post*-citation-filter.
* Under the shipped 5-voter (`drop_l2_2`) / escalate config the B-080 structural
  filter drops **0** findings from the selected MAP set — verified two
  independent ways (below). (B-082's residual "8 `invalid_sentence_id`" was a
  pre-5-voter / 6-voter measurement; the shipped selection contains none of
  them.)

### Fix
`batch/runner.py::finalize()` now detects the whole-handle empty-`chunk_map`
case (the rebuild path) up front: it skips the per-chunk citation filter and
emits **one DEBUG line** —
`"citation filter skipped: rebuild handle carries no per-chunk source index (N chunks); see B-083"` —
instead of N WARNINGs. The per-chunk warning in `citation_drop_indices` is left
intact, so the live pipeline (where `chunk_map` is populated and a *single*
empty chunk among populated ones is genuinely suspicious) still warns as before.
Deliberately **not** carrying the source index into the cached JSON +
`chunk_map`: it buys nothing given the verified 0 drops, and would bloat every
per-paper artifact with the full sentence index.

### Verification
* Bridge `--validate --all` (related15): bridge replay (citation **OFF**) ==
  sweep `_replay` (citation **ON**) for **all 15 papers**, 2294 == 2294 total
  findings, every paper `MATCH`.
* Direct `_replay` citation ON/OFF diff under the shipped config:
  `citation ON=2294  OFF=2294  drops=0`.
* After the fix, a rebuild emits a single per-paper DEBUG note (suppressed at the
  default INFO level) instead of one WARNING per chunk.

## Bug 84 — no-db rebuild runs DB-coupled post-run steps (row-audit + incremental corpus-relate)

### Status / Severity / Surface
Fixed (2026-06-22) · Low (cosmetic for the funnel; wasteful + leaky isolation) ·
`scripts/rebuild_from_cached_map.py --no-db` → batch `finalize()`
(`pipeline/stages/summarization/batch/runner.py`). Surfaced during the held-out
5-voter funnel rebuild, alongside [B-083](#bug-83--rebuild_from_cached_map-citation-filter-empty-chunk-warning-is-cosmetic).

### Symptom
Running the held-out rebuild with `--no-db`:

```
[PMC12129600_HIS-87-35] RESOLVE — done → 59 final rules
[PMC12129600_HIS-87-35] Post-run row-count audit — 5 table(s) below expected:
  [FAIL] rows:sum_map_findings  0 rows for pmcid=PMC12129600_HIS-87-35 (expected ≥1)
  … (all 5 sum_* tables FAIL, every paper)
CORPUS RELATE [PMC12129600_HIS-87-35]: comparing 44 new rules × 1729 existing rules
[corpus] RELATE gate rejected 1570878/1570878 pairs
```

### Evidence
* `--no-db` sets `pipeline_run_db_id=None` (`rebuild_from_cached_map.main`), so
  every per-stage persister no-ops — **but the runner still holds a live DB
  connection**, because grounding needs `replace_verbatim_from_db` to read the
  cited source paragraphs. So `self._db is not None` stays true.
* The post-run row-count audit (`batch/runner.py`, Issue G) gated only on
  `self._db is not None` → it queried `sum_*` tables that were deliberately
  never written → `[FAIL] 0 rows` for all 5 tables of all 15 papers.
* `_corpus_relate_incremental` gated only on `self._db is None` (early-return);
  it calls `relate_incremental`, whose "existing" pool is
  `_load_rules_from_db(new_pmcid, db)` — **the DB**, not the rebuild's
  `--summaries-dir`. The DB held the *related15* corpus, so `existing = 1729`
  (related15's exact `canonical=final` count). `RelateStage.relate(all_rules)`
  then enumerates all pairs of the union: `44 + 1729 = 1773`,
  `C(1773, 2) = 1 570 878`. The cheap subject/outcome gate rejects all of them
  before any NLI runs → 0 relations, but the enumeration runs 15× (once per
  paper) and `_replace_for_pmcid` issues a 0-row delete per paper.

### Diagnosis
`--no-db` was implemented as "create no `sum_pipeline_run` row" (null
`pipeline_run_db_id`) rather than "touch no DB", because the runner legitimately
needs DB **reads**. The two DB-coupled *post-run* steps keyed off the wrong
signal (`self._db`) and so ran in a mode where they are meaningless: the audit
checks rows that by-construction don't exist, and the incremental corpus-relate
pools against whatever *other* corpus happens to be in the DB. Neither affects
any reported number — the **E04 funnel is per-paper** and reads neither the
audit nor corpus-relate — but both are wasted work and break the intended
isolation of a held-out split from the shared corpus tables.

### Fix
Gate both on `pipeline_run_db_id is not None` in `finalize()`:
* row-count audit → `if self._db is not None and pipeline_run_db_id is not None:`
* incremental corpus-relate → `if pipeline_run_db_id is not None: self._corpus_relate_incremental(...)`

Real DB runs (`pipeline_run_db_id` set) are unchanged. The final file-based
`_run_corpus_relate → relate_from_dir` pass — which pools rules from the split's
*own* dir and writes the isolated `<split>/corpus_relations.json` — is untouched
and is the authoritative cross-paper result; it is constructed with `db=None`
(`run_paper._run_corpus_relate`), so it never persisted to the DB and needed no
change. No separate "skip persist under `--no-db`" edit was required.

### Verification
`python3 -m py_compile` clean. A subsequent `--no-db` rebuild emits neither the
per-table `[FAIL] 0 rows` audit block nor the `44 new × 1729 existing` /
`1570878 pairs` incremental-relate lines; the per-paper funnel output
(`N final rules`, the per-paper JSON) is byte-identical (the removed steps never
fed it). Held-out cross-paper relations still produced by the final
`relate_from_dir` pass over `out/summaries/heldout15/summaries`.

## Bug 85 — "Docling baseline" is not off-the-shelf Docling

**Status / Severity / Surface.** Observed / Low / Thesis §4.1 (Results,
document-extraction) and the eval E01 sweep variant `01_docling`.

**Symptom.** §4.1 labels sweep variant `01_docling` the "Docling baseline" and
states "the figure-extraction behavior of Docling was not modified," implying the
40 % table strict-F1 is what stock Docling scores. It is not: every sweep variant,
`01_docling` included, runs the full document-extraction pipeline.

**Evidence.** `scripts/eval/run_all_sweeps.py:179-208` — `01_docling` enables
two-pass ghost-text (`two_pass.enabled=True`), region masking, artifact filtering,
and the caption/cropping machinery; it only *disables* the table-handling knobs
(footnote-expand, reconstruct, geometry drops, TATR). Figures additionally pass
through `nearest_caption`, `min_figure_pts` icon/size filtering, and sub-figure
merge in `media_cropper.py`. So `01_docling` = "the pipeline with Docling-only
table detection and table-handling off," not vanilla Docling.

**Diagnosis.** The table *detection* in `01_docling` is genuinely Docling's, so the
detection/footnote numbers are ≈ stock; but caption (via `nearest_caption`) and
figure FP-suppression are pipeline contributions credited to "Docling."

**Mitigation (measurement, not code).** Built a true off-the-shelf baseline —
`scripts/eval/baseline_offtheshelf_docling.py` (stock `DocumentConverter`, native
`doc.pictures`/`doc.tables`, native `caption_text`, no pipeline scaffolding),
variant `00_docling_offtheshelf`, labels seeded from `01_docling` via
`scripts/eval/transfer_offtheshelf_labels.py`, then fully hand-labelled. **Final**
(all 34 tables + 139 figures labelled): stock-Docling **tables 36.6 % strict-F1**
(13/21/24) vs `01_docling` 40.0 % vs v18 83.8 %; **figures 44.7 %** (40/99/0) vs
pipeline 84.0 %. Tables: both stock and `01_docling` fail the footnote dimension
(foot P ~44 %) → capped ~37–40 %, so the 40→83.8 gain is **footnote-driven**;
`nearest_caption` lifts the table baseline ~3 pts (40.0 vs 36.6) — scaffolding
inflates it *slightly* (caption), not *not at all*; 36.6 % is the low end of the
predicted [36.6–42.3 %]. Figures: stock over-emits (139 vs 76; 73 icon-FPs vs 14 →
crop P 44.6 % vs 81.6 %) and under-captions (caption P 60.6 % vs 90.3 %) → the
pipeline's icon filter + sub-figure merge + `nearest_caption` are worth ~39 strict-F1
pts. See RESULTS.md **E01b**.

**Verification.** Done for the measurement (re-scored clean, 0 unlabelled / 0
unrecognised). Remaining: the §4.1 prose reframe (rename `01_docling` → "pipeline
baseline, Docling-only table detection"; correct the "Docling figure behaviour was
not modified" claim; add the off-the-shelf row) — drafted for paste, tracked in
THESIS.md TODOs. Note the annotation-contamination incident (B-086) hit during this
work and was reverted.

## Bug 86 — annotate.py share-map propagation corrupts peer variants for caption-divergent crops

**Status / Severity / Surface.** Observed / Low / `eval/annotate.py --variant` +
`eval/annotations/share_map.json`.

**Symptom.** Hand-labelling `00_docling_offtheshelf` silently modified 51 *other*
variants' tracked annotation files; their scores shifted (e.g. `01_docling` figure
strict-F1 84.0→80.3) although only the off-the-shelf variant was annotated.

**Evidence.** `git status` showed `M` on nearly every `eval/annotations/<variant>/
json_figures.json` + several `json_tables_docling.json`; `git diff` showed pipeline
figures flipped `"correct"` → `"correct figure, no caption"` — exactly the
off-the-shelf labels.

**Diagnosis.** `annotate.py` propagates each label to peer variants sharing the crop
key `p{page}_{type}_{x1}_{y1}` (`share_map.json`, built across the pipeline sweep).
The off-the-shelf crops share keys with the pipeline variants but carry **different
caption metadata** (stock attaches no caption where `nearest_caption` does), so a
rubric label that encodes caption-correctness is **not** valid for the peers. The
propagation's "shared crop ⇒ shared label" assumption breaks whenever caption/footnote
attachment is variant-specific.

**Mitigation.** Reverted with `git checkout HEAD -- eval/annotations/` (the
off-the-shelf dir is untracked → its labels survived); re-scored clean. When
annotating a caption-divergent variant in future, revert tracked `eval/annotations/`
afterward, or annotate with share-map propagation disabled. **Re-annotating
`00_docling_offtheshelf` will re-corrupt the peers.**

**Verification.** Post-revert score: `01_docling` back to figures 84.0 % / tables
40.0 %, v18 83.8 %; off-the-shelf 44.7 % / 36.6 % unchanged. `git status
eval/annotations/` clean except the untracked off-the-shelf dir.

## Bug 87 — hybrid-detector docstring says "IoU" but merge is boolean-overlap

**Status / Severity / Surface.** Fixed (2026-07-01) / Low / PDF extraction,
`pipeline/stages/pdf_text_extraction/table_detectors/hybrid_detector.py` module
docstring.

**Symptom.** The docstring said `HybridTableDetector` "merges overlapping bounding
boxes using iterative **IoU-based** union", implying an IoU-threshold merge that
does not exist.

**Evidence.** The `_merge` method calls `parsers.layout_utils.merge_rects`
(`hybrid_detector.py:96,124`), whose own docstring states: *"Merges on
`Rect.intersects` (boolean overlap), **not on IOU threshold**. Any non-zero
overlap absorbs the smaller rect into the union."* (`parsers/layout_utils.py:103-106`).
A repo-wide search found **no** IoU threshold: no `iou` / `overlap_thresh` /
`min_overlap` code path and no config knob. `merge_rects` is a shared helper reused
by region masking (`region_masker.py`), two-pass redaction (`two_pass_extractor.py`),
and the hybrid detector — all boolean-overlap.

**Diagnosis.** Documentation-only inaccuracy in one docstring; the merge behaviour
(iterative boolean-overlap union) was correct and intentional (`merge_rects`
explicitly notes it is "not on IOU threshold"). The choice was never A/B-tested
against IoU — the E01 detector sweep varied detector / threshold / footnote-expand /
drop flags, not the merge rule — but boolean-overlap is the sensible default for a
union-style hybrid (fuse both detectors' boxes for the same table into one).
The thesis text ("the overlapping bounding boxes are merged into one") already
described the real behaviour correctly.

**Fix.** Corrected two docstrings in `hybrid_detector.py`: the module docstring
(line 6, "IoU-based union" → "boolean-overlap union", now explicitly "NOT an IoU
threshold — see `merge_rects`") and the `detect()` return doc (line 65,
"IoU-merged regions" → "overlap-merged regions"). No behaviour change; no
thesis-text change.

**Verification.** `merge_rects` unchanged; `HybridTableDetector` runtime output
unchanged (docstring-only edits). The only remaining "IoU" occurrence in
`hybrid_detector.py` is the intentional clarifying phrase "NOT an IoU threshold";
no docstring now *claims* IoU-based merging.

## Bug 88 — stats_writer stage-failure timing dropped by undefined `exc`

**Status / Severity / Surface.** Fixed (2026-07-11) / Medium / PDF extraction,
`pipeline/stages/pdf_text_extraction/outputs/stats_writer.py` →
`DocStatsCollector.stage()`.

**Symptom.** When a pipeline stage raised, the per-document stats JSON came out
with `stage_timings: []` — the failed stage's timing entry (with `ok=False` and
the exception type in `error`) was missing. `status="failed"`, `failed_stage`, and
top-level `error` were still written, so the loss was silent unless you inspected
`stage_timings` directly.

**Evidence.** Three tests reproduce it:
`test_stats_writer.py::test_collector_stage_captures_failure_and_reraises`
(`IndexError: list index out of range` on `stage_timings[0]`) and
`test_run_document_failed_doc_stats.py` ×2
(`assert len(data["stage_timings"]) == 1` → got 0; partial-progress case lost the
failing `STEP2_TABLE_DETECTION` entry). The swallowed inner error is visible in the
captured log: `NameError: name 'exc' is not defined` raised from the recording
lambda at `stats_writer.py:163`, suppressed by `_safe()` at line 253.

**Diagnosis.** The context manager caught the stage exception with a bare
`except Exception:` (no binding) but the recording lambda referenced
`type(exc).__name__`. Python raised `NameError` **inside** the `except` block; that
`NameError` was routed through `_safe()` (which logs-and-suppresses record ops), so
the `stage_timings.append(...)` never ran. Control then hit the bare `raise`, which
correctly re-raised the *original* stage exception — masking the fact that the
timing record had been lost. Because `_safe()` is designed to never let a stats op
break the pipeline, the defect produced no traceback in normal runs.

**Fix.** `except Exception:` → `except Exception as exc:` (`stats_writer.py:160`),
binding the type name to a local *before* the lambda:
```python
except Exception as exc:
    err_name = type(exc).__name__
    self._safe(lambda: self._stats.stage_timings.append(
        _StageTiming(name=name, seconds=time.monotonic() - t, ok=False, error=err_name)))
```
**Do not** collapse this back to `error=type(exc).__name__` inside the lambda: ruff's
`F841` autofix (`ruff check --fix`) does not see the closure use, judges `exc` unused,
and strips `as exc` — which silently re-introduces this exact bug. That regression
actually happened once (a `ruff --fix` run reverted the first fix; the `err_name`
local now makes `exc` an unmistakable direct use so the autofix leaves it alone). See
also the 2026-07-11 ruff-config changelog row in STRUCTURE.md.

**Verification.** `python -m pytest tests/pdf_text_extraction/test_stats_writer.py
tests/pdf_text_extraction/test_run_document_failed_doc_stats.py` → 45 passed
(was 3 failing). Full suite green (1406 passed); `ruff check .` clean.

## Bug 89 — test modules leak `NLP_HISTO_DISABLE_UMLS` process-wide

**Status / Severity / Surface.** Fixed (2026-07-11) / Medium / Test infra,
`tests/knowledge_extraction/test_persist_voter_outputs.py`,
`tests/knowledge_extraction/test_batch_persistence.py`.

**Symptom.** `tests/knowledge_extraction/test_phase_a_gate.py::test_normalize_stage_normalizes_entities`
**passed in isolation** (≈88 s; the scispaCy/UMLS linker loads and resolves
`"CD31 expression"` → canonical `"CD31 Antigens"`) but **failed in a full-suite
run** (the whole suite finished in ≈31 s — UMLS never loaded). Classic
order-dependent flake: pass alone, fail together.

**Evidence.** Reproduced deterministically by running the polluter before the
victim:
`pytest tests/knowledge_extraction/test_persist_voter_outputs.py
tests/knowledge_extraction/test_phase_a_gate.py::test_normalize_stage_normalizes_entities`
→ the normalize test fails in ~3 s (UMLS disabled, so the outcome entity is never
canonicalised). `grep -rn "os.environ.setdefault.*DISABLE_UMLS" tests/` found two
module-level setters.

**Diagnosis.** Both modules did
`os.environ.setdefault("NLP_HISTO_DISABLE_UMLS", "1")` at **import time** to skip
the linker load they don't need. `os.environ` mutation is process-global and
`setdefault` never resets it, so once pytest imported either module the kill-switch
stayed on for the rest of the session — disabling UMLS for every later test,
including the one test that genuinely needs it. The victim's assertion was correct;
the failure was pure cross-test pollution. (Sibling modules like
`test_relate_cui_gate.py` set the same var via `monkeypatch.setenv`, which
auto-reverts — those never leaked.)

**Fix.** Replaced both module-level setters with an autouse fixture
(`monkeypatch.setenv(...)`, reverted after each test). Separately,
`norecursedirs`-excluded the gitignored, vendored `pdffigures2/` tree in
`pyproject.toml` — pytest had been collecting that third-party tool's own tests,
which fail at import (`module 'pdffigures2.evaluation.datasets' has no attribute
'DATASETS'`) and are not part of this project's suite.

**Verification.** Polluter-then-victim ordering now passes (UMLS loads, ~59 s).
Full suite runs clean at 1406 passed / 0 failed in ≈152 s (UMLS loads exactly
once, as intended) — vs the earlier 31 s "fast" run that silently skipped it.

## Topic — stale-test sweep (2026-07-11)

A full `pytest` run showed 29 failures. Exactly **two** were live defects
(B-088, B-089). The remaining **27 were stale tests** asserting behaviour that had
been deliberately superseded by shipped changes; the production code was correct in
every case. Fixed by updating the tests to the current contracts (no product-code
changes beyond B-088). Grouped by root cause:

| Tests | Superseded-by (shipped change) | Test update |
|-------|-------------------------------|-------------|
| `test_cli_cropping_flags` (×8), `test_drop_tables_inside_figures::test_main_*` (×3) | `runner.main()` pre-filters the PDF list and dispatches via `ParallelBatchRunner.run_paths(paths)`, not `run(pdf_dir)` (using `run` would re-glob and discard the filtering). Also `CroppingConfig.expand_tables_with_footnotes` default frozen `False→True` on 2026-05-21. | Added `run_paths()` to the `_CapturedConfig` stand-ins; updated the `expand_tables_with_footnotes` default assertions to `True`. |
| `test_corpus_relate` (×4) | `RelateStage.relate()` returns a 3-tuple `(relations, raw_pairs, skipped_pairs)`; `corpus_relate.py` unpacks 3. | Mocks changed from `return_value=[fake_relation]` / `[]` to `([fake_relation], [], [])` / `([], [], [])`. |
| `test_config_loader::test_summarization_section_independent` | `MapConfig.reject_theta` default changed `0.2→0.0` on 2026-06-10 (hard-REJECT drop off by default). | Assertion updated to `0.0`. |
| `test_inspector` (×2, polarity-mismatch) | Refactor 77bb1ac replaced `assertion_status=="uncertain"` with `direction=="unclear"` and dropped `assertion_status`; the trigger field was stripped from the fixtures but never replaced. | Added `direction:"unclear"` to both fixture objects. |
| `test_dual_matcher_metrics::test_rank_tiebreaks_*` (×1), `test_screen_refine` economy (×2) | Selection cost axis switched `escalate_rate → _cost_frac` (price-weighted L2+L3 escalation from `n_chunks`/`n_l2_invoked`/`n_l3_invoked`). Fixtures only set `escalate_rate`, so `_cost_frac` returned 0 and the tiebreak/Pareto frontier collapsed. | Fixtures now synthesise the count fields so `_cost_frac` reproduces the intended cost. |
| `test_checkpoint_resume` (×2) | `map_theta` stage pins `BEST_VOTER_SUBSET="drop_l2_2"` (calibration selection 2026-06-22) → 7-tuple cell keys; production stamps `voter_subset` into every persisted row. | Synthetic rows now stamp `voter_subset` (matching production), so cell/row keys agree.

Additionally, 2 pytest failures were the vendored `pdffigures2/` package's own
import-broken tests — not part of this suite; excluded via `norecursedirs`
(see B-089).

## Bug 90 — null-dep tests crash pytest-randomly teardown reseed

**Status / Severity / Surface.** Fixed (2026-07-11) / Medium / Test infra,
`tests/pdf_text_extraction/test_b027_seed_and_cache.py`
(`test_seed_pipeline_handles_missing_torch`, `…_missing_numpy`) × the newly-added
`pytest-randomly` plugin.

**Symptom.** With `pytest-randomly` installed, a full-suite run under `--randomly-seed=1`
produced **1 failed, 1229 passed, 353 errors** (vs 0 under deterministic order).
Every one of the 353 errors was identical:
`ModuleNotFoundError: import of torch halted; None in sys.modules`.

**Evidence.** The traceback pins it to *teardown*, not the test body:
```
ERROR at teardown of test_seed_pipeline_handles_missing_torch
  pytest_randomly/__init__.py:_reseed
    thinc/util.py:fix_random_seed
      torch/random.py:manual_seed
        → ModuleNotFoundError: import of torch halted; None in sys.modules
```
`grep -c "import of torch halted"` = 353 — a single root cause, not 353 independent bugs.

**Diagnosis.** The two tests simulate a missing optional dependency by setting
`sys.modules["torch"]` / `["numpy"] = None` (Python's "this import must fail"
sentinel) via `monkeypatch.setitem`. `pytest-randomly` reseeds all registered RNGs
(`random`, `numpy`, `faker`, and — because scispaCy/thinc is installed — `torch` via
`thinc.fix_random_seed`) at **every test's setup and teardown**. Its teardown reseed
hook runs *before* `monkeypatch`'s finalizer restores `sys.modules`, so it calls
`torch.manual_seed` while `sys.modules["torch"] is None` → `ModuleNotFoundError`. That
exception aborts teardown, so monkeypatch's undo never runs and the `None` sentinel
**leaks permanently**; every subsequent test's own teardown reseed then hits the same
wall → the cascade. Only manifests with `pytest-randomly` present (no reseed hook, no
bug without it).

**Fix.** Restore the nulled module inside the test body with `try/finally` (helper
`_null_module_during(name)`) instead of relying on `monkeypatch.setitem`, so
`sys.modules` is real again *before* the teardown reseed fires. The test still runs
`_make_runner` with the module nulled (the behaviour under test is unchanged).

**Verification.** `--randomly-seed=1` → 1406 passed / 0 errors (was 353 errors);
seeds 42 / 1234 / 9999 also clean. See also [Bug 91](#bug-91--get_nlp-caches-the-umls-disable-decision-poisoning-later-tests)
(the seed=1 `1 failed`, a separate order-dependence surfaced by the same probe).

## Bug 91 — `get_nlp()` caches the UMLS disable-decision, poisoning later tests

**Status / Severity / Surface.** Fixed (2026-07-11) / Medium / Summarisation /
test infra, `pipeline/stages/knowledge_extraction/umls_resources.py` → `get_nlp()`
singleton, exposed via `pytest-randomly`.

**Symptom.** Under `--randomly-seed=1`, `test_phase_a_gate::test_normalize_stage_normalizes_entities`
failed and the whole suite finished in ~28 s — i.e. scispaCy **never loaded at all**
(a real load is ~83 s). The test needs the linker to canonicalise
`"CD31 expression" → "CD31 Antigens"`; it silently got `None`.

**Evidence.** `get_nlp()` (`umls_resources.py:100-111`) short-circuits on a cached
`_AVAILABLE` flag: `if _AVAILABLE is not None: return _NLP`. `_AVAILABLE` is set to
`False` the first time the function is entered while `umls_disabled()` is true, and the
env var (`NLP_HISTO_DISABLE_UMLS`) is **never read again**. All `DISABLE_UMLS` tests
live in `tests/knowledge_extraction/`, alongside the victim; randomized order let one
of them probe the singleton first.

**Diagnosis.** The singleton is process-wide and load-once by design (double-loading the
UMLS KB OOM-kills the pipeline). Its first caller's env decision is frozen into
`_AVAILABLE`. A `DISABLE_UMLS` test (kill-switch on via `monkeypatch`) that calls
`get_nlp()` caches `_AVAILABLE=False`; when its `monkeypatch` reverts the env var the
cache does **not** follow, so every later caller — including the one test that genuinely
needs UMLS — gets `None`. This is a distinct, deeper mechanism than the env-var leak of
[B-089](#bug-89--test-modules-leak-nlp_histo_disable_umls-process-wide): even with
perfectly-scoped env fixtures, the *singleton* still caches the first decision. Correct
in production (the env var is fixed at process start), so a runtime code change would be
over-engineering — this is a test-isolation concern.

**Fix.** Autouse fixture `_unpoison_umls_singleton` in
`tests/knowledge_extraction/conftest.py`: after each test, if `_AVAILABLE is False`
(a disabled/failed cache — never a successful load, which is `True`), call
`umls_resources._reset_for_tests()` so the next real caller re-probes the env and loads
cleanly. Zero reload cost — a `False` cache means nothing was loaded to discard.

**Verification.** `--randomly-seed=1` now loads scispaCy once (~158 s) and passes
1406/1406; seeds 42 / 1234 / 9999 clean. Deterministic-order run also still green.

## Bug 92 — pytest-randomly × thinc seed overflow breaks default `python -m pytest`

**Status / Severity / Surface.** Fixed (2026-07-11) / High / Test infra, the
`pytest-randomly` plugin (adopted 2026-07-11) × thinc's random-seed entry point.

**Symptom.** A plain `python -m pytest` (no `--randomly-seed`) reported **1680 errors**
alongside 540 passed:
`ValueError: Seed must be between 0 and 2**32 - 1`, one at nearly every test's setup
and teardown. Passed clean under the explicit seeds used during development
(`--randomly-seed=1/42/1234/9999`).

**Evidence.** Forcing a large seed reproduces it deterministically:
`pytest eval/silver/tests/test_embedder_backoff.py --randomly-seed=4294967290`
→ every test errors (8 errors for 4 tests = setup + teardown). The registered
reseeders are exactly one:
`python -c "from importlib.metadata import entry_points; print(list(entry_points(group='pytest_randomly.random_seeder')))"`
→ `thinc.api:fix_random_seed`. In `pytest_randomly/__init__.py::_reseed`, line 158
clamps the built-in numpy reseed (`np_random.seed(seed % 2**32)`) but line 164 calls
each entry-point reseeder with the **raw** `seed`.

**Diagnosis.** pytest-randomly reseeds every test with
`seed = base_seed + crc32(nodeid)` — up to ~2**33. It clamps its own numpy reseed but
NOT third-party entry-point reseeders. thinc's `fix_random_seed(seed)` (pulled in by
scispaCy) forwards the value straight to `numpy.random.seed`, which rejects anything
> 2**32-1. The default base seed is `random.Random().getrandbits(32)` (0…2**32-1); when
it lands high, `base + crc32(nodeid)` overflows for most tests → the cascade. Tiny
explicit seeds keep `base + crc32` under 2**32 for almost every test, which is why the
B-089/B-090/B-091 verification (all small seeds) never tripped it. A regression
introduced by adopting `pytest-randomly`.

**Fix.** Repo-root `conftest.py` `pytest_configure` pre-populates
`pytest_randomly.entrypoint_reseeds` (the module global pytest-randomly lazily builds on
first reseed) with clamped wrappers — each entry-point reseeder receives `seed % 2**32`,
mirroring pytest-randomly's own numpy clamp. Runs before the first reseed at collection.
thinc/torch/numpy still get reseeded deterministically, just in range.

**Verification.** Post-fix, `--randomly-seed=4294967290` and `4294967295` (both
previously all-error) pass the `test_embedder_backoff.py` set; full-suite run under the
large seed and under a plain default `python -m pytest` (large random base seed) both
green at 1406 passed / 0 errors.

## Bug 93 — NORMALIZE test flakes when the scispaCy UMLS linker S3 fetch fails

**Status / Severity / Surface.** Observed / Low / Test infra + network dependency,
`tests/knowledge_extraction/test_phase_a_gate.py::test_normalize_stage_normalizes_entities`.

**Symptom.** One full-suite run (2026-07-12, `python -m pytest -q`) reported
`1 failed, 1402 passed`; the single failure was
`AssertionError: assert 'CD31 expression' == 'CD31 Antigens'` at `test_phase_a_gate.py:228`.
Re-running the test in isolation: **1 passed**.

**Evidence.** Captured log for the failing test:
`WARNING umls_resources: UMLS: linker unavailable — downstream stages will skip CUI work:
HTTPSConnectionPool(host='s3-us-west-2.amazonaws.com', port=443): Max retries exceeded …
NameResolutionError("Failed to resolve 's3-us-west-2.amazonaws.com'")`.

**Diagnosis.** The scispaCy `en_core_sci_lg` UMLS **linker** downloads KB resources
(e.g. `umls_semantic_type_tree.tsv`) from S3 on first load when not already cached. In that
run the sandbox could not resolve `s3-us-west-2.amazonaws.com` (DNS/network unavailable),
so `get_linker()` returned unavailable and NORMALIZE skipped CUI normalisation, leaving the
raw surface form `'CD31 expression'` instead of the UMLS-canonical `'CD31 Antigens'`. This
is a **network-dependent flake**, distinct from B-091's order-triggered `_AVAILABLE`
cache-poisoning (already fixed): here the linker genuinely could not fetch its resource. Not
caused by the concurrent `chore(scripts): remove superseded model-connection checker`
deletion — an unreferenced script; the test passes in isolation.

**Mitigation.** None applied here; tracked separately from the Phase 5 · Commit 4 deletion,
which made no test/code change. Candidate follow-ups: pre-cache the scispaCy linker
resources so the suite needs no network, or `pytest.skip` the UMLS-linker-dependent
assertion when the linker is unavailable.

**Verification.** Isolation re-run passed (1 passed); a controlled full-suite re-run on the
staged-deletion state (seed=1) was run as the Phase 5 · Commit 4 flake check.

---

## Bug 94 — Stage-walker inspection CLIs are not directly runnable

**Status / Severity / Surface.** Fixed (2026-07-13) · Medium ·
`scripts/inspect/inspect_map_normalize.py`,
`scripts/inspect/inspect_normalize_group.py`,
`scripts/inspect/inspect_phase123_pipeline.py` — direct execution.

**Symptom.** The documented commands do not work:

```bash
python scripts/inspect/inspect_map_normalize.py PMC10047158
python scripts/inspect/inspect_normalize_group.py PMC7150310_main
python scripts/inspect/inspect_phase123_pipeline.py PMC7150310_main
```

(these exact forms appear in each script's own docstring, in `HOW_TO_RUN.md`, and in
`REPOSITORY_GUIDE.md`). They raise `ModuleNotFoundError` before doing any useful work.

**Evidence (fully static — the scripts were never executed).**

1. *sys.path.* The repository is **not installed as a package** (`import database` from
   outside the repo fails). Python sets `sys.path[0]` to the **script's directory**, not the
   cwd — proven with a probe script run from the repo root but located elsewhere:
   `sys.path[0] = <probe dir>`, `repo root on sys.path? False`, `import database ->
   ModuleNotFoundError`. So a direct run puts `scripts/inspect/` on the path, and both
   `import database` and `import pipeline` fail.
2. *Stale module paths.* Four lazily-imported modules moved during the `stages/` +
   `grounding/` refactors and were never updated in these scripts. File-existence check:

   | Imported by the scripts | Exists? | Actual location |
   |---|---|---|
   | `…knowledge_extraction.map_stage` | no | `…knowledge_extraction.stages.map_stage` |
   | `…knowledge_extraction.normalize_stage` | no | `…stages.normalize_stage` |
   | `…knowledge_extraction.group_stage` | no | `…stages.group_stage` |
   | `…knowledge_extraction.grounding_filter` | no | `…grounding.grounding_filter` |
   | `…knowledge_extraction.cache` / `.models`, `database`, `database.models` | yes | unchanged |

   These fail **even with** the repo root on `sys.path`, so a bootstrap alone is not enough.

**Diagnosis.** Two independent defects, both invisible to the usual gates: the repo-local
imports are **lazy** (inside `main()` / helpers), so `py_compile`, `ruff`, AST checks and a
module-level import-smoke all pass. No test imports these scripts. The sibling tools in the
same directory (`inspect_pipeline_output.py`, `viewer_server.py`) *do* carry the standard
bootstrap, as do `run_paper.py` and ~21 other scripts — these three were simply never given
one, and were never updated when the stage modules moved.

**No API drift.** Every imported symbol still exists and every call site matches the current
signatures — `MapStage(voter_llms, level2_voter_llms, escalation_llm, theta, chunk_size)` +
`.process(sentences, pmcid, cache=)`, `GroundingFilter(threshold=)` + `._pipe`,
`score_findings(findings, nli_pipe=)`, `NormalizeStage().normalize(...)`,
`GroupStage().group(...)`, `PipelineCache(Path)`, `is_groupable(nf)`. Only the import paths
are wrong.

**Proposed fix (imports only).** In each of the three scripts:

```python
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

placed before the repo-local imports, **plus** correcting the four stale module paths
(`.stages.` / `.grounding.`). No change to CWD behaviour (`out/inspect/` and all input/output
paths stay cwd-relative), CLI arguments, defaults, DB queries, LLM prompts/model config, or
printed output. No `scripts/inspect/__init__.py`; no package-install requirement.

**Safety.** `inspect_map_normalize.py` and `inspect_phase123_pipeline.py` make **real paid
Vertex/LLM calls** when executed. During this investigation **no script was run** and **no
database connection, network/API call, LLM/Vertex client, model download, cache write, or
output file** was created or modified.

**Related (not fixed here).** A bounded sweep of tracked `scripts/**.py` found 8 further
self-documented, bootstrap-less scripts affected by defect (1) only — `compare_docling_options`,
`compare_policies`, `compare_prefilter`, `copy_relevant_files`, `eval_policy`,
`fit_routing_threshold`, `select_policy`, `two_pass_extract` — plus `create_tui_gin_index`
(undocumented). `list_paper_sizes.py` documents `PYTHONPATH=. python …`, so its documented
command works. 23 scripts already carry a bootstrap.

**Fix.** `fix(scripts): make stage inspection CLIs directly runnable` (`a6d2927`, 2026-07-13).
Added the standard `_REPO_ROOT = Path(__file__).resolve().parents[2]` + `sys.path.insert`
bootstrap to all three scripts (after the stdlib block, before third-party imports), and
retargeted the nine stale repo-local imports to `…knowledge_extraction.stages.*` and
`…knowledge_extraction.grounding.grounding_filter`. Imports only — no CWD, CLI, default,
DB-query, prompt/model, threshold, constructor-argument, output-path or printed-output
change; no `scripts/inspect/__init__.py`.

**Verification.** Static, by necessity (two of the three make paid Vertex/LLM calls, so
end-to-end execution was deliberately **not** performed — the scripts were never run and
`main()` was never called):

* `python -m py_compile` on all three — pass.
* `importlib.util.find_spec` — all eight repo-local modules resolve
  (`…cache`, `…grounding.grounding_filter`, `…stages.{map,normalize,group}_stage`,
  `…models`, `database`, `database.models`); nothing instantiated.
* Import-smoke from a **non-repository cwd**: each script imported with `main()` never
  called, each `_REPO_ROOT` landed on `sys.path`, and `import database` then resolved —
  previously `ModuleNotFoundError`. This directly demonstrates the repaired path.
* Static checks: bootstrap precedes every repo-local import in each file; no `chdir`
  anywhere; `out/inspect/` and all input/output paths remain cwd-relative and untouched.
* Repo-wide search: zero executable stale imports remain in any tracked `.py`.
* `ruff check .` clean; full `python -m pytest` 1404 passed; `git diff --check` clean.

No database connection, network/API/LLM call, model download, cache write, or inspection
output occurred at any point.

**Follow-up (still open).** The bounded sweep's other findings are **not** fixed by this
commit: eight self-documented, bootstrap-less scripts remain broken under their own
documented bare invocation (`compare_docling_options`, `compare_policies`,
`compare_prefilter`, `copy_relevant_files`, `eval_policy`, `fit_routing_threshold`,
`select_policy`, `two_pass_extract`), plus `create_tui_gin_index` (undocumented, same
defect). `list_paper_sizes` is unaffected in practice because it documents
`PYTHONPATH=. python …`. None of these appears in project documentation. Track as a
separate bounded repair if/when that work is scheduled.

---

## Bug 95 — Direct-run scripts under `scripts/` lack a repository-root bootstrap

**Status / Severity / Surface.** Fixed (2026-07-13, 3e8c819) · Medium · nine scripts directly under `scripts/`.

**Symptom.** Eight of the nine self-document a bare direct command that cannot work as
written, e.g.:

```bash
python scripts/eval_policy.py ...
python scripts/two_pass_extract.py path/to/paper.pdf
python scripts/copy_relevant_files.py
```

They raise `ModuleNotFoundError` at import, before any argument parsing.
(`create_tui_gin_index.py` carries no usage line but has the identical defect.)

**Evidence (fully static — no script was executed, `main()` never called).**

1. *sys.path.* For `python scripts/<name>.py`, Python sets `sys.path[0]` to the **script's
   directory** (`<repo>/scripts`), not the cwd; the repo is not installed as a package.
   With `<repo>/scripts` as `sys.path[0]`, `importlib.util.find_spec` returns **None** for
   `pipeline`, `database`, `parsers` and `named_entity_recognition`.
2. *All repo-local imports are at **module level*** (unlike B-094's lazy imports), so the
   failure is immediate on import.
3. With `_REPO_ROOT = Path(__file__).resolve().parents[1]` inserted, all seven imported
   module paths resolve.

**No stale imports.** Every imported module path and symbol exists in the current tree
(`…pdf_text_extraction.components.{layout_extractor,text_assembler,two_pass_extractor}`,
`…pdf_text_extraction.config` incl. `OcrEngine`, `…pdf_text_extraction.models.scored_node`
incl. `TwoPassResult`, `…knowledge_extraction.routing.{routing_dataset,policy}`,
`parsers.layout_utils`, `database`, `named_entity_recognition.enums`). The repair is
**bootstrap-only** — this is the key difference from B-094.

**Side effects if run** (documented so they are never invoked casually):

| Script | Behaviour when executed |
|---|---|
| `create_tui_gin_index.py` | **DB DDL** — `CREATE INDEX IF NOT EXISTS idx_entity_semantic_types … USING GIN` + `commit()`; mutates the schema |
| `copy_relevant_files.py` | DB reads + **file copies** (`shutil.copy2`) |
| `compare_docling_options.py`, `compare_prefilter.py`, `two_pass_extract.py` | **Docling / two-pass extraction** over PDFs; write outputs |
| `compare_policies.py`, `eval_policy.py`, `fit_routing_threshold.py`, `select_policy.py` | Read the routing dataset / policy store; write reports |

**None makes paid LLM calls.** No database connection, extraction run, file copy, model
load, or output write occurred during this investigation.

**Ruff.** `# noqa: E402` is **not** needed: `pyproject.toml` already declares
`[tool.ruff.lint.per-file-ignores] "scripts/**" = ["E402"]`, added precisely because the
documented bootstrap pattern forces repo-local imports below `sys.path.insert(...)`.

**Proposed fix.** Add, immediately before the module-level repository-local imports:

```python
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

adding `import sys` where absent (5 scripts) and `from pathlib import Path` where absent
(2 scripts). Imports only — no change to CWD semantics, CLI arguments, defaults, database
queries, extraction settings, copying behaviour, output locations, logging, or function
structure. No `scripts/__init__.py`, no wrappers, no `PYTHONPATH` instructions.

**Separate finding — un-migrated DDL (not a bootstrap issue).** No Alembic migration
creates `idx_entity_semantic_types`; `create_tui_gin_index.py` is the only path to that
index, i.e. an **out-of-band schema change** in an otherwise Alembic-managed schema. Fixing
its bootstrap does not address that. The index arguably belongs in a migration — flagged
for a separate decision (repair, retire, or migrate).

### Bug 95 — approved repair scope (2026-07-13)

The repair is split three ways:

1. **Eight scripts — standard bootstrap repair (approved, Phase 8 · Commit 1).**
   `compare_docling_options`, `compare_policies`, `compare_prefilter`, `copy_relevant_files`,
   `eval_policy`, `fit_routing_threshold`, `select_policy`, `two_pass_extract`. Bootstrap
   only (`parents[1]`), plus `import sys` where absent (5) and `from pathlib import Path`
   where absent (1). No stale imports, no new `# noqa`, no behavioural change.

2. **`create_tui_gin_index.py` — HELD for a schema-governance decision.** It is the only
   one of the nine that **mutates the database schema** (`CREATE INDEX … USING GIN` +
   `commit()`), and no Alembic migration creates `idx_entity_semantic_types`, so it is an
   out-of-band DDL path in an otherwise Alembic-managed schema. Deliberately **left
   unrepaired** pending a separate decision: (a) replace with an Alembic migration,
   (b) retain as an explicitly documented administrative utility, or (c) retire.

3. **`list_paper_sizes.py` — no defect in practice.** It has the same missing bootstrap but
   documents `PYTHONPATH=. python scripts/list_paper_sizes.py`, so **its documented command
   works**. Excluded from the repair; a consistency-only cleanup at most.

**Verification (planned, no paid/DB/model/extraction work).** `py_compile` each changed
script; static checks for `parents[1]`, bootstrap placed before repo-local imports, no
`chdir`, unchanged relative-path expressions; `find_spec` resolution of every repo-local
module without instantiating clients/sessions/models; import-smoke only after AST proves
module-level code is side-effect-free (all nine qualify — module level is limited to
logging setup, `load_dotenv`, dataclass/config construction, and `Path(...)`); then
`ruff check .`, full `python -m pytest`, `git diff --check`.

---

## Bug 96 — Runner test writes stats into the real `out/` tree

**Status / Severity / Surface.** Observed · Low ·
`tests/pdf_text_extraction/test_b027_seed_and_cache.py` ×
`PathConfig.run_metadata_dir` (`pipeline/stages/pdf_text_extraction/config.py:101`).

**Symptom.** A plain `python -m pytest` creates/overwrites
`out/run_metadata/PMC1_stats.json` in the working repository.

**Evidence.** The file's contents identify it as test output:

```json
{ "pmcid": "PMC1",
  "pdf_path": ".../pytest-of-emir/pytest-116/test_runner_skip_false_does_no0/fake.pdf" }
```

`test_runner_skip_false_does_not_use_cache` (via `_build_runner_with_mocks`) supplies
`tmp_path / "fake.pdf"` as the input and `"PMC1"` as the pmcid, but leaves
`cfg.paths.run_metadata_dir` at its default `Path("out/run_metadata")`, so
`DocStatsCollector` writes into the repository's real output tree.

**Diagnosis.** Test-isolation gap, not a pipeline defect: the input path is isolated but the
*output* path is not. `out/run_metadata/` otherwise contains ~1070 genuine run artifacts
from real pipeline runs, so the stray `PMC1_stats.json` pollutes meaningful local output.

**Impact.** Cosmetic/hygiene only. `out/` is gitignored (`.gitignore:246`) and the file is
untracked, so nothing can leak into a commit. No production behaviour is affected.

**Fix (not applied).** Point the test's `cfg.paths.run_metadata_dir` (and any sibling output
dirs it exercises) at `tmp_path`, matching the other runner tests. Do **not** change the
production default in `config.py` — `out/run_metadata` is the correct runtime location.

**Disposition of the artifact.** `out/run_metadata/PMC1_stats.json` was **left in place**,
not deleted: it is provably test-generated, but the directory holds real artifacts and the
file is harmless where it is.

**Relation to Phase 8.** Discovered while verifying `fix(scripts): bootstrap direct-run
utility scripts` (`3e8c819`). **Unrelated to that commit** — none of the eight bootstrapped
scripts references `run_metadata`; the write came from the test suite.

---

## Bug 97 — `semantic_types` column type diverges between ORM and Alembic

**Status / Severity / Surface.** Observed · **High** · `entities.semantic_types`;
`database/models.py:294`; `alembic/versions/0006_add_entity_semantic_types.py:25`;
`scripts/create_tui_gin_index.py`.

**Symptom.** Two database-initialisation paths produce **different column types**:

| Path | Resulting type |
|---|---|
| `create_tables()` → `Base.metadata.create_all` (ORM, `models.py:294` `Column(ARRAY(Text))`) | `TEXT[]` |
| `alembic upgrade head` (revision `0006`: `ADD COLUMN IF NOT EXISTS semantic_types VARCHAR`) | `VARCHAR` |

`0006` is the **only** migration touching the column; the chain is linear `0001 → 0014`
(head `0014`) and no later revision alters it.

**Why it went unnoticed.** `0006` uses `ADD COLUMN IF NOT EXISTS`, so on a database that
already had the column — e.g. the live development DB, created by the one-off
`database/migrations/add_semantic_types.py` that `0006`'s docstring says it "absorbs" (that
script is no longer in the tree) — the revision is a **silent no-op**. The live DB therefore
almost certainly holds `TEXT[]` and behaves correctly, while a *fresh* Alembic-initialised
database silently gets `VARCHAR`.

**Consequences on an Alembic-built database.**

1. Array-overlap queries cannot work against `varchar`:
   * `named_entity_recognition/export_disease_entities.py:126` — `Entity.semantic_types.op('&&')(array(semantic_types))`
   * `scripts/copy_relevant_files.py:37` — same `&&` operator
2. `scripts/create_tui_gin_index.py` would **error**: `CREATE INDEX … USING GIN
   (semantic_types)` has **no default operator class for `varchar`** (GIN needs an array,
   `jsonb`, `tsvector`, or a `pg_trgm` opclass).
3. The ORM and the physical schema disagree, so `Entity.semantic_types` round-trips
   differently depending on how the DB was built.

**Sub-issue — unmanaged schema mutation.** `idx_entity_semantic_types` is declared in **no**
Alembic revision and in **no** `__table_args__`. `scripts/create_tui_gin_index.py` is the
only path to it: an out-of-band `CREATE INDEX` + `commit()` in an otherwise
Alembic-managed schema. (Separately, the "GIN full-text index on `text_content`" asserted in
`.claude/CLAUDE.md` exists in neither `database/models.py` nor any migration — the
documentation claim is unverified.)

**Operational notes on the index itself.** The statement is idempotent
(`IF NOT EXISTS`); it is **not** `CONCURRENTLY`, so it takes a lock that blocks writes to
`entities` for the duration of the build — potentially long on a large table. Note that
`CREATE INDEX CONCURRENTLY` **cannot** run inside a transaction block, so an Alembic
revision would need an autocommit block for it.

**Impact.** Low in the current thesis environment (the live DB is reused via `.env` and
appears to hold `TEXT[]`), but a genuine **clean-clone / fresh-database reproducibility
defect**: `alembic upgrade head` does not reproduce the working schema.

**Fix direction (not applied).** Reconcile the column type first — either correct `0006` to
create `TEXT[]` (safe only if no deployed DB depends on the current text) or add a
corrective revision `0015` that converts `VARCHAR → TEXT[]` where needed — and only then
settle ownership of the GIN index. **The index question cannot be decided independently of
the column type.**

**Safety.** Investigation was fully static: no PostgreSQL connection, no `alembic` command,
no SQL executed, and `scripts/create_tui_gin_index.py` was never run, imported, or edited.

### Bug 97 — revised after the migration-design investigation (2026-07-13)

**The VARCHAR state is unreachable on every supported initialisation path, and the real
defect is different (and arguably worse) than first recorded.**

*Decisive new evidence (all static — no DB, no Alembic, no SQL executed):*

1. **No migration creates the core tables.** The chain root `0001_add_pipeline_runs`
   (`down_revision = None`) creates only `pipeline_runs`, and revisions `0002`–`0014` create
   only `sum_*` / `llm_judge_cache`. `entities`, `documents`, `text_elements`, `figures`,
   `tables` and the junction tables exist **only** in the ORM (`Base.metadata.create_all`
   via `create_tables()`). The single reference to `entities` anywhere in the chain is
   `0006`'s `op.drop_column` in *downgrade*.
2. **`alembic upgrade head` cannot initialise a database from scratch.** On an empty DB it
   fails at **revision `0001`**, whose `pipeline_runs.document_id` is
   `sa.ForeignKey("documents.id")` — `documents` does not exist yet.
3. Therefore the only workable sequence is `create_tables()` (ORM) **then**
   `alembic upgrade head`. The ORM always builds `semantic_types` as `ARRAY(Text)` →
   **`TEXT[]`**, so `0006`'s `ADD COLUMN IF NOT EXISTS … VARCHAR` is a **guaranteed no-op**.
   `0006` can only ever fire on a database whose `entities` table predates the column in the
   ORM — a historical shape that no supported path can produce today.
4. **Even hypothetically, a VARCHAR-backed column could hold only NULL.** The *sole* writer
   is `named_entity_recognition/ner.py:351`
   (`session.bulk_insert_mappings(Entity, entities_to_save)`), binding
   `mapping["semantic_types"]` = `list(concept.types)` (a Python `list[str]`) or `None`.
   Compiled offline against the PostgreSQL dialect, SQLAlchemy emits
   `INSERT INTO entities (…, semantic_types, …) VALUES (…, ARRAY['T047', 'T191'], …)` — a
   genuine `text[]` expression, **not** a string. PostgreSQL has no implicit
   `text[] → varchar` cast, so such an INSERT raises a type error; only the `None` branch
   (rendered as plain `NULL`) would succeed. No JSON, CSV, comma-joined or single-value
   serialisation exists anywhere in the tracked tree, and an empty list never occurs
   (`ner.py` writes `None` when `concept.types` is falsy).

*Consequences for the planned migration:* **no data-conversion migration is required**, and
the feared `semantic_types::TEXT[]` ambiguity does not arise. Preparing revision `0015` is
**safe but unnecessary**; it would add a permanently no-op revision to a chain whose head
(`0014`) is referenced in tracked documentation.

*What is actually broken (re-scoped):*

* **`0006` declares the wrong type** (`VARCHAR` vs the ORM's `ARRAY(Text)`) — latent and
  currently unreachable, but it would create a broken column if it ever did fire.
* **The documented initialisation command is wrong.** `README.md:320` claims
  "**21 Alembic-managed tables** (`alembic upgrade head`)" and `.claude/CLAUDE.md:166/351/439`
  present `alembic upgrade head` as the create/init path. In fact **7 of the 21 tables are
  ORM-created**, and `alembic upgrade head` alone fails on an empty database. This is the
  genuine clean-clone reproducibility defect.
* `create_tables()` is therefore **required**, not "legacy/programmatic reset" as documented.

*Unchanged:* `scripts/create_tui_gin_index.py` remains on hold (its GIN SQL would be invalid
on a `varchar` column, though that state is now shown to be unreachable) and stays excluded
from Phase 8 · Commit 1. The `.claude/CLAUDE.md` claim of a "GIN full-text index on
`text_content`" is **false** — it exists in neither `database/models.py` nor any migration.

*Safety:* no PostgreSQL connection, no `alembic` command, no SQL executed; no migration
generated; `0006`, `database/models.py` and `scripts/create_tui_gin_index.py` unmodified.

### Bug 97 — final disposition (2026-07-13)

**No migration will be added.** The suspected populated-`VARCHAR` state is **unreachable**
on every supported initialisation path, so a conversion migration (`0015`) would be
permanently redundant. Revision `0006`, `database/models.py`, `scripts/create_tui_gin_index.py`
and every other Alembic revision remain **unchanged**.

**The actual defect is inaccurate initialisation/schema documentation**, now corrected in
`README.md` and `.claude/CLAUDE.md` (`docs(db): correct schema initialization guidance`).
The verified ownership model:

* `Base.metadata` defines **all** tables; `create_tables()` (`Base.metadata.create_all`)
  builds the whole schema and is what initialises a new database.
* The Alembic chain owns **incremental change only**. It cannot initialise an empty
  database (revision `0001` declares `pipeline_runs.document_id →
  sa.ForeignKey("documents.id")`, and `documents` does not exist yet), and its
  `op.create_table` calls are **unguarded** (no `checkfirst`, no `IF NOT EXISTS`), so they
  also collide with a schema `create_tables()` has just built in full — `pipeline_runs`,
  `llm_judge_cache` and all 12 `sum_*` tables are defined in **both** the ORM and the chain.
* Therefore: **new DB** → `create_tables()` + `alembic stamp head`; **existing DB behind
  head** → `alembic upgrade head`.

**Also corrected:** the false claim in `.claude/CLAUDE.md` of a "GIN full-text index on
`text_content`" — no such index exists in `database/models.py` or in any revision.

**Still open / separate:** `scripts/create_tui_gin_index.py` remains under review (retain as
an optional admin utility, move to legacy, or delete) and is still excluded from Phase 8 ·
Commit 1. Its `idx_entity_semantic_types` GIN index is declared in no migration and no
`__table_args__`.

**Safety.** No PostgreSQL connection, no `alembic` command, and no SQL was executed at any
point in this investigation or the documentation fix.

---

## Bug 98 — Alembic head does not reproduce the ORM schema (stamping is unsafe)

**Status / Severity / Surface.** Observed · **High** · `database/models.py` vs
`alembic/versions/0001…0014`; `pipeline_runs.narrative_summary`; `sum_corpus_relations`
indexes.

**Symptom.** A database built purely by `alembic upgrade head` is **not** the schema the
application needs, and a database built by `create_tables()` is **not** what the Alembic
chain describes. Therefore `create_tables()` + `alembic stamp head` is **not** a safe
fresh-database procedure, and must not be documented as one.

**Evidence (static: AST parse of every revision + offline `Base.metadata` inspection; no DB,
no Alembic, no SQL).**

1. **`pipeline_runs.narrative_summary` exists only in the ORM.** `database/models.py:354`
   declares `narrative_summary = Column(Text, nullable=True)`, and **no** Alembic revision
   creates it. Yet it is **production code**: written in
   `pipeline/stages/knowledge_extraction/persistence.py:636` (via `runner.py:947` and
   `batch/runner.py:953`) and read in `scripts/inspect/inspect_pipeline_output.py:527`.
   Because SQLAlchemy selects **all** mapped columns when loading a `PipelineRun`, any ORM
   read against an Alembic-built database would emit
   `SELECT … pipeline_runs.narrative_summary …` and fail with *column does not exist*.
   ⇒ **`alembic upgrade head` alone yields a database the pipeline cannot use.**
2. **Index names diverge on `sum_corpus_relations`.** Revision `0005` creates
   `ix_cor_corpus_run`, `ix_cor_scope`, `ix_cor_rule_a`, `ix_cor_rule_b`, `ix_cor_pmcid_a`,
   `ix_cor_pmcid_b` (6). The ORM (`models.py:829-835`) declares `ix_screl_run`,
   `ix_screl_scope`, `ix_screl_pmcid_a`, `ix_screl_pmcid_b`, `ix_screl_rule_a`,
   `ix_screl_rule_b` **and `ix_screl_type`** (an extra index on `relation_type`), plus
   `ix_sum_corpus_relations_corpus_run_id`. The chain has never heard of `ix_screl_*`; the
   ORM has never heard of `ix_cor_*`.
3. **The chain contains real data migrations.** `0009_split_canonical_scope` runs
   `UPDATE sum_canonical_rules SET is_conflicted = (canonical_scope = 'conflicted'), …` and
   an equivalent `UPDATE sum_corpus_relations …`. These are no-ops on an empty database, but
   they confirm the chain carries history a stamped database never receives.

**What *does* match.** All 12 `sum_*` tables and `llm_judge_cache` have **identical column
sets** between chain-head and ORM (including `0007`'s drop, `0009`'s split, and `0011`/`0012`
additions), and the five columns `0014` adds to `figures` are present in both.

**Consequences for stamping.** `alembic stamp head` marks `0001…0014` as applied without
running them. On an ORM-created database that assertion is **false**: the objects `0005`
would have created (`ix_cor_*`) do not exist, and an extra ORM index does. Any future
revision that references those names (`op.drop_index("ix_cor_scope")`, etc.) would fail, and
any future `op.add_column("pipeline_runs", "narrative_summary")` would fail with *column
already exists*.

**Also relevant.** `alembic stamp` is used **nowhere** in the tracked repository; nothing
calls `create_tables()` at application startup; and no test or deployment workflow depends on
Alembic revision state.

**Fix direction (not applied).** The migration chain must be made to describe the real
schema — either a **baseline revision autogenerated from `Base.metadata`**, or targeted
revisions that add `pipeline_runs.narrative_summary` and reconcile the
`sum_corpus_relations` index names. Until then, documentation must **not** prescribe
`alembic stamp head`.

**Safety.** No PostgreSQL connection, no `alembic` command, no SQL executed.

### Bugs 97 & 98 — immediate action taken (2026-07-13)

**Immediate action: a documentation-only correction** (`docs(db): correct schema
initialization guidance`, tracked files `README.md` + `database/ENV_LOADING.md`).
**Schema reconciliation is deferred to a separate phase.**

* **No stamp guidance.** `alembic stamp head` is **not** documented and must not be used:
  the ORM-created schema and Alembic head are **not** equivalent, so stamping would falsely
  assert equivalence and could break future name-based migrations.
* **No migration `0015`** is created in this commit; the VARCHAR→array conversion theory is
  abandoned (that state is unreachable — see B-097).
* **Parity failure recorded (B-098):** `pipeline_runs.narrative_summary` exists in the ORM
  (`models.py:354`) and in active runtime code (`persistence.py:636`, `runner.py:947`,
  `batch/runner.py:953`, read by `inspect_pipeline_output.py:527`) but is created by **no**
  revision; `sum_corpus_relations` indexes diverge (`ix_cor_*` in the chain vs `ix_screl_*`
  + an extra `ix_screl_type` in the ORM); revision `0009` contains genuine `UPDATE`
  backfills that a stamped database would never receive.
* **Documented model:** `create_tables()` (`Base.metadata.create_all`) initializes a new
  database; `alembic upgrade head` is bounded to an existing database initialized under the
  project's historical setup and known to be behind the head; `database/models.py` is
  authoritative.
* `.claude/CLAUDE.md` corrections remain **local-only** (that file is gitignored), so the
  tracked README must carry everything a fresh clone needs.

**Safety.** No database, Alembic, or SQL command was executed at any point.

### Bug 98 — refreshed full parity inventory (2026-07-13, Phase 8 · Commit 2B investigation)

A **complete** re-comparison (AST replay of every revision vs. offline `Base.metadata`)
found **two divergences beyond the three already recorded**, and corrected one false
positive. All drift is confined to `pipeline_runs`, `entities.semantic_types`, and
`sum_corpus_relations`.

| # | Divergence | ORM | Alembic head |
|---|---|---|---|
| 1 | `pipeline_runs.narrative_summary` | `Column(Text, nullable=True)` (`models.py:354`) | **absent** |
| 2 | `entities.semantic_types` | `ARRAY(Text)` → `TEXT[]` | `VARCHAR` (`0006`; unreachable — see B-097) |
| 3 | `sum_corpus_relations` indexes | `ix_screl_{run,scope,pmcid_a,pmcid_b,rule_a,rule_b}` + **`ix_screl_type`** (`relation_type`) | `ix_cor_{corpus_run,scope,pmcid_a,pmcid_b,rule_a,rule_b}` — same columns, **different names**; no `relation_type` index |
| 4 | **NEW** — `sum_corpus_relations` unique constraint | `uq_sum_corpus_relation` (`models.py:836`) | `uq_corpus_relation` (`0005:62`) — same columns `(corpus_run_id, relation_id)`, **different name** |
| 5 | **NEW** — ORM-internal redundancy | **duplicate index on `corpus_run_id`**: `Column(..., index=True)` → `ix_sum_corpus_relations_corpus_run_id` **plus** explicit `Index("ix_screl_run", "corpus_run_id")` (`models.py:795, 829`) | chain has one (`ix_cor_corpus_run`) |

**False positive corrected:** `uq_sum_map_voter_output` **does** match — it is declared
inline inside `op.create_table` in `0013:50` (an earlier extractor only scanned
`op.create_unique_constraint`).

**Everything else matches**: all other `sum_*` tables, `llm_judge_cache`, and the five
columns `0014` adds to `figures` have identical definitions.

**`narrative_summary` re-characterised.** It is a **legacy column of the retired REDUCE
stage** — `models.py:353` states it is *"always NULL on the current pipeline. Retained for
schema/migration stability."* Both runners always pass `narrative_summary=None`
(`runner.py:691`, `batch/runner.py:714`), and `persistence.py:635` only assigns it when
non-None, so it is **never written**. It *is* read (`inspect_pipeline_output.py:527`) and,
decisively, SQLAlchemy **SELECTs every mapped column** when loading a `PipelineRun` — so on
an Alembic-only database any `PipelineRun` read still fails with *column does not exist*.
The parity break is therefore real even though the column is semantically dead.

**Revision 0009 backfills (must be preserved for historical databases).** It adds
`is_conflicted`/`study_coverage` to `sum_canonical_rules` and `is_conflicted_{a,b}`/
`study_coverage_{a,b}` to `sum_corpus_relations`, **UPDATEs them from the old
`canonical_scope` string**, sets them `NOT NULL`, then drops `canonical_scope`. A fresh
`create_all()` database never has `canonical_scope`, so the backfill is meaningless there —
but a database at `0008` requires it.

**Drift-test infrastructure: none exists.** No `testcontainers`, no `pytest-postgresql`, no
Docker Compose, no Dockerfile, no CI workflow is tracked (only `psycopg2-binary`). Alembic's
`compare_metadata` requires a `MigrationContext` bound to a **live** connection, so a full
parity check needs real PostgreSQL; an AST-replay-vs-`Base.metadata` comparison (as used
here) is the only check possible with **zero** database access.

**Safety.** No schema mutation, no migration generated, no PostgreSQL connection, no Alembic
command, no SQL executed.

### Bugs 97 & 98 — HISTORICAL RECONSTRUCTION REVERSES THE B-097 CONCLUSION (2026-07-13)

**B-097's "VARCHAR is unreachable" finding was WRONG.** Git history proves the VARCHAR state
is not only reachable but was **actually created and populated in this repository**:

| Date | Commit | Event |
|---|---|---|
| 2026-01-08 | `e292812` | ORM declares `semantic_types = Column(String)` (**VARCHAR**). One-off `database/migrations/add_semantic_types.py` runs `ALTER TABLE entities ADD COLUMN semantic_types VARCHAR NULL`. |
| 2026-01-08 | `e292812` | `scripts/add_semantic_types_from_cui.py` **populates it**: `ent.semantic_types = ", ".join(concept.types)` → values like `"T047, T191"`. |
| 2026-01-26 | `1198026` | ORM changed `String` → **`ARRAY(Text)`** — **with no migration**. *This is the drift-introducing commit.* |
| 2026-04-13 | `a76e73f` | Revision `0006` written as `VARCHAR`, faithfully reproducing the one-off; the one-off script is deleted. |

**Classification: PROVABLY REACHABLE — and provably populated, with a known serialization**
(`", ".join(types)` → comma-and-space separated). Consequences:

* A **direct `semantic_types::TEXT[]` cast would fail** (`"T047, T191"` is not a valid array
  literal → *malformed array literal*).
* **`ARRAY[semantic_types]` would silently corrupt** — it yields the 1-element array
  `{"T047, T191"}`, so `&&` overlap on `'T047'` would never match.
* A conversion migration is therefore **required**, and must parse the one proven format
  (`string_to_array(semantic_types, ', ')`) while **failing loudly** on anything else.

**Live database inference (not proven — no connection made).** The current dev DB almost
certainly holds `TEXT[]`: the array-overlap queries (`export_disease_entities`,
`copy_relevant_files`) and list reads (`paper_selection/fingerprints`) work, which is
impossible against `varchar`. So the column was converted at some point (manually, or by
`drop_tables()/create_tables()`). The **repository's historical path**, however, still
produces `varchar` — and `0006` still creates `varchar` today for any database reaching it
without the column.

**`sum_corpus_relations` naming — the deployed names are probably the *Alembic* ones.** The
table was introduced in `a76e73f` in **both** `models.py` and revision `0005`. The project's
documented practice was `alembic upgrade head`, so `0005` most likely created the table with
`ix_cor_*` + `uq_corpus_relation`; `create_all()` skips existing tables, so the ORM's
`ix_screl_*`, `uq_sum_corpus_relation` and the `index=True`-derived
`ix_sum_corpus_relations_corpus_run_id` may exist **only in ORM metadata, never in any
database**. **No application, test, or documentation code references any of these objects by
name** (verified) — so renaming the *ORM* to match history is a zero-DDL metadata fix, and is
strictly safer than renaming live database objects.

**`narrative_summary`** was introduced in `a82a113` (2026-04-13) and *was* genuinely written
by the REDUCE stage until the Phase 3 retirement commits (`2e70807`, `c4c3572`, `d98a310`,
2026-07-12) removed that path. The model comment ("populated by the retired REDUCE stage") is
therefore **historically justified**, not stale.

**Status:** historical reconstruction complete; `semantic_types` reachability **no longer
assumed**; `narrative_summary` preserve-vs-remove and ORM-name-vs-migration-name decisions
open; Alembic baseline graph feasibility unresolved. **No database, migration, or SQL
operation was run.**

### Bug 98 — static drift characterization added; FOUR NEW DIVERGENCES FOUND (2026-07-13)

`tests/database/test_schema_drift.py` (fail-closed AST replay of revisions 0001–0014 vs
`Base.metadata`) is prepared. Building it **immediately surfaced four divergences that the
manual analysis had missed** — exactly what a fail-closed lint is for. They are **not** added
to the approved set; they are reported for review first.

| # | Object | Alembic chain | ORM |
|---|---|---|---|
| 7 | `pipeline_runs.status` | `server_default="running"` (**DDL**) | `default="running"` (**Python-side only**) |
| 8 | `pipeline_runs.started_at` | `server_default=sa.func.now()` (**DDL**) | `default=func.now()` (**Python-side only**) |
| 9 | `sum_corpus_relations.scope_check_result` | `server_default="scope_unknown"` (**DDL**) | `default="scope_unknown"` (**Python-side only**) |
| 10 | `sum_corpus_relations.scope_note` | **`nullable=False`, `server_default=''`** | **`nullable=True`**, no default |

Consequence: an Alembic-migrated table carries real `DEFAULT` clauses (and, for `scope_note`,
`NOT NULL`) that an ORM-created table does **not**. #10 is the most substantive — the two
initialisation paths disagree on **nullability**. A row inserted without `scope_note` succeeds
on an ORM-created database and **fails on a migrated one**.

Also confirmed: `*.created_at` is **not** drift — the chain's `sa.text('now()')` and the ORM's
`server_default=text('now()')` normalize to the same `now()`; an earlier raw-source comparison
made them look divergent.

Note the distinction the test enforces: `Column(default=...)` is **Python-side** (applied by
SQLAlchemy at INSERT) and is *not* DDL; `server_default=` **is** DDL. Divergences 7–9 are
therefore real physical-schema differences, even though application-level behaviour coincides
when writes always go through the ORM.

**Status:** the test currently **fails on exactly these four** (all other assertions and all
parser self-tests pass). Approval is required to add them to the approved divergence set, or
to fix the ORM instead. No database, migration, or SQL command was run.

### Bug 98 — drift characterization complete (2026-07-13, Phase 8 · Commit 2D)

`tests/database/test_schema_drift.py` now characterizes **all ten** known divergences and
passes (17 tests). **Characterization is not endorsement** — every entry below is still a
pending defect:

* **`sum_corpus_relations.scope_note` (substantive).** Chain: `nullable=False`,
  `server_default=''`. ORM: `nullable=True`, no default. Recorded as **two separate**
  records (`nullability_mismatch` + `server_default_mismatch`), never collapsed. An INSERT
  omitting `scope_note` **succeeds** on an ORM-created database and **fails** on a migrated
  one.
* **Three client-vs-server default mismatches** (`pipeline_runs.status`,
  `pipeline_runs.started_at`, `sum_corpus_relations.scope_check_result`). The chain sets a
  real DDL `DEFAULT`; the ORM only sets `Column(default=...)`, which SQLAlchemy applies at
  INSERT and which produces **no DEFAULT clause** in `CREATE TABLE`. The test keeps
  `Column(default=…)` and `server_default=…` in separate fields and records the *ownership*
  mismatch (`client_default_vs_server_default`) even when the values coincide — they are
  never normalized to equal.
* The six `ix_screl_* ↔ ix_cor_*` index renames, the ORM-only `ix_screl_type`, the duplicate
  `ix_sum_corpus_relations_corpus_run_id`, the `uq_sum_corpus_relation ↔ uq_corpus_relation`
  rename, `pipeline_runs.narrative_summary`, and the `entities.semantic_types` type
  divergence — all as previously recorded.

The lint fails closed: unknown `op.*`, unknown/modified raw SQL, non-literal `op.execute`,
unmodelled `Column()` or `alter_column()` kwargs, and unsupported type expressions all raise.
Any *new* divergence fails with "stop and review — do not just add it".

**Not covered** (still needs disposable PostgreSQL): operator classes, expression indexes,
resolved server-default DDL, enum evolution, dialect-implicit behaviour, data-dependent
effects, locking.

No database, Alembic, or SQL command was run.

---

## Bug 99 — Fresh-database setup omitted the ownership requirement

### Status / Severity / Surface

**Observed** · Medium · Docs / setup — `README.md` § *2. Set Up Database*, PostgreSQL
role privileges and database ownership.

### Symptom

The guarded clean-room verification of the newly documented fresh-database workflow
(`docs(db): document fresh-database initializer`, `772fcf4`) **halted at step 0** — the
disposable database could not be created at all:

```
createdb: error: database creation failed: ERROR:  permission denied to create database
[cleanup] nothing created; nothing to drop.
```

The initializer therefore never ran; nothing about it was verified.

### Evidence

Read-only inspection with the configured credentials (`local_db_user`, password never
printed):

| role | superuser | createdb |
|---|---|---|
| `postgres` | **true** | **true** |
| `abitutor_user` | false | false |
| **`local_db_user`** (the configured `DB_USER`) | false | **false** |
| `tagger_user` | false | false |

`SELECT datname, pg_get_userbyid(datdba) FROM pg_database` — the real development
database **`nlp_histo` is owned by `local_db_user`**. Server version: **14.8**.

So the configured application role owns the development database yet cannot create
databases: `nlp_histo` was never produced by any "`createdb` as `DB_USER`" path. An
administrator created it and assigned ownership — precisely the step the README did not
document.

### Diagnosis

Two separate facts, previously conflated:

1. **Creating the database requires an administrative role.** `CREATE DATABASE` needs
   `CREATEDB` or `SUPERUSER`. The application role does not have it and does not need
   it. Creating through an admin role is the *normal* arrangement, not a workaround.
2. **The application role must be able to create tables *inside* the new database.**
   The README's original example — `createdb -U <postgres-user> <database-name>` —
   assigned **no owner**. A supervisor who then connects as a non-owner `DB_USER` and
   runs `python -m database.init_db` fails at table creation with
   `permission denied for schema public` on **PostgreSQL 15+**, where `PUBLIC` no
   longer holds `CREATE` on schema `public`.

This server is 14.8, where `PUBLIC` *does* still hold `CREATE` on `public` — so the
defect is **masked locally**. Even a fully successful clean-room run on this machine
**would not have detected it**. The bug was found only because the *creation* step
failed first, for the unrelated privilege reason above.

### Fix

Documented ownership explicitly in `README.md` (commit
`docs(db): document database ownership requirement`):

```bash
createdb -U <admin-role> -O <db-user> <database-name>
```

* `<admin-role>` — any role permitted to create databases (do **not** assume `postgres`).
* `<db-user>` — must equal `DB_USER` in `.env`.
* `-O <db-user>` makes the application role the **owner**, which is what lets
  `database.init_db` create the ORM tables. It is **version-independent**: it does not
  rely on the permissive pre-15 default `public`-schema privileges.

A concise alternative is noted (an administrator may instead grant the configured role
sufficient creation rights), but ownership is the recommended setup. No broad
`GRANT ALL` is recommended, and no version-specific reliance on default `public`-schema
privileges is documented.

**Not fixed here (follow-up):** the connection-failure hint in `database/init_db.py:374`
still prints `createdb {database}` without an owner. It is a runtime *diagnostic hint*,
not a setup procedure, and source changes were out of scope for the documentation
commit.

### Verification

**PASSED (2026-07-13).** Clean-room run against a disposable database
(`nlp_histo_cleanroom_20260713T142721Z_13bbb4`), created by an administrator with
`createdb -U postgres -O local_db_user …`; **every application operation then ran as
`DB_USER` with only `DB_NAME` overridden** (`.env` untouched):

| step | result |
|---|---|
| owner check | `local_db_user` — exactly `DB_USER` |
| `python -m database.init_db` | empty → **21 tables created**, verified, exit 0 |
| re-run (idempotence) | *"Schema already initialized — nothing to create"*, exit 0 |
| `--check-only` | valid, created nothing, exit 0 |
| `--smoke` | ORM round trip passed, **transaction rolled back**, exit 0 |
| rollback proof | `documents=0`, `SMOKE_%`-prefixed `documents=0`, `text_elements=0` |
| `entities.semantic_types` | `ARRAY` / `_text` → **`TEXT[]`**, as the ORM declares (contrast B-097: Alembic would have produced `VARCHAR`) |
| `pipeline_runs.narrative_summary` | present, `text` (ORM-only column — B-098) |
| `alembic_version` | **absent** — confirming Alembic was not involved in initialisation |
| cleanup | dropped **by the owner `DB_USER`**; the admin role was *not* needed again |

Confirms the ownership fix: with `-O <db-user>` the application role can create every ORM
table **and** drop the database afterwards, without ever holding `CREATEDB`. Development
database `nlp_histo` verified still present and never connected to; `.env` unchanged;
working tree clean. No Alembic, extraction, model download, corpus ingest, or paid API
call occurred.

**Fresh-database reproducibility is now verified end-to-end** for the documented
procedure. Caveat: this server is PostgreSQL **14.8**, so the run does not itself
*prove* the PG15+ `public`-schema behaviour — but `-O` makes the outcome
version-independent by construction, which is precisely why it is the documented path.

During investigation nothing was mutated: the disposable database was never created, the
cleanup trap correctly dropped nothing, the development database was never connected to,
`.env` was unchanged, and no Alembic, extraction, model-download, or paid-API operation
occurred. Only read-only `pg_roles` / `pg_database` queries against the `postgres`
maintenance database were issued.

---

## Bug 100 — Drafted appendix lists nonexistent silver artifact paths

### Status / Severity / Surface

**Superseded (2026-07-13)** · Low · Thesis drafting — `docs/thesis/reference/appendix_reference.md`
(gitignored), artifact-location tables at `:134-136` and `:225-228`.

**The defect surface no longer exists.** The user intentionally deleted the gitignored
`docs/thesis/` drafting tree on 2026-07-13, which is where — and *only* where — the wrong
paths lived. No correction was applied, and none is possible: there is nothing left to
correct. This entry is retained solely so the verified path table below is on record, and
the same wrong paths are not reintroduced if the appendix is ever authored into `.tex`.

### Symptom

The drafted appendix tells a reader to look for the silver-evaluation artifacts under
`eval/silver/`. Nothing is there — `eval/silver/` contains **no `.jsonl` and no
`.sqlite` at all**. A supervisor following the appendix would find nothing.

### Evidence

Draft rows vs. verified reality:

| Draft row | Verified correct path | Class |
|---|---|---|
| `eval/silver/pipeline_findings.jsonl` | `eval/data/pipeline_findings_related15.jsonl` (`evaluate.py:49`) | ignored generated artifact — **not currently on disk** |
| `eval/silver/silver_findings.jsonl` | `eval/data/silver_findings_related15.jsonl` (`evaluate.py:48`) | ignored generated artifact (exists) |
| `eval/silver/silver_findings_test.jsonl` | `eval/data/silver_findings_heldout15.jsonl` | ignored generated artifact (exists) — **renamed** |
| `eval/silver/embedding_cache_openai.sqlite` | `eval/data/embedding_cache_openai.sqlite` (`matcher.py:50`) | cache, ignored (exists) |
| `eval/silver/embedding_cache_gemini.sqlite` | `eval/data/embedding_cache_gemini.sqlite` (`matcher.py:51`) | cache, ignored (exists) |
| `eval/silver/map_primer/` | `eval/data/map_primer/` (`map_theta_sweep.py:95`) | directory, ignored (exists) |
| `eval/silver/map_primer_test/` | `eval/data/map_primer_heldout15/` | directory, ignored (exists) — **renamed** |

Only `eval/data/source_cases_related15.jsonl` is a **tracked** file; every artifact in
the table above is a gitignored generated artifact or cache.

### Diagnosis

Two independent errors, which is why the naive fix is unsafe:

1. **Wrong directory** — the artifacts live under `eval/data/`, not `eval/silver/`.
2. **Wrong filenames** — `silver_findings_test.jsonl` → `silver_findings_heldout15.jsonl`
   and `map_primer_test/` → `map_primer_heldout15/`. A blind
   `s|eval/silver/|eval/data/|` substitution would produce two paths that still do not
   exist, plus one (`pipeline_findings_related15.jsonl`) that is a valid code default
   but is **not currently materialised** — it is regenerated by
   `python -m eval.silver.data.export_pipeline`.

**The thesis itself is not affected.** `docs/histo_thesis/pages/appendix.tex` is still
the untouched LaTeX template boilerplate, and `main.tex:52,54` carries
`% \appendix{}` / `% \input{pages/appendix}` — **commented out**. The thesis therefore
has no appendix, and this table has never been compiled into it. The defect is confined
to gitignored drafting material (and its generated derivative
`docs/thesis/build/thesis_final.md`); it would only reach the thesis if that draft is
converted to `.tex` as-is.

### Fix

**None applied — none required.** The drafting tree that carried the error was deleted, and
the thesis itself never contained the table (see *Diagnosis*). The durable output of this
investigation is the verified path table under *Evidence*: it is the authority to use when
the appendix is actually written. Because the `related15` / `heldout15` suffixes
are dataset-specific, and the primer directory and both embedding caches are
CLI-overridable (`--primer-dir`; cache overrides at `map_theta_sweep.py:1251,1261`), the
appendix should quote the **exact canonical path used by the documented reproduction**
rather than a bare directory or a pattern.

### Verification

Confirmed 2026-07-13. `git grep` over all 126 tracked files of `docs/histo_thesis/` for
`silver_findings` / `embedding_cache` / `map_primer` / `pipeline_findings` / `eval/silver` /
`eval/data` returns **zero hits** — the tracked thesis never carried an obsolete path. No
tracked file anywhere in the repository contains one. `find` confirms
`appendix_reference.md` and `thesis_final.md` are absent from the tree.

No thesis source was edited (none needed editing). No artifact was generated, no experiment,
embedding, or API call was run, no evaluation code or cache default was altered, and the
tracked tree is unchanged.

---

## Bug 101 — `generate --help` crashes on an unescaped percent

**Status / Severity / Surface** — Fixed (2026-07-13, commit `f2581ac`) · Low · Eval CLI.
`eval/silver/generate.py:52` (now `eval/silver/generation/generate.py:52`) × argparse
help expansion.

> Reconstructed 2026-07-16 from the catalogue entry, commit `f2581ac`, and the code — the
> catalogue row linked here but the section had never been written. No claim below is new:
> each is re-verified against the tree, and the bug is **not** reopened.

### Symptom

```
$ python -m eval.silver.generate --help
TypeError: %c requires int or char
```

The CLI itself ran fine — only `--help` was unusable, which is why it went unnoticed.

### Evidence

`generate.py:52` declared:

```python
help="Use Anthropic batch API (~50% cheaper). Re-run to check status."
```

Reproduced in isolation, 2026-07-16, with a three-line snippet — no repository code
involved, confirming the mechanism is argparse's and not the module's:

```python
p = argparse.ArgumentParser()
p.add_argument('--batch', action='store_true', help='Use Anthropic batch API (~50% cheaper).')
p.format_help()          # → TypeError: %c requires int or char
```

Confirmed pre-existing at HEAD when found: the crash was identical before and after the
Phase-11A bootstrap removal, whose only diff was a deleted `sys.path.insert`.

### Diagnosis

argparse expands help text through `self._get_help_string(action) % params`. A literal
`%` in a help string is therefore read as the start of a format specifier: `% c` in
`50% cheaper` becomes `%c`, which demands an int or char and gets a dict. Nothing about
the module is at fault — any argparse help string containing a bare `%` fails the same
way.

### Fix

Escape the percent as `~50%% cheaper`, argparse's documented escape, which renders as the
intended `~50% cheaper`. Shipped in `f2581ac` ("fix(eval): escape percent in generate
help"), a one-line change to a single file.

Deliberately kept out of `refactor(eval): normalize silver import and repository paths`,
which was forbidden from altering CLI behaviour, arguments, or defaults.

### Verification

Re-verified 2026-07-16 against the current tree (the module has since moved to
`eval/silver/generation/generate.py`; the escape moved with it):

```
generate.py:52 → help="Use Anthropic batch API (~50%% cheaper). Re-run to check status."
python -m eval.silver.generation.generate --help → exit 0
```

No other eval CLI is affected — the other six `--help` invocations exit 0.

---

## Bug 102 — Chapter-9 offline replay requires the repo root as cwd

**Status / Severity / Surface** — Observed · Medium · Reproducibility.
`eval/silver/analysis/map_theta_sweep.py:94` × `scripts/thesis/run_chapter9_offline_replay.py`.

### Symptom

Run the Chapter-9 offline replay from anywhere other than the repository root and it
fails partway through, after several analyses have already succeeded:

```
cd /tmp
python3 /Users/…/nlp-histo/scripts/thesis/run_chapter9_offline_replay.py
…
[…] --- analysis: 05_bootstrap_ci_cascade_vs_sonnet ---
voter cache not found: eval/data/map_primer/voter_cache.json
Run `python -m eval.silver.analysis.map_theta_sweep prime` (with --source/--primer-dir) then `collect`.
```

From the repository root the same command completes normally, so the dependency never
surfaces in ordinary use.

### Evidence

`eval/silver/analysis/map_theta_sweep.py:94`:

```python
PRIMER_DIR   = Path("eval/data/map_primer")
```

A bare relative path — it resolves against the **current working directory**, not the
repository root. `git show HEAD:eval/silver/analysis/map_theta_sweep.py` contains the same
line, so this is pre-existing.

### Diagnosis

Two separate mechanisms are easy to conflate here, and only one of them is at fault:

* **Import resolution** — fine. The script's `sys.path` bootstrap resolves the repository
  root from `__file__`, so `import pipeline` / `import eval.…` work from any cwd. The
  script imports and starts executing correctly from `/tmp`.
* **Data-path resolution** — broken. `PRIMER_DIR` (and the sibling report-output defaults
  in the same module) are cwd-relative, so the *inputs* are only found when cwd happens to
  be the repository root.

The failure is therefore a working-directory contract, not a packaging or bootstrap
problem. It was discovered during the Phase-11C bootstrap validation, whose
outside-the-repository execution test exists precisely to expose hidden cwd dependence.

Secondary observation: `run_chapter9_offline_replay.py` has no argparse/`--help` handling,
so invoking it with `--help` runs the full replay instead of printing usage.

### Fix

**Not fixed** — deliberately deferred. The Phase-11C bootstrap refactor that surfaced it was
rolled back in full (too much churn for no reproducibility gain), and fixing a data-path
contract inside a rollback would have mixed two unrelated concerns.

Reserved as an explicit item for the **clean-room reproducibility pass**, which should:

1. anchor `PRIMER_DIR` to the repository root (`REPO_ROOT / "eval" / "data" / "map_primer"`,
   via the existing `eval/paths.py` anchor) rather than to the cwd;
2. audit the other cwd-relative defaults in the same module (e.g. the `eval/reports/` sweep
   output path) in the same change — they share the failure mode;
3. decide explicitly whether repo-root-cwd is a *supported contract* (documented and
   asserted at startup with a clear error) or a bug to remove. Either is defensible; the
   current state — an undocumented requirement with a confusing downstream symptom — is not.

### Verification

Not applicable (open). The reproduction above is the acceptance test: after the fix, running
the replay from `/tmp` with an absolute path must locate the primer cache and behave
identically to a repo-root run.

---

## Bug 103 — E13 `scope_aware` arm cannot measure scope-awareness

**Status / Severity / Surface** — Observed (2026-07-14) · Medium · Eval design.
`eval/silver/experiments/E13_nli_ablation/evaluate.py:120-121` ×
`eval/data/relation_claim_pairs_300.jsonl`.

### Symptom

E13 reports two RELATE input modes on the synthetic 300-pair set: `predicate_only`
(accuracy 0.927) and `scope_aware` (0.923). Read at face value — and as §4.5/§5.6 originally
read it — this says scope-awareness is *mildly harmful* yet is used in production anyway, on
faith. That reading is wrong: the arm cannot express the mechanism it appears to test.

### Evidence

Production builds the NLI text **per rule** (`relate_stage.py:291`, `_build_scope_prefix`),
so each rule contributes its own `Scope` and the two sides of a pair can differ:

```
"[scope: disease=AciCCIS] TP53 is absent"   vs   "[scope: disease=AciCC] TP53 is present"
```

The comparability gate matches on subject / outcome / category / relation type but **not** on
scope, so scope divergence is precisely the signal the tag exists to surface (the AciCC /
AciCCIS false-SUPPORT case recorded in THESIS.md as the motivation for the flag).

E13 instead feeds the synthetic set's **pair-level** `disease_or_entity` to *both* claims:

```python
(build_shim(r["claim_a"], r.get("disease_or_entity")),      # evaluate.py:120-121
 build_shim(r["claim_b"], r.get("disease_or_entity")))
```

and the shim's own docstring (`:71`) says `scope.disease_subtype = the shared entity`. So in
all 300 pairs both claims receive an identical prefix.

The dataset is **not** at fault. The generation prompt
(`eval/prompts/relation_pairs/batch_01_prompt.txt:58`) defines the field as:

> `disease_or_entity`: a short disease/entity/context label **for the pair**. For SUPPORTING
> and CONTRADICTING pairs, this should usually be the **shared** disease … For UNRELATED
> pairs, use the **closest common context**.

It is a generation scaffold — a shared context is what makes a SUPPORTING/CONTRADICTING pair
constructible at all — and Opus produced exactly that. The defect is downstream reuse of the
scaffold as the model input tag.

Worst on the 63 `different_entity` UNRELATED pairs, where the tag asserts sameness against the
gold label:

```
scope: "Lung neoplasia"
  A: ALK rearrangements in lung adenocarcinoma are detected by break-apart FISH…
  B: Small cell lung carcinoma frequently shows inactivation of RB1 and TP53…
  gold: UNRELATED — "Different lung tumor types and different molecular topics."
```

### Diagnosis

The arm measures *"classifier accuracy when an identical prefix is prepended to both claims"*.
That is a real and useful measurement — it establishes that the bracketed, non-natural-language
prefix does not degrade a cross-encoder fine-tuned on MNLI/MedNLI sentence pairs (0.927 →
0.923, i.e. dilution only). It is **not** a measurement of whether scope *helps*, because a
prefix identical on both sides cannot signal a subtype difference. Similar-claims-different-
context does not occur in the dataset by construction.

**The numbers are correct.** `predicate_only` never reads scope and is untouched; the headline
0.927, the per-class table, the threshold sweep, the grounding results and §4.2's relation
counts all stand. Only the *interpretation* of the `scope_aware` column changes.

A claim-derived scope (scispaCy/UMLS over each claim) does **not** repair the arm: the synthetic
claims are fluent sentences that already name their disease, so extracting and re-prepending it
would test "does restating the disease help". In production the tag matters because
CANONICALIZE's abstracted `predicate_text` ("CD30 is expressed") has *dropped* the disease —
the tag re-injects information the model would otherwise never see.

### Fix

**Not fixed** — deliberately, at thesis submission (2026-07-14). No code, data, or numbers were
changed. The thesis prose was corrected instead:

* §4.5 (`04_results/05_nli_in_grounding_and_relation_classification.tex`) — the `scope_aware`
  arm is now framed as a check that the production input format does not degrade the classifier,
  followed by an explicit statement that the set cannot test whether scope *helps*, because both
  claims in a pair share one context label.
* §5.6 (`05_discussion/06_nli_as_shared_backbone.tex`) — the bullet is retitled "Scope-aware
  input is a design choice that the evaluation cannot test", and concludes that scope-awareness
  is **untested rather than disconfirmed**.
* §3.4.7 (`03_methodology/04_knowledge_extraction/07_relate.tex`) — no longer promises per-mode
  relation counts or cross-mode label stability (neither exists); the verbatim axis is declared
  available but unevaluated.
* §3.2.6 (`03_methodology/02_corpus/06_relation_classification_evaluation_set.tex`) — the
  indirect-analysis claim now points at §4.2, the only place corpus relation counts appear.

Two follow-ups are recorded in THESIS.md ##TODOs (minimal-pair regeneration; exhaustive
gate-passing-pair audit on production rules).

### Verification

Not applicable (open by design). The acceptance test for a future fix is that the
`scope_aware` arm must contain pairs whose two sides carry **different** scopes and whose gold
label depends on that difference — otherwise the arm cannot, in principle, discriminate.

---

## Bug 104 — Spaced bracket citation markers (`[1, 2, 3]`) are not stripped

**Status / Severity / Surface** — Observed (2026-07-14) · Low · Doc extraction.
`parsers/text_processing.py:318` (`remove_citations`).

### Symptom

Citation markers typeset with spaces survive text assembly and are stored in
`text_elements.text_content`. Markers typeset without spaces are removed correctly.

### Evidence

The bracket rule is digits-only, with no whitespace allowed anywhere:

```python
# parsers/text_processing.py:318
# Bracket-style citations: [1], [1,2], [1-29], [3,11,21,22], [1–3]
cleaned = re.sub(r'\[\d+(?:[,–\-]\d+)*\]', '', cleaned)
```

Measured on the production corpus (35 896 text elements, 977 documents):

```
elements with spaced   '[1, 2' markers surviving : 561   (1.6%)
elements with unspaced '[1,2]' markers surviving :   0
```

The split is total, not partial: a paper that spaces its citation lists keeps *every* marker,
a paper that does not is cleaned completely. The other rules in `remove_citations` (URL strip,
after-period / after-comma / standalone superscript runs, double-space collapse) are unaffected.

### Diagnosis

Publisher house styles differ on whether a bracketed citation list is set as `[1,2,3]` or
`[1, 2, 3]`. The regex encodes only the former. Nothing downstream re-checks for citation
markers, so the leftovers propagate: `text_elements` → MAP chunks → possibly into
`verbatim_support` strings and therefore into the NLI/grounding inputs. Severity is Low: the
markers are cosmetic noise inside otherwise-correct sentences; no structure, path, or provenance
is broken.

### Fix

Untested one-liner — permit optional whitespace while keeping the digits-only constraint so
`[Table 1]` and `[Fig. 2]` still do not match:

```python
cleaned = re.sub(r'\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]', '', cleaned)
```

**Deliberately not applied at thesis submission (2026-07-14).** The fix only takes effect on
re-ingest, and re-ingesting the corpus would invalidate every downstream artifact the thesis
reports on (MAP findings, silver labels, all §4 metrics). Sequencing for a future fix: patch the
regex → re-ingest → regenerate silver → re-run the evaluation battery, as one change, not
piecemeal.

**Thesis text is not affected.** §3.3 states that markers "such as `[1,2,3]`" are detected and
removed, which is accurate for the form the regex handles and claims no exhaustiveness. Writing
the example as `[1, 2, 3]` would have made the sentence false — the spaced form is precisely
what is *not* removed.

Note: the project `CLAUDE.md` describes `remove_citations` as stripping "`[1, 2, 3]`-style
citation markers" (with spaces). That documentation line is wrong; the code is authoritative.

### Verification

Not applicable (open). Acceptance test for the fix: after re-ingest, the corpus query

```sql
SELECT count(*) FROM text_elements WHERE text_content ~ '\[[0-9]+, [0-9]+';
```

must return 0 (currently 561), while `[Table 1]`-style bracketed references must survive.

---

## Topic — HOW_TO_RUN clean-room verification (2026-07-16)

B-105 … B-110 all come from one exercise: executing every command in
`docs/HOW_TO_RUN.md` against this tree, under the constraint that **no money may be
spent**, to answer whether a supervisor can reproduce the thesis from the document alone.

**Method.** Every run was wrapped in a guard that intercepts `socket.getaddrinfo` and
raises on any billable host (`api.openai.com`, `api.anthropic.com`,
`generativelanguage.googleapis.com`, `*.openai.azure.com`, Mistral, DeepSeek, …) while
allowing free model downloads, logging every resolved hostname. This converts "is it
free?" from a claim into a measurement: a paid call cannot silently succeed, and the
run's full network footprint is recorded. Across every command in this exercise the only
hosts contacted were `huggingface.co` and `s3-us-west-2.amazonaws.com`, and **zero** paid
blocks fired.

A first attempt blocked *all* sockets. That was too blunt — it also blocked the free
model downloads, which silently changed the results and produced two spurious diffs.
That misfire is itself the evidence for B-107, and the reason the guard was narrowed to
a paid-host denylist. **Lesson for future verification: block what bills, not what
moves.** A guard that changes the run's behaviour cannot be used to verify the run.

**Headline result — the replay reproduces.** With only paid hosts blocked,
`nlp-histo replay chapter9 --artifact-root .` regenerated **9 CSVs, all byte-identical**
to `out/thesis_results/chapter9_offline_replay/`, with exactly the two documented
non-regenerating files (`04_theta_heatmap.csv`,
`10_cascade_vs_sonnet_gap_ci_per_case.csv`) absent. §10's "a complete run writes exactly
9 CSVs" is accurate.

**Two apparent discrepancies that are not defects**, recorded so they are not
re-investigated:

* **E04 prints `MAP=2294` against its `_E03_FROZEN` reference of 2280.** Different
  config, not drift: `_FROZEN_PROFILES = {"6voter_frozen", "real"}`
  (`cardinalities.py:49`), but the current `out/summaries` holds profile `real_5` — the
  **5-voter** subset (E08 winner, Claude-Haiku dropped). The 2280 constant came from the
  6-voter cascade. The script prints its own output as "the config-robust readout".
* **E14 reports `strict_f1_optimal = 0.7128` against a related15 reference of 0.7160.**
  By design — heldout15 vs related15. The printed generalization gap (−0.0032) *is* the
  result.

**Coverage.** Verified: §3 (all 15 `--help` invocations exit 0), §4 (all 8 env vars are
genuinely read in `src/`), §5 (`db check` and `db init` both exit 0 against live
PostgreSQL; `db init` is idempotent and dropped nothing), §6 failure path, §9 (both
commands executed as written under `--dry-run`, exit 0 — see B-105), §10, §11, §12.
**Not** verified: §2's exact venv sequence (this machine runs the framework 3.12
install), `db init` from an empty database, §6's happy path, §7/§8 live runs (flag names
confirmed present in the forwarded parsers — `runner.py:1084` `--pdf-dir`,
`runner.py:1091` `--out-root`, `batch_ner.py:288` `--entity-cache` — but no run was
executed, as both write to the database), and `E14 --theta-frontier`.

**Second lesson, from B-105.** The two documented paid commands had been written by
*inspection* — they were never executed, because executing them looked like it would cost
money. Both were broken (a flag that does not exist; two omitted required arguments), and
§9 meanwhile claimed "flag parsing" had been verified. `--dry-run` ("Print config and exit
without API calls", `knowledge.py:427`) would have caught all three defects for free. **A
cost constraint is not a reason to leave a command unverified — it is a reason to find the
free way to verify it.** Every paid command should be checked with `--dry-run` under the
denylist guard before it is documented.

---

## Bug 105 — Neither documented `knowledge` command could run

**Status / Severity / Surface** — Fixed (2026-07-16) · Medium · Docs.
`docs/HOW_TO_RUN.md` §9 × `src/nlp_histo/workflows/knowledge.py:407,422,460`.

### Symptom

**Both** commands in §9 exit 2 at argparse, without running:

```
$ nlp-histo knowledge --profile cheap --pmcid PMC1448691 --sync --health-check no
nlp-histo: error: unrecognized arguments: --pmcid

$ nlp-histo knowledge --profile real --all --source-cases eval/data/source_cases_related15.jsonl
nlp-histo: error: the following arguments are required: --health-check
   … and after adding it:
nlp-histo: error: one of the arguments --sync --batch is required
```

### Evidence

Three independent defects in one six-line section:

| Doc said | Parser says |
|---|---|
| `--pmcid PMC1448691` | `knowledge.py:407` — `pmcid` is a **positional** (`nargs="?"`); there is no `--pmcid` flag |
| (cmd 2) no `--health-check` | `:460` — `required=True` |
| (cmd 2) no mode flag | `:422` — `add_mutually_exclusive_group(required=True)` over `--sync` / `--batch` |

The section also asserted "*Verified here:* `--help`, **flag parsing**, and the
missing-file preflight". That claim was false: `--help` and the preflight had been
checked, but the documented command lines themselves never were — had they been run once,
all three defects would have surfaced immediately.

The **legacy** pre-packaging doc had it right. Gitignored
`docs/readmes/HOW_TO_RUN.md:437-440` (that file was deleted in the 2026-07-17 docs
consolidation — it was gitignored, so it is not in git history either; the quote below
is now the only surviving record of it):

```
# Live summarisation — --sync/--batch AND --profile AND --health-check are all required
python scripts/run_paper.py PMC7150310_main --sync  --profile cheap --health-check no
python scripts/run_paper.py --from-selection … --batch --profile real --health-check no
```

— positional PMCID, both required flags, and `--batch` for the full-corpus `real` run.
So this is a **regression introduced by the packaging-era doc rewrite**, not a
long-standing gap.

### Diagnosis

Documentation defect only, and self-limiting: argparse rejects before any provider is
constructed, so a supervisor copy-pasting these loses time, not money. The required-ness
is deliberate and correct — it is the property that stops a paid run starting by accident
(§9's own stated rationale). The parser is right; the doc was wrong.

Root cause is method, not typing: §9 was written by inspection rather than execution,
because executing it looked like it would cost money. It would not have — `--dry-run`
(`knowledge.py:427`, "Print config and exit without API calls") resolves the entire
invocation for free. The safe verification tool existed and went unused.

### Fix

§9 now: uses the positional PMCID; passes `--health-check no` and `--batch`/`--sync` on
both lines; states that all three arguments are required with no defaults; documents
`--dry-run` as the free way to check a paid invocation before running it; and records
that both commands were verified by execution rather than inspection.

### Verification

Both commands, **exactly as documented**, with `--dry-run` appended and run under a
`getaddrinfo` guard that raises on any billable host:

```
$ nlp-histo knowledge PMC1448691 --profile cheap --sync --health-check no --dry-run
  L1  gpt-4.1-nano … L3  gpt-4.1-mini            exit=0

$ nlp-histo knowledge --all --profile real --batch --health-check no \
      --source-cases eval/data/source_cases_related15.jsonl --dry-run
  L1 … L3  claude-sonnet-4-6                      exit=0
```

Both resolve their full cascade config and exit 0. Zero paid hosts contacted.

---

## Bug 106 — §10 overstated the replay as "offline" with "no model inference"

**Status / Severity / Surface** — Fixed (2026-07-16) · Medium · Docs.
`docs/HOW_TO_RUN.md` §10 × `nlp-histo replay chapter9`.

### Symptom

§10 read: "Offline: no API key, no database, no model inference, no cost." Two of the
four claims are false — the replay reaches the network and runs model inference.

### Evidence

Under a `getaddrinfo` guard logging every resolved host, a full replay contacted:

```
RESOLVE:s3-us-west-2.amazonaws.com:443    scispaCy en_core_sci_lg + UMLS KB
RESOLVE:huggingface.co:443                pritamdeka/PubMedBERT-MNLI-MedNLI
```

and logged `RELATE: loading NLI model 'pritamdeka/PubMedBERT-MNLI-MedNLI' on
device='mps'` — inference, locally, on the GPU. "No API key" / "no cost" both re-confirmed
true: zero paid-host blocks fired and both frozen embedding caches reported **0 cache
misses**.

### Diagnosis

The claim holds *once both models are in the local cache*, which is true on any machine
that has run the pipeline — so it was never falsified in day-to-day use. It fails exactly
for the audience §10 is written for: a supervisor on a cold cache.

### Fix

§10 now separates "no paid provider is ever called" (true, with the two free hosts named
and the verification date) from "offline" (false on a cold cache: several hundred MB of
model downloads, then genuinely no network), and cross-links B-107.

### Verification

Re-read of §10 against the two `RESOLVE:` logs and the `0 cache misses` lines.

---

## Bug 107 — Replay silently degrades when the UMLS linker is unreachable

**Status / Severity / Surface** — Fixed (2026-07-16) · High · Reproducibility.
`nlp-histo replay chapter9` × `entities/umls_resources.py` (linker load).

> **Scope of the fix.** The *silent degradation* — the actual bug — is fixed: the replay
> refuses and exits 3 instead of reporting success on wrong numbers. *Offline capability*
> is deliberately **not** addressed (the sidecar-based cache fallback below remains a
> possible future improvement). The replay still requires a free network fetch; it now
> says so and fails loudly instead of guessing.

### Symptom

With no network, the replay **warns once and exits 0**, writing plausible-but-wrong
numbers. Nothing in the exit code, the CSVs, or `manifest.json` marks the run degraded.

### Evidence

Three runs, same tree, same artifact root:

| Run | Network | Result |
|-----|---------|--------|
| 1 | all blocked | exit 0, **8 CSVs**; `06` and `12` differ from baseline; `11` skipped |
| 2 | paid hosts blocked only | exit 0, **9 CSVs**, **9/9 byte-identical** to baseline |
| baseline | (frozen, 2026-07-14) | — |

Run 1's only signal was:

```
WARNING   UMLS: linker unavailable — downstream stages will skip CUI work: <download failed>
```

The diff between runs 1 and 2 is exactly the CUI-dependent analyses
(`06_exp_f_test_split.csv`, `12_real_profile_grounding_polarity.csv`). Run 2 proves the
normal path is unaffected and fully reproducible.

### Diagnosis

**Corrected 2026-07-16 after a faithful offline test.** The run-1 evidence above was
gathered with a guard that raised a custom `RuntimeError`, which library fallback paths
(`except (ConnectionError, OSError)`) never catch — so run 1 overstated the blast radius.
Re-tested by raising `socket.gaierror` (what a real machine with no DNS actually
raises), the picture is an **asymmetry**, and the surviving half is worse than first
filed:

* **HuggingFace NLI model — caches correctly, loads offline.** Verified: the tokenizer/
  model load from `~/.cache/huggingface` under a real `gaierror`, with or without
  `HF_HUB_OFFLINE=1`. Run 1's `status=error` on `11_nli_input_four_mode_ab.csv` was an
  **artifact of the RuntimeError guard**, not real behaviour. On a genuinely offline
  machine `11_` regenerates fine — so the one loud symptom that might have tipped a user
  off **does not actually occur**.
* **scispaCy UMLS linker — fails offline even with a complete cache, and fails
  silently.** This is real and reproducible.

**Why pre-fetching cannot fix it.** scispaCy keys its cache on the *live* ETag:
`get_from_cache` issues an unconditional `requests.head(url)`
(`scispacy/file_cache.py:119`), reads `ETag` (`:126`), and `url_to_filename` builds the
cache name as `sha256(url) + "." + sha256(etag)` (`:53-68`). With no network the ETag is
unobtainable, so the filename cannot be computed, so a **byte-complete 2.1 GB cache on
disk is unfindable**. Confirmed on this machine: all five linker artifacts are cached
(`nmslib_index.bin` 759 MB, `umls_2022_ab_cat0129.jsonl` 658 MB,
`tfidf_vectors_sparse.npz` 516 MB, `concept_aliases.json` 276 MB, `tfidf_vectorizer.joblib`),
each with a sidecar recording its url + etag — and the offline load still dies fetching
`tfidf_vectors_sparse.npz`. **No amount of pre-downloading makes the replay
offline-capable.**

So the corrected severity is **higher**, not lower: the trigger is *any* offline run
(warm cache or cold — cache warmth is irrelevant), and the only symptom is a single
WARNING line, because the NLI stage that failed loudly in run 1 does not fail on a real
offline machine. The networked path remains byte-exact (9/9 CSVs), so this bites the
clean-room reproduction and nothing else.

### Fix

**Shipped — fail hard, before anything is written.** Chosen over stamping
`degraded=true` into `manifest.json`: a manifest flag still leaves nine
authoritative-looking CSVs on disk for someone to read without opening the manifest,
which is the same trap one indirection deeper. Refusing before the output directory
exists makes a bad run leave *nothing* to misread.

The gate lives in the **replay**, not in `umls_resources`, because the two callers want
opposite things: the live per-paper pipeline is *allowed* to skip CUI work and carry on
(that is the documented contract of `get_nlp()` returning `None`), while the replay
exists to reproduce published numbers and must not. So `get_nlp()` is untouched and the
strictness is opt-in:

* `umls_resources.require_umls(context=…, affected_outputs=…)` — probes the loader and
  raises the new `UmlsUnavailableError` unless it is usable.
* `umls_resources._FAILURE_REASON` + `failure_reason()` — `_AVAILABLE=False` alone could
  not distinguish "deliberately disabled via the kill-switch" from "tried to load and
  broke", and those need different messages. The kill-switch path says so plainly rather
  than blaming the network.
* `replay._require_umls_or_refuse()` is called from `configure()` **after** artifact
  validation and **before** `OUT_DIR.mkdir(...)`; `main()` maps the error to exit **3**
  (`0` ran · `2` artifact tree unusable · `3` UMLS unavailable).

The error replays the real exception chain, names `06_exp_f_test_split` and
`12_real_profile_grounding_polarity` as the outputs that would have been wrong, states
that a warm cache does not help and why (the ETag lookup), and notes that the required
fetch is free — so nobody reads "needs network" as "will cost money".

### Not fixed — offline capability (possible future improvement)

The replay still needs a free network fetch. Making it genuinely offline would mean
bypassing scispaCy's ETag-keyed lookup: every cached file has a sidecar (`<name>.json`)
recording `{"url": …, "etag": …}`, so an offline-tolerant `cached_path` could, on
`requests.head` failure, scan the cache dir for the sidecar whose `url` matches and
return that file — no network, no re-download. That would free a clean-room run from
depending on an AI2 S3 bucket staying up with stable ETags. Deliberately **out of scope
here**: it is a behavioural change to a third-party cache path, and the networked path is
free and byte-exact. **Do not attempt to fix this by pre-fetching** — the files are
already fetched; see the Diagnosis.

### Verification

Against the **real** offline failure, not a mock — DNS forced to `socket.gaierror`, the
way a disconnected machine actually fails (the original investigation used a custom
`RuntimeError`, which library `except (ConnectionError, OSError)` fallbacks never catch,
and which therefore manufactured a failure that did not match reality):

```
$ python3 true_offline.py console:nlp-histo replay chapter9 --artifact-root . --output-dir OUT
error: The chapter-9 replay requires the UMLS entity linker, which is unavailable.
  ConnectionError: HTTPSConnectionPool(host='s3-us-west-2.amazonaws.com', …
    NameResolutionError(… [Errno 8] nodename nor servname provided …))
  Affected outputs …  - 06_exp_f_test_split.csv / .json
                      - 12_real_profile_grounding_polarity.csv / .json
EXIT=3                     ← was 0
OUT/                       ← never created
```

Success path unchanged: with network, the same command exits **0** and regenerates
**9/9 CSVs byte-identical** to the frozen baseline, zero paid hosts contacted.

Regression tests — `tests/knowledge_extraction/entities/test_umls_require.py` (6) builds
its fixture by pointing `socket.getaddrinfo` at a `gaierror` and issuing a real
`requests.head` against scispaCy's actual S3 artifact URL, then asserting the captured
exception *is* a `requests.exceptions.ConnectionError` (a RuntimeError-only mock would
re-introduce the original blind spot). It fails the load at `add_pipe("scispacy_linker")`,
not `spacy.load` — faithful to the observed path, and necessary: `requests`' error
subclasses `OSError`, so failing `spacy.load` would be swallowed by get_nlp()'s
`except OSError: "not installed, trying next"` and misreported as "No scispaCy model
found". Coverage: `get_nlp()` still returns `None` (live-pipeline contract intact);
`require_umls` raises on real offline failure; the message is actionable; the success path
is quiet; the kill-switch reports itself distinctly. Plus 3 in `tests/cli/test_cli.py`:
exit 3 + no output directory, the gate fires before `mkdir`, and the available path still
exits 0. Suite: **1565 passed**, ruff clean.

---

## Bug 108 — `grounding.py` overwrites a tracked thesis artifact

**Status / Severity / Surface** — Observed (2026-07-16) · Medium · Eval reproducibility.
`eval/sweeps/grounding.py` × `eval/results/grounding_sweep.md` (tracked).

### Symptom

Running the §12-documented `python eval/sweeps/grounding.py` rewrites a tracked file with
different numbers and dirties the worktree.

### Evidence

`git diff eval/results/grounding_sweep.md` after one run:

```
- created_at: 2026-05-16T16:06:45Z      → + 2026-07-16T13:11:46Z
- pmcids (5): …                          → + pmcids (15): …
- pipeline_config_hashes (1): 149023b8…  → + cfb56a0289b557be
```

### Diagnosis

**Input drift, not non-determinism.** The script pins no input: it consumes whatever is
in `out/summaries`, which now holds 15 papers where the committed snapshot was built from
5. Its output therefore tracks the corpus, not the thesis. The rewritten numbers
correspond to no published table.

### Fix

Not fixed; restored with `git checkout eval/results/grounding_sweep.md` (worktree clean).
Options, to be decided with the other frozen-artifact paths in the module: pin the input
set (a `--source`/`--run-id` filter matching the committed snapshot), default the output
to an untracked path, or untrack the `.md` and treat it as regenerable output.

### Mitigation

§12 documents the overwrite and the `git checkout` recovery.

---

## Bug 109 — Free, cached experiments hard-require an unused `GOOGLE_API_KEY`

**Status / Severity / Surface** — Fixed (2026-07-16, ec11eec) · Medium · Eval UX.
`eval/silver/analysis/map_context.py` × frozen embedding caches.

### Symptom

`python -m eval.silver.experiments.E14_heldout.heldout_eval` exits immediately with
`GOOGLE_API_KEY not set` — on a run that makes no API call.

### Evidence

`_load_map_context` resolves the key before it ever consults the cache
(`map_context.py:81-83`; the OpenAI branch at `:94` is identical). Yet the same run, once
a key is present, reports:

```
Agreement embed pre-warm: 15273 unique claims, 0 cache misses
```

and contacts no Google host under the denylist guard. The key is dead weight.

### Diagnosis

Eager embedder construction. Every developer machine has the key in `.env`, so the
requirement never bit anyone — but it makes a free, fully-cached command unrunnable for
a supervisor without a Google account, and the error sends them to buy an API key rather
than telling them the truth, which is that the run is free and complete offline.

### Fix

Fixed (ec11eec). With `strict_cache_only=True` the gemini/openai branches in
`map_context.py` build a `_NoLiveEmbedding` and never resolve an API key, so a cache miss
raises instead of silently billing. E14/E03/E10/E11/E12 pass `strict_cache_only=True` (e.g.
`heldout_eval.py`), so the offline replays serve every embedding from the cache with no
credential and no client. (The originally-proposed lazy-construction approach was not the one
taken; the eager path still exists in the non-strict `else` branch for live callers.)

### Mitigation

§12 notes the requirement and that any non-empty value satisfies it.

---

## Bug 110 — Stale config-loader tests assert pre-calibration defaults

**Status / Severity / Surface** — Fixed (2026-07-16) · Low · Tests.
`tests/test_config_loader.py` × `configs/run.yaml`.

### Symptom

`python -m pytest` → **2 failed, 1552 passed**:

```
FAILED tests/test_config_loader.py::test_agreement_scorer_kind_loaded_from_run_yaml
    assert 'hybrid' == 'embedding'
FAILED tests/test_config_loader.py::test_hybrid_config_loaded_from_run_yaml
    assert 0.15 == 0.25
```

### Evidence

The tracked `configs/run.yaml` deliberately pins the calibrated winner, and says so
in its own comments:

```yaml
scorer_kind: hybrid         # E06 family_refine pin (was 'embedding').
alignment_strategy: greedy  # E06 family_refine pin (was 'soft_max').
```

Corroborated independently by `eval/silver/experiments/E14_heldout/heldout_eval.py:67-74`,
whose `_frozen_spec()` — documented as "the E05–E08 winner scorer (= configs/run.yaml /
E03 `_frozen_spec`)" — hard-codes `w_category=0.15, w_embedding=0.30, w_entity=0.50,
w_evidence=0.05`.

### Diagnosis

**The config is right; the tests are stale.** They encode the defaults that were
superseded when the sweep winner was pinned, and were never updated. Test-only — no
runtime behaviour is wrong.

### Fix

Both assertions updated to the calibrated values (`hybrid`; 0.15/0.30/0.50/0.05), and
their docstrings now say that `configs/run.yaml` pins the E06 winner rather than the
dataclass defaults — so the next reader does not "correct" them back.

The sibling `test_hybrid_config_defaults` (0.25/0.40/0.25/0.10) was deliberately **left
untouched**: it asserts the `HybridConfig` *dataclass* defaults, which are a different
thing from what the YAML ships, and it correctly passes. Only the two `_from_run_yaml`
tests were wrong.

HOW_TO_RUN §11 tells the supervisor to run `pytest`; leaving these red would greet a
clean-room reproduction with two failures over a config that is in fact correct — a
credibility cost out of proportion to the defect.

### Verification

```
$ python -m pytest tests/test_config_loader.py -q     →  33 passed
$ python -m pytest -q                                 →  1554 passed, 0 failed  (~4 min)
$ ruff check .                                        →  All checks passed!
```

1552 + the 2 repaired = 1554; no other test changed. `docs/HOW_TO_RUN.md` §11 now
records the green result.

---

## Bug 111 — The CLI's own documented passthrough help did not work

**Status / Severity / Surface** — Fixed (2026-07-16) · Medium · CLI.
`src/nlp_histo/cli/main.py:186-192` (`_split_forwarded`) × `:128` (ingest help string).

### Symptom

The exact invocation the CLI tells you to use exits 2:

```
$ nlp-histo ingest -- --help
usage: nlp-histo [-h] [--pdf-dir PDF_DIR] [--glob GLOB] … [--main-pdf-only | --no-main-pdf-only]
nlp-histo: error: unrecognized arguments: -- --help          # exit 2

$ nlp-histo ner extract -- --help
nlp-histo: error: unrecognized arguments: -- --help          # exit 2
```

### Evidence

`ingest` and `ner extract|merge|export` define no options of their own — everything after
the command path is sliced out of `argv` by `_split_forwarded` and handed to the
workflow's parser. `nlp-histo ingest --help` therefore *deliberately* prints this CLI's
short stub rather than the runner's options; for `knowledge` that interception is what
guarantees the PAID warning is seen. The stub points elsewhere for the real list
(`main.py:128`):

> Options are passed through to the extraction runner; see `nlp-histo ingest -- --help`.

That command did not work. So the workflow's real options — `--pdf-dir`, `--out-root`,
`--entity-cache` and ~30 more — were **undiscoverable through the CLI**: the usage line
appeared only as a side-effect of the error.

### Diagnosis

`_split_forwarded` forwarded the `--` verbatim:

```python
rest = argv[consumed:]           # ["--", "--help"]
return list(path), rest          # runner.main(["--", "--help"])
```

argparse reads a bare `--` as its **positional separator**, so the `--help` after it is
demoted from an option to a positional — and the runner's parser has no positional to
absorb it. Hence `unrecognized arguments: -- --help`.

Found by executing the documented hint rather than grepping for flag names. Every flag
the doc mentions *did* exist, which is precisely why an inspection-based check missed
this — the same lesson as B-105: **verify commands by running them, not by reading them.**

### Fix

`_split_forwarded` now treats a leading `--` as an explicit "everything after this
belongs to the workflow" marker and **drops** it before forwarding:

```python
if rest[:1] == ["--"]:
    return list(path), rest[1:]      # ingest -- --help  ->  runner.main(["--help"])
if rest[:1] and rest[0] in ("-h", "--help"):
    return argv, []                  # ingest --help     ->  this CLI's stub (cost warning)
```

The two-level behaviour is now intentional and documented in the function's docstring:
plain `--help` is the CLI's (preserving the `knowledge` cost warning), `-- --help` is the
workflow's.

### Verification

```
$ nlp-histo ingest -- --help        → runner options, exit 0
$ nlp-histo ner extract -- --help   → batch_ner options (incl. --entity-cache), exit 0
$ nlp-histo knowledge --help        → still prints the PAID warning, exit 0
$ nlp-histo ingest --pdf-dir X --out-root Y   → still forwards (no regression)
```

Regression tests added in `tests/cli/test_cli.py`:
`test_explicit_passthrough_reaches_the_workflow_help` (asserts the `--` is dropped and
`["--help"]` reaches the runner) and `test_plain_help_after_command_is_the_clis_own`
(asserts the runner is *not* dispatched, protecting the cost warning). The existing CLI
tests covered flag forwarding (`test_ner_extract_forwards_entity_cache_flag`) but never
the `--` form, which is how this survived. Suite: 1554 passed → 1556 passed.

---

## Bug 112 — The replay's paid-call guard misses the Gemini embedding cache

**Status / Severity / Surface** — Fixed (2026-07-16, commit `8d0c5c5`) · High ·
Cost / reproducibility. `workflows/replay.py:132,163` (`REQUIRED_ARTIFACTS`) ×
`eval/silver/analysis/map_context.py:36,90`.

> The finding below is preserved as written when it was Observed. The **Fix** section
> records what shipped; the Symptom/Evidence/Diagnosis describe the defect as it existed
> before `8d0c5c5` and are deliberately left in the past tense rather than erased.

### Symptom

None today — and that is the point. The replay is free here because both caches happen to
be complete. An artifact tree with the OpenAI cache but not the Gemini one passes
validation, starts confidently, and spends money.

### Evidence

The 2026-07-16 run log opens two caches:

```
SQLite embedding cache: 58895 entries at eval/data/embedding_cache_openai.sqlite
SQLite embedding cache: 87942 entries at eval/data/embedding_cache_gemini.sqlite   ← unvalidated
Agreement embed pre-warm: 15273 unique claims, 0 cache misses
```

Only the first is in the required set:

```python
# replay.py:132 / :163
RequiredArtifact(Path("eval") / "data" / "embedding_cache_openai.sqlite", …)
FROZEN_EMBEDDING_CACHE = Path("eval") / "data" / "embedding_cache_openai.sqlite"
```

The second arrives via a different door — `_load_map_context("gemini", embed_cache_path=None)`:

```python
# map_context.py:30,36,90
from eval.paths import REPO_ROOT
_FROZEN_GEMINI_CACHE = REPO_ROOT / "eval" / "data" / "embedding_cache_gemini.sqlite"
path = Path(embed_cache_path) if embed_cache_path else _FROZEN_GEMINI_CACHE
```

and a miss bills:

```python
# map_theta_sweep.py:679
if miss_texts:
    new_embs = embed_fn(miss_texts)          # live AgreementGeminiEmbedder → PAID
```

`make_embedding_cache` creates an empty SQLite when the file is absent, so a missing cache
does not error — it misses on all 15 273 claims and embeds every one.

### Diagnosis

Two defects, and fixing only the first would give a false sense of safety:

1. **Incomplete required set.** §10 sells the preflight as the reason a missing cache
   "can never" cause paid calls in a free workflow. That holds for OpenAI and not for
   Gemini. The guarantee is written as absolute and is half-enforced.
2. **Wrong anchor.** `_FROZEN_GEMINI_CACHE` resolves against `eval.paths.REPO_ROOT` — the
   *repository* — not the replay's `--artifact-root`. So `--artifact-root`, the flag whose
   entire purpose is to say where the data lives, does not govern this input. A replay
   pointed at a copied tree reads the repository's cache instead (or misses, outside one).
   Same family as B-102.

Why it survived: every developer machine has both caches, complete, in the repository —
so the happy path is genuinely free and the hole never shows. It would surface on exactly
the clean-room reproduction the replay exists to serve.

### Fix

**Shipped in `8d0c5c5`** — both halves together, because either alone would give false
assurance: adding the gemini cache to `REQUIRED_ARTIFACTS` while the path still resolved
against `REPO_ROOT` would have validated a file the run does not read.

* **Anchoring.** All four `_load_map_context` call sites now pass `embed_cache_path`
  explicitly, from `frozen_embedding_cache(kind)`, resolved under `--artifact-root`. The
  repo-anchored `_FROZEN_*` fallbacks remain in `map_context` for its other (repository-only)
  callers, but the replay no longer reaches them. No new packaged→eval dependency was
  introduced: `_load_map_context` already accepted the parameter — the replay was passing
  `None`.
* **Entry-level preflight.** `validate_embedding_cache_entries()` derives the required
  texts from the voter cache with the same extraction the production pre-warm uses, then
  checks every entry in **both** caches. Existence is not enough: an empty or partial cache
  is a valid SQLite database and `make_embedding_cache` creates one on demand. Corrupt or
  unreadable caches fail closed. Runs before `OUT_DIR.mkdir`; exit **4**; reports counts
  and paths only — never claim text, never 15k keys.
* **Strict cache-only.** `strict_cache_only=True` constructs **no provider at all**, so a
  race, a malformed row, or a preflight that failed to enumerate something raises
  `CacheOnlyViolation` instead of billing. The guarantee is structural, not contingent on
  an unset API key.

`get_nlp()`-style permissiveness is preserved for repository-only callers: `map_context`'s
default remains `strict_cache_only=False`, so the sweeps and experiments are unaffected.

**Not fixed (unchanged):** B-102's cwd-relative `PRIMER_DIR` is the same anchoring family
and is still open. It does not affect the replay, which takes an explicit `--artifact-root`.

### Verification

```
tree without gemini cache        → exit 2 (artifact validation), nothing written
tree with EMPTY gemini cache     → exit 4 "gemini: 15273 of 15273 required entries missing"
real tree, network on            → exit 0, 9/9 CSVs byte-identical, 0 paid hosts
```

The **15 273** figure is the load-bearing check: it equals the run's own
`Agreement embed pre-warm: 15273 unique claims`, confirming the preflight enumerates
exactly the set the run embeds rather than an approximation of it.

Regression tests — `tests/cli/test_replay_embedding_cache_preflight.py` (15): missing
OpenAI cache, missing Gemini cache, incomplete OpenAI, incomplete Gemini, empty cache,
corrupt cache, non-default `--artifact-root`, complete-cache success, exit 4 + no output
directory, gate-before-mkdir ordering, and two proving no provider is constructed and that
an unexpected runtime miss raises rather than bills. Two fixture notes worth keeping: the
voter-cache fixture must satisfy `AuditableSummary` in full (validation failures are
silently skipped, so an under-specified fixture yields zero claims and tests nothing), and
a corrupt-cache test must delete the `-wal` sidecar — SQLite otherwise recovers the rows
and reports a healthy cache. `tests/test_replay_artifacts.py`'s fixture grew with the new
required artifact; its coverage test caught the change, as designed. Suite: 1581 passed.

---

## Bug 113 — An explicit env file is silently overridden by inherited DB variables

**Status / Severity / Surface** — Fixed (2026-07-16, commit `30755f8`) · High · Data integrity / safety.
`src/nlp_histo/database/db_connection.py:22-25` × `ENV_LOADING.md:159-169`.

### Symptom

A command explicitly pointed at a test database connects to production instead, with no
warning:

```
$ NLP_HISTO_ENV_FILE=/tmp/ingest_test.env  python -c "...print(engine.url)"
  database = nlp_histo          ← the established 977-paper corpus
                                ← /tmp/ingest_test.env says DB_NAME=new_local_db
```

### Evidence

Demonstrated during the §7 ingest verification (2026-07-16). The harness asserted its
target before connecting, and the assertion fired:

```
AssertionError: WRONG TARGET: nlp_histo
  host=localhost port=5432 user=local_db_user database=nlp_histo
```

The one-PDF ingest was seconds from writing into the production corpus. The only thing
that stopped it was a hand-written assert in the verification script — the library itself
never says which database it is about to use. Re-running with `env -u DB_NAME …` resolved
correctly to `new_local_db`.

The shell had `DB_NAME=nlp_histo` exported from an earlier `source .env`.

### Diagnosis

`db_connection.py:22-25` resolves *which file* to load, then loads it:

```python
_explicit = os.getenv("NLP_HISTO_ENV_FILE")
_found = _explicit or find_dotenv(usecwd=True)
if _found and Path(_found).exists():
    load_dotenv(_found)          # python-dotenv defaults to override=False
```

`override=False` means values already in the environment are never replaced. So
`NLP_HISTO_ENV_FILE` selects the file and the file then loses every variable the shell
already defines.

**The precedence is deliberate and documented — this is not an ordering bug.**
`ENV_LOADING.md:159-169`:

> **Environment Variable Precedence**
> 1. **Actual environment variables** (highest priority)
> 2. **.env file** (loaded by dotenv)
> 3. **Default values** (hardcoded fallbacks)
>
> ```bash
> export DB_NAME=custom_db
> alembic current  # Uses 'custom_db', not the .env value
> ```

Users are explicitly invited to override file values from the environment. A blanket
`override=True` would break that contract and is **not** the fix.

The defect is **silence**, not ordering. `NLP_HISTO_ENV_FILE` reads as an intentional,
explicit selection — "use *this* configuration" — and when it loses to ambient state,
nothing says so: no warning, no conflict detection, no echo of the resolved target. Two
reasonable behaviours (env-wins; explicit-file-wins) are indistinguishable to the
operator at the moment it matters.

Contributing factors:

* **Untested.** `tests/test_runtime_paths.py:39-43` (`test_env_file_override`) asserts only
  that `_env_path()` returns the explicit path — file *selection*, never value
  *precedence*. The conflict case has no coverage.
* **Misleading documentation.** `docs/HOW_TO_RUN.md` §4 describes the variable as
  "explicit path to the `.env` file", which reinforces the wrong mental model and says
  nothing about it losing to inherited variables.
* **Likelihood is not low in practice.** It requires `DB_*` to be exported already — which
  `source .env`, direnv, CI secrets, docker-compose and IDE run-configurations all do.

### Fix

**Shipped in `30755f8`** — options 1 and 2 below, both narrow; the documented
precedence is unchanged. Options 3 (documentation) shipped in `423cf05`; option 4 remains
deferred.

1. **Conflict detection (recommended).** When `NLP_HISTO_ENV_FILE` is set *and* a `DB_*`
   variable in that file disagrees with the inherited environment, fail loudly (or warn
   with the winner named). This keeps the documented env-wins precedence intact for the
   ordinary discovered-`.env` case, and only speaks up when an *explicit* selection is
   being contradicted — the exact situation that produced the near-miss. It cannot break
   anyone whose environment and file agree.
2. **Echo the resolved non-secret target.** `db init`/`db check` already print
   `Target: user@host:port/database`; the write paths (`ingest`, `ner`) do not. Printing
   it once at connect time makes the mistake visible at the moment it is made, and is a
   pure addition.
3. **Documentation.** §4 to state the precedence and the footgun explicitly.
4. **`--env-file` CLI flag with defined precedence** (an explicit flag legitimately
   outranking the environment) — larger surface, deferred unless wanted.

Rejected: global `override=True` (breaks the documented contract of
`ENV_LOADING.md:159-169`).

### Verification

Until a fix ships, every database-writing verification in this pass proves its target
first:

```python
u = get_db_connection().engine.url
assert u.database == EXPECTED, f"WRONG TARGET: {u.database}"
print(f"{u.username}@{u.host}:{u.port}/{u.database}")   # password never printed
```

That guard is what caught this, and it is a cheap habit independent of the eventual fix.

---

## Bug 114 — Document bibliographic columns are never populated

**Status / Severity / Surface** — Observed (2026-07-16) · Low · Known limitation /
unused fields. `database/models.py:32-34` ×
`pdf_text_extraction/outputs/db_ingester.py:84-89`.

### Symptom

`documents.title`, `documents.journal` and `documents.publication_year` are `NULL` for
every paper ever ingested.

### Evidence

The established corpus, 2026-07-16:

```
with_title=0  with_journal=0  with_year=0  with_text_source=977  total=977
```

Reproduced on the independently verified one-PDF ingest into a fresh database
(`title=<null>`), so it is a property of the pipeline, not of a historical run. Noted
during the §7 verification precisely because it *looked* like a defect in that run — it
is not; it matches production exactly.

### Diagnosis

`db_ingester.py:84-89` never sets them:

```python
doc = Document(
    pmcid=pmcid,
    filename=f"{pmcid}.pdf",
    file_path=str(pdf_path) if pdf_path else f"{pmcid}.pdf",
    text_source="pdf",
)
```

All three columns are nullable (`title = Column(Text)`), so nothing errors. No other code
writes them — `scripts/seed_fake_papers.py:157` sets `title`, but only for synthetic
papers. The `text_source` default of `'xml'` suggests the columns date from an
XML-ingest design that would have carried this metadata; the PDF path never acquired it.

**Nothing depends on them.** The only consumer is Pipeline C's paper selection, which
carries `title` through `loaders.py:152` → `fingerprints.py:234` → `export.py:162,210`
into an export field that is simply `null`. It is never used in scoring, similarity or
selection, and no document promises the field is populated. The thesis numbers do not read
these columns.

### Fix

**Deliberately not fixed** during the verification pass: adding speculative title
extraction would touch the ingest path with no consumer demanding it, and every reported
number would be unaffected. Classified as an unused-field / known-limitation cleanup item.

Options when someone picks it up:

* populate from the PMC XML in `files/organized_xmls/` (already downloaded, and the
  authoritative source for bibliographic metadata) during ingest;
* populate from Docling's detected document title (weaker — it is a layout heuristic);
* drop the three columns and the dead `title` plumbing in `eval/paper_selection/`.

### Verification

None required — no behaviour changed. If it is ever fixed, `with_title` should rise from
0 and the `paper_selection` export should stop emitting `"title": null`.

---

## Topic — `pipeline_runs` after ingest: not a defect (closed 2026-07-16)

The §7 ingest verification observed `pipeline_runs=0` after a successful run and asked
whether ingest should register a run. **It should not.** Recorded so the question is not
re-opened:

* `PipelineRun`'s own docstring (`database/models.py:319-325`) scopes it:
  *"Tracks one execution of **KnowledgeExtractionRunner.process()** for a single paper.
  Every stage-level persistence table … will FK here, making this the root of all lineage
  tracing."* Its children are the `sum_*` knowledge-extraction tables.
* The only writer is `knowledge_extraction/persistence.py:601`.
* The established database corroborates it: **33 `pipeline_runs` rows against 977 ingested
  documents** — the row count tracks knowledge-extraction runs, not ingests.
* Ingest is not without provenance: it writes `out/run_metadata/run_{ISO}_{uuid}.json`
  via `RunManifestWriter`, plus per-document `{pmcid}_stats.json`.
* No documentation claims ingest writes to `pipeline_runs`.

So the empty table is the designed contract, and `docs/HOW_TO_RUN.md` §7 records it as
such rather than as a caveat. No bug filed; no behaviour changed.

---

## Bug 115 — `ner merge` and `export` filtered on a model name that is never stored

**Status / Severity / Surface** — Fixed (2026-07-16, commit `8d6cc9c`) · Medium · NER.
`ner/merge_entities_by_umls.py` + `ner/export_disease_entities.py` (`--model` default) ×
`ner/ner.py:215`.

### Symptom

`ner extract` populates the table; the next documented command finds nothing, and says so
cheerfully:

```
$ nlp-histo ner extract --entity-cache …
  Summary: 1 Processed | 0 Skipped | 0 Errors        ← 865 entities, 749 with CUIs

$ nlp-histo ner merge
  No entities with UMLS mappings found.              ← exit 0
$ nlp-histo ner export
  No disease entities found matching the criteria.   ← exit 0
```

### Evidence

The database plainly holds them, under one name — and only one:

```sql
-- isolated verification database, after ner extract
core_sci_lg | umls | n=865
-- the ESTABLISHED corpus
core_sci_lg | n=1792440          ← every entity ever extracted
```

spaCy's ground truth:

```
meta[lang] = en
meta[name] = core_sci_lg        ← ner.py:215 stores THIS
```

while both consumers declared:

```python
parser.add_argument('--model', default='en_core_sci_lg', …)   # matches 0 rows
```

Passing the stored value made merge work immediately (749 occurrences → 762 files),
confirming the diagnosis before any change was made.

### Diagnosis

The `en_` in `en_core_sci_lg` is the *package* name; `meta["name"]` omits the language
prefix. `ner.py` knew this — `:178` carries the comment *"exposes via `nlp.meta["name"]`
("core_sci_lg" for en_core_sci_lg)"* and its skip-check hardcoded `"core_sci_lg"`
correctly. The two readers were written against the package name and never reconciled, so
writer and readers disagreed for the corpus's entire lifetime.

Why it survived: the failure is a **silent no-op**. Exit 0, a plausible message
("No entities … found"), and no consumer downstream to notice the missing files. Same
family as B-107 — an empty result that is indistinguishable from a real one.

### Fix

Two parts, in `8d6cc9c`:

1. **One identifier.** `enums.DEFAULT_MODEL_NAME = "core_sci_lg"` — documented as spaCy's
   `meta["name"]`, not the package name — now feeds `ner.py`'s skip-check and both
   consumers' `--model` default. Explicit `--model` overrides are untouched.
2. **Emptiness must be honest.** `model_filter.check_model_filter()` raises
   `NoMatchingEntitiesError` (exit 1) when the requested model matches nothing *while the
   database holds entities under other names*, listing what is available and explaining the
   prefix. It stays silent for a genuinely empty corpus, or when no `--model` was given —
   those are honest empty results. The distinction is the whole point: "no data" and
   "wrong question" must not look alike.

Choosing to change the *readers* rather than the writer was deliberate: making `ner.py`
store `en_core_sci_lg` would orphan 1 792 440 existing rows and demand a migration, for no
gain.

### Verification

Isolated database (production neither re-run nor mutated), documented CLI throughout:

```
ner extract  → 865 entities · 749 with CUIs · all 14 text elements
ner merge    → 749 occurrences · 762 files       (was: "No entities found", exit 0)
ner export   → 89 disease CUIs · 178 files       (was: "No entities found", exit 0)
ner merge --model en_core_sci_lg → exit 1, names 'core_sci_lg' as available
ner merge --model core_sci_lg    → exit 0 (explicit override still works)
```

Output is meaningful, not merely present: `C0002989 "Epithelioid hemangioma of skin"` for
a paper on cutaneous epithelioid angiomatous nodule. Tests:
`tests/test_ner_model_name.py` (11) pin the constant against spaCy's actual metadata, the
writer/reader agreement, and every branch of the mismatch check. Suite: 1592 passed.

### Blast radius — audited read-only, 2026-07-16

**Nothing depends on the missing output; no reported number changes.**

* No thesis chapter references `umls_entities*` / `disease_entities*`.
* `eval/paper_selection` reads the `entities` **table** directly (`loaders.py:116-121`) and
  **never filters on `model_name`** — it was never affected. Its
  `PaperFingerprint.disease_entities` is a keyword-derived set (`fingerprints.py:265,304`):
  a name collision with the export directory, not a dependency.
* The only consumer of those directories is
  `legacy/langchain-summarization/count_tokens.py` — quarantined, superseded, not a live
  path.
* `README.md:282-283` documents the output locations (documentation only).

That absence of consumers is itself the explanation for the bug's longevity: the command
produced nothing, and nothing was waiting for it.

---

## Bug 116 — The wheel install block omits the dependency step

**Status / Severity / Surface** — Fixed (2026-07-16) · Low · Docs / packaging.
`docs/HOW_TO_RUN.md` §2 × `pyproject.toml` (no `[project] dependencies`).

### Symptom

Following §2's wheel block verbatim yields a package that imports and prints help, then
fails on the first command that does anything:

```
$ pip install dist/nlp_histo-0.1.0-py3-none-any.whl
Successfully installed nlp-histo-0.1.0
$ nlp-histo --help          → exit 0
$ nlp-histo db check        → ModuleNotFoundError: No module named 'sqlalchemy'
```

### Evidence

Fresh venv outside the repository, Python 3.12.0, wheel only:

```
pip install --dry-run …/nlp_histo-0.1.0-py3-none-any.whl
  Would install nlp-histo-0.1.0          ← nothing else
wheel METADATA Requires-* lines: 1
  Requires-Python: <3.13,>=3.10          ← no Requires-Dist at all
```

Everything the wheel *does* promise works: `import nlp_histo` resolves from
`site-packages` (not the checkout, with `PYTHONPATH` unset), all 11 command/subcommand
`--help` invocations exit 0, and both packaged resources load —
`model_prices.json` (1437 B), `nli_models.yaml` (1558 B).

### Diagnosis

`pyproject.toml` declares no `dependencies`, by design: `requirements.txt` is the pinned,
tested set, and CLAUDE.md names it "the tested source of truth". The editable block in §2
reflects that correctly — `pip install -r requirements.txt` **then** `pip install -e .
--no-deps`. The wheel block does not: it says "no source tree needed" (true of the
*package*) and lists only the wheel install, implying self-sufficiency the wheel never had.

Neither the packaging nor the dep-less wheel is wrong. The documentation is.

### Fix

§2's wheel block now installs `requirements.txt` alongside the wheel, states plainly that
the wheel carries no dependencies and why, and records the verification — including the
`ModuleNotFoundError` as *expected* behaviour for a wheel-only install rather than a
defect to be reported.

### Verification

The commands above, from a venv outside the repository. Not exercised: installing
`requirements.txt` into that venv (a multi-GB torch/docling resolution on a
disk-constrained machine) — so the wheel's packaging is verified end-to-end and its
dependency resolution is not; §2's inventory says so.

---

## Bug 117 — `acquire download` reported success after downloading nothing

**Status / Severity / Surface** — Fixed (2026-07-16) · Medium · Acquisition.
`cli/main.py:40` (`_acquire`) × `acquisition/downloader.py:120`.

### Symptom

Every requested paper fails; the command says it is done and exits 0:

```
$ nlp-histo acquire download --pmcid-file one_pmcid.txt --output-dir tarballs
Looking up PMC10047158...
❌ Failed to download https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/fa/c8/PMC10047158.tar.gz: 404
Done — 0 tarball(s) in tarballs.
$ echo $?
0
```

### Evidence

`download_papers()` counts what it fetched and returns it (`downloader.py:120`); the CLI
threw the number away:

```python
download_papers(args.pmcid_file, args.output_dir, overwrite=args.overwrite)
...
return 0            # unconditional
```

Observed live against NCBI on 2026-07-16 with a single PMCID.

### Diagnosis

A count of zero out of one requested is a failure, and nothing distinguished it from a
successful no-op. The per-PMCID reason *was* printed — but a script, a CI job, or a
`&&` chain sees only the exit code, and this one said success. The same shape as B-107
(replay degraded silently) and B-115 (`ner merge` matched nothing silently): the failure
is visible only to a human reading the log.

### Fix

`_acquire` now compares fetched against requested and exits 1 when a non-empty request
yields nothing, referring the reader to the per-PMCID reasons above it.

Two cases deliberately stay 0:

* **Partial success.** Papers outside the OA subset are reported individually
  (`⚠️ {pmcid} is not in the Open Access Subset`) and are a normal, expected outcome of a
  bulk fetch — failing the run on them would make the common case red.
* **An empty PMCID file.** Nothing requested, nothing fetched: vacuously fine.

### Verification

Against the live 404 (exit 1, previously 0) and by tests covering total failure, partial
success and the empty-file edge (`tests/cli/test_cli.py`). The existing dispatch test's
stub returned `None` — an artifact of the discarded-return contract — and now returns a
count, matching the real signature.

---

## Topic — NCBI OA packages are unreachable (upstream blocker, 2026-07-16)

**From-scratch corpus acquisition (§6) is currently impossible, and not because of
anything in this repository.** NCBI's OA API advertises package paths that exist on
neither its HTTPS nor its FTP server.

### Evidence — bounded diagnostic, 5 PMCIDs, one API call + one HEAD each, no retries

Stratified across the corpus's full range by NCBI's own "Last Updated" field, every one
of them currently advertised by the official OA API as having a package:

| PMCID | OA index date | HTTPS status | content-type |
|---|---|---|---|
| PMC2680278 | 2010-12-27 | **404** | `text/html` |
| PMC8395919 | 2021-08-28 | **404** | `text/html` |
| PMC7155696 | 2023-11-12 | **404** | `text/html` |
| PMC11791731 | 2025-02-11 | **404** | `text/html` |
| PMC12687694 | 2025-12-10 | **404** | `text/html` |

**HTTPS 200: 0/5.** Every response is an HTML error page, not an archive. Fifteen years
of publication dates fail identically, so this is not a handful of withdrawn papers.

The API still hands out `ftp://` links, exactly as `downloader.py:107` assumes:

```
api        = ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz
rewritten  = https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz → 404
```

### The rewrite is NOT the cause — refuted, not assumed

The obvious suspect was `downloader.py:107-109`, which rewrites `ftp://` → `https://` on
the assumption that "the hosts serve the same paths over TLS". One FTP probe of one
failed case (the authorised limit) refutes it:

```
ftplib → FTP(ftp.ncbi.nlm.nih.gov), anonymous login: OK
ftp.size("/pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz")
  → 550 /pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz: No such file or directory
```

The FTP server is up and accepts the connection; the file is simply not there. **FTP and
HTTPS fail identically**, so the scheme rewrite is sound and `acquire download` is
behaving correctly — it faithfully requests what NCBI tells it to request, and NCBI does
not have it.

### Diagnosis

NCBI's `oa.fcgi` API, and the copy of NCBI's own index committed at
`files/oa_file_list.csv`, both point at `oa_package/**` paths that no longer resolve on
either protocol. The most likely explanation is that the OA package tree has been
relocated or retired while the API continues to advertise legacy paths; that hypothesis
is **not** tested here and should not be treated as established.

Local code is exonerated on the available evidence:

* the rewrite is not at fault (FTP fails the same way);
* the paths are not mis-derived — they match NCBI's own committed index byte-for-byte;
* B-117's exit-code bug is real and fixed, but is a *reporting* defect, not the cause.

### Consequences

* **§6 `acquire download` was blocked at the time of this note** (later fixed — see the
  B-118 update below), for any paper in the bounded sample. A supervisor rebuilding the corpus
  from scratch was blocked until the upstream layout was identified.
* **`acquire unpack` is unverifiable** as a side effect: no tarball can be obtained, and
  none remain on disk (they are deleted after `organize`).
* **Nothing downstream is affected.** The existing corpus (1132 PDFs, 977 ingested) is on
  disk, and every other documented command — ingest, NER, replay, the experiments — was
  verified against it.

### Superseded by B-118 (fixed 2026-07-16, 27ed0f8)

**Update:** this was subsequently fixed — B-118 changed `acquisition/downloader.py`
`candidate_urls()` to follow NCBI's OA packages to their relocated `/pub/pmc/deprecated/`
tree, so `acquire download` works again. The original "not fixed, deliberately" analysis
below predates that fix.

Original analysis: no acquisition code had been changed at that point. The evidence identifies where the fault is **not**
(our rewrite, our path derivation) but does not establish the current NCBI layout, and
guessing at a replacement URL would be speculation dressed as a fix. Settling it needs
someone to determine where OA packages now live — e.g. by listing `/pub/pmc/` on the FTP
server, or consulting current NCBI documentation — which is outside the authorised
request budget. Until then `acquire download` fails honestly and loudly (B-117), which is
the correct behaviour for a dependency that has moved.

---

## Bug 118 — NCBI relocated its OA packages and its API still advertises the old paths

**Status / Severity / Surface** — Fixed (2026-07-16, commit `27ed0f8`) · High ·
**mitigation expires August 2026**. `acquisition/downloader.py` (`candidate_urls`) ×
NCBI `oa.fcgi`.

### Symptom

`acquire download` fails for every paper. §6 — building the corpus — is impossible.

### Evidence

5 PMCIDs, stratified across the corpus by NCBI's own "Last Updated" field, each currently
advertised by the OA API as having a package. **0/5** returned an archive:

| PMCID | OA index date | HTTPS | content-type |
|---|---|---|---|
| PMC2680278 | 2010-12-27 | 404 | `text/html` |
| PMC8395919 | 2021-08-28 | 404 | `text/html` |
| PMC7155696 | 2023-11-12 | 404 | `text/html` |
| PMC11791731 | 2025-02-11 | 404 | `text/html` |
| PMC12687694 | 2025-12-10 | 404 | `text/html` |

Fifteen years of publication dates failing identically ruled out withdrawn papers.

NCBI's readme (`/pub/pmc/readme.txt`, updated 2026-04-10) states it outright:

> "All legacy files for the PMC Article Datasets were moved to a new temporary directory
> named **"deprecated"**. All legacy files on the FTP Service will be removed in
> **August 2026**."

Listing confirms it — `/pub/pmc/` now holds only `deprecated/`, `PMC-ids.csv.gz` and
`readme.txt`; `oa_package/` is gone from the top level and present under `deprecated/`:

```
FTP, binary mode:
  FOUND    7,556,375 bytes  /pub/pmc/deprecated/oa_package/e5/a1/PMC8395919.tar.gz
  MISSING  550 No such file  /pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz   ← API advertises this
HTTPS HEAD:
  404  text/html            …/pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz
  200  application/x-gzip   …/pub/pmc/deprecated/oa_package/e5/a1/PMC8395919.tar.gz  (7556375)
```

### Diagnosis

NCBI restructured its FTP distribution and `oa.fcgi` was left advertising the old tree.
Our code faithfully requested exactly what it was told to request.

**The local rewrite was suspected and cleared, on evidence rather than argument.**
`downloader.py` rewrites the advertised `ftp://` link to `https://`; the obvious theory was
that this assumption had lapsed. One FTP probe of the *original* advertised URL refuted
it — `550 No such file or directory` — proving both protocols fail identically and the
scheme rewrite is sound. The paths were also not mis-derived: they match NCBI's own index
byte-for-byte.

### Fix

`candidate_urls()` returns the advertised URL first, then the same URL with
`/pub/pmc/` → `/pub/pmc/deprecated/`. `download_package()` tries them in order and only
the final candidate reports its failure.

Advertised-first is deliberate: the advertised path is what NCBI *intends* to serve, so
the fallback disappears by itself if their API is repaired. Hard-coding `deprecated/`
would pin the project to a directory NCBI has announced it will delete.

### ⚠ This mitigation expires — August 2026

NCBI will remove the legacy files. After that both candidates 404 and acquisition fails
loudly (B-117), which is correct: the successor is the AWS OA service
(https://pmc.ncbi.nlm.nih.gov/tools/cloud/, https://registry.opendata.aws/ncbi-pmc),
reachable over HTTPS/S3 without a login. Migrating means a new endpoint and a new index —
`oa.fcgi` is not part of it — so it is a real piece of work, tracked in THESIS.md.

The expiry is stated in the module docstring, in `candidate_urls()`, at the call site, in
the run output (`↪ via NCBI's relocated legacy tree — temporary`), in `HOW_TO_RUN.md` §6,
and in the TODO. A fix with a four-week life that looked permanent would be its own trap.

### Verification

Live NCBI, the exact documented commands, isolated from the established corpus:

```
nlp-histo acquire download → exit 0 · 7.2 MB · "↪ via NCBI's relocated legacy tree"
nlp-histo acquire unpack   → exit 0 · valid tar (17 members) · 1 PDF + 1 XML extracted
nlp-histo acquire organize → exit 0 · 1 PDF + 1 XML organized
```

`files/organized_pdfs` still holds its original 1132 PDFs. Unit tests cover candidate
order, the scheme rewrite, no double-rewrite of an already-relocated URL, no fallback for
non-PMC URLs, fallback on 404, **no** fallback when the advertised URL works, and loud
failure when both are gone. Suite: 1637 passed.

---

## Bug 119 — AWS and FTP disagreed about document identity

**Status / Severity / Surface** — Fixed (2026-07-16, commit `163cf91`) · High · Identity.
`acquisition/downloader.py` (AWS naming) × `pdf_text_extraction/runner.py:1009,1028`,
`batch.py:201`.

### Terminology (used consistently from here on)

* **PMC accession** — the bare NLM value: `PMC8395919`.
* **document ID** — this project's composite identifier: `PMC8395919_dermatopathology-08-00036`.
  This is what `documents.pmcid` stores.

### Symptom

The same paper, acquired two ways, becomes two documents:

```
ftp → organized_pdfs/PMC8395919_dermatopathology-08-00036.pdf → id PMC8395919_dermatopathology-08-00036
aws → organized_pdfs/PMC8395919.1.pdf                          → id PMC8395919.1
```

Not an error — a duplicate row. Nothing would have reported it.

### Diagnosis

AWS names objects `PMC8395919.1.pdf`, where `.1` is the article version; the FTP tarball
carried the publisher's filename. Ingest derives identity from the PDF's stem
(`runner.py:1028`, `batch.py:201`, and the skip-check at `:1009`), so the naming
difference *is* an identity difference. Making AWS the default (`489e42e`) turned
"resume an FTP corpus with AWS" into a normal operation, at which point a documentation
note saying not to mix sources stops being a mitigation.

### Fix

**1. Authoritative naming.** The JATS XML records the publisher's filename:

```xml
<self-uri content-type="pmc-pdf" xlink:href="dermatopathology-08-00036.pdf"/>
```

The AWS route fetches the XML **first**, resolves that, and writes `unpack`'s exact
layout — `corpus/<PMCID>/dermatopathology-08-00036.pdf` + `<PMCID>.nxml`. `organize` is
unchanged and yields the same document ID from either source.

**2. Fail rather than invent.** No `pmc-pdf` self-uri, several disagreeing ones, or
unparsable XML → the article fails (`UnresolvablePdfName`). A fabricated name would
produce an ID that silently disagrees with the FTP-derived one, which is the bug.

**3. The self-uri is untrusted.** It is publisher-supplied text that becomes a path we
write to. Rejected outright, not repaired: URLs (`https://…`, `//…`), absolute paths
(`/etc/…`, `C:/…`), `..` traversal, empty values, non-PDF values. A nested reference
(`supplementary/paper.pdf`) reduces to its filename only after it is known relative and
traversal-free, so the last component cannot escape the paper directory.

**4. Shared safety net.** `nlp_histo.document_id.canonical_document_id()` at all three
derivation sites, replacing a bare `Path.stem`. It edits *only* `.<digits>` directly
after `PMC<digits>`, and only when the token ends there or `_` follows:

| input | output |
|---|---|
| `PMC8395919.1` | `PMC8395919` |
| `PMC8395919.1_dermatopathology-08-00036` | `PMC8395919_dermatopathology-08-00036` |
| `PMC8395919_dermatopathology-08-00036` | unchanged |
| `PMC8395919_paper.v2.final` | unchanged (publisher dots survive) |
| `PMC8395919.1.2` | unchanged (not an NLM shape — refuse to guess) |
| `not-a-pmcid.1` | unchanged |

**5. Adjacent correctness, fixed while here.** Version selection is numeric — a lexical
`max` picks `.9` over `.10` and serves a superseded article. S3 listing follows
pagination to exhaustion — a page caps at 1000 keys, and for a figure-heavy article the
silently dropped object could be the PDF.

**6. Provenance.** The AWS key, publisher filename, version and resulting document ID go
to `corpus/<PMCID>/_source.json` — unrecoverable from the renamed files, and ignored by
`organize`.

### Why the composite ID was preserved

The bare accession is arguably the *right* identifier — the composite is an accident of
tarball naming. But it is load-bearing history: **977 corpus rows**, the frozen replay
artifacts (`out/summaries/summaries/PMC10100421_HIS-82-393.json`) and the silver labels
(`case_id: "PMC11649514_HIS-86-204|3562"`) are all keyed on it. Normalising would orphan
every one and invalidate numbers verified byte-identical this week. Recorded as optional
**post-thesis migration debt** in THESIS.md — it would require coordinated database,
artifact, label, provenance and evaluation migration — **not** a bug and not a
reproduction step.

### Verification

Live, both routes, the documented commands:

```
aws → corpus/PMC8395919/{dermatopathology-08-00036.pdf, PMC8395919.nxml, _source.json}
      organize → PMC8395919_dermatopathology-08-00036.pdf
ftp → organize → PMC8395919_dermatopathology-08-00036.pdf
MATCH
```

Tests: `tests/test_document_identity.py` (35) — the parser's whole table, self-uri
resolution, ten hostile href values, numeric v10>v9, pagination.
`tests/test_cross_source_identity.py` (5) — drives both routes end-to-end offline with
identical PDF bytes and asserts identical pre-organize layout, identical organized
filename, identical document ID, resume-without-duplication, refusal without a self-uri,
and that `_source.json` does not disturb `organize`. Suite: **1685 passed**, ruff clean.

**Cleanup needed: none.** Zero versioned document IDs exist in either database
(`nlp_histo` 0/977, `new_local_db` 0/1) — the AWS work never reached ingest. The
established corpus is untouched at 977 documents / 35 896 text elements.

---

## Bug 120 — dry-run hardcoded the required API keys

**Status: Fixed (2026-07-17) / Severity: Medium / Surface:** `workflows/knowledge.py:575`

### Symptom

`nlp-histo knowledge … --dry-run` is documented — in `HOW_TO_RUN.md` §9, in `README.md`,
and in the fix note for [B-105](#bug-105--neither-documented-knowledge-command-could-run) —
as the free way to resolve a paid invocation's config before spending money. For **every**
profile it ended with the same line:

```
Env vars required: GOOGLE_API_KEY, ANTHROPIC_API_KEY
```

### Evidence

The line was a string literal, printed directly beneath a *correct*, profile-derived voter
table — so the output contradicted itself. Resolved needs vs. what it claimed:

| profile | providers actually used | truly required | printed |
|---|---|---|---|
| `cheap` | gemini, openai | `GOOGLE_API_KEY, OPENAI_API_KEY` | `GOOGLE_API_KEY, ANTHROPIC_API_KEY` |
| `real` | claude, gemini, openai | all three | `GOOGLE_API_KEY, ANTHROPIC_API_KEY` |
| `real_5` | claude, gemini, openai | all three | `GOOGLE_API_KEY, ANTHROPIC_API_KEY` |
| `haiku_only` | claude | `ANTHROPIC_API_KEY` | `GOOGLE_API_KEY, ANTHROPIC_API_KEY` |

Wrong for 4 of 4. The `cheap` case is the sharpest: all four of its voters are OpenAI or
Gemini, yet the message omits `OPENAI_API_KEY` entirely and demands an Anthropic key the
profile never touches.

### Diagnosis

`_dry_run` derived the model table from the resolved `profile` object but not the env-var
line, which predated the profile system and was never revisited when `real_5` /
`haiku_only` were added or when OpenAI voters entered `cheap`. Nothing tested the line,
because the assertion it makes is about the *operator's environment*, not the program's.

Same class as B-105: a claim on the paid-command surface asserted rather than executed.
B-105 fixed the *commands*; this line survived because it is printed by the very tool
B-105's verification used.

### Fix

Derive the set from the resolved profile:

```python
providers = {v.provider for v in (*profile.l1_voters, *profile.l2_voters, l3)}
required = sorted(env_by_provider.get(p, f"<unknown provider {p!r}>") for p in providers)
```

An unknown provider renders as `<unknown provider 'x'>` rather than being silently
dropped — a missing key must never be invisible, which is the defect being fixed.

### Verification

All four profiles run under `--dry-run` (exit 0, no paid host contacted) and each printed
line compared against an expectation computed independently from the provider sets: 4/4
match. `README.md`'s "three direct-API keys" claim — true only for `real`/`real_5` — was
corrected in the same change to point at `--dry-run` as the per-profile authority.

---

## Bug 121 — the documented hierarchical query could not run

**Status: Fixed (2026-07-17) / Severity: Medium / Surface:** `database/models.py:10,79`

### Symptom

`.claude/CLAUDE.md` lists this under **Critical Patterns**, as the way to exploit the
hierarchy that `TextElement` exists to store:

```python
# Hierarchical query — all paragraphs under "Methods" anywhere in path
session.query(TextElement).filter(TextElement.path_list.contains(['Methods']))
```

Run against the corpus, it raises:

```
NotImplementedError: ARRAY.contains() not implemented for the base ARRAY type;
please use the dialect-specific ARRAY type
```

### Evidence

`models.py:79` declared the column as:

```python
path_list = Column(ARRAY(Text), nullable=False)  # PostgreSQL array
```

with `ARRAY` imported at line 10 from SQLAlchemy's **generic** namespace
(`from sqlalchemy import …, ARRAY, …`). The generic `ARRAY` is dialect-agnostic and its
`.contains()` raises unconditionally — SQLAlchemy cannot know the containment syntax
without a dialect.

Two details show this was a slip, not a decision: the column's own comment already reads
`# PostgreSQL array`, and line 12 already imported `JSON, JSONB` **from the postgresql
dialect**. The intent was PostgreSQL throughout; one name came from the wrong module.

### Diagnosis

It survived because **nothing in the pipeline queries by path.** `path_list` is written by
ingest and read back whole (`path_string` at `models.py:115` joins it for display). No
production code calls an array operator, so no test covered one. The array operators exist
purely for a human exploring the corpus — the supervisor's use case, and the one path with
no automated coverage.

The DDL is identical either way, which is why the schema never hinted at the problem.
Compiled against the postgresql dialect, both types emit:

```
CREATE TABLE … ( p TEXT[] )
```

The difference is only in operator compilation: the dialect type renders
`path_list @> ARRAY['Methods']::TEXT[]`; the generic type refuses.

### Fix

Import `ARRAY` from the dialect alongside the `JSON`/`JSONB` already taken from there:

```python
from sqlalchemy.dialects.postgresql import ARRAY, JSON, JSONB
```

**No migration.** The emitted DDL is byte-identical, so existing databases and the hosted
corpus dump (`nlp-histo-corpus.sql.gz`) are unaffected — this changes what SQLAlchemy will
*compile*, not what PostgreSQL stores. All other `ARRAY(Text)` columns (`Entity.semantic_types`,
the `sum_*` provenance arrays) gain the same operators for free.

### Verification

The documented query, run live against the 977-paper corpus, before and after:

| | result |
|---|---|
| before | `NotImplementedError` |
| after | **86** text elements under a `Methods` heading |

with `path_list.contains()` compiling to `@>`. Sanity-checked that the generic type does
support `.any('Methods')` (`= ANY (…)`), confirming nothing that previously worked depended
on the generic behaviour. Full suite re-run after the change.

Found while verifying REPRODUCE.md's claim that the reader ends up with "a working corpus
database you can query": the file demonstrated only `count(*)`, so the first genuine query
tried was the one CLAUDE.md documents — which failed.

---

## Bug 122 — requirements.txt was a freeze of an unrelated interpreter

**Status: Fixed (2026-07-17) / Severity: Medium / Surface:** `requirements.txt`

### Symptom

404 pinned entries. `pyproject.toml` declares **zero** dependencies, so this file is the
only dependency source in the project — everyone who installs gets all 404.

### Evidence

The dependency closure over every top-level import in `src/`, `eval/`, `scripts/` and
`tests/` (32 distinct imports → 45 distributions) covers **218** of the 404 entries. The
remaining 186 are reachable from nothing:

| category | examples |
|---|---|
| other projects' packages | `bcrypt`, `cleo`, `configobj`, `conllu`, `fastapi`, `gurobipy`, `appnope`, `easyocr` |
| dev tooling nothing runs | `black`, `flake8`, `mypy`, `ipython`, `ipykernel`, `debugpy` |
| unused pytest plugins | `pytest-benchmark`, `-codspeed`, `-randomly`, `-recording`, `-socket`, `-xdist`, `-cov` |
| **11 unused spaCy models** | `de_core_news_{sm,md}`, `es`, `fr`, `it`, `ja`, `ko`, `pl`, `pt`, `ru`, `sv` |

The models are the headline: the project is English-only histopathology and loads exactly
two, `en_core_sci_lg` (36 references) and `en_core_sci_sm` (27 — real production sites at
`knowledge_extraction/runner.py:888` for sentence splitting and
`pdf_text_extraction/runner.py:493` for two-pass). `en_core_web_sm` occurs once, in a
**comment**; `en_core_sci_md` once, as a mocked string in a test.

The cause is recorded in CLAUDE.md's own gotchas: `python` here resolves to the system
framework interpreter, not a venv. A `pip freeze` there captures every project on the
machine.

### Diagnosis

Static analysis alone is **not sufficient** to prune this file — two entries are invisible
to it:

* **`psycopg2-binary`** is never imported. `db_connection.py:66` builds a bare
  `postgresql://` URL and SQLAlchemy resolves the DBAPI from the scheme. Dropping it yields
  a clean install and a runtime failure. (`asyncpg` *was* correctly dropped — no async
  driver is named.)
* **`build`** is never imported; §12 documents `python -m build`.

### Fix

Filter the original file by the closure, **preserving every version pin**. The pins are the
thesis's provenance: they are the versions the published numbers were produced with.

`requirements.in` now records the direct set with a reason per entry. `requirements.txt`
remains the pinned lock and is what you install.

**A first attempt was rejected.** Resolving an *unpinned* direct list produced a working
environment — 200 packages, `1697 passed` — but pip had upgraded `transformers` 4.57.3 →
**5.8.1**, `pandas` 2.1.4 → **3.0.3**, `torch` 2.9.1 → 2.13.0, and dropped `sentencepiece`
entirely. That is the trap this bug exists to document: **a green test suite is not evidence
of reproducibility.** The tests do not exercise the NLI models whose entailment decisions
the grounding filter — and therefore the published numbers — depend on. A transformers major
bump could move them silently.

### Verification

Three successive clean venvs, each installing the candidate file and reporting any package
pip had to resolve itself (an unpinned transitive = uncontrolled version):

| attempt | result |
|---|---|
| unpinned `requirements.in` | works, but `transformers` → 5.8.1, `pandas` → 3.0.3 — **rejected** |
| closure-filtered, pinned | 11 unpinned extras — the closure missed docling's chunking stack (`semchunk`, `mpire`, `tree-sitter*`, `dill`, `multiprocess`), which pip then resolved at drifted versions |
| **+ those 11 at original pins** | **0 unpinned extras** — the lock is complete |

Final state: **229 pins** (from 404), install 1m13s, `nlp-histo --help` exit 0, the
`postgresql://` URL resolves to the `psycopg2` driver, `torch 2.9.1` / `transformers
4.57.3` intact, **1697 passed**, `ruff` clean.

Found while pruning at the user's request, ahead of a supervisor's clean-clone run — every
one of those 404 lines is something they would otherwise download.

---

## Bug 123 — E14's heldout15 inputs were missing from the reproduction bundle

**Status: Fixed (2026-07-17) / Severity: Medium / Surface:** `scripts/make_reproduction_bundle.py`, `docs/REPRODUCE.md` Step 8

### Symptom

A clean clone + the downloaded bundle, running REPRODUCE.md Step 8 exactly:

```
python -m eval.silver.experiments.E14_heldout.heldout_eval --theta-frontier
voter cache not found: /Users/emir/histo-test/eval/data/map_primer_heldout15/voter_cache.json
```

E04, the other Step 8 experiment, ran fine.

### Evidence

The bundle's member list (`REPLAY_MEMBERS`) was built to mirror `replay.REQUIRED_ARTIFACTS`
— what `nlp-histo replay chapter9` needs. That is the *related15* primer
(`eval/data/map_primer/voter_cache.json`). But Step 8 also runs `E14_heldout`, and E14 reads
a **different** primer and silver set:

```
eval/data/map_primer_heldout15/voter_cache.json     16 MB
eval/data/silver_findings_heldout15.jsonl           1.1 MB
```

Neither was in the bundle, and neither was tracked in git, so a clean clone had no way to
obtain them. The bundle's scope (the replay's artifacts) and the doc's promise (the free
track, including E14) had silently diverged.

### Diagnosis

Not a cost regression. E14 constructs its scorer with `strict_cache_only=True`
(`heldout_eval.py`), so any embedding not already in the frozen gemini cache raises rather
than issuing a paid call — the same guarantee as [B-112](#bug-112--the-replay-embedding-cache-preflight).
Running E14 against the shipped gemini cache confirmed the caches themselves were complete:

```
SQLite embedding cache: 87942 entries at eval/data/embedding_cache_gemini.sqlite
Agreement embed pre-warm: 11148 unique claims, 0 cache misses
strict_f1_optimal = 0.7128   loose_f1_optimal = 0.8837
GENERALIZATION GAP (heldout − related15) = -0.0032
```

0 misses, exit 0, and the exact values REPRODUCE.md documents. So the embeddings were all
present; only the two *input* files (primer + silver) were missing.

### Fix

Commit the two files to git rather than re-cut and re-upload the 1.2 GB bundle for 1 MB of
data (they gzip to ~1.0 MB combined). This is the same home as the already-tracked
`source_cases_related15.jsonl`, and it means a clone carries them — so E14 runs against the
**existing** uploaded bundle, no re-upload, no checksum change.

The gitignore re-includes **only** the clean `voter_cache.json`. Its directory also holds
`voter_cache.contaminated.json` — a data-leakage variant that must never ship — and an
unused `primer.json`; both stay ignored. The rule negates the directory, re-ignores its
contents, then narrows to the one file (git will not descend into a wholesale-ignored
directory, the same lesson as `out/*`).

`make_reproduction_bundle.py` keeps its original member list, with a comment recording that
the heldout15 inputs live in git by design, so a future bundle does not redundantly carry
them.

**Trade-off, accepted deliberately:** the heldout15 primer now lives in git while its
related15 twin remains in the bundle. Inconsistent, but it avoids a 1.2 GB re-upload for a
1 MB fix, and the reproduction works either way.

### Verification

With the two files tracked, a clean clone carries them; E14 was confirmed free and correct
(0.7128 / -0.0032) against the already-shipped gemini cache. The contaminated variant and
`primer.json` were verified to remain ignored.

---

## Bug 124 — offline experiments demanded a key they never used

**Status: Fixed (2026-07-17) / Severity: Medium / Surface:** `E03/E10/E11/E12` × `eval/silver/analysis/map_context.py`

### Symptom

REPRODUCE.md Track A promises "no API key". But four of the experiments it lists (or should
list) died in a keyless clone:

```
GOOGLE_API_KEY not set
```

E12 was the first caught, because it was already in Step 8.

### Evidence

`_load_map_context(...)` has two paths (`map_context.py:106`): with `strict_cache_only=True`
it builds a keyless `NoLiveEmbedding`; otherwise it builds a live `GeminiEmbedder(api_key)`,
which requires a key at construction. E03, E10, E11 and E12 all called it **without** the
flag:

```python
ctx = _load_map_context("gemini", embed_cache_path=None)   # → live embedder → needs a key
```

Yet each is a documented offline replay, and each pre-warms the agreement embeddings with
**0 cache misses** — so the live embedder it constructs is never actually called. Pure
constructor theatre, the same defect as [B-109](#bug-109--the-replay-needed-a-credential-it-never-used)
and [B-112](#bug-112--the-replay-embedding-cache-preflight).

E14 was already immune — it passes `strict_cache_only=True`. E04 and E13 construct no
embedder at all.

### Diagnosis

It escaped the Step-8 verification because those runs happened on the author's machine,
where `.env` supplies the keys. `load_dotenv(override=False)` means an explicitly-set env var
wins, so re-running with every provider key blanked reproduces the keyless clone exactly —
and that is where the failure surfaces. A command verified only with credentials present is
not verified for a track whose whole premise is that there are none.

### Fix

Add `strict_cache_only=True` at the four call sites, mirroring E14. The mechanism already
existed; this is not new code, just the correct flag. Because strict-cache-only changes
behaviour only on a cache *miss* (it raises instead of billing) and these replays have none,
the reported numbers are unaffected.

### Verification

All four re-run with every provider key blanked (`OPENAI_API_KEY= GOOGLE_API_KEY= …`):

| exp | keyless | misses | value |
|---|---|---|---|
| E03 | exit 0 | 0 | retention 0.838 @0.50, best 0.7160 |
| E10 | exit 0 | 0 | cascade 0.7160 vs single-Sonnet 0.7129 |
| E11 | exit 0 | 0 | cascade 0.7160, paper-level bootstrap B=10000 |
| E12 | exit 0 | 0 | LOO attribution CSV |

E04, E13, E14 also confirmed keyless (0.7128 for E14). Suite 1697 green, ruff clean. The fix
unblocked E10 and E11 — free but previously keyless-broken — which are now added to
REPRODUCE.md Step 8.

---

## Bug 125 — E02c and E09 were free but their inputs were unshipped

**Status: Fixed (2026-07-17) / Severity: Low / Surface:** `docs/REPRODUCE.md` Step 8 / Step 11

### Symptom

REPRODUCE.md listed E02c and E09 among the experiments that "cannot reproduce from the free
track". True at the time — but only because a total of ~0.7 MB of inputs was missing, not
because anything about them was paid or PDF-bound.

### Evidence

* **E09** (`cost_quality_frontier`) constructs no embedder (`0` occurrences of
  `_load_map_context`) — a pure pandas re-analysis. It reads the frozen calibration-sweep
  CSVs from E06c / E07 / E08b. Those sweeps are the heavy calibration runs a reader does not
  re-execute (their result is already `configs/run.yaml`), and their CSVs were gitignored.
* **E02c** (`rule_provenance_heldout`) is a keyless read-only DB walk that reads the held-out
  per-paper summary JSONs at `out/summaries/heldout15/summaries/*.json` — also gitignored,
  and not in the bundle (which carries only `out/summaries/summaries`).

Both verified free before shipping: run with every provider key blanked,

```
E09  → quality 0.7160@23.66, knee 0.7067@21.80, economy 0.5433@3.38   (exit 0)
E02c → heldout15: 15 papers, all 100.00% carry-rate                   (exit 0)
```

matching the EXPERIMENTS.md registry.

### Fix

Commit the inputs to git — the B-123 choice, avoiding a 1.2 GB bundle re-upload for < 1 MB:

* 15 held-out summary JSONs (7.2 MB on disk, ~0.6 MB git-packed).
* 3 frozen sweep CSVs — `E06c_voter_subset_refine`, `E07_map_theta`, `E08b_map_theta_shipped`
  (~52 KB).

The gitignore negations are surgical: **only** those files. The bundle's own
`out/summaries/summaries` stays ignored (it ships in the archive); the held-out
`batch_handles` and `corpus_relations` intermediates stay ignored; every other
`eval/reports/` artifact stays ignored.

REPRODUCE.md Step 8 now lists **eight** free bundle-based experiments (was seven), and Step
11 lists **three** DB provenance experiments (was two).

*(Later revision: a subsequent edit moved E03 (grounding sweep) behind the DB restore — its
grounding step needs the corpus paragraphs — so Step 8 now lists **seven** cache-only
experiments and Step 11's DB group covers **four** (E03 + E02/E02b/E02c). The B-125 fix itself
is unchanged.)*

### Verification

With the inputs tracked, both run keyless from what a clean clone carries. E09 reads the
committed frozen CSVs (no re-run of the calibration sweeps); E02c reads the committed
summaries against the restored corpus DB. E01 is deliberately *not* covered here — it needs
the 27 rubric PDFs, a licensing decision recorded in the Decisions log.


---

## Bug 126 — flatten_to_csv crashed on its own success message

**Status: Fixed (2026-07-17) / Severity: Low / Surface:** `eval/silver/experiments/E01_doc_extraction/flatten_to_csv.py:120`

### Symptom

The documented E01 reproduction wrote its CSV and then died:

```
E01 flatten … ValueError: 'eval/reports/E01_doc_extraction/…_PR.csv' is not in the
subpath of '/Users/…/nlp-histo'
```

Exit 1, full traceback — but the CSV was already written and correct.

### Diagnosis

```python
print(f"E01 flatten: {n} rows -> {csv_path.relative_to(_REPO_ROOT)}")
```

`_REPO_ROOT` is absolute; `csv_path` is whatever `--csv-out` was, or the default derived from
`--json`. A reader naturally passes a **relative** path (`eval/reports/…`), and
`Path.relative_to` requires both operands absolute and nested — so it raises. The failure is
*after* `flatten()` has written the file: a crash on success.

### Fix

Resolve before comparing, and fall back to the raw path when the target is outside the repo:

```python
try:
    shown = csv_path.resolve().relative_to(_REPO_ROOT)
except ValueError:
    shown = csv_path
print(f"E01 flatten: {n} rows -> {shown}")
```

### Verification

Both invocations — a relative `--csv-out`, and the default (no `--csv-out`) — now exit 0, and
the regenerated CSV is byte-identical to the committed frozen
`figtable_extraction_sweep_rerun_27pdf_20260604_PR.csv`. ruff clean.
