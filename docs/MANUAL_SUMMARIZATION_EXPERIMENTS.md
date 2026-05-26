### MAP stage

For the MAP stage of the summarization pipeline, we have a lot of thresholds that we should fine-tune our model. There is a randomly selected dataset consisting of 15 papers, and the fine-tuning is done on that dataset. 

In the SummarizationConfig, we already have:

map.theta: Accept threshold
map.reject_theta: Reject threshold
map.chunk_size: Chunk window (sentences per LLM call)
map.chunk_overlap: Sliding-window overlap

routing.enable_router: False = legacy L1→L2→L3; True = router L1→L3
routing.router_single_voter_policy: Router N=1 handling. 'escalate' (default) → L3; 'keep' → accept lone vetted voter.
routing.legacy_single_voter_policy: Legacy AgreementChecker N=1 handling. 'keep' (default; preserves prior implicit
                                    behaviour — synthetic confidence=1.0) vs 'escalate' (treat N=1 as low-evidence
                                    and route up the cascade). Done 2026-05-26.

> **Config layout v2 (2026-05-26).** The three routing-policy fields above moved from `MapConfig` into a dedicated
> `RoutingConfig` because they are agreement-decision parameters, not MAP extraction parameters (answers the inline
> question: "is it possible to have a router config with the single_voter_policy separately so it looks cleaner?"
> — yes, done as a pure refactor with no behaviour change). Defaults preserved verbatim; strict YAML loader rejects
> the v1 paths (`summarization.map.enable_router` etc.) with a clear error. See `STRUCTURE.md` pipeline changelog
> 2026-05-26 + `THESIS.md` Decisions log.
--------------------------------------------------------------------------------

#### AgreementConfig

These parameters control how agreement between MAP voters is scored.

agreement.tau:
Pairwise similarity values below this threshold are treated as zero before computing coverage. Higher tau makes agreement stricter.

agreement.count_alpha:
Penalty strength for voter outputs with very different finding counts. This helps prevent one voter emitting many findings while another emits one broad finding from looking artificially aligned.

agreement.reuse_weight:
Penalty for concentrated reuse, where one finding is reused to match many findings from another voter. Higher values punish many-to-one matching more strongly.

agreement.contradiction_weight:
Penalty applied when comparable findings appear to disagree in polarity/numeric direction.

> **Wiring (verified 2026-05-26):** all four fields flow `config.py` → `configs/run.yaml` (lines 128–132) → `SummarizationRunner.__init__` / `BatchSummarizationRunner.__init__` → `EmbeddingSimilarityStrategy.from_config(cfg.agreement)`. No hardcoded overrides on the path. Tests pin defaults + YAML round-trip + override + `from_config` propagation (`tests/test_config_loader.py::test_agreement_*`). The calibration sweep (`map_theta_sweep.py`, `run_summarization_sweeps.py`) builds `AgreementConfig()` with dataclass defaults explicitly so the sweep grids are independent of the production YAML — this is by design (sweeps iterate over weight variants).


--------------------------------------------------------------------------------

### TODO: these are not in the config, promote to a config field

Scorer kind: EmbeddingSimilarityStrategy vs HybridStructuredSimilarity
Voter profile: real / cheap. we can continue with real. 

Embedder: OpenAI text-embedding-3-small / Gemini. The embeddings are currently being generated using Gemini. 

max_tokens: Currently 16384. Shorter might cause problems about not generating the full correct output, longer might mean the models keep generating if there is a bug on the LLM side, so no early exit. 

--------------------------------------------------------------------------------

### The following are not in the SummarizationConfig either, promote these to a config field

grounding_floor: Default = 0.50. Post-embedding multiply if a finding's grounding-pass fraction is low. Only fires when an AgreementContext is supplied (router path)

#### TODO: check 

Hybrid blend weights: 

w_category / w_embedding / w_entity / w_evidence
Default = 0.25 / 0.40 / 0.25 / 0.10

#### TODO: check and think about how to find the optimal weights 

Polarity Conflict: Escalate iff {positive, negative} comparable pair. 
These trigger forced escalation. 

#### TODO: maybe change this to a flag, so we can enable and disable after fine-tuning and testing on the 15 related paper dataset

--------------------------------------------------------------------------------

