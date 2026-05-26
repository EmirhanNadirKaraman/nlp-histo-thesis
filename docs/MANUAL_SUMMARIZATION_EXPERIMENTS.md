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

#### Agreement

agreement.tau: Pairwise similarity below tau zeroed before coverage
##### TODO: check what this is doing

agreement.count_alpha: Count-mismatch exponent, 0.25 → 4:1 finding-count ratio ≈ 0.71×
##### TODO: check 

agreement.reuse_weight: Reuse-concentration weight, up to 15% reduction when default = 0.15 is used. 

##### TODO: check 

agreement.contradiction_weight: Polarity/numeric contradiction weight


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

