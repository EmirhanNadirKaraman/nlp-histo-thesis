# Experiment results — explanations + interpretation

Narrative companion to the registry in `docs/EXPERIMENTS.md` (which maps
experiment → CSV → command → status). This file explains **what each run
experiment tested and what its numbers mean**. The raw CSVs live in the
`E##_*/` subfolders alongside this file.

> **Update protocol:** whenever a new experiment runs, add a section here with
> **What / Result / Interpretation / Artifact**, and flip its row to *done* in
> `docs/EXPERIMENTS.md`. Keep both in sync in the same change.

> **B-074 correction (2026-06-17), RESOLVED:** an enum-stringification bug in
> `_finding_to_pipeline` under-counted strict-F1 in proportion to early-accept rate
> (heavy at low θ, ≈0 at θ0.9). Fixed (commit `c2a9b58`); the frozen pin **held** (E07
> argmax still θ0.9). **All θ-sensitive experiments re-run on the fix: E07, E06c, E09,
> E10, E11, E12** (numbers below are post-fix). Fixed-θ verdicts (E06b/E08/E03)
> unaffected (both arms equally corrupted). Net effect: low-θ strict-F1 rose sharply →
> flatter θ-curve, economy config much stronger, and single-Sonnet (E10/E11) no longer
> beaten by the cascade.

**Dataset:** all results below are **in-sample calibration** on `related15`
(15-paper ILP cluster, 454 cases / 2 294 frozen-config MAP findings; Opus silver
labels). Selection metric is `strict_f1_optimal` (strict-F1 under the optimal /
Hungarian one-to-one matcher); greedy is a reported diagnostic only. Out-of-sample
generalization is the separate held-out test (E14, **done** — the 5-voter config generalizes,
heldout15 strict-F1 0.7128 vs calibration 0.7160, gap −0.0032).

## Frozen config of record (result of E05–E08b)

`configs/run.yaml` `summarization`, pinned from the calibration below:

| knob | value | from |
|---|---|---|
| cascade | legacy AgreementChecker (router off), **5 voters** (Claude-Haiku dropped from L2) | E06b/E06c |
| scorer / alignment | **hybrid / greedy** | E06 |
| hybrid blend (cat/emb/ent/evi) | **0.15 / 0.30 / 0.50 / 0.05** (entity-heavy) | E06 |
| embedder / tau | gemini / 0.15 | E06 |
| θ / reject-θ | **0.9 / 0.2** | E07 / E08b |
| gates (single-voter / polarity) | **escalate** / force-escalate true | E08 |
| grounding threshold | 0.5 | E03 |
| **headline** | **strict-F1 0.7160** @ 95.6 % escalation | E08 / E08b |

> **5-voter / escalate update (2026-06-22):** the config of record changed from the earlier
> all-6-voter / keep config. (1) E06c on the corrected (gemini-embedder) operating points showed
> dropping **Claude-Haiku** from L2 is strict-F1-equivalent at ~18 % lower per-chunk cost; the
> 5-voter cascade is **selected on calibration** for cost + **evaluator-independence** (the L1/L2
> voting becomes provider-independent of the Opus silver labeller — see E06b/E06c). (2) E08 found
> the single-voter gate is no longer inert at 5 voters: **escalate** wins (0.7160 vs keep 0.7147).
> The voter drop is a **selection** over the 6-voter primer (re-prime avoided), so `_make_voters`
> and E12 stay on 6 voters; map_theta (E07) is the keep θ-selection at the default gate, and **E08b**
> (`map_theta_shipped`) regenerates the θ-curve at the escalate gate — the curve E09 reads.

---

# RQ3 — MAP agreement-cascade calibration (E05–E08)

Screen→refine design: pick the structure block first (cheap, broad), refine only
the finalists' applicable weights × full threshold grid, re-confirm thresholds,
then gates. The dependency order avoids tuning weights under a structural knob
(alignment) that hasn't been chosen yet.

## E05 — structure screen
**What:** all `{embedder × scorer × alignment}` structures at default weights over
a coarse θ grid (120 cells, 12 structures); Pareto-aware finalist selection over
(strict-F1 ↑, escalate ↓).
**Result:** **hybrid beats embedding** on every embedder (best hybrid 0.700–0.712
vs best embedding 0.686–0.702); `gemini/hybrid` and `openai/hybrid` lead. All top
structures peak at θ=0.9 on the coarse grid (escalation-heavy). Finalists carried
to E06 included the eventual quality structure (`gemini/hybrid/greedy`) and an
economy candidate (`openai/hybrid/soft_max`).
**Interpretation:** the structured-similarity scorer (category+embedding+entity+
evidence) is worth its complexity over plain embedding similarity; embedder and
one-to-one vs soft alignment matter far less at high θ (everything escalates).
**Artifact:** `E05_structure_screen/sweep_20260617T004005.csv`

## E06 — family refine (weights × θ)
**What:** each finalist's applicable weights (tau, hybrid blend; soft-max-only
penalties where relevant) × the full θ/reject grid (1 197 cells).
**Result:** **winner `gemini/hybrid/greedy`, entity-heavy blend (0.15/0.30/0.50/
0.05), θ=0.9/reject=0.1 → strict-F1 0.7133, loose-F1 0.8861, escalation 0.962.**
**Interpretation:** the entity overlap signal dominates the blend (w_entity 0.50);
the max-F1 operating point is ≈ "escalate almost everything to the L3 premium
model." This is the **quality** config. A cheaper **economy** operating point
(`openai/hybrid/soft_max`, θ=0.4) sits on the same Pareto frontier as the
cost-efficient alternative.
> *B-074 note:* this 1197-cell sweep was **not re-run** (expensive; the selection is
> robust — E07 reconfirms θ0.9 as argmax, and structure/weight selection is at high θ
> where corruption was ~0). The 6-voter winner is ≈0.7133; the voter set is then pruned to
> **5 voters** (E06b/E06c, drop Haiku) and the gate set to escalate (E08), so the final shipped
> headline is **0.7160** (E08b) — the **structure/weight selection here is unchanged**; the voter-set
> and gate refinements happen in the downstream stages.
**Artifact:** `E06_family_refine/sweep_20260617T121203.csv` *(pre-fix sweep; selection valid)*

## E06b — voter-subset screen (leave-one-voter-out, fixed θ)
**What:** drop each of the 6 voters in turn (L1: gemini-flash-lite, gpt-4o-mini,
gpt-4.1-nano; L2: gemini-flash, gpt-4.1-mini, claude-haiku-4.5) × quality/economy/
knee configs, at each config's fixed θ (21 cells). Scored with the voter-subset-aware
per-chunk cost (`eval.silver.experiments.E06b_voter_subset_cost`, which charges the
*reduced* tier price for a dropped voter — unlike `escalation_breakdown`'s full-ensemble cost).
**Result (2026-06-22, corrected gemini configs):** drops are NOT uniformly bad — at fixed θ,
**dropping Claude-Haiku from L2 at the quality config is strict-F1-neutral (+0.0016) at −18 % cost**,
and two L1 OpenAI drops (gpt-4.1-nano, gpt-4o-mini) Pareto-dominate `all` at the economy config.
(This supersedes the earlier "every single-voter drop loses strict-F1", which was measured on the
pre-Jun-22 openai/θ0.4 economy point.)
**Interpretation:** voter pruning *is* a viable cost lever for the 5-voter cascade — the cheap
OpenAI L1 voters dilute at the economy end, and Haiku is redundant at the quality end. Fixed-θ
domination is necessary but not sufficient; E06c re-confirms over the full θ grid.
**Artifact:** `E06b_voter_subset_screen/sweep_20260622T131114.csv` (+ `_cost.csv` sidecar)

## E06c — voter-subset refine (frontier check, full θ)
**What:** the fixed-θ-dominating drops (quality/drop-haiku, economy/drop-nano, economy/drop-4o-mini)
+ their `all` baselines, swept over the full θ/reject grid (105 cells), to test domination on the
cost-quality frontier per config.
**Result (2026-06-22):** the drops dominate across θ. **quality/drop-haiku owns the max-F1 operating
point — 0.7147 vs all 0.7131 (≥ F1, −18 % cost)**, owning the high-F1 end of the frontier; the two
economy L1 drops dominate `all` almost everywhere (21/24 frontier points each).
**Interpretation:** **adopt the 5-voter cascade (drop Claude-Haiku, `BEST_VOTER_SUBSET="drop_l2_2"`)** —
selected on the calibration set. The strict-F1 gain (+0.0016) is noise; the case is **cost** (−18 %)
+ **evaluator-independence** (the 5-voter L1/L2 ensemble is entirely non-Anthropic, so its agreement
with the Opus silver is not a same-provider artefact; L3 stays Sonnet). Decision made entirely on
`related15`; E14 *reports* the held-out generalization but does not gate the pin. The drop is a
selection over the 6-voter primer (no re-prime) — `_make_voters`/E12 stay on 6 voters.
**Artifact:** `E06c_voter_subset_refine/sweep_20260622T143146.csv` (+ `_cost.csv` sidecar)

## E07 — θ / reject-θ selection (default gate)
**What:** sweep θ × reject for the pinned 5-voter quality structure at the DEFAULT gate (keep) —
the θ-SELECTION, before the gate is chosen (21 cells, forward protocol).
**Result (2026-06-22, 5-voter):** **max strict-F1 at θ=0.9 / reject=0.2 (0.7147)** at the keep gate.
θ-curve monotone, flat at the top (θ0.8→0.9 only +0.009) → diminishing returns. reject=0.2 weakly
dominates 0.1 (the (0.0,0.1] band is empty for the 5-voter ensemble — reject 0.0≡0.1 — and the
lowest-consensus chunk shifted into (0.1,0.2]; the 6-voter config had the empty band at (0.1,0.2]).
**Interpretation:** pins `BEST_THETA=0.9`, `BEST_REJECT_THETA=0.2`. E07 stays the θ-selection record
at the default gate; the shipped θ-curve at the chosen gate is E08b. Escalation has clear diminishing
returns (feeds E09).
**Artifact:** `E07_map_theta/sweep_20260622T153848.csv` *(keep — the θ-selection record)*

## E08 — gate ablations (single-voter policy × polarity hard-fail)
**What:** `legacy_single_voter_policy ∈ {keep, escalate}` × `force_escalate_on_polarity_conflict ∈
{True, False}` at the pinned θ (4 cells, 5-voter).
**Result (2026-06-22, 5-voter):** the gates are **NOT inert** (unlike the 6-voter cascade).
`single_voter_policy=escalate` wins: **0.7160** (escalate/true) vs 0.7147 (keep/true); polarity barely
moves it (~+0.0001).
**Interpretation:** with only 2 L2 voters the degenerate N=1-at-L2 case is more common; `escalate`
(send the lone voter to Sonnet rather than trust it) is the argmax **and** the principled choice (a
lone voter carries no agreement signal). **Adopt escalate** → `BEST_SINGLE_VOTER_POLICY="escalate"`,
run.yaml `routing.legacy_single_voter_policy=escalate`. The gain over keep is noise-level — adopted on
principle + calibration argmax, not on the number.
**Artifact:** `E08_map_gates/sweep_20260622T154808.csv`

## E08b — shipped config's θ-curve (map_theta_shipped)
**What:** θ × reject re-swept at the SHIPPED gate (escalate, E08's choice) — the 5-voter shipped
config's θ-curve (21 cells). CHARACTERIZATION for E09's frontier, NOT a θ re-selection.
**Result (2026-06-22):** θ0.9/reject0.2 → **strict-F1 0.7160**, escalation 0.956 — confirms θ0.9
remains the argmax at the escalate gate (the gate is gate-invariant for θ) and fixes the headline.
**Interpretation:** the curve E09 reads (not E07). The θ pin is unchanged; this only regenerates the
shipped config's θ-curve. Forward protocol: E07 = keep θ-selection record, E08b = escalate shipped curve.
**Artifact:** `E08b_map_theta_shipped/sweep_20260622T164928.csv`

## E09 — cost-quality frontier
**What:** the SHIPPED 5-voter config's θ-curve (E08b, escalate) as a strict-F1-vs-cost frontier.
Cost = price-weighted per-tier invocations with the **5-voter tier prices** (L1=1.75, L2=**4.80**
[Haiku excluded], L3=18.00 USD/1M tok); escalate_rate is a robustness cross-check. Offline. (The old
two-structure quality+economy join is dropped: the voter sets differ now and E06c has no drop_l2_2
economy — the shipped config's own θ-curve IS the frontier, voter-consistent end to end.)
**Result (2026-06-22, 5-voter):** operating points on the single curve (anchor reproduces E08 at 0.7160):
- **quality** θ0.9/r0.2 → **0.7160 @ cost/chunk 23.66** (esc 0.956)
- **knee** (λ=0.20) θ0.8/r0.2 → **0.7067 @ 21.80** (esc 0.867)
- **economy** θ0.3/r0.0 → **0.5433 @ 3.38** (esc 0.068)
- **balanced** (ε-constraint, floor 0.60; added 2026-08-09) θ0.7/r0.2 → **0.6575 @ 16.40** (esc 0.622)
Per-tier-cost frontier == escalate-rate frontier (robust to cost model).
**Interpretation:** monotone with **clear diminishing returns** — knee→quality buys < 1 strict-F1
point for ~10 % more cost. Escalation is the dominant cost lever. The cascade frontier is itself
dominated by single models (E10/E11): single-Sonnet ties the quality point at lower cost.
**Artifact:** `E09_cost_quality/frontier_20260622T165135.csv` (frozen, 3 points) ·
`E09_cost_quality/frontier_20260809T133949.csv` (re-run with `balanced`; anchor θ0.9/r0.2 = 0.7160 OK)

## E09b — λ sensitivity of the knee operating point
**What:** the `knee` is `argmax(strict_f1_optimal − λ·cost)` with `COST_LAMBDA = 0.20`, a value
pinned in commit `e13dc1c` (2026-06-17) with no recorded rationale and never swept. This asks for
which λ the selected cell stays the same. Score is linear in λ per cell, so the argmax over λ is
the upper envelope of a line arrangement — band edges are solved exactly as chord slopes, not
sampled. Pure re-analysis of two frozen CSVs; no API calls, no cost.
**Result (2026-08-09):** the shipped λ reproduces both documented knees exactly, so the method is
validated — but **both bands are narrow at the top**:
- **E09 frontier** (`cost_norm`): knee = θ0.8/r0.2, sf1 **0.7067 @ 21.80** — stable for
  λ ∈ **[0.1186, 0.2102)**. Headroom −0.0814 / **+0.0102**. Only three cells ever win: quality
  (λ < 0.1186), knee, then economy (λ ≥ 0.2102) — there is no smooth family of knees.
- **E06 structures** (`cost_frac`, 1 020 cells): knee = `blend_embedding_heavy` θ0.8/r0.0,
  sf1 **0.6744 @ cost 0.6324** — stable for λ ∈ **[0.1909, 0.2081)**. Headroom −0.0091 / **+0.0081**.
**Interpretation:** λ = 0.20 sits ~5 % below the E09 breakpoint and ~4 % below the E06 one. At
λ ≥ 0.2102 the E09 knee collapses onto the economy cell (0.5433 @ 3.38) and the designation stops
being a knee at all. The band edges are the frontier's own chord slopes — 0.2102 is the marginal
quality-per-cost from economy→knee, 0.1186 from knee→quality — so the narrowness is a property of
the frontier's near-linearity, not of the code. **This does not touch the shipped configuration:**
production is the *quality* point (θ0.9/r0.2, argmax strict-F1), which is λ-independent. λ selects
a reporting point and, in E06, one of three candidates entering the E06b voter screen — the arm
that was ultimately adopted is the quality arm.
**Follow-up — the binding limit is the method, not the value of λ (2026-08-09).** Weighted-sum
scalarization can only ever return a vertex of the **upper convex hull** of the (cost, F1) point
set; any Pareto point in a concave stretch is below some chord and is never the argmax for *any*
λ ∈ [0, ∞). On the E09 frontier: 11 distinct cells, **7 Pareto-optimal, but only 3 on the hull** —
exactly the three bands the λ sweep found. The other four Pareto points are permanently
unreachable:

| cell | strict-F1 | $/chunk | vs quality |
|---|---|---|---|
| θ0.4/r0.0 | 0.5444 | 3.53 | — |
| θ0.5/r0.0 | 0.5531 | 4.82 | 79.6 % cheaper, −0.1629 F1 |
| θ0.6/r0.2 | 0.5960 | 9.48 | 59.9 % cheaper, −0.1200 F1 |
| θ0.7/r0.2 | 0.6575 | 16.40 | 30.7 % cheaper, −0.0585 F1 |

And the knee is barely a middle ground: θ0.8 sits at **92 % of the quality cost** (21.80 vs 23.66)
for 0.0093 less F1, so the whole \$3.38–\$21.80 span is unrepresented by the three reported points.
The **ε-constraint** method — already used for the `economy` point (Haimes 1971) — reaches the
concave region directly: floor 0.60 → θ0.7 at 30.7 % cheaper for 0.0585 F1; floor 0.58 → θ0.6 at
59.9 % cheaper for 0.1200 F1. This is the textbook weighted-sum limitation stated in Boyd, the
source the thesis already cites for the scalarization. **Still no effect on the shipped config**
(the λ-independent quality point); it affects only how the frontier is summarised.
**Follow-up 2 — ε-constraint floor sweep (2026-08-09).** Same treatment for the *other* selection
constant: `point(ε) = argmin cost s.t. strict_f1 ≥ ε`, solved exactly (the pick changes precisely
when ε crosses a Pareto cell's F1, so no floor can fall between grid points).

| ε floor range | cell | strict-F1 | $/chunk | esc | vs quality | λ can reach it? |
|---|---|---|---|---|---|---|
| (0.0000, 0.5433] | θ0.3/r0.0 | 0.5433 | 3.38 | 0.068 | 85.7 % cheaper, −0.1727 | yes ← `economy` |
| (0.5433, 0.5444] | θ0.4/r0.0 | 0.5444 | 3.53 | 0.074 | 85.1 % cheaper, −0.1716 | **no** |
| (0.5444, 0.5531] | θ0.5/r0.0 | 0.5531 | 4.82 | 0.124 | 79.6 % cheaper, −0.1629 | **no** |
| (0.5531, 0.5960] | θ0.6/r0.2 | 0.5960 | 9.48 | 0.317 | 59.9 % cheaper, −0.1200 | **no** |
| (0.5960, 0.6575] | θ0.7/r0.2 | 0.6575 | 16.40 | 0.622 | 30.7 % cheaper, −0.0585 | **no** ← `balanced` |
| (0.6575, 0.7067] | θ0.8/r0.2 | 0.7067 | 21.80 | 0.867 | 7.9 % cheaper, −0.0093 | yes (= the knee) |
| (0.7067, 0.7160] | θ0.9/r0.2 | 0.7160 | 23.66 | 0.956 | — (is quality) | yes |

**ε-constraint reaches all 7 Pareto cells; the weighted sum reaches only the 3 hull vertices.** Any
floor above 0.7160 is infeasible. Shipped floors: `economy` ε=0.50 → θ0.3, unchanged for
ε ∈ (0, 0.5433] (headroom −0.5000/+0.0433); `balanced` ε=0.60 → θ0.7, unchanged for
ε ∈ (0.5960, 0.6575] (headroom **−0.0040**/+0.0575). Note the asymmetry: 0.60 sits just 0.004 above
the θ0.6 boundary, so a marginally lower floor buys θ0.6 at 9.48 — that is a different *choice*
(cheaper, −0.12 F1), not a degeneracy like the λ collapse.
**Artifact:** `E09_cost_quality/lambda_sensitivity_20260809.csv` ·
`E09_cost_quality/epsilon_floor_sweep_20260809.csv` · `scripts/eval/lambda_sensitivity.py` ·
`scripts/eval/epsilon_floor_sweep.py`

## E10 — single-model baselines vs cascade (5-voter)
**What:** each of the 7 voter models run ALONE (no voting, no escalation) vs the frozen **5-voter**
cascade (escalate gate), scored vs silver; cost/chunk = 1 call × model price. Offline.
**Result (2026-06-22, 5-voter):** cascade **0.7160 @ cost 23.66** vs single-Sonnet **0.7129 @ 18.0** —
within 0.003 at **24 % lower cost**. Then haiku-4.5 0.6558 @ 4.80, gpt-4.1-mini 0.5957 @ 2.0,
gemini-flash 0.5761 @ 2.8, gemini-flash-lite 0.5721 @ 0.5, gpt-4o-mini 0.5394 @ 0.75,
gpt-4.1-nano 0.4570 @ 0.50.
**Interpretation:** the cascade is the top-scoring system, but **single-Sonnet is within noise at
lower cost**. At ~96 % escalation the cascade output ≈ Sonnet's. The 5-voter cascade is **cheaper than
the old 6-voter** (23.66 vs 28.5 — the Haiku drop paying off) but still costs more than Sonnet. E11
adds the significance test.
**Artifact:** `E10_baselines/baselines_20260622T165331.csv`

## E11 — cascade vs single-model paired bootstrap (5-voter)
**What:** paper-level (clustered) paired bootstrap, B=10 000, of cascade − each single model on
strict-F1 (papers the independent unit). Offline.
**Result (2026-06-22, 5-voter):** the cascade **significantly beats every cheaper model** (all 95 %
CIs > 0): gpt-4.1-nano +0.259, gpt-4o-mini +0.177, gemini-flash-lite +0.144, gemini-flash +0.140,
gpt-4.1-mini +0.120, haiku-4.5 +0.060. But vs **single-Sonnet: Δ +0.0031, 95 % CI [−0.0002, +0.0068]
— NOT significant** (the CI only just includes 0).
**Interpretation:** the ensemble+escalation genuinely extracts more than any cheaper model, but
**does not significantly beat Sonnet-alone** — a real equivalence, not underpower (cheap-model Δs are
large and cleanly resolved at the same n=15). With the cascade costing 24 % more, it is **not
cost-justified vs single-Sonnet**. The 5-voter point estimate sits marginally *above* Sonnet (the
6-voter sat marginally below); neither is significant, so the cascade and single-Sonnet are
statistically indistinguishable in quality. Defensible value lies in non-F1 dimensions — chiefly the
**provider independence** of the 5-voter L1/L2 voting from the Opus silver labeller, plus vendor
diversity and the audit trail. Caveat: silver is Opus and Sonnet ≈ Opus, so a human metric could move this.
**Artifact:** `E11_bootstrap/bootstrap_20260622T165546.csv`

## E12 — leave-one-voter-out attribution (per-model contribution)
**What:** drop each L1/L2 voter, re-replay the cascade, at the *diagnostic* regime
(`embedding`/`soft_max`, gemini embedder, θ 0.5/0.6/0.7 — where the ensemble, not
escalation, decides). L3 excluded. Two readings: **decision** (Δ strict-F1, N=3→N=2)
and **attribution** (most-disagreeable = drop leaving highest remaining agreement).
Read the *deltas*, not the absolute F1 (0.53–0.60 is the diagnostic config, not the
0.71 production config).
**Result:** dropping a voter *changes the early-accept rate* (unlike the fixed-config
E06b/c), so the two arms escalate differently. **Dropping either agreeable L1 OpenAI voter
(`gpt-4o-mini`, `gpt-4.1-nano`) helps strict-F1 at every θ — and more as θ rises:**
+0.042 / +0.043 at θ0.5, up to +0.063 / +0.061 at θ0.7, while their escalation climbs
~0.08→0.13. `gemini-flash-lite` is the disagreement-driver (most disagreeable at every θ);
dropping it helps only slightly and also rises with θ (+0.003 → +0.005). The three L2 voters
(`gemini-flash`, `gpt-4.1-mini`, `claude-haiku`) are noise-level to drop (−0.002 to ~0 across
θ). The small negative on `claude-haiku` here (−0.0018 @θ0.7) is within noise and is **not**
the basis for the shipped haiku drop — that decision is made in E06c/E08 at the production
operating point (θ0.9 / escalate), where drop-haiku is +0.0016 vs keep-all and the case is
cost + evaluator-independence, not a strict-F1 gain. There is **no** low-θ regime where
dropping the cheap L1 voters hurts.
**Interpretation:** the agreeable cheap voters' consensus increasingly lands on chunks that
*should* escalate, so removing them forces escalation to Sonnet → "drop helps" = *escalate
more → approach Sonnet* (the same phenomenon as E10/E11, per-voter); the effect grows with θ
because a higher bar makes more of that cheap consensus wrong. The cheap voters' agreement is
the weak link relative to Sonnet. **No production change:** diagnostic mid-θ config; production
θ0.9 escalates ~96 % so this is moot — consistent with E06b/c's "keep all." E12 is the
mechanistic close-up of *why* production escalates aggressively.
**Artifact:** `E12_voter_loo/20260717T180222.csv` *(6-voter LOO by design — `_make_voters` stays 6 — so the verdict is unchanged: no production change)*

---

# RQ2 — Grounding filter

## E03 — grounding-threshold sweep + sensitivity

**What:** NLI entailment of each MAP finding's `claim` from its cited `verbatim_support`
(PubMedBERT-MNLI-MedNLI); sweep the drop threshold over the frozen-config (5-voter / escalate)
findings and score survivors vs silver (offline, no API — replays the cached voter_cache filtered
to `drop_l2_2`; n_pipeline **2294** at threshold None, vs 6-voter 2280).
**Result (2026-06-22, 5-voter):** scores are **bimodal** — most findings well-grounded, ~14 % < 0.3
(claim not entailed by its **real cited paragraph**). Best silver strict-F1 is at the filter **off**
(**0.7160**, retention 100 %) — equal to the cascade headline (the sweep replays the same 5-voter
config). Raising the threshold trades **recall** for groundedness; at the pinned **0.5**:
strict-F1 **0.6520**, **retention 83.8 % (1923/2294)**, precision near-flat (0.872→0.875),
recall 0.900→0.757. Strict-F1 declines smoothly (~6.4 points off→0.5).
**Interpretation:** the filter is a **source-faithfulness guard**, not an F1 optimizer — at 0.5 it
drops the ~16 % tail whose claim is not entailed by its **real cited paragraph**, costing ~6 strict-F1
points with essentially no silver-precision gain (0.872→0.875). The near-flat precision means the
dropped tail is **not** silver-false: grounding and silver-agreement are orthogonal — a finding can
match the Opus silver yet still fail to be entailed by its own source. 0.5 is kept for **groundedness**
(a knowledge base should not hold claims unsupported by their cited evidence), a faithfulness criterion
silver-F1 does not reward, and the choice is robust to the exact value.
**Artifact:** `E03_grounding/sweep_20260622T174122.csv` (5-voter, DB-paragraph grounding)

## E04 — knowledge-extraction cardinalities (the funnel)
**What:** descriptive aggregation of per-stage item counts over the 15 related15
papers (MAP findings → grounded → normalized → groupable → canonical rules →
relations → final rules), summed from each paper's `rejection_summary` + list
lengths. Offline, no API.
**Result (frozen 5-voter cascade — `drop_l2_2`, escalate gate, θ0.9/reject0.2, grounding 0.5; runs on the bridged→rebuilt corpus in `out/summaries/summaries`):**
**2294 MAP findings → 1923 grounded** (83.8 % retention) → 1860 normalized (−3.3 % dedup)
→ 1820 groupable (−2.2 % non-groupable) → **1747 canonical rules** (rules/finding **0.939**)
→ **13 intra-paper + 12 cross-paper = 25 relations** (SUPPORT 22 / **CONTRADICT 3**) → **1747 final rules** (116.5/paper).
**Interpretation:** (1) **minimal consolidation** — ≈1 finding → 1 canonical rule;
NORMALIZE/GROUP/CANONICALIZE are near-identity in volume (findings already atomic/distinct).
(2) **relations are sparse** — RELATE fires ~1 intra-paper relation/paper (+12 cross-paper
corpus-wide from 1747 rules), and only **3 are CONTRADICT** (~1 genuine) — exactly why the
relation classifier (E13) is evaluated on the synthetic 300, not the corpus.
> Re-run on the **5-voter frozen-config corpus** (bridge→rebuild output in
> `out/summaries/summaries`) — supersedes the 6-voter aggregation (MAP 2280 → 1729 final;
> `cardinalities_20260618T220332.csv`). MAP total (2294) matches the 5-voter E03/E08b
> reference, and grounding retention 0.838 matches E03 @0.5 **exactly** (1923/2294) — the
> E03 sweep grounds real DB paragraphs, not LLM paraphrases (B-079).
**Artifact:** `E04_cardinalities/cardinalities_20260622T210109.csv`

---

# RQ1 — Document extraction

## E01 — document-extraction sweep + rubric F1
**What:** PDF → figure/table extraction sweep (32 config variants: Docling / TATR /
hybrid detection × confidence × footnote-expand × drop/premask/reconstruct), scored
against the 27-PDF human rubric (`eval/label_rubric.yaml` — crop / caption / footnote
dims; **strict** = all dims correct). 2026-06-04 rerun on the post-exclusion 27-PDF set.
**Result:** winner **variant 18** (`hybrid` detection @0.99 + footnote-expand 1.2):
tables crop-F1 91.9 %, **strict-F1 83.8 %** (31/6/6); figures crop-F1 89.9 %, strict-F1
84.0 %. vs the Docling baseline (variant 01): tables strict-F1 **40.0 % → 83.8 %**,
driven by footnote precision **43.8 % → 91.4 %** (crop was already ~91 %).
**Interpretation:** crop/bbox detection was already strong (~90 %); the bottleneck was
**footnote inclusion** in table crops — extending the table region downward
(footnote-expand 1.2) over a hybrid (Docling ∪ TATR) detector more than doubled table
strict-F1. Figures have no footnote dimension and were stable at 84 %/90 % throughout.
This is the pinned production doc-extraction config. *(Correction 2026-06-19, B-078:
figure strict-F1 is **invariant at 84.0 %** across all 32 variants — the sweep re-tunes
only table detection/cropping, so the reconstruct/merge variants 28–32 leave figures
unchanged and only move table scores, mostly regressively (28→79.5 %, 31→44.2 %,
32→80.5 %). The 84 % figure ceiling is bounded by 14 decorative-icon crop-FPs that no
swept knob removes; icon suppression — not any swept variant — is the route to a higher
figure score. The earlier "variants 28–32 → 100 % figure strict-F1" note was a stale
number absent from the cited artifact.)*
**Artifact:** `E01_doc_extraction/figtable_extraction_sweep_rerun_27pdf_20260604_PR.md`

## E01b — off-the-shelf Docling baseline (vs the pipeline "Docling baseline")
**What:** the sweep's `01_docling` is **not** vanilla Docling — it is the full
document-extraction pipeline (two-pass, masking, artifact filter, `nearest_caption`,
size/icon filter, cropping) with only the table-handling knobs disabled. This baseline
runs *stock* Docling (`DocumentConverter`, native `doc.pictures`/`doc.tables`, native
`caption_text`) with none of that scaffolding, crops the native bboxes, and scores on the
same 27-PDF rubric. Script `scripts/eval/baseline_offtheshelf_docling.py`; labels seeded
from `01_docling` via `scripts/eval/transfer_offtheshelf_labels.py`; variant
`00_docling_offtheshelf`.
**Result (T1, automatic):** stock Docling emits **139 figures** (42 captioned, 30 %) vs
the pipeline's 76 (58, 76 %), and **34 tables** (30 captioned) vs 33 (33). Table detect
rate 34/37 = 91.9 %.
**Result (T2, FINAL — all 34 tables + 139 figures hand-labelled, 0 unlabelled, 0
unrecognised):**

| | strict-F1 | crop F1 | caption P | foot P | icon FPs |
|---|---|---|---|---|---|
| off-the-shelf tables | **36.6 %** (13/21/24) | 90.1 % | 90.6 % | 43.8 % | — |
| `01_docling` tables | 40.0 % (14/19/23) | 91.4 % | 93.8 % | 43.8 % | — |
| v18 pinned tables | 83.8 % | 91.9 % | 94.3 % | 91.4 % | — |
| off-the-shelf figures | **44.7 %** (40/99/0) | 61.7 % | 60.6 % | n/a | **73** |
| `01_docling` / all-pipeline figures | 84.0 % (55/21/0) | 89.9 % | 90.3 % | n/a | 14 |

**Interpretation:**
- **Tables — footnote-capped, ≈ stock.** Stock 36.6 % vs `01_docling` 40.0 %, both far
  below v18 83.8 %. Both fail the footnote dimension (foot P ~44 %), capping strict-F1
  ~37–40 %; the 40→83.8 jump is **footnote-driven** (foot P 43.8 %→91.4 %), not a
  weak-baseline artifact. `nearest_caption` lifts the table baseline ~3 pts (40.0 vs
  36.6) — the scaffolding inflates the table baseline *slightly* (caption), not *not at
  all*. The 36.6 % lands at the low end of the earlier [36.6–42.3 %] bound (all 6
  disagreement tables were strict-FP).
- **Figures — pipeline adds ~39 pts.** Stock 44.7 % vs pipeline 84.0 %. Stock over-emits
  (139 vs 76; **73 icon-FPs** vs 14 → crop P 44.6 % vs 81.6 %) and under-captions (caption
  P 60.6 % vs 90.3 %). So "Docling figure behaviour was not modified" is misleading: the
  pipeline keeps Docling's figure *detection* defaults but its icon/size filter,
  sub-figure merge, and `nearest_caption` are worth ~39 strict-F1 points.

**⚠️ Annotation-contamination incident (resolved).** Annotating `00_docling_offtheshelf`
with `eval/annotate.py --variant` triggered `share_map.json` label propagation onto the
*peer* sweep variants (same crop key, but stock has no caption where the pipeline does),
silently corrupting 51 tracked annotation files (e.g. `01_docling` figures 84.0→80.3).
Reverted via `git checkout HEAD -- eval/annotations/` (the off-the-shelf dir is untracked,
so survived); re-scored clean. **Re-annotating this variant will re-corrupt peers** —
revert tracked `eval/annotations/` afterward, or annotate with propagation disabled.
**Artifact:** `out/sweeps/00_docling_offtheshelf/` + `score_pdf_variants.py` rows
(`00_docling_offtheshelf`).

## E02 — provenance preservation (corpus DB)
**What:** corpus-level metric of the pipeline's provenance claim — can every extracted
artifact be traced to source? Read-only DB queries over 977 documents / 35 896 text
elements / 4 479 figures / 1 960 tables, plus a qualitative trace.
**Result:**
- **TEXT provenance — essentially complete:** **100 %** of paragraphs carry a
  well-formed `unique_path` (PMCID/section/position), 98 % a `path_list` section
  array, 100 % non-empty text. Every finding/rule traces to its source section.
- **MEDIA provenance — via caption + crop + cross-ref:** tables 100 % caption / 100 %
  crop; figures 86 % caption / 100 % crop. In-text citations resolved for **58.9 % of
  tables** and **53.6 % of figures** (linked to ≥1 citing paragraph); 12.8 % of
  paragraphs cite a figure/table (3 878 fig-refs, 1 701 tab-refs).
- **Coordinate provenance — FIXED (B-075 + migration 0014):** `page_number`/`bbox` were
  initially **0 %** (the ingester dropped what the cropper computed; figures had no
  columns at all). After the forward ingester fix, migration 0014 (figure columns), and a
  media-JSON backfill, **both tables and figures are 100 % page_number + 100 % valid bbox**
  (locatable in the source PDF, Docling coords). Residual no-source gaps: `table_content`,
  `section_context`.
**Interpretation:** the DB now preserves a **complete coordinate-grounded provenance
chain — section address + caption + crop + page/bbox + cross-reference** — so a rule
traces to its paragraph's section path *and* to the exact PDF region (page + bbox) of the
cited table/figure (qualitative example: `PMC10047408…/3. Results/0` → "Table 1.
Morphometry…" → page 3, bbox [165,137,559,76], crop PNG). The coordinate drop was a
one-line ingester omission (tables) + a missing-column schema gap (figures), both now
fixed (B-075, migration 0014) and backfilled corpus-wide (1960 tables + 4479 figures).
Remaining: table content + media `section_context` (no source in the crop path) — minor.
**Artifact:** `E02_provenance/provenance_20260618T120844.csv`

## E02b — final-rule provenance carry-rate (RQ2)
**What:** the *rule*-side companion to E02 (which measures document/media provenance).
Quantifies the claim of `04_knowledge_extraction/09_provenance_model.tex`: every emitted
`FinalRule` traces back to ≥1 real source paragraph. Read-only walk of the four-link chain
over persisted `sum_*` rows — FinalRule → CanonicalRule (`canonical_id`) → NormalFinding
(`member_normal_ids`) → span (`text_element_id`) → TextElement → Document — resolved
per `pipeline_run_id` (ids are run-scoped). No API.
**Result:**
- **Carry-rate 100 % — 1729 / 1729** final rules across the **15 `related15` production
  runs** trace to ≥1 source paragraph; **per-run all 100 %** (40–163 rules/paper). The
  18 pre-summarization runs (0 final rules) are excluded.
- **Failure funnel empty** — no rule breaks at any link (A canonical / B normal /
  C span-te / D text-element / E document).
- **Completeness:** `member_normal_ids` **100 %** resolved (1801/1801); spans carrying a
  non-null `text_element_id` **100 %** (1878/1878).
- **0 cross-paper** — every rule's source paragraph belongs to the rule's own PMCID.
**Interpretation:** the knowledge-extraction provenance chain is **empirically complete**,
not just guaranteed by construction — the typed back-pointers hold for every rule, and no
rule leaked via a malformed-evidence span (the one nullable link, `SumNormalFindingSpan.
text_element_id`, is non-null everywhere in these runs). This is the RQ2 rule-level
counterpart to E02's document/media 100 %, and lets `09_provenance_model.tex` cite an
empirical carry-rate instead of a by-construction argument. Caveat: measured over the
DB-persisted `sum_*` runs (which match the `related15` set); provenance is a property of
the pipeline code, so this is representative of every run.
**Artifact:** `E02b_rule_provenance/rule_provenance_20260702T173213.csv`

## E02c — final-rule provenance carry-rate, held-out (RQ5 / RQ2)
**What:** the generalization counterpart to E02b — does the rule-provenance chain hold on
papers **outside** the calibration cluster? The held-out summarization runs were never
DB-persisted (offline replay), so this reads the on-disk pipeline summaries
(`out/summaries/heldout15/summaries/`, frozen config hash `cfb56a0289b557be` = E14's
5-voter shipped config) and resolves each final rule to a paragraph via the canonical
rule's serialized `representative_text_element_id` (the best-grounded member's paragraph
pointer — exactly "the identifier of the paragraph it came from" in `09_provenance_model`),
checking that id against the DB `text_elements` (held-out papers are in the 977-corpus DB).
`related15` is re-measured by the same JSON path as a cross-check. No API, no re-run.
**Result:**
- **Held-out carry-rate 100 % — 1273 / 1273** final rules across the 14 held-out papers that
  emit rules (PMC8221904 emits 0 → n/a); **per-paper all 100 %**, failure funnel empty, 0
  cross-paper, all 1273 also carry `member_normal_ids`. The 1273 total matches E14's held-out
  funnel.
- **`related15` JSON cross-check 100 % — 1747 / 1747** (same shipped 5-voter config).
- **Config note:** the related15 JSON total (1747, 5-voter) exceeds E02b's DB total (1729)
  because the DB still holds the **superseded 6-voter** run (E04: 5-voter 1747 supersedes
  6-voter 1729). Carry-rate is therefore 100 % across **both** configs (6-voter DB via E02b's
  full-span chain; 5-voter JSON via E02c's representative-paragraph pointer) and **both**
  datasets.
**Interpretation:** rule-level provenance **generalizes** — every emitted rule on unseen
papers traces to a real source paragraph in its own paper, same as on the calibration set.
Combined with E02b, provenance completeness is now empirically confirmed on calibration
(RQ2) *and* held-out (RQ5), under the shipped config. E02c resolves via the canonical
representative paragraph rather than E02b's full member-span set (per-finding spans were not
persisted for held-out); both establish "final rule → ≥1 real source paragraph", and the
related15 cross-check agrees with E02b's DB-path 100 %.
**Artifact:** `E02c_rule_provenance_heldout/rule_provenance_heldout_20260702T175134.csv`

## E13 — NLI relation-classification ablation (synthetic 300, RQ4)
**What:** direct evaluation of the RELATE NLI classifier on a synthetic **300 claim-pair**
set (Opus 4.7 batch API, forced structured output, balanced **100/100/100**
SUPPORTING/CONTRADICTING/UNRELATED) — needed because the corpus has too few real
contradictions to measure it (frozen-cascade corpus run: **3 CONTRADICT corpus-wide**, ~1
genuine). Two NLI-input modes (`predicate_only`, `scope_aware`) at the **frozen RelateConfig
0.50/0.50**; verbatim modes excluded (synthetic claims have no real source sentence). Gold
mapped SUPPORTING/CONTRADICTING/UNRELATED → SUPPORT/CONTRADICT/UNRELATED. Threshold sweep is
**diagnostic only — no tuning** (thesis: out-of-sample).
**Result:**
- **predicate_only @ 0.50/0.50 (primary): accuracy 0.927, macro-F1 0.927.** Per-class F1:
  SUPPORT 0.958 (P **1.000**), CONTRADICT 0.922, UNRELATED 0.902. Error mass = the
  UNRELATED↔CONTRADICT boundary (8 UNREL→CON, 6 CON→UNREL); SUPPORT never false-fired.
- **scope_aware @ 0.50/0.50: accuracy 0.923, macro-F1 0.924 — marginally worse (Δ−0.003).**
- **By difficulty (predicate):** easy 0.960 (n=149) → medium 0.902 (n=123) → **hard 0.857
  (n=28)**. scope_aware is **better on hard** (0.893 vs 0.857) despite being worse overall.
- **Diagnostic sweep:** best macro-F1 = 0.9307 (pred @ ent0.4/con0.9) vs default 0.9273 →
  **+0.003 only**; 0.50/0.50 is within ~0.3 % of optimum.
**Interpretation:** the classifier is strong on the clean synthetic set (~93 %), and the
**sweep shows 0.50/0.50 is near-optimal → no calibration warranted** — keeps the
out-of-sample property and the corpus run at 0.50/0.50 stands (decision over the
"calibrate the thresholds?" question). Scope-aware NLI adds nothing in aggregate but helps
the hard/ambiguous cases (where the corpus lives). The set is **easy-skewed (149/123/28)**,
so 0.93 is an optimistic ceiling — hard accuracy (0.857, small n) is more corpus-
representative. Discussion caveat: synthetic is cleaner than the corpus (cf. corpus
contradiction precision ≈ 1/3 on the 3 real CONTRADICTs).
**Artifact:** `E13_nli_ablation/20260622T165812.csv` (re-run 2026-06-22; cascade-independent — predicate-only acc 0.927 / macro-F1 0.927, scope_aware 0.923, unchanged) + dataset
`eval/data/relation_claim_pairs_300.jsonl` (api_batch, claude-opus-4-7, content_hash
67cf283c280179f0; calibration/evaluation splits 150/150).

## E14 — held-out generalization test (heldout15, RQ5) — THE HEADLINE
**What:** does the **5-voter / escalate** config calibrated on related15 (E06–E08b, strict-F1 0.7160)
generalize to a **disjoint** 15-paper test set? Replays the frozen config (θ0.9/reject0.2, hybrid
entity-heavy, **Claude-Haiku dropped** via `drop_l2_2` over the 6-voter heldout primer — no re-prime,
escalate gate) on the **heldout15** primer and scores MAP findings vs `silver_findings_heldout15.jsonl`
(1657 silver). Zero voter API (replay). Operating point **selected on related15, reported on heldout15**.
**Result (2026-06-22, 5-voter):**
- **heldout15 @ frozen θ0.9/r0.2: strict_f1_optimal = 0.7128** (loose 0.884, P 0.880 / R 0.888;
  n_silver 1657 / n_pipeline 1672). **related15 = 0.7160** (loose 0.886, P 0.872 / R 0.900) **→ gap −0.0032**
  (within noise); **every metric within ~1 pt** (the full-metric comparison, not just strict-F1).
- **At θ0.9 escalate = 0.990** (virtually every chunk → L3/Sonnet).
- **Descriptive θ-frontier** (`heldout_eval.py --theta-frontier`, `heldout_20260625T082104.csv`):
  θ0.4→0.5453, 0.5→0.5707, 0.6→0.6115, 0.7→0.6589, 0.8→0.7033, 0.9→0.7128 — reproduces the related15
  monotone-then-flattening shape (fig `theta_strict_f1_heldout.{pdf,png}`, overlaid with the calibration
  curve) → **the cost-quality trade-off generalizes**. Descriptive only; operating point fixed from related15.
**Interpretation:** **the frozen 5-voter config generalizes — no overfit to the calibration cluster**
(held-out ≈ calibration, gap −0.0032, within noise). E14 reproduces the E10/E11 structural fact
out-of-sample: at ~99 % escalation the cascade is effectively **single-Sonnet** on held-out papers too.
Caveats: (1) −0.003 is "no meaningful difference," not "held-out is worse"; (2) generalization **against
Opus silver** — same circularity (a human metric would test it); (3) the operating point is selected on
related15, never on heldout15 (which stays the unbiased test).
**Artifact:** `E14_heldout/heldout_20260622T170151.csv` (5-voter; primer
`eval/data/map_primer_heldout15/` filtered to `drop_l2_2`); θ-frontier in
`E14_heldout/heldout_20260625T082104.csv`, plot `E14_heldout/theta_strict_f1_heldout.{pdf,png}`
(`plot_theta_curve_heldout`).

### E14 held-out downstream funnel (descriptive, isolated)
**What:** to complement the scored MAP number with a held-out *funnel* (no held-out
gold beyond silver), the frozen θ0.9 cascade MAP was bridged from the heldout15 primer
(zero API; per-paper findings validated == sweep θ0.9 replay) into an **isolated** store
`out/summaries/heldout15/` and the local downstream stages re-run there with `--no-db`
(`bridge_populate_corpus --primer … --out-dir …` + `rebuild_from_cached_map --profile real_5
--summaries-dir … --no-db`). DB-isolation: `--no-db` nulls `pipeline_run_db_id`, so no held-out
pmcid is written to the shared corpus tables, and the post-run audit + incremental corpus-relate
are now gated off in that mode (B-084); related15 `out/summaries/summaries` untouched.
**Result (heldout15, frozen 5-voter config — `drop_l2_2`, escalate, θ0.9/reject0.2, grounding 0.5):** **1672 MAP → 1379 grounded (82.5 %) → 1342 normalized
(−2.7 % dedup) → 1314 groupable (−2.1 % non-groupable) → 1273 canonical = 1273 final** (rules/finding **0.949**, 84.9/paper);
**4 relations** (4 intra-paper, 0 cross-paper, **0 CONTRADICT**). Funnel MAP=1672 **matches E14's
scored n_pipeline (1672)**: the held-out bridge was regenerated and re-validated == sweep replay on
all 15 papers (an earlier rebuild had recorded 1662 from a divergent bridge pass; superseded). One
paper (PMC8221904) yields 0 canonical rules, so corpus-relate pools 14.
**Interpretation:** the downstream funnel **shape generalizes at every stage** — held-out
grounding retention 82.5 % (related15 83.8 %), dedup −2.7 % (related15 −3.3 %), rules/finding
**0.949** (related15 0.939, near-identical), ≈1 finding→1 rule, relations equally sparse (related15 25,
heldout 4, 0 held-out contradictions). Structural generalization, not just strict-F1. Descriptive only
(silver, no held-out gold), reported in §9.6 beside the scored E14 MAP strict-F1.
**Artifact:** `E04_cardinalities/cardinalities_20260622T214622.csv` (`--summaries-dir
out/summaries/heldout15/summaries --source …heldout15.jsonl`); relations
`out/summaries/heldout15/corpus_relations.json` (2026-06-22, 1273 rules, 4 relations).

# Pending (no results yet)
- **E15** silver-vs-human validity (rubric27) — LLM; no human-label resources → write up as a limitation.
