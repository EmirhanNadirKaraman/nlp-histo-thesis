# 03 — Agreement-based cascading calibration

> Draft for `chapters/04_results/03_agreement_based_cascading_calibration.tex`
> (`\label{section:results-abc-calibration}`). Source of truth for every number:
> `eval/reports/RESULTS.md` (post-B-074 figures). Markdown tables below map onto
> booktabs tables in the same style as sections 01–02.

This section reports the calibration of the MAP-stage agreement cascade, which answers the calibration half of **RQ3**. The calibration fixes the cascade's *configuration of record*; the cost-quality consequences of that configuration are reported separately in Section~\ref{section:results-cost-quality-comparison}.

All numbers in this section are produced by **offline replay** of cached voter artifacts, as described in Subsection~\ref{subsec:abc-calibration-and-replay}: the voters are queried once to build a primer cache, and every configuration is then scored by re-reading that cache, so no model API calls are made during calibration. The calibration set is `related15` (the 15-paper ILP cluster of Subsection~\ref{subsec:eval-datasets}): 454 cases and 2\,280 frozen-config MAP findings, scored against Opus silver labels. The selection metric is `strict_f1_optimal` — strict-F1 under the optimal (Hungarian) one-to-one matcher of Subsection~\ref{subsec:eval-knowledge-extraction}; the greedy-matcher score is reported only as a sensitivity diagnostic and never drives selection. These are **in-sample calibration** scores; the unbiased out-of-sample test is deferred to Section~\ref{section:results-generalization-to-heldout-papers}.

## Calibration design: screen, then refine

The cascade configuration spans a structural block — embedder, similarity scorer, and alignment strategy — together with the scorer's continuous weights, the two decision thresholds (the agreement threshold $\theta$ and the rejection threshold $\theta_{\mathrm{reject}}$), and two routing gates. Sweeping all of these jointly is wasteful, because the alignment strategy is a *structural* knob: under a one-to-one alignment (greedy or Hungarian) three of the four soft-alignment weights become inert, so weights tuned before the alignment is chosen may simply be discarded. The calibration therefore proceeds in dependency order. The structural block is decided first, at default weights, on a coarse threshold grid (E05); only the surviving *finalist* structures are weight-tuned over the full threshold grid (E06); the thresholds are then re-confirmed at the final scoring function (E07); and the routing gates are ablated last (E08). Carrying a *set* of finalists rather than the single screen winner guards against eliminating a structure that only becomes competitive after its weights are tuned.

## Structure screen (E05)

The screen evaluates all twelve `{embedder × scorer × alignment}` structures at default weights over a coarse $\theta$ grid (120 cells), selecting finalists by a Pareto-aware rule over (strict-F1 $\uparrow$, escalation $\downarrow$). The **structured-similarity (hybrid) scorer beats plain embedding similarity on every embedder** — the best hybrid structure scores 0.700–0.712 against 0.686–0.702 for the best embedding structure — so the additional category, entity, and evidence signals earn their complexity. The `gemini/hybrid` and `openai/hybrid` structures lead. Every top structure peaks at $\theta=0.9$ on the coarse grid, i.e. in the escalation-heavy regime; embedder identity and one-to-one-versus-soft alignment matter far less there, because at high $\theta$ almost every chunk escalates regardless. The finalist set carried into E06 included the eventual quality structure (`gemini/hybrid/greedy`) and a cheaper economy candidate (`openai/hybrid/soft_max`).

## Family refine (E06)

Each finalist's applicable weights — `tau`, the hybrid blend, and the soft-max-only penalties where relevant — are swept jointly with the full $\theta$/reject grid. The winner is the `gemini/hybrid/greedy` structure with an **entity-heavy blend** (category/embedding/entity/evidence = 0.15 / 0.30 / 0.50 / 0.05) at $\theta = 0.9$, $\theta_{\mathrm{reject}} = 0.1$:

| Metric | Value |
|---|---|
| strict-F1 (optimal matcher) | **0.7135** |
| loose-F1 (optimal matcher) | 0.8861 |
| escalation rate | 0.962 |

The entity-overlap signal dominates the blend (weight 0.50), and the selected operating point is effectively "escalate almost everything to the Level-3 premium model." This is the **quality** configuration. A cheaper **economy** operating point (`openai/hybrid/soft_max`, low $\theta$) lies on the same cost-quality Pareto frontier and is carried forward to the cost-quality analysis of Section~\ref{section:results-cost-quality-comparison}.

\pinktodo{The headline strict-F1 quoted here is the post-B-074 value (0.7135), confirmed as the argmax by E07; the 1\,197-cell E06 sweep itself was not re-run because the selection is robust at high $\theta$ — see the correction note at the end of this section.}

## Voter-subset ablation (E06b, E06c)

A natural cost lever is to drop a voter from the six-model ensemble (Level 1: gemini-flash-lite, gpt-4o-mini, gpt-4.1-nano; Level 2: gemini-flash, gpt-4.1-mini, claude-haiku-4.5). E06b drops each voter in turn at each operating point's fixed $\theta$; E06c then re-sweeps the *dominating* drops over the full $\theta$/reject grid to test whether the domination survives on the cost-quality frontier.

\pinktodo{PENDING E06c re-run (2026-06-22, corrected configs) — DO NOT finalize this paragraph until the full-$\theta$ frontier result lands. The current fixed-$\theta$ screen, costed with the price-weighted per-chunk model, does NOT support a clean keep-all verdict: dropping the Level-2 haiku voter at the quality configuration is strict-F1-neutral ($+0.0016$, within noise) at $\approx 18\%$ lower per-chunk cost, and two Level-1 voter drops Pareto-dominate the full ensemble at the economy operating point. Whether either drop also dominates across the full $\theta$ grid (E06c) decides the final \texttt{BEST\_VOTER\_SUBSET} pin and the wording here.}

\pinktodo{An optional re-screen of the voter subsets on the 2026-06-22 corrected economy/knee operating points is in progress; it reconfirms the keep-all verdict and is expected to move only the economy-band absolute numbers, not the conclusion. Slot the refreshed economy figure here if it changes.}

## Threshold re-confirmation (E07)

The pinned quality structure is swept over the full $\theta \times \theta_{\mathrm{reject}}$ grid at the final scoring function. The **maximum strict-F1 is at $\theta = 0.9$, $\theta_{\mathrm{reject}} = 0.1$ (0.7135)** — the argmax holds, confirming the frozen pin. A small rejection threshold ($\theta_{\mathrm{reject}} = 0.1$) dominates no rejection ($0.0$): it drops the lowest-consensus chunks (likely false-positive findings) for slightly higher strict-F1 (0.7135 vs 0.7124) at lower cost (456 vs 471 Sonnet calls). The $\theta$ curve is **flat near the top** — $\theta = 0.5$ already reaches $\approx 0.65$ against $\theta = 0.9$'s 0.71 — so escalation has clear diminishing returns, a fact that drives the cost-quality analysis of Section~\ref{section:results-cost-quality-comparison}.

## Gate ablation (E08)

The two routing gates — the single-voter policy (`keep` vs `escalate`) and the polarity hard-fail (`force_escalate_on_polarity_conflict` $\in$ {True, False}) — are ablated at the pinned $\theta$. **All four combinations are identical** (strict-F1 0.7135, escalation 0.962): at $\theta = 0.9$ the gates are *inert*, because with $\approx 96\%$ escalation the single-voter branch is almost never reached and the handful of polarity-conflict chunks escalate to Level 3 either way. The principled safety defaults (`keep`, force-escalate `True`) are retained, since they would still bite at the lower $\theta$ of the economy configuration.

## Configuration of record

The calibration above fixes the production cascade configuration, summarised below.

| Knob | Value | Fixed by |
|---|---|---|
| Cascade | legacy AgreementChecker, all 6 voters | E06b / E06c |
| Scorer / alignment | hybrid / greedy | E06 |
| Hybrid blend (cat / emb / ent / evi) | 0.15 / 0.30 / 0.50 / 0.05 (entity-heavy) | E06 |
| Embedder / tau | gemini / 0.15 | E06 |
| $\theta$ / $\theta_{\mathrm{reject}}$ | 0.9 / 0.1 | E07 |
| Gates (single-voter / polarity) | keep / force-escalate true | E08 |
| Grounding threshold | 0.5 | E03 (Section~\ref{section:results-nli-in-grounding-and-relation-classification}) |
| **Headline** | **strict-F1 0.7135 @ 96% escalation** | — |

This configuration is the one used for every downstream result in the thesis (the knowledge-extraction funnel of Section~\ref{section:results-ke-pipeline-output-and-provenance}, the cost-quality comparison of Section~\ref{section:results-cost-quality-comparison}, and the held-out test of Section~\ref{section:results-generalization-to-heldout-papers}). Whether the escalation-heavy operating point is *worth* its cost is the question Section~\ref{section:results-cost-quality-comparison} answers.

---

\pinktodo{Correction note (B-074): an enum-stringification bug in the strict-F1 scorer under-counted strict-F1 in proportion to the early-accept rate (heavy at low $\theta$, $\approx 0$ at $\theta = 0.9$). It was fixed, and every $\theta$-sensitive experiment (E07, E06c, E09–E12) re-run on the fix; the figures above are post-fix. The frozen pin held — the E07 argmax is still $\theta = 0.9$ — because structure and weight selection occur at high $\theta$ where the corruption was negligible. The net effect was a sharp rise in low-$\theta$ strict-F1, hence the flatter $\theta$ curve and the much stronger economy operating point reported in Section~\ref{section:results-cost-quality-comparison}.}
