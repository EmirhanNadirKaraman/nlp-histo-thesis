"""
score_distribution.py — distribution of the per-chunk agreement score
(``primary``) that ``reject_theta`` thresholds in the ABC cascade.

WHAT THIS MEASURES
``reject_theta`` rejects a chunk when its primary score ``<= reject_theta``
(``agreement/checker.py:143``). The score is::

    primary = bundle.confidence or bundle.embedding_agreement or 0.0   # checker.py:140

and is only consulted for chunks that REACH the primary gate (``decision is
None``). Chunks that short-circuit earlier BYPASS reject_theta and are excluded
from the gate population, else they inflate the band with phantom counts:
  * 0 voters       → ESCALATE (checker.py:85)
  * 1 voter        → KEEP @ confidence=1.0 (checker.py:95, single_voter_policy=keep)
  * polarity clash → ESCALATE @ force_escalate=True (checker.py:118)

WHY IT IS θ-INDEPENDENT (and why that matters)
theta/reject_theta only CLASSIFY ``primary``; they never compute it. So the L1
score distribution is fixed across the whole θ grid. A chunk with primary in
(0.1, 0.2] is the only thing that can make reject_theta=0.1 differ from 0.2 — so
the share of gate scores in that band is exactly the sensitivity of the reject
prune. The structure_screen sweep showed reject 0.1≡0.2 byte-identical only for
θ≥0.7; this histogram is θ-independent, so it closes that gap for θ=0.30–0.60 too.

This script is MECHANISM ILLUSTRATION, not new validation: the 60-cell
byte-identical sweep result already forces the (0.1,0.2] share to ~0 (any gate
chunk in that band would have flipped 0.1→0.2). The genuinely new information is
the distribution SHAPE — expect bimodal (clusters near 0 and near 1, sparse
middle), which is *why* reject_theta is insensitive in the band.

Usage:
  python -m eval.silver.analysis.score_distribution                      # all finalists, band (0.1,0.2]
  python -m eval.silver.analysis.score_distribution --band 0.1 0.2       # explicit band
  python -m eval.silver.analysis.score_distribution --bins 20
  python -m eval.silver.analysis.score_distribution --structure gemini embedding greedy
  # model_validate prints "Repaired category…" to stdout (B-016) — filter it:
  python -m eval.silver.analysis.score_distribution 2>/dev/null | grep -v "Repaired category"
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from eval.silver.analysis.map_context import _load_map_context
from eval.silver.analysis.map_theta_sweep import _build_scorer
from eval.silver.analysis.run_new_summarization_sweeps import (
    FINALIST_STRUCTURES,
    _default_spec,
    _refine_specs,
)


def _replay_scores(voter_cache: dict, checker) -> dict:
    """Replay L1/L2 chunk scoring; collect ``primary`` for gate-reaching chunks.

    Returns per-tier score lists plus the bypassed-chunk tally (counted at L1,
    where every chunk is scored). L2 scores come only from chunks that have
    cached L2 voters (i.e. escalated during priming) — a θ-dependent subset, so
    L1 is the clean θ-independent headline; L2 is supplementary.
    """
    from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary

    out = {"l1": [], "l2": [], "n_chunks": 0, "empty": 0, "single": 0, "polarity": 0}
    for entry in voter_cache.values():
        for chunk_id in entry.get("chunk_map", {}):
            src = entry.get("source_texts", {}).get(chunk_id, "")
            for tier in ("l1", "l2"):
                raw = entry.get(tier, {}).get(chunk_id) or []
                voters = [AuditableSummary.model_validate(d) for d in raw if d]
                if tier == "l1":
                    out["n_chunks"] += 1
                    if len(voters) == 0:
                        out["empty"] += 1
                    elif len(voters) < 2:
                        out["single"] += 1
                if len(voters) < 2:
                    continue  # bypasses the primary gate (empty→ESCALATE, single→KEEP)
                bundle = checker.compute(voters, source_text=src)
                is_polarity = bool(
                    bundle.score_details
                    and bundle.score_details.get("hard_fail_reason") == "polarity_conflict"
                )
                if is_polarity:
                    if tier == "l1":
                        out["polarity"] += 1
                    continue  # force_escalate=True → bypasses the reject gate
                # Replicate checker.py:140 EXACTLY (the `or` chain: 0.0/None fall through).
                primary = bundle.confidence or bundle.embedding_agreement or 0.0
                out[tier].append(float(primary))
    return out


def _band_count(scores: list[float], lo: float, hi: float) -> int:
    """Count scores in (lo, hi] — the reject lo↔hi difference band."""
    return sum(1 for s in scores if lo < s <= hi)


def _stats(scores: list[float]) -> tuple[float, float, float]:
    if not scores:
        return (0.0, 0.0, 0.0)
    s = sorted(scores)
    mid = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
    return (s[0], mid, s[-1])


def _histogram(scores: list[float], bins: int, lo: float, hi: float) -> None:
    if not scores:
        print("  (no gate-population scores)")
        return
    counts = [0] * bins
    for s in scores:
        counts[min(int(s * bins), bins - 1)] += 1
    peak = max(counts) or 1
    n = len(scores)
    for i, c in enumerate(counts):
        a, b = i / bins, (i + 1) / bins
        # Mark buckets overlapping the reject band.
        mark = "◄band" if (a < hi and b > lo) else ""
        bar = "█" * round(40 * c / peak)
        print(f"  [{a:.2f},{b:.2f})  {c:6d}  {100*c/n:5.1f}%  {bar} {mark}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--structure", nargs=3, metavar=("EMB", "SCORER", "ALIGN"),
                    action="append", help="limit to this finalist (repeatable); default = all")
    ap.add_argument("--band", nargs=2, type=float, default=[0.1, 0.2], metavar=("LO", "HI"),
                    help="reject difference band (lo, hi]; default 0.1 0.2")
    ap.add_argument("--bins", type=int, default=20, help="histogram bins over [0,1] (default 20)")
    ap.add_argument("--refine", action="store_true",
                    help="use the REAL family_refine weight variants (_refine_specs) instead of "
                         "the default-weight spec — for off-default band spot-checks")
    ap.add_argument("--variant-filter", default=None,
                    help="comma-separated substrings; with --refine, keep only variants whose "
                         "spec name contains one (e.g. 'tau_0.3,blend_,contradiction_weight_0.4')")
    args = ap.parse_args(argv)

    lo, hi = args.band
    structures = [tuple(s) for s in args.structure] if args.structure else list(FINALIST_STRUCTURES)

    # Group by embedder so each embedder's context/cache loads once.
    by_emb: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for emb, sk, al in structures:
        by_emb[emb].append((sk, al))

    from nlp_histo.pipeline.stages.knowledge_extraction.agreement import AgreementChecker

    subs = args.variant_filter.split(",") if args.variant_filter else None
    per_struct: dict[str, dict] = {}
    for emb, combos in by_emb.items():
        ctx = _load_map_context(emb, embed_cache_path=None)
        for sk, al in combos:
            if args.refine:
                specs = _refine_specs(emb, sk, al)
                if subs:
                    specs = [s for s in specs if any(sub in s.name for sub in subs)]
            else:
                specs = [_default_spec(sk, al)]
            for spec in specs:
                label = spec.name if args.refine else f"{emb}/{sk}/{al}"
                scorer = _build_scorer(spec, ctx.agreement_embed_fn)
                checker = AgreementChecker(
                    scorer, theta=0.9, reject_theta=0.1,
                    single_voter_policy="keep", force_escalate_on_polarity_conflict=True,
                )
                per_struct[label] = _replay_scores(ctx.voter_cache, checker)

    # Per-structure table (L1 gate population)
    print("\nPer-chunk L1 agreement score (`primary`) — gate population only")
    print(f"reject difference band = ({lo}, {hi}]\n")
    hdr = f"{'structure / variant':44} {'n_gate':>7} {'in band':>8} {'band %':>7} {'min':>6} {'med':>6} {'max':>6}"
    print(hdr)
    print("-" * len(hdr))
    pooled: list[float] = []
    for label, rec in per_struct.items():
        l1 = rec["l1"]
        pooled += l1
        nb = _band_count(l1, lo, hi)
        mn, md, mx = _stats(l1)
        pct = 100 * nb / len(l1) if l1 else 0.0
        print(f"{label[:44]:44} {len(l1):>7} {nb:>8} {pct:>6.2f}% {mn:>6.3f} {md:>6.3f} {mx:>6.3f}")

    # Bypassed-chunk accounting (pooled, from L1 pass)
    tot = sum(r["n_chunks"] for r in per_struct.values())
    emp = sum(r["empty"] for r in per_struct.values())
    sng = sum(r["single"] for r in per_struct.values())
    pol = sum(r["polarity"] for r in per_struct.values())
    gate = len(pooled)
    print(f"\nChunk accounting (pooled over {len(per_struct)} structures, L1):")
    print(f"  total chunk-evals       {tot}")
    print(f"  bypass: empty (0 voters){emp:>8}   single (1 voter){sng:>8}   polarity-escalated{pol:>8}")
    print(f"  reached primary gate    {gate}  ({100*gate/tot if tot else 0:.1f}% of evals)")

    # Headline + pooled histogram
    nb = _band_count(pooled, lo, hi)
    pct = 100 * nb / gate if gate else 0.0
    print(f"\n>>> HEADLINE: {nb}/{gate} = {pct:.3f}% of gate scores fall in ({lo}, {hi}]")
    print(f"    → that share is the exact sensitivity of reject_theta to a {lo}→{hi} change.\n")
    print(f"Pooled L1 score distribution ({args.bins} bins):")
    _histogram(pooled, args.bins, lo, hi)

    # L2 supplementary
    pooled_l2 = [s for r in per_struct.values() for s in r["l2"]]
    if pooled_l2:
        nb2 = _band_count(pooled_l2, lo, hi)
        pct2 = 100 * nb2 / len(pooled_l2)
        print(f"\nL2 (supplementary; escalated-during-priming chunks): "
              f"{nb2}/{len(pooled_l2)} = {pct2:.3f}% in ({lo}, {hi}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
