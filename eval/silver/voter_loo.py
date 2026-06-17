#!/usr/bin/env python3
"""voter_loo.py — index-based, model-LABELLED leave-one-voter-out cascade diagnostic.

For each L1/L2 voter SLOT, drop that voter from the primed cache and re-replay the
legacy L1→L2→L3 cascade, to see how the cascade changes when a model is absent.

Index→model is stable by construction (collect places results by the custom_id
voter index via ``slot[vi]``; ``_make_voters()`` is a static list). A self-guard
asserts the cache still matches the current voter set, so a regenerated cache can
never silently mislabel index→model.

TWO questions, TWO comparisons — do not conflate them:
  * DECISION ("should we drop model X?"):  baseline **N=3** vs drop-X **N=2**.
    The N=3→2 change IS the real consequence of dropping X. Headline metric =
    ``strict_f1_optimal`` (does removing it improve silver-measured quality),
    with ``escalate_rate`` for cost; ``early_accept_rate`` is the *why*.
  * ATTRIBUTION ("which voter is most disagreeable?"):  the N=2 drops WITHIN a
    level, ranked against each other (N cancels). The drop with the HIGHEST
    ``early_accept_rate`` leaves the best-agreeing remaining pair → the dropped
    voter was the most disagreeable one.

Offline replay over the primed voter cache + warm embedding cache — this is a
DIAGNOSTIC, not calibration; no API beyond embedding-cache misses (a LOO reuses a
subset of already-embedded claims, so misses are negligible). L3 is excluded: it
is the escalation fallback, not an agreement voter.

Usage:
  python -m eval.silver.voter_loo --cases 15                     # fast smoke
  python -m eval.silver.voter_loo                                # full (θ 0.5/0.6/0.7)
  python -m eval.silver.voter_loo --scorer embedding --theta 0.6 --theta 0.7
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.map_theta_sweep import (  # reused engine + voter definitions
    REPORTS_DIR,
    ScorerSpec,
    _make_voters,
    _write_csv,
    run_sweep,
)
from eval.silver.matcher import SIMILARITY_THRESHOLD
from eval.silver.run_summarization_sweeps import _load_map_context
from pipeline.stages.summarization.config import AgreementConfig


def _voter_labels() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Canonical index→(provider, model) for L1 and L2, from _make_voters()."""
    L1, L2, _ = _make_voters()
    return ([(v.provider, v.model) for v in L1], [(v.provider, v.model) for v in L2])


def _max_slots(cache: dict, level: str) -> int:
    return max((len(lst) for e in cache.values()
               for lst in e.get(level, {}).values() if lst), default=0)


def _assert_cache_matches_voters(cache: dict, n_l1: int, n_l2: int) -> None:
    """Fail loud if the cache's per-level slot count != the current voter set — a
    regenerated cache primed with different voters would make index→model labels
    WRONG, and a mislabelled LOO yields confidently-wrong conclusions."""
    ml1, ml2 = _max_slots(cache, "l1"), _max_slots(cache, "l2")
    if ml1 != n_l1 or ml2 != n_l2:
        raise SystemExit(
            f"Cache voter-slot count ({ml1} L1 / {ml2} L2) != current _make_voters "
            f"({n_l1} L1 / {n_l2} L2). The cache was primed with a DIFFERENT voter set, "
            "so index→model labels would be wrong. Re-prime or align _make_voters first."
        )


def _drop_voter(cache: dict, level: str, idx: int) -> dict:
    """New cache with ``entry[level][chunk][idx] = None`` for every chunk — i.e. drop
    voter ``idx`` on the RAW None-padded list (``_replay`` then compacts it). All other
    levels / l3 / source structure are SHARED, and the source cache is never mutated."""
    out: dict = {}
    for sid, e in cache.items():
        lvl = {
            cid: ([None if j == idx else d for j, d in enumerate(lst)] if lst else lst)
            for cid, lst in e[level].items()
        }
        out[sid] = {**e, level: lvl}
    return out


def _run_variant(ctx, cache, spec, thetas, args) -> list[dict]:
    return run_sweep(
        voter_cache=cache,
        silver_by_case=ctx.silver_by_case,
        embedder=ctx.embedder,
        embed_cache=ctx.embed_cache,
        sim_threshold=args.sim_threshold,
        scorer_specs=[spec],
        thetas=thetas,
        reject_thetas=[args.reject_theta],
        split=args.split,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        agreement_embed_fn=ctx.agreement_embed_fn,
        single_voter_policies=["keep"],
        force_escalate_on_polarity_conflict_grid=[True],
    )


_CSV_FIELDS = [
    "dropped", "level", "idx", "model", "n_voters",
    "embedder", "scorer", "alignment", "theta", "reject_theta",
    "strict_f1_optimal", "f1_optimal", "precision_optimal", "recall_optimal",
    "strict_f1_greedy", "f1_greedy",
    "early_accept_rate", "escalate_rate", "n_pipeline", "n_silver",
]


def _f(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Model-labelled leave-one-voter-out cascade diagnostic (offline replay)."
    )
    ap.add_argument("--embedder", default="gemini", choices=["gemini", "openai"])
    ap.add_argument("--scorer", default="embedding", choices=["embedding", "hybrid"])
    ap.add_argument("--alignment", default="soft_max", choices=["soft_max", "greedy", "hungarian"])
    ap.add_argument("--theta", type=float, action="append",
                    help="Repeatable. Default 0.5 0.6 0.7 (where agreement is sensitive).")
    ap.add_argument("--reject-theta", type=float, default=0.10)
    ap.add_argument("--level", default="both", choices=["l1", "l2", "both"])
    ap.add_argument("--cases", type=int, default=None,
                    help="Limit to the first N cache cases (quick smoke).")
    ap.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-fraction", type=float, default=0.8)
    ap.add_argument("--embed-cache", default=None)
    args = ap.parse_args()

    thetas = args.theta or [0.5, 0.6, 0.7]
    if args.reject_theta >= min(thetas):
        raise SystemExit(f"--reject-theta {args.reject_theta} must be < min theta {min(thetas)}")

    L1_lab, L2_lab = _voter_labels()
    ctx = _load_map_context(args.embedder, embed_cache_path=args.embed_cache)
    _assert_cache_matches_voters(ctx.voter_cache, len(L1_lab), len(L2_lab))

    cache = ctx.voter_cache
    if args.cases:
        cache = dict(list(cache.items())[: args.cases])
        print(f"[smoke] limited to {len(cache)} case(s)")

    spec = ScorerSpec(args.scorer, args.scorer,
                      AgreementConfig(scorer_kind=args.scorer, alignment_strategy=args.alignment))

    # baseline (all voters) + one drop per L1/L2 slot
    variants: list[tuple] = [("ALL", None, None, "(all voters)", 3)]
    if args.level in ("l1", "both"):
        for i, (p, m) in enumerate(L1_lab):
            variants.append((f"-L1[{i}]", "l1", i, f"{p}/{m}", 2))
    if args.level in ("l2", "both"):
        for i, (p, m) in enumerate(L2_lab):
            variants.append((f"-L2[{i}]", "l2", i, f"{p}/{m}", 2))

    all_rows: list[dict] = []
    for label, level, idx, model, nvot in variants:
        print(f"running {label:10s} {model} ...")
        c = cache if level is None else _drop_voter(cache, level, idx)
        rows = _run_variant(ctx, c, spec, thetas, args)
        for r in rows:
            r.update({"dropped": label, "level": level or "", "idx": "" if idx is None else idx,
                      "model": model, "n_voters": nvot, "embedder": args.embedder,
                      "scorer": args.scorer, "alignment": args.alignment})
        all_rows.extend(rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = REPORTS_DIR / "E12_voter_loo"   # matches docs/readmes/EXPERIMENTS.md
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{ts}.csv"
    _write_csv(csv_path, all_rows, _CSV_FIELDS)

    for theta in thetas:
        rt = [r for r in all_rows if abs(float(r["theta"]) - theta) < 1e-9]
        base = next(r for r in rt if r["dropped"] == "ALL")
        drops = [r for r in rt if r["dropped"] != "ALL"]
        print(f"\n{'=' * 96}")
        print(f"θ={theta:.2f}  scorer={args.scorer}/{args.alignment}  embedder={args.embedder}"
              f"   (baseline N=3, drops N=2)")
        print('=' * 96)
        print("DECISION — drop model X?   Δ vs ALL; headline = strict_f1_optimal (escalate = cost)")
        print(f"  {'variant':9s} {'model':34s} {'strictF1':>8} {'ΔstrF1':>8} {'f1opt':>7} "
              f"{'early':>6} {'esc':>6}")
        print(f"  {'ALL N=3':9s} {'(all voters)':34s} {_f(base,'strict_f1_optimal'):8.4f} "
              f"{'--':>8} {_f(base,'f1_optimal'):7.4f} {_f(base,'early_accept_rate'):6.3f} "
              f"{_f(base,'escalate_rate'):6.3f}")
        for r in drops:
            d = _f(r, 'strict_f1_optimal') - _f(base, 'strict_f1_optimal')
            print(f"  {r['dropped']:9s} {r['model']:34s} {_f(r,'strict_f1_optimal'):8.4f} "
                  f"{d:+8.4f} {_f(r,'f1_optimal'):7.4f} {_f(r,'early_accept_rate'):6.3f} "
                  f"{_f(r,'escalate_rate'):6.3f}")

        for lvl_name, lvl_key in (("L1", "l1"), ("L2", "l2")):
            ld = [r for r in drops if r["level"] == lvl_key]
            if not ld:
                continue
            print(f"\n  ATTRIBUTION {lvl_name} — most disagreeable (drops within {lvl_name}, "
                  f"highest early_accept ⇒ removed voter was the odd one out):")
            for r in sorted(ld, key=lambda r: _f(r, 'early_accept_rate'), reverse=True):
                print(f"    drop {r['model']:34s} early={_f(r,'early_accept_rate'):.3f}  "
                      f"strictF1={_f(r,'strict_f1_optimal'):.4f}  esc={_f(r,'escalate_rate'):.3f}")

    print(f"\nCSV → {csv_path}")
    print("Note: ~4 already-sparse chunks drop to N=1, where single_voter_policy='keep' "
          "auto-accepts — a small, variant-dependent early-accept inflater.")


if __name__ == "__main__":
    main()
