#!/usr/bin/env python3
"""E04 — knowledge-extraction per-stage cardinalities (the extraction "funnel").

DESCRIPTIVE (RQ2, §9.2), not a calibration: aggregates, over the related15 papers,
how many items survive each summarisation stage —

    MAP findings → grounded → normalized (dedup) → groupable → canonical rules
                 → relations → final rules

All counts already live in each paper's pipeline-output JSON (`rejection_summary`
+ top-level list lengths), so this is a pure aggregation — no API, no model loads.

Reads a directory of per-paper pipeline JSONs (default the cached out/summaries).
NOTE on config: the aggregate reflects whatever MAP config produced those JSONs —
it reports the `cascade_profile` it finds and warns if it is not the frozen
6-voter cascade. For frozen-config absolute counts, regenerate the downstream over
the frozen MAP first; the per-stage *rates* (dedup %, groupable %, rules/finding)
are far more config-robust than the absolute counts.

Usage:
  python -m eval.silver.experiments.E04_cardinalities.cardinalities
  python -m eval.silver.experiments.E04_cardinalities.cardinalities --summaries-dir out/summaries/summaries
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.analysis.map_theta_sweep import REPORTS_DIR  # noqa: E402

_DEFAULT_SUMMARIES = _REPO_ROOT / "out" / "summaries" / "summaries"
_DEFAULT_SOURCE = _REPO_ROOT / "eval" / "data" / "source_cases_related15.jsonl"
# Frozen-config MAP+grounding reference from E03 (related15, θ0.9/r0.1, grounding 0.5,
# DB-paragraph grounding = production path; now matches this funnel's grounded count, B-079):
_E03_FROZEN = {"map_findings": 2280, "grounded_at_0_5": 1911}
# Cascade-profile names that ARE the frozen 6-voter cascade (E06–E08 winner).
# "real" is the production label for that cascade; "6voter_frozen" is a legacy alias.
_FROZEN_PROFILES = {"6voter_frozen", "real"}

_PER_PAPER_FIELDS = [
    "pmcid", "cascade_profile", "grounding_threshold",
    "map_findings", "grounding_rejected", "normalized", "non_groupable",
    "canonical_rules", "relations", "final_rules",
]


def _g(d: dict, *keys, default=0):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="E04 knowledge-extraction funnel (offline aggregation).")
    ap.add_argument("--summaries-dir", default=str(_DEFAULT_SUMMARIES),
                    help="dir of per-paper pipeline JSONs (default out/summaries/summaries).")
    ap.add_argument("--source", default=str(_DEFAULT_SOURCE),
                    help="related15 source_cases jsonl — restricts to its pmcids.")
    args = ap.parse_args()

    pmcids = {json.loads(line)["pmcid"] for line in open(args.source)}
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(args.summaries_dir, "*.json"))):
        if os.path.basename(path)[:-5] not in pmcids:
            continue
        d = json.loads(Path(path).read_text())
        if d.get("status") != "success":
            continue
        rs = d.get("rejection_summary", {}) or {}
        mc = d.get("audit_trail", {}).get("map_chunks", [])
        n_map = int(_g(rs, "map_findings_total", default=sum(len(c.get("findings", [])) for c in mc)))
        rows.append({
            "pmcid": d.get("pmcid", os.path.basename(path)[:-5]),
            "cascade_profile": d.get("map_run_metadata", {}).get("cascade_profile", "?"),
            "grounding_threshold": _g(rs, "grounding_threshold", default=""),
            "map_findings": n_map,
            "grounding_rejected": int(_g(rs, "map_grounding_rejected")),
            "normalized": int(_g(rs, "normal_findings_total", default=n_map)),
            "non_groupable": int(_g(rs, "non_groupable_total")),
            "canonical_rules": len(d.get("canonical_rules") or []),
            "relations": len(d.get("relations") or []),
            "final_rules": len(d.get("final_rules") or []),
        })

    if not rows:
        raise SystemExit(f"no related15 pipeline JSONs found in {args.summaries_dir}")

    def S(k):
        return sum(r[k] for r in rows)
    n_papers = len(rows)
    profiles = sorted({r["cascade_profile"] for r in rows})
    tot = {k: S(k) for k in ("map_findings", "grounding_rejected", "normalized",
                             "non_groupable", "canonical_rules", "relations", "final_rules")}
    grounded = tot["map_findings"] - tot["grounding_rejected"]
    groupable = tot["normalized"] - tot["non_groupable"]

    print(f"E04 knowledge-extraction funnel — {n_papers} related15 papers")
    print(f"cascade_profile(s) in source JSONs: {profiles}")
    if any(p not in _FROZEN_PROFILES for p in profiles):
        print("  ⚠️ NOT the frozen 6-voter cascade — absolute counts are config-specific; "
              "rates below are the config-robust readout. (Frozen MAP+grounding ref from E03 shown.)")
    else:
        print("  ✓ frozen 6-voter cascade ('real' profile, θ0.9/reject0.1) — absolute counts are config-final.")

    def line(label, n, denom=None):
        pct = f"  ({n/denom:6.1%} of prev)" if denom else ""
        print(f"  {label:24} {n:6d}{pct}")
    print("\nFUNNEL (summed over papers):")
    line("MAP findings", tot["map_findings"])
    line("→ grounded (kept)", grounded, tot["map_findings"])
    line("→ normalized (dedup)", tot["normalized"], grounded)
    line("→ groupable", groupable, tot["normalized"])
    line("→ canonical rules", tot["canonical_rules"], groupable)
    line("→ relations", tot["relations"])
    line("→ final rules", tot["final_rules"])

    print("\nRATES (config-robust):")
    print(f"  grounding retention    {grounded/tot['map_findings']:.3f}")
    print(f"  normalize dedup        {1 - tot['normalized']/grounded:.3f}  (fraction merged/removed)")
    print(f"  non-groupable          {tot['non_groupable']/tot['normalized']:.3f}")
    print(f"  canonical rules/finding {tot['canonical_rules']/tot['normalized']:.3f}")
    print(f"  relations / paper       {tot['relations']/n_papers:.2f}")
    print(f"  final rules / paper     {tot['final_rules']/n_papers:.1f}")

    print(f"\nfrozen-config MAP reference (E03): {_E03_FROZEN['map_findings']} findings → "
          f"{_E03_FROZEN['grounded_at_0_5']} grounded @0.5 "
          f"(this run's MAP={tot['map_findings']} under {profiles})")

    out_dir = REPORTS_DIR / "E04_cardinalities"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"cardinalities_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_PER_PAPER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["pmcid"]):
            w.writerow(r)
        w.writerow({"pmcid": "TOTAL", "cascade_profile": "+".join(profiles), **tot})
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
