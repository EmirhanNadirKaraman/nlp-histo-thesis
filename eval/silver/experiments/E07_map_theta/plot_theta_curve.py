#!/usr/bin/env python3
"""E07 — plot strict-F1 vs theta for the 5-voter cascade (calibration figure).

Draws ``strict_f1_optimal`` against the agreement threshold ``theta`` at the
shipped ``reject_theta = 0.2``, for both single-voter gates:

  * keep   gate (default)  — the E07 ``map_theta`` sweep
  * escalate gate (shipped) — the E08b ``map_theta_shipped`` sweep

The curve is monotone and flattens above ``theta = 0.8`` (the diminishing-returns
point that motivates the cost-quality analysis of Section 04), and both gates peak
at the pinned ``theta = 0.9`` (escalate marginally higher: the E08 finding).

Reads the timestamped sweep CSVs from ``eval/reports/E07_map_theta/`` and
``eval/reports/E08b_map_theta_shipped/`` and writes ``theta_strict_f1.{pdf,png}``
back into ``eval/reports/E07_map_theta/``. Offline — CSV reports only, no API/DB.

    python -m eval.silver.experiments.E07_map_theta.plot_theta_curve
"""
from __future__ import annotations

import csv
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = _REPO_ROOT / "eval" / "reports"
E07_DIR = REPORTS / "E07_map_theta"            # keep gate (default)
E08B_DIR = REPORTS / "E08b_map_theta_shipped"  # escalate gate (shipped)
REJECT_THETA = "0.2"                           # the pinned rejection threshold


def _latest_csv(d: Path) -> Path:
    csvs = sorted(d.glob("sweep_*.csv")) or sorted(d.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no sweep CSV in {d}")
    return csvs[-1]


def _theta_curve(path: Path, reject: str = REJECT_THETA):
    """(theta, strict_f1_optimal) points at the fixed reject_theta, sorted by theta."""
    pts: dict[float, float] = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("reject_theta") or "").strip() not in (reject, f"{float(reject)}"):
                continue
            try:
                pts[float(r["theta"])] = float(r["strict_f1_optimal"])
            except (KeyError, ValueError, TypeError):
                continue
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


def main() -> int:
    e07, e08b = _latest_csv(E07_DIR), _latest_csv(E08B_DIR)
    kx, ky = _theta_curve(e07)   # keep gate
    ex, ey = _theta_curve(e08b)  # escalate gate
    if not kx and not ex:
        print(f"no rows at reject_theta={REJECT_THETA} in {e07.name} / {e08b.name}")
        return 1
    print(f"keep  (E07,  {e07.name}): {len(kx)} points")
    print(f"escal (E08b, {e08b.name}): {len(ex)} points")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if ex:
        ax.plot(ex, ey, "-o", color="#1f77b4", lw=1.8, ms=4.5,
                label="escalate gate (shipped)")
    if kx:
        ax.plot(kx, ky, "--s", color="#888888", lw=1.3, ms=3.5,
                label="keep gate (default)")

    # Mark the pinned theta = 0.9 (argmax under both gates).
    if ex:
        peak = ey[-1]
        ax.axvline(0.9, color="#d62728", lw=0.8, ls=":", alpha=0.7)
        ax.annotate(
            f"pinned $\\theta=0.9$\nstrict-F1 {peak:.4f}",
            xy=(0.9, peak), xytext=(0.55, peak - 0.075),
            fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
        )

    ax.set_xlabel(r"agreement threshold $\theta$")
    ax.set_ylabel("strict-F1 (optimal matcher)")
    ax.set_title(
        r"5-voter cascade: strict-F1 vs $\theta$ (at $\theta_{\mathrm{reject}}=0.2$)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = E07_DIR / f"theta_strict_f1.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
