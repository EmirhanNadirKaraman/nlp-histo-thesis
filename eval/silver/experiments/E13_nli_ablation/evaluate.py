#!/usr/bin/env python3
"""E13 — relation-classification NLI ablation on the synthetic 300 claim-pair set.

RQ4 (thesis §9.5). Evaluates the RELATE NLI classifier directly against silver
labels, using the EXACT production NLI logic from
``pipeline/stages/summarization/current_stages/relate_stage.py`` (no
reimplementation): ``_get_nli_pipe``, ``_nli_scores``, ``_build_nli_text``,
``_classify_pair``.

Two input modes (verbatim modes are excluded — synthetic claims have no real
source sentence, so verbatim would just fall back to the predicate):
  * predicate_only  — NLI over the bare claim
  * scope_aware     — NLI over "[scope: disease=<entity>] <claim>"

Gold-label mapping (dataset keeps canonical labels; mapped to the classifier's
{SUPPORT, CONTRADICT, UNRELATED} space for scoring):
  SUPPORTING → SUPPORT, CONTRADICTING → CONTRADICT, UNRELATED → UNRELATED.

PRIMARY reported result = the default RelateConfig thresholds (0.50/0.50). The
entailment×contradiction threshold sweep is DIAGNOSTIC sensitivity analysis only
— it does NOT tune RelateConfig or the frozen pipeline (per the thesis: no
hyperparameter is tuned on this set).

Note: ``_classify_pair`` has a corpus-structural same-polarity guard that reads
``rule.direction``. Synthetic free-text claims have no structured direction, so
the shims carry ``direction=None`` — this neutralizes the guard (contradiction
always permitted), isolating the NLI classifier, which is what this ablation
measures.

  python -m eval.silver.experiments.E13_nli_ablation.evaluate
  python -m eval.silver.experiments.E13_nli_ablation.evaluate --limit 30   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.summarization.config import RelateConfig  # noqa: E402
from pipeline.stages.summarization.current_stages.relate_stage import (  # noqa: E402
    _build_nli_text,
    _classify_pair,
    _get_nli_pipe,
    _nli_scores,
)

DATASET = _REPO_ROOT / "eval" / "data" / "relation_claim_pairs_300.jsonl"
REPORTS_DIR = _REPO_ROOT / "eval" / "reports" / "E13_nli_ablation"

GOLD_MAP = {"SUPPORTING": "SUPPORT", "CONTRADICTING": "CONTRADICT", "UNRELATED": "UNRELATED"}
CLASSES = ["SUPPORT", "CONTRADICT", "UNRELATED"]
MODES = ["predicate_only", "scope_aware"]
SWEEP = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
_DEFAULTS = RelateConfig()  # frozen pipeline thresholds (0.50 / 0.50)


# ── rule-shim ────────────────────────────────────────────────────────────────
def build_shim(claim: str, entity: str | None):
    """Minimal CanonicalRule stand-in for the relate_stage NLI helpers.

    predicate_text = the claim; representative_verbatim = None (no verbatim mode);
    scope.disease_subtype = the shared entity (other scope fields None);
    direction = None (neutralizes the corpus same-polarity contradict guard).
    """
    scope = SimpleNamespace(
        disease_subtype=entity or None, tissue_site=None, assay_method=None,
        treatment_context=None, endpoint=None, study_design=None,
        biomarker_cutoff=None, cohort_n=None,
    )
    return SimpleNamespace(
        predicate_text=claim, representative_verbatim=None, scope=scope, direction=None,
    )


# ── metrics (pure python, no sklearn dependency) ─────────────────────────────
def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    m = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def per_class_prf(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in CLASSES:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(prf: dict[str, dict[str, float]]) -> float:
    return sum(prf[c]["f1"] for c in CLASSES) / len(CLASSES)


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0


# ── scoring ──────────────────────────────────────────────────────────────────
def score_mode(records: list[dict], scope_aware: bool, pipe) -> list[dict]:
    """Compute bidirectional NLI scores once per mode (threshold-independent).

    Returns per-pair {gold, shim_a, shim_b, scores_ab, scores_ba}.
    """
    shims = [
        (build_shim(r["claim_a"], r.get("disease_or_entity")),
         build_shim(r["claim_b"], r.get("disease_or_entity")),
         GOLD_MAP[r["gold_label"]])
        for r in records
    ]
    pairs_ab = [
        (_build_nli_text(a, scope_aware=scope_aware), _build_nli_text(b, scope_aware=scope_aware))
        for a, b, _ in shims
    ]
    pairs_ba = [(hyp, prem) for prem, hyp in pairs_ab]
    scores_ab = _nli_scores(pairs_ab, pipe)
    scores_ba = _nli_scores(pairs_ba, pipe)
    return [
        {"gold": g, "shim_a": a, "shim_b": b, "scores_ab": sab, "scores_ba": sba}
        for (a, b, g), sab, sba in zip(shims, scores_ab, scores_ba)
    ]


def classify_all(scored: list[dict], ent_thr: float, con_thr: float) -> tuple[list[str], list[str]]:
    gold, pred = [], []
    for s in scored:
        lab = _classify_pair(s["shim_a"], s["shim_b"], s["scores_ab"], s["scores_ba"], ent_thr, con_thr)
        gold.append(s["gold"])
        pred.append(lab.value if lab is not None else "UNRELATED")
    return gold, pred


def _print_result(title: str, gold: list[str], pred: list[str]) -> dict:
    prf = per_class_prf(gold, pred)
    acc, mf1 = accuracy(gold, pred), macro_f1(prf)
    print(f"\n{title}")
    print(f"  accuracy={acc:.4f}   macro-F1={mf1:.4f}")
    print(f"  {'class':12} {'P':>7} {'R':>7} {'F1':>7} {'n':>5}")
    for c in CLASSES:
        d = prf[c]
        print(f"  {c:12} {d['precision']:7.3f} {d['recall']:7.3f} {d['f1']:7.3f} {int(d['support']):5d}")
    cm = confusion(gold, pred)
    print(f"  confusion (rows=gold, cols=pred): {'':>4}" + "".join(f"{c[:5]:>7}" for c in CLASSES))
    for g in CLASSES:
        print(f"    {g[:9]:>9}: " + "".join(f"{cm[g][p]:7d}" for p in CLASSES))
    return {"accuracy": acc, "macro_f1": mf1, "prf": prf}


def main() -> None:
    ap = argparse.ArgumentParser(description="E13 — NLI relation-classification ablation (no API).")
    ap.add_argument("--dataset", default=str(DATASET), help="path to relation_claim_pairs_300.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N pairs (smoke)")
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        print(f"Dataset not found: {ds}\nRun validate_and_merge first.")
        raise SystemExit(1)
    records = [json.loads(ln) for ln in ds.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"E13 NLI relation-classification ablation — {len(records)} pairs from {ds.name}")
    print(f"PRIMARY thresholds (frozen RelateConfig): entailment={_DEFAULTS.entailment_threshold} "
          f"contradiction={_DEFAULTS.contradiction_threshold}")

    pipe = _get_nli_pipe()
    rows: list[dict] = []

    # ── per mode: score once, then default (primary) + diagnostic sweep ──
    cached: dict[str, list[dict]] = {}
    primary: dict[str, dict] = {}
    for mode in MODES:
        scope_aware = (mode == "scope_aware")
        scored = score_mode(records, scope_aware, pipe)
        cached[mode] = scored
        gold, pred = classify_all(scored, _DEFAULTS.entailment_threshold, _DEFAULTS.contradiction_threshold)
        res = _print_result(
            f"[PRIMARY] mode={mode}  thr=({_DEFAULTS.entailment_threshold},"
            f"{_DEFAULTS.contradiction_threshold})",
            gold, pred,
        )
        primary[mode] = res
        for c in CLASSES:
            d = res["prf"][c]
            rows.append({
                "phase": "primary", "mode": mode, "ent_thr": _DEFAULTS.entailment_threshold,
                "con_thr": _DEFAULTS.contradiction_threshold, "is_default": True,
                "accuracy": round(res["accuracy"], 4), "macro_f1": round(res["macro_f1"], 4),
                "class": c, "precision": round(d["precision"], 4),
                "recall": round(d["recall"], 4), "f1": round(d["f1"], 4), "support": int(d["support"]),
            })

    # ── scope-aware vs predicate-only (at default) ──
    print("\n── scope-aware vs predicate-only (default thresholds) ──")
    d_acc = primary["scope_aware"]["accuracy"] - primary["predicate_only"]["accuracy"]
    d_mf1 = primary["scope_aware"]["macro_f1"] - primary["predicate_only"]["macro_f1"]
    print(f"  Δaccuracy(scope−pred) = {d_acc:+.4f}   Δmacro-F1 = {d_mf1:+.4f}")

    # ── DIAGNOSTIC threshold sweep (NOT tuning — does not change RelateConfig) ──
    print("\n── DIAGNOSTIC threshold sweep (sensitivity only; pipeline NOT retuned) ──")
    for mode in MODES:
        scored = cached[mode]
        best = None
        for ent in SWEEP:
            for con in SWEEP:
                gold, pred = classify_all(scored, ent, con)
                prf = per_class_prf(gold, pred)
                acc, mf1 = accuracy(gold, pred), macro_f1(prf)
                is_def = (ent == _DEFAULTS.entailment_threshold and con == _DEFAULTS.contradiction_threshold)
                for c in CLASSES:
                    dd = prf[c]
                    rows.append({
                        "phase": "sweep", "mode": mode, "ent_thr": ent, "con_thr": con,
                        "is_default": is_def, "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                        "class": c, "precision": round(dd["precision"], 4), "recall": round(dd["recall"], 4),
                        "f1": round(dd["f1"], 4), "support": int(dd["support"]),
                    })
                if best is None or mf1 > best[2]:
                    best = (ent, con, mf1, acc)
        print(f"  {mode:14}: best-by-macroF1 @ ent={best[0]} con={best[1]} → "
              f"macro-F1={best[2]:.4f} acc={best[3]:.4f}   "
              f"(default macro-F1={primary[mode]['macro_f1']:.4f})")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = REPORTS_DIR / f"{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "phase", "mode", "ent_thr", "con_thr", "is_default", "accuracy", "macro_f1",
            "class", "precision", "recall", "f1", "support",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
