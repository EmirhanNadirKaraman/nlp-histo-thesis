"""
Generate a standalone HTML dashboard from sweep report CSVs.

Reads the latest CSVs from eval/reports/ and renders a self-contained HTML file
with Chart.js P/R/F1 curves, RELATE stacked-bar charts, and best-threshold callouts.
No server required — open directly in any browser.

Usage:
  python -m eval.silver.dashboard
  python -m eval.silver.dashboard --reports eval/reports --out eval/reports/dashboard.html
  python -m eval.silver.dashboard --grounding path/to/grounding.csv --sweep path/to/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path("eval/reports")


# ── CSV loading ───────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _latest_matching(reports_dir: Path, pattern: str) -> Path | None:
    matches = sorted(reports_dir.glob(pattern))
    return matches[-1] if matches else None


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pipeline Threshold Sweep Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; color: #1a1a1a; padding: 24px; }}
  h1  {{ font-size: 1.5rem; margin-bottom: 6px; }}
  .meta {{ font-size: 0.8rem; color: #666; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
            gap: 24px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  .card h2 {{ font-size: 1rem; margin-bottom: 4px; }}
  .card .subtitle {{ font-size: 0.75rem; color: #888; margin-bottom: 16px; }}
  .best-box {{ background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 8px;
                padding: 12px 16px; margin-bottom: 16px; font-size: 0.85rem; }}
  .best-box strong {{ color: #2e7d32; }}
  canvas {{ max-height: 280px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 12px; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #e0e0e0; text-align: right; }}
  th {{ background: #f9f9f9; font-weight: 600; text-align: left; }}
  td:first-child {{ text-align: left; }}
  tr.best-row td {{ background: #fff9c4; font-weight: 600; }}
  .no-data {{ color: #aaa; font-style: italic; font-size: 0.85rem; padding: 16px 0; }}
</style>
</head>
<body>
<h1>Pipeline Threshold Sweep Dashboard</h1>
<div class="meta">Generated {timestamp} &nbsp;|&nbsp; split: {split} &nbsp;|&nbsp;
sim-threshold: {sim_threshold}</div>

<div class="grid">

<!-- ── Eval matching threshold (sweep.py) ── -->
<div class="card" id="card-eval">
  <h2>Eval Matching Threshold</h2>
  <div class="subtitle">Cosine similarity cutoff for silver ↔ pipeline finding matching.
    Set SIMILARITY_THRESHOLD in eval/silver/matching/matcher.py to the best value.</div>
  {eval_best_box}
  <canvas id="chart-eval"></canvas>
  {eval_table}
</div>

<!-- ── Grounding threshold ── -->
<div class="card" id="card-grounding">
  <h2>Grounding Threshold</h2>
  <div class="subtitle">NLI entailment score cutoff (GroundingConfig.threshold).
    Set in pipeline/stages/knowledge_extraction/config.py.</div>
  {grounding_best_box}
  <canvas id="chart-grounding"></canvas>
  {grounding_table}
</div>

<!-- ── RELATE entailment threshold ── -->
<div class="card" id="card-relate-ent">
  <h2>RELATE Entailment Threshold</h2>
  <div class="subtitle">Entailment score for SUPPORT / SCOPE_QUALIFY classification.
    RelateConfig.entailment_threshold in config.py. Shown at contradiction_threshold=0.50.</div>
  <canvas id="chart-relate-ent"></canvas>
  {relate_ent_table}
</div>

<!-- ── RELATE contradiction threshold ── -->
<div class="card" id="card-relate-con">
  <h2>RELATE Contradiction Threshold</h2>
  <div class="subtitle">Both-direction contradiction score cutoff for CONTRADICT classification.
    RelateConfig.contradiction_threshold in config.py. Shown at entailment_threshold=0.50.</div>
  <canvas id="chart-relate-con"></canvas>
  {relate_con_table}
</div>

</div><!-- .grid -->

<script>
const evalData        = {eval_json};
const groundingData   = {grounding_json};
const relateData      = {relate_json};

function prcChart(canvasId, labels, datasets) {{
  const ctx = document.getElementById(canvasId);
  if (!ctx || !labels.length) return;
  new Chart(ctx, {{
    type: "line",
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        x: {{ title: {{ display: true, text: ctx.closest(".card").querySelector("h2").textContent }} }},
        y: {{ min: 0, max: 1, title: {{ display: true, text: "Score" }} }},
      }},
    }},
  }});
}}

function barChart(canvasId, labels, datasets) {{
  const ctx = document.getElementById(canvasId);
  if (!ctx || !labels.length) return;
  new Chart(ctx, {{
    type: "bar",
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        x: {{ stacked: true }},
        y: {{ stacked: true, title: {{ display: true, text: "Relation count" }} }},
      }},
    }},
  }});
}}

// ── Eval matching chart ──
if (evalData.length) {{
  const labels = evalData.map(r => r.threshold);
  prcChart("chart-eval", labels, [
    {{ label: "Precision", data: evalData.map(r => r.precision),
       borderColor: "#1976d2", backgroundColor: "#1976d222", fill: true, tension: 0.3 }},
    {{ label: "Recall",    data: evalData.map(r => r.recall),
       borderColor: "#388e3c", backgroundColor: "#388e3c22", fill: true, tension: 0.3 }},
    {{ label: "F1",        data: evalData.map(r => r.f1),
       borderColor: "#f57c00", backgroundColor: "#f57c0022", fill: true, tension: 0.3 }},
    {{ label: "Strict F1", data: evalData.map(r => r.strict_f1),
       borderColor: "#7b1fa2", backgroundColor: "#7b1fa222", fill: true,
       borderDash: [5,3], tension: 0.3 }},
  ]);
}}

// ── Grounding chart ──
if (groundingData.length) {{
  const labels = groundingData.map(r => r.grounding_threshold);
  prcChart("chart-grounding", labels, [
    {{ label: "Precision", data: groundingData.map(r => r.precision),
       borderColor: "#1976d2", backgroundColor: "#1976d222", fill: true, tension: 0.3 }},
    {{ label: "Recall",    data: groundingData.map(r => r.recall),
       borderColor: "#388e3c", backgroundColor: "#388e3c22", fill: true, tension: 0.3 }},
    {{ label: "F1",        data: groundingData.map(r => r.f1),
       borderColor: "#f57c00", backgroundColor: "#f57c0022", fill: true, tension: 0.3 }},
    {{ label: "Strict F1", data: groundingData.map(r => r.strict_f1),
       borderColor: "#7b1fa2", backgroundColor: "#7b1fa222", fill: true,
       borderDash: [5,3], tension: 0.3 }},
  ]);
}}

// ── RELATE entailment chart (at contradiction_threshold=0.50) ──
const relateEntSlice = relateData.filter(r => r.contradiction_threshold == 0.50);
if (relateEntSlice.length) {{
  const labels = relateEntSlice.map(r => r.entailment_threshold);
  barChart("chart-relate-ent", labels, [
    {{ label: "Support",       data: relateEntSlice.map(r => r.n_support),
       backgroundColor: "#4caf50" }},
    {{ label: "Scope qualify", data: relateEntSlice.map(r => r.n_scope_qualify),
       backgroundColor: "#2196f3" }},
    {{ label: "Contradict",    data: relateEntSlice.map(r => r.n_contradict),
       backgroundColor: "#f44336" }},
    {{ label: "Unrelated",     data: relateEntSlice.map(r => r.n_unrelated),
       backgroundColor: "#bdbdbd" }},
  ]);
}}

// ── RELATE contradiction chart (at entailment_threshold=0.50) ──
const relateConSlice = relateData.filter(r => r.entailment_threshold == 0.50);
if (relateConSlice.length) {{
  const labels = relateConSlice.map(r => r.contradiction_threshold);
  barChart("chart-relate-con", labels, [
    {{ label: "Contradict",    data: relateConSlice.map(r => r.n_contradict),
       backgroundColor: "#f44336" }},
    {{ label: "Support",       data: relateConSlice.map(r => r.n_support),
       backgroundColor: "#4caf50" }},
    {{ label: "Scope qualify", data: relateConSlice.map(r => r.n_scope_qualify),
       backgroundColor: "#2196f3" }},
    {{ label: "Unrelated",     data: relateConSlice.map(r => r.n_unrelated),
       backgroundColor: "#bdbdbd" }},
  ]);
}}
</script>
</body>
</html>
"""


# ── Table & best-box builders ─────────────────────────────────────────────────

def _prf_table(rows: list[dict], label_key: str, label_header: str) -> str:
    if not rows:
        return '<p class="no-data">No data — run the sweep first.</p>'
    best = max(rows, key=lambda r: float(r.get("f1", 0)))
    cells = []
    for r in rows:
        is_best = r[label_key] == best[label_key]
        cls = ' class="best-row"' if is_best else ""
        cells.append(
            f"<tr{cls}>"
            f"<td>{r[label_key]}</td>"
            f"<td>{float(r.get('precision',0)):.3f}</td>"
            f"<td>{float(r.get('recall',0)):.3f}</td>"
            f"<td>{float(r.get('f1',0)):.3f}</td>"
            f"<td>{float(r.get('strict_f1',0)):.3f}</td>"
            f"<td>{r.get('n_pipeline','')}</td>"
            "</tr>"
        )
    return (
        "<table>"
        f"<tr><th>{label_header}</th><th>Precision</th><th>Recall</th>"
        f"<th>F1</th><th>Strict F1</th><th>Pipeline N</th></tr>"
        + "".join(cells)
        + "</table>"
    )


def _best_box(rows: list[dict], label_key: str) -> str:
    if not rows:
        return ""
    best = max(rows, key=lambda r: float(r.get("f1", 0)))
    return (
        f'<div class="best-box">'
        f"Best F1: <strong>{label_key}={best[label_key]}</strong> &nbsp; "
        f"F1={float(best['f1']):.3f} &nbsp; "
        f"Precision={float(best.get('precision',0)):.3f} &nbsp; "
        f"Recall={float(best.get('recall',0)):.3f}"
        f"</div>"
    )


def _relate_table(rows: list[dict], fix_key: str, fix_val: str,
                  vary_key: str, vary_header: str) -> str:
    slice_ = [r for r in rows if r.get(fix_key) == fix_val]
    if not slice_:
        return '<p class="no-data">No data — run relate sweep first.</p>'
    cells = []
    for r in slice_:
        cells.append(
            f"<tr>"
            f"<td>{r[vary_key]}</td>"
            f"<td>{r.get('n_support','')}</td>"
            f"<td>{r.get('n_scope_qualify','')}</td>"
            f"<td>{r.get('n_contradict','')}</td>"
            f"<td>{r.get('n_unrelated','')}</td>"
            f"<td>{r.get('n_total_pairs','')}</td>"
            "</tr>"
        )
    return (
        "<table>"
        f"<tr><th>{vary_header}</th><th>Support</th><th>Scope qualify</th>"
        f"<th>Contradict</th><th>Unrelated</th><th>Total pairs</th></tr>"
        + "".join(cells)
        + "</table>"
    )


# ── JSON serialiser for chart data ────────────────────────────────────────────

def _to_chart_json(rows: list[dict], numeric_keys: list[str]) -> str:
    out = []
    for r in rows:
        item = {}
        for k, v in r.items():
            if k in numeric_keys:
                try:
                    item[k] = float(v)
                except (ValueError, TypeError):
                    item[k] = None
            else:
                item[k] = v
        out.append(item)
    return json.dumps(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep dashboard HTML")
    parser.add_argument("--reports",  default=str(REPORTS_DIR))
    parser.add_argument("--out",      default=None,
                        help="Output HTML path (default: eval/reports/dashboard_<ts>.html)")
    parser.add_argument("--grounding", default=None, help="Path to grounding_sweep CSV")
    parser.add_argument("--sweep",     default=None, help="Path to eval matching sweep CSV")
    parser.add_argument("--relate",    default=None, help="Path to relate_sweep CSV")
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # ── Load CSVs ──
    def _load(explicit: str | None, pattern: str, label: str) -> list[dict]:
        path = Path(explicit) if explicit else _latest_matching(reports_dir, pattern)
        if path is None or not path.exists():
            print(f"[dashboard] No {label} CSV found — section will be empty.", file=sys.stderr)
            return []
        print(f"[dashboard] Loading {label}: {path}")
        return _load_csv(path)

    eval_rows      = _load(args.sweep,     "sweep_*.csv",          "eval-matching sweep")
    grounding_rows = _load(args.grounding, "grounding_sweep_*.csv","grounding sweep")
    relate_rows    = _load(args.relate,    "relate_sweep_*.csv",   "relate sweep")

    # Filter eval rows to a single embedder if both present (prefer gemini)
    if eval_rows and "embedder" in eval_rows[0]:
        gemini_rows = [r for r in eval_rows if r.get("embedder") == "gemini"]
        eval_rows = gemini_rows if gemini_rows else eval_rows

    split         = grounding_rows[0].get("split", "dev") if grounding_rows else "dev"
    sim_threshold = grounding_rows[0].get("sim_threshold", "—") if grounding_rows else "—"

    # ── Build HTML fragments ──
    eval_best_box     = _best_box(eval_rows, "threshold")
    grounding_best_box = _best_box(grounding_rows, "grounding_threshold")

    eval_table      = _prf_table(eval_rows, "threshold", "Threshold")
    grounding_table = _prf_table(grounding_rows, "grounding_threshold", "Grounding thr.")
    relate_ent_table = _relate_table(relate_rows, "contradiction_threshold", "0.5",
                                     "entailment_threshold", "Ent. thr.")
    relate_con_table = _relate_table(relate_rows, "entailment_threshold", "0.5",
                                     "contradiction_threshold", "Con. thr.")

    prf_keys = ["precision", "recall", "f1", "strict_f1"]
    eval_json      = _to_chart_json(eval_rows,      prf_keys + ["threshold"])
    grounding_json = _to_chart_json(grounding_rows, prf_keys + ["grounding_threshold"])
    relate_json    = _to_chart_json(
        relate_rows,
        ["entailment_threshold", "contradiction_threshold",
         "n_support", "n_contradict", "n_scope_qualify", "n_unrelated", "n_total_pairs"],
    )

    html = _HTML.format(
        timestamp=timestamp,
        split=split,
        sim_threshold=sim_threshold,
        eval_best_box=eval_best_box,
        grounding_best_box=grounding_best_box,
        eval_table=eval_table,
        grounding_table=grounding_table,
        relate_ent_table=relate_ent_table,
        relate_con_table=relate_con_table,
        eval_json=eval_json,
        grounding_json=grounding_json,
        relate_json=relate_json,
    )

    out_path = Path(args.out) if args.out else reports_dir / f"dashboard_{timestamp}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard → {out_path}")
    print(f"Open with:  open {out_path}")


if __name__ == "__main__":
    main()
