"""
viewer_server.py
────────────────
Live Flask server that serves the pipeline inspector from PostgreSQL.
No static HTML regeneration needed — pages render fresh on every request.

Usage
-----
    PYTHONPATH=. python scripts/inspect/viewer_server.py
    PYTHONPATH=. python scripts/inspect/viewer_server.py --port 8080
    PYTHONPATH=. python scripts/inspect/viewer_server.py --host 0.0.0.0 --port 5000

Then open http://localhost:5000 in your browser.

Routes
------
  GET /                         Index page — all runs grouped by PMCID
  GET /paper/<pmcid>            Redirect to latest successful run for that PMCID
  GET /paper/<pmcid>/<run_id>   Per-paper inspector (rendered from DB)
  GET /api/runs                 JSON list of all runs (newest first)
  GET /api/paper/<pmcid>/<run_id>  JSON raw data dict for one run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from flask import Flask, Response, abort, redirect, url_for
except ImportError:
    print("ERROR: flask is required.  pip install flask", file=sys.stderr)
    sys.exit(1)

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("ERROR: jinja2 is required.  pip install jinja2", file=sys.stderr)
    sys.exit(1)

# ── Bootstrap path so we can import from project root ─────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from inspect_pipeline_output import (  # noqa: E402
    build_context_from_db,
    build_corpus_index,
    build_corpus_index_from_db,
    _compute_flags,
    _load_corpus_relations,
)

app = Flask(__name__)
_LOW_GS_THRESHOLD = 0.4

# Corpus index built once at startup from corpus_relations.json (if found).
# Reloaded via GET /admin/reload-corpus without restarting the server.
_corpus_index: dict = {}


def _load_corpus_index(path: Path | None) -> dict:
    if path is None:
        return {}
    data = _load_corpus_relations(path)
    if data is None:
        return {}
    n = data.get("total_relations", "?")
    cp = data.get("cross_paper_count", "?")
    print(f"Corpus relations loaded: {n} total, {cp} cross-paper from {path}")
    return build_corpus_index(data)


_CORPUS_RELATIONS_PATH: Path | None = None


# ── Jinja2 env (reuse the same templates as the static generator) ─────────────

def _make_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=True,
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_db():
    from database import get_db_connection  # noqa: PLC0415
    return get_db_connection()


def _index_rows_from_db(session) -> list[dict]:
    """
    Return one row per (pmcid, run) for the index page, newest run first.
    Counts come from the DB directly (no JSON loading).
    """
    from sqlalchemy import func  # noqa: PLC0415
    from database.models import (  # noqa: PLC0415
        PipelineRun, SumFinalRule, SumCanonicalRule,
        SumRelation, SumMapFinding,
    )

    # All runs newest-first
    runs = (
        session.query(PipelineRun)
        .order_by(PipelineRun.started_at.desc())
        .all()
    )

    # Counts per pipeline_run_id — one query each (fast with indexed FK)
    def _counts(model, id_col):
        rows = (
            session.query(id_col, func.count().label("n"))
            .group_by(id_col)
            .all()
        )
        return {r[0]: r[1] for r in rows}

    final_counts  = _counts(SumFinalRule,    SumFinalRule.pipeline_run_id)
    canon_counts  = _counts(SumCanonicalRule, SumCanonicalRule.pipeline_run_id)
    rel_counts    = _counts(SumRelation,      SumRelation.pipeline_run_id)
    map_counts    = _counts(SumMapFinding,    SumMapFinding.pipeline_run_id)

    # Contradicted / flagged final-rule counts
    contradicted_by_run: dict[int, int] = {}
    flagged_by_run: dict[int, int] = {}
    fr_rows = (
        session.query(
            SumFinalRule.pipeline_run_id,
            SumFinalRule.is_contradicted,
            SumFinalRule.subject_entity,
            SumFinalRule.outcome_entity,
            SumFinalRule.predicate_text,
        ).all()
    )
    for run_id_fk, is_contradicted, subj, outc, pred in fr_rows:
        if is_contradicted:
            contradicted_by_run[run_id_fk] = contradicted_by_run.get(run_id_fk, 0) + 1
        fr_dict = {
            "subject_entity": subj,
            "outcome_entity": outc,
            "predicate_text": pred,
        }
        if _compute_flags(fr_dict, _LOW_GS_THRESHOLD):
            flagged_by_run[run_id_fk] = flagged_by_run.get(run_id_fk, 0) + 1

    index_rows = []
    for run in runs:
        rid = run.id
        contradicted = contradicted_by_run.get(rid, 0)
        flagged      = flagged_by_run.get(rid, 0)
        low_gs       = 0  # too expensive to compute inline; skipped in live view
        warnings: list[str] = []
        if flagged:
            warnings.append(f"{flagged} flagged")
        if contradicted:
            warnings.append(f"{contradicted} contradicted")

        index_rows.append({
            "pmcid":        run.pmcid,
            "run_id":       run.run_id,
            "html_path":    f"/paper/{run.pmcid}/{run.run_id}",
            "status":       run.status,
            "started_at":   run.started_at.isoformat() if run.started_at else "",
            "finished_at":  run.finished_at.isoformat() if run.finished_at else "",
            "map_findings": map_counts.get(rid, 0),
            "canonical":    canon_counts.get(rid, 0),
            "final":        final_counts.get(rid, 0),
            "relations":    rel_counts.get(rid, 0),
            "flagged":      flagged,
            "contradicted": contradicted,
            "low_gs":       low_gs,
            "warnings":     " · ".join(warnings),
        })

    return index_rows


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    db = _get_db()
    with db.session_scope() as session:
        rows = _index_rows_from_db(session)

    env = _make_env()
    template = env.get_template("pipeline_batch_index.html.jinja2")
    html = template.render(
        rows=rows,
        source_dir="PostgreSQL",
        total=len(rows),
        corpus_data=None,
        live_mode=True,
    )
    return Response(html, mimetype="text/html")


@app.route("/paper/<pmcid>")
def paper_latest(pmcid: str):
    """Redirect to the latest successful run for this PMCID."""
    from database.models import PipelineRun  # noqa: PLC0415
    db = _get_db()
    with db.session_scope() as session:
        run = (
            session.query(PipelineRun)
            .filter_by(pmcid=pmcid, status="success")
            .order_by(PipelineRun.started_at.desc())
            .first()
        )
        if run is None:
            # Fall back to any run for this PMCID
            run = (
                session.query(PipelineRun)
                .filter_by(pmcid=pmcid)
                .order_by(PipelineRun.started_at.desc())
                .first()
            )
        if run is None:
            abort(404, description=f"No runs found for PMCID {pmcid!r}")
        run_id = run.run_id

    return redirect(url_for("paper_run", pmcid=pmcid, run_id=run_id))


@app.route("/paper/<pmcid>/<run_id>")
def paper_run(pmcid: str, run_id: str):
    """Per-paper inspector rendered live from DB."""
    db = _get_db()
    with db.session_scope() as session:
        # Prefer DB corpus index; fall back to JSON-loaded index if DB is empty.
        corpus_connections = build_corpus_index_from_db(session) or _corpus_index or None
        ctx = build_context_from_db(
            pmcid, run_id, session,
            low_gs_threshold=_LOW_GS_THRESHOLD,
            corpus_connections=corpus_connections,
        )

    if ctx is None:
        abort(404, description=f"Run {run_id!r} not found for PMCID {pmcid!r}")

    # Inject available run list for the run-selector dropdown
    from database.models import PipelineRun  # noqa: PLC0415
    db = _get_db()
    with db.session_scope() as session:
        all_runs = (
            session.query(PipelineRun.run_id, PipelineRun.status, PipelineRun.started_at)
            .filter_by(pmcid=pmcid)
            .order_by(PipelineRun.started_at.desc())
            .all()
        )
    ctx["available_runs"] = [
        {
            "run_id":     r.run_id,
            "status":     r.status,
            "started_at": r.started_at.isoformat() if r.started_at else "",
        }
        for r in all_runs
    ]
    ctx["current_run_id"] = run_id
    ctx["live_mode"] = True

    env = _make_env()
    template = env.get_template("pipeline_inspector.html.jinja2")
    html = template.render(**ctx, diff_mode=False)
    return Response(html, mimetype="text/html")


@app.route("/admin/reload-corpus")
def reload_corpus():
    """Reload corpus_relations from DB (and JSON fallback) without restarting."""
    global _corpus_index
    _corpus_index = _load_corpus_index(_CORPUS_RELATIONS_PATH)
    # Also check DB so next paper_run request gets fresh data
    db = _get_db()
    with db.session_scope() as session:
        db_index = build_corpus_index_from_db(session)
    n_db  = len(db_index)
    n_json = len(_corpus_index)
    return Response(
        f"<p>Reloaded. DB: {n_db} canonical rule(s) with cross-paper connections. "
        f"JSON fallback: {n_json}.</p>"
        f'<a href="/">← Back to index</a>',
        mimetype="text/html",
    )


# ── JSON API ───────────────────────────────────────────────────────────────────

@app.route("/api/runs")
def api_runs():
    """Return all pipeline runs as JSON (newest first)."""
    from database.models import PipelineRun  # noqa: PLC0415
    db = _get_db()
    with db.session_scope() as session:
        runs = (
            session.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .all()
        )
        payload = [
            {
                "id":          r.id,
                "run_id":      r.run_id,
                "pmcid":       r.pmcid,
                "status":      r.status,
                "started_at":  r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "error":       r.error,
            }
            for r in runs
        ]
    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        mimetype="application/json",
    )


@app.route("/api/paper/<pmcid>/<run_id>")
def api_paper(pmcid: str, run_id: str):
    """Return the raw reconstructed data dict for one run as JSON."""
    from inspect_pipeline_output import build_data_from_db  # noqa: PLC0415
    db = _get_db()
    with db.session_scope() as session:
        data = build_data_from_db(pmcid, run_id, session)
    if data is None:
        abort(404, description=f"Run {run_id!r} not found for PMCID {pmcid!r}")
    return Response(
        json.dumps(data, indent=2, ensure_ascii=False),
        mimetype="application/json",
    )


# ── Error pages ────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return Response(
        f"<h2>404 Not Found</h2><p>{e.description}</p>"
        f'<a href="/">← Back to index</a>',
        status=404,
        mimetype="text/html",
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    global _corpus_index, _CORPUS_RELATIONS_PATH

    parser = argparse.ArgumentParser(
        description="Live pipeline inspector server (reads from PostgreSQL).",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode (auto-reload on code changes)")
    parser.add_argument("--corpus-relations", metavar="PATH",
                        help="Path to corpus_relations.json for cross-paper connections. "
                             "Auto-detected from out/summaries/ if not provided.")
    args = parser.parse_args()

    # ── Resolve corpus_relations.json ─────────────────────────────────────
    if args.corpus_relations:
        _CORPUS_RELATIONS_PATH = Path(args.corpus_relations)
    else:
        # Auto-detect common locations relative to project root
        candidates = [
            _ROOT / "out" / "summaries" / "corpus_relations.json",
            _ROOT / "out" / "summaries" / "summaries" / "corpus_relations.json",
            _ROOT / "corpus_relations.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                _CORPUS_RELATIONS_PATH = candidate
                break

    if _CORPUS_RELATIONS_PATH:
        _corpus_index = _load_corpus_index(_CORPUS_RELATIONS_PATH)
    else:
        print("No corpus_relations.json found — cross-paper connections disabled. "
              "Pass --corpus-relations PATH to enable.")

    print(f"Starting viewer server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
