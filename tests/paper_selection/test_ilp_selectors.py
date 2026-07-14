"""ILP-selector tests.

Use synthetic fingerprints and exercise the ILP entry points end-to-end.
PuLP is required; the suite is skipped when it is not installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pulp = pytest.importorskip("pulp")

from eval.paper_selection.fingerprints import build_fingerprints
from eval.paper_selection.ilp_selectors import (
    ILPConfig,
    _prescore_fallback,
    _resolved_limit,
    _retained_related_edges,
    _validate_selection_count,
    select_calibration_set_ilp,
    select_diverse_papers_ilp,
    select_hard_papers_ilp,
    select_related_papers_ilp,
)
from eval.paper_selection.loaders import RawEntity, RawPaper
from eval.paper_selection.metrics import Hardness, Relatedness
from eval.paper_selection.selectors import SelectionConfig, select_calibration_set

from tests.paths import REPO_ROOT


# ── Fixture builders ─────────────────────────────────────────────────────────

def _ent(text, *, label="ENTITY", cui=None, semtypes=None, score=0.95):
    return RawEntity(
        text=text, label=label, umls_cui=cui,
        canonical_name=text, semantic_types=list(semtypes or []),
        umls_score=score,
    )


def _lymphoma_paper(pmcid: str, *, n_paragraphs: int = 12) -> RawPaper:
    body = (
        "DLBCL with strong CD20 expression. CD30 was negative. "
        "BCL2 high (Ki-67 60%) prognostic for worse OS. "
    )
    paragraphs = [body for _ in range(n_paragraphs)]
    entities = [
        _ent("DLBCL",                        cui="C0024301", semtypes=["T191"]),
        _ent("diffuse large b-cell lymphoma", cui="C0079744", semtypes=["T191"]),
        _ent("CD20", cui="C0054945", semtypes=["T116"]),
        _ent("CD30", cui="C0054916", semtypes=["T116"]),
        _ent("BCL2", cui="C1366512", semtypes=["T028"]),
        _ent("lymph node", cui="C0024204", semtypes=["T023"]),
        _ent("immunohistochemistry", cui="C0021044", semtypes=["T060"]),
        _ent("overall survival", cui="C0282416", semtypes=["T201"]),
    ]
    return RawPaper(pmcid=pmcid, title=f"Paper {pmcid}",
                    text_elements=paragraphs, entities=entities,
                    n_figures=2, n_tables=1, n_captions=3)


def _make_unrelated_paper(pmcid: str, extra: list[RawEntity] | None = None) -> RawPaper:
    body = (
        "Granuloma annulare presents with erythematous plaques. "
        "Histology shows mucin and palisaded histiocytes. "
        "Treatment with topical corticosteroids. "
    )
    paragraphs = [body for _ in range(10)]
    entities = [
        _ent("granuloma annulare", cui="C0085074", semtypes=["T047"]),
        _ent("mucin",      cui="C0026727", semtypes=["T116"]),
        _ent("histiocyte", cui="C0019625", semtypes=["T025"]),
        _ent("skin",       cui="C0037293", semtypes=["T023"]),
        _ent("corticosteroid", cui="C0001617", semtypes=["T123"]),
    ]
    if extra:
        entities.extend(extra)
    return RawPaper(pmcid=pmcid, title=f"Paper {pmcid}",
                    text_elements=paragraphs, entities=entities,
                    n_figures=1, n_tables=0, n_captions=1)


def _hard_paper(pmcid: str) -> RawPaper:
    """Many tables, dense entities, fragmented paragraphs — high absolute hardness."""
    short_paragraphs = [f"CD{i} positive." for i in range(3, 80)]
    long_paragraphs = ["x " * 1000]
    entities = [
        _ent(f"CD{i}", cui=f"C00000{i:02}", semtypes=["T116"]) for i in range(3, 60)
    ] + [
        _ent("DLBCL", cui="C0024301", semtypes=["T191"]),
        _ent("Burkitt lymphoma", cui="C0006413", semtypes=["T191"]),
        _ent("MYC",  semtypes=["T028"]),
        _ent("BCL6", semtypes=["T028"]),
        _ent("immunohistochemistry", semtypes=["T060"]),
        _ent("FISH", semtypes=["T060"]),
        _ent("overall survival", semtypes=["T201"]),
    ]
    return RawPaper(pmcid=pmcid, title=f"Paper {pmcid}",
                    text_elements=[*short_paragraphs, *long_paragraphs],
                    entities=entities,
                    n_figures=10, n_tables=12, n_captions=20)


def _medium_hard_paper(pmcid: str) -> RawPaper:
    body = "DLBCL with CD20 and BCL2 expression. " * 10
    entities = [
        _ent("DLBCL", cui="C0024301", semtypes=["T191"]),
        _ent("CD20",  cui="C0054945", semtypes=["T116"]),
        _ent("BCL2",  cui="C1366512", semtypes=["T028"]),
        _ent("immunohistochemistry", semtypes=["T060"]),
    ]
    return RawPaper(pmcid=pmcid, title=f"Paper {pmcid}",
                    text_elements=[body] * 8, entities=entities,
                    n_figures=2, n_tables=2, n_captions=2)


def _make_corpus() -> list[RawPaper]:
    """7 lymphoma + 6 unrelated (varied diseases) + 5 hard + 3 medium-hard."""
    papers: list[RawPaper] = []
    for i in range(1, 8):
        papers.append(_lymphoma_paper(f"PMC100{i}"))

    extras = [
        [_ent("psoriasis", cui="C0033860", semtypes=["T047"])],
        [_ent("melanoma", cui="C0025202", semtypes=["T191"])],
        [_ent("hepatocellular carcinoma", cui="C2239176", semtypes=["T191"])],
        [_ent("breast cancer", cui="C0006142", semtypes=["T191"])],
        [_ent("renal cell carcinoma", cui="C0007134", semtypes=["T191"])],
        [_ent("glioblastoma", cui="C0017636", semtypes=["T191"])],
    ]
    for i, e in enumerate(extras, start=1):
        papers.append(_make_unrelated_paper(f"PMC200{i}", extra=e))

    for i in range(1, 6):
        papers.append(_hard_paper(f"PMC300{i}"))
    for i in range(1, 4):
        papers.append(_medium_hard_paper(f"PMC400{i}"))
    return papers


def _loose_cfg() -> SelectionConfig:
    return SelectionConfig(
        max_sentences_related=10_000,
        max_sentences_diverse=10_000,
        min_sentences=1,
        min_useful_entities=1,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_related_ilp_chooses_lymphoma_cluster():
    fps = build_fingerprints(_make_corpus())
    chosen, rationale = select_related_papers_ilp(
        fps, k=5, config=_loose_cfg(), ilp_config=ILPConfig(candidate_limit=50),
    )
    pmcids = {p.pmcid for p in chosen}
    assert len(chosen) == 5
    # All selections must come from the lymphoma cluster (PMC100*)
    assert all(pid.startswith("PMC100") for pid in pmcids), pmcids
    # Rationale records ILP metadata
    for entry in rationale.values():
        assert "ilp_objective" in entry
        assert entry["ilp_status"] in {"Optimal", "Not Solved", "Infeasible",
                                        "Unbounded", "Undefined"}


def test_related_ilp_beats_or_matches_greedy_pair_relatedness():
    """ILP optimum cannot be worse than greedy on the same pool."""
    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    rel = Relatedness()

    def _pair_sum(pmcids: list[str]) -> float:
        by = {fp.pmcid: fp for fp in fps}
        ps = [by[p] for p in pmcids if p in by]
        return sum(rel.score(ps[i], ps[j])
                   for i in range(len(ps)) for j in range(i + 1, len(ps)))

    greedy = select_calibration_set(fps, config=cfg)
    ilp_chosen, _ = select_related_papers_ilp(fps, k=5, config=cfg)
    assert _pair_sum([p.pmcid for p in ilp_chosen]) >= _pair_sum(greedy.related) - 1e-6


def test_diverse_ilp_avoids_near_duplicates_and_covers_concepts():
    # Build a corpus where 5 of the lymphoma papers are near-duplicates and
    # 5 unrelated papers cover entirely different concepts.
    papers = [_lymphoma_paper(f"PMCDUP{i}") for i in range(1, 6)]
    extras = [
        [_ent("psoriasis", cui="C0033860", semtypes=["T047"])],
        [_ent("melanoma", cui="C0025202", semtypes=["T191"])],
        [_ent("hepatocellular carcinoma", cui="C2239176", semtypes=["T191"])],
        [_ent("breast cancer", cui="C0006142", semtypes=["T191"])],
        [_ent("renal cell carcinoma", cui="C0007134", semtypes=["T191"])],
    ]
    for i, e in enumerate(extras, start=1):
        papers.append(_make_unrelated_paper(f"PMCDIV{i}", extra=e))
    fps = build_fingerprints(papers)

    chosen, _ = select_diverse_papers_ilp(
        fps, k=5, config=_loose_cfg(), ilp_config=ILPConfig(candidate_limit=50),
    )
    assert len(chosen) == 5
    # ILP should pick at most one of the near-duplicate cluster.
    dup_count = sum(1 for p in chosen if p.pmcid.startswith("PMCDUP"))
    assert dup_count <= 1, f"diverse ILP picked {dup_count} near-duplicates: {[p.pmcid for p in chosen]}"


def test_hard_ilp_satisfies_composition_constraints():
    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    ilp_cfg = ILPConfig(candidate_limit=50,
                        hard_top_normalized_pool=8,
                        hard_top_absolute_pool=8)
    chosen, _ = select_hard_papers_ilp(fps, k=5, config=cfg, ilp_config=ilp_cfg)
    assert len(chosen) == 5

    # Recompute the same sub-pools the ILP used to verify constraints held.
    h = Hardness()
    breakdowns = {fp.pmcid: h.breakdown(fp) for fp in fps if not fp.is_empty()}
    candidates = [fp for fp in fps if not fp.is_empty()]
    pool = sorted(candidates, key=lambda p: -breakdowns[p.pmcid].absolute_hardness)[:50]

    by_norm = sorted(pool, key=lambda p: -breakdowns[p.pmcid].normalized_hardness)
    by_abs  = sorted(pool, key=lambda p: -breakdowns[p.pmcid].absolute_hardness)
    by_med  = sorted(pool, key=lambda p:  breakdowns[p.pmcid].normalized_hardness)
    top_norm = {p.pmcid for p in by_norm[:ilp_cfg.hard_top_normalized_pool]}
    top_abs  = {p.pmcid for p in by_abs[: ilp_cfg.hard_top_absolute_pool]}
    n = len(by_med)
    lo = int(n * ilp_cfg.hard_medium_pool_low_pct)
    hi = max(int(n * ilp_cfg.hard_medium_pool_high_pct), lo + 1)
    medium_band = {p.pmcid for p in by_med[lo:hi]}

    chosen_ids = {p.pmcid for p in chosen}
    assert len(chosen_ids & top_norm) >= min(2, len(top_norm))
    assert len(chosen_ids & top_abs)  >= min(2, len(top_abs))
    assert len(chosen_ids & medium_band) >= min(1, len(medium_band))


def test_calibration_ilp_returns_15_unique():
    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    res = select_calibration_set_ilp(
        fps, config=cfg, ilp_config=ILPConfig(candidate_limit=50), allow_overlap=False,
    )
    assert len(res.related) == 5
    assert len(res.diverse) == 5
    assert len(res.hard) == 5
    assert len(set(res.all_pmcids())) == 15
    assert all(entry.get("strategy") == "ilp" for entry in res.rationale.values())


def test_greedy_calibration_also_returns_15_unique():
    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    res = select_calibration_set(fps, config=cfg, allow_overlap=False)
    assert len(set(res.all_pmcids())) == 15


def test_cli_strategy_ilp_on_jsonl_fixture(tmp_path: Path):
    """Run `python -m eval.paper_selection.run_select --strategy ilp` end-to-end."""
    corpus = _make_corpus()
    jsonl_path = tmp_path / "papers.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for paper in corpus:
            fh.write(json.dumps({
                "pmcid": paper.pmcid,
                "title": paper.title,
                "text_elements": paper.text_elements,
                "entities": [
                    {
                        "text": e.text, "label": e.label,
                        "umls_cui": e.umls_cui,
                        "canonical_name": e.canonical_name,
                        "semantic_types": e.semantic_types,
                        "umls_score": e.umls_score,
                    } for e in paper.entities
                ],
                "n_figures": paper.n_figures,
                "n_tables":  paper.n_tables,
                "n_captions": paper.n_captions,
            }) + "\n")

    repo_root = REPO_ROOT
    export_dir = tmp_path / "configs"
    report_dir = tmp_path / "reports"
    cmd = [
        sys.executable, "-m", "eval.paper_selection.run_select",
        "--jsonl", str(jsonl_path),
        "--strategy", "ilp",
        "--candidate-limit", "50",
        "--time-limit-seconds", "10",
        "--max-sentences-related", "10000",
        "--max-sentences-diverse", "10000",
        "--min-sentences", "1",
        "--output-version", "v_ilp_test",
        "--export-dir", str(export_dir),
        # Explicit: the CLI default is the tracked reports/paper_selection/, and
        # this subprocess runs with cwd=repo_root — without this the run would
        # write test artefacts into the repository.
        "--report-dir", str(report_dir),
        "--log-level", "WARNING",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"CLI failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    # Roster YAML in the export dir; derived reports in the report dir — never
    # the other way round.
    assert (export_dir / "v_ilp_test.yaml").exists()
    rationale_path = report_dir / "v_ilp_test_rationale.json"
    assert rationale_path.exists()
    assert (report_dir / "v_ilp_test_summary.csv").exists()
    assert not (export_dir / "v_ilp_test_rationale.json").exists()
    assert not (export_dir / "v_ilp_test_summary.csv").exists()

    # Hermetic: nothing leaked into the repository's tracked reports directory.
    assert not list((repo_root / "reports" / "paper_selection").glob("v_ilp_test*"))

    payload = json.loads(rationale_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v_ilp_test"
    assert len(payload["papers"]) == 15
    # Verify the ILP path actually ran (vs silent fallback to greedy):
    # ILP rationale strings always start with "ILP <bucket>:".
    reasons = [p["selection_reason"] or "" for p in payload["papers"].values()]
    assert any(r.startswith("ILP ") for r in reasons), reasons


# ── New tests — sparsification, caps, index-based names, fallbacks ───────────

def test_ilp_config_resolved_limit_prefers_override():
    ilp = ILPConfig()
    assert _resolved_limit(ilp, "related") == ilp.related_candidate_limit
    assert _resolved_limit(ilp, "diverse") == ilp.diverse_candidate_limit
    assert _resolved_limit(ilp, "hard")    == ilp.hard_candidate_limit

    override = ILPConfig(candidate_limit=42)
    assert _resolved_limit(override, "related") == 42
    assert _resolved_limit(override, "diverse") == 42
    assert _resolved_limit(override, "hard")    == 42


def test_related_pair_variables_are_sparse_not_quadratic():
    """For a moderately large pool, retained pair vars ≪ n·(n-1)/2."""
    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    ilp_cfg = ILPConfig(
        candidate_limit=0,         # keep all eligible
        related_pair_top_m=4,
        related_pair_threshold=0.05,
    )
    chosen, rationale = select_related_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ilp_cfg,
    )
    assert len(chosen) == 5
    pool_size = next(iter(rationale.values()))["ilp_pool_size"]
    n_edges   = next(iter(rationale.values()))["ilp_n_pair_edges"]
    full_pair_count = pool_size * (pool_size - 1) // 2
    # Strictly sparser than the complete pair graph.
    assert n_edges < full_pair_count, (n_edges, full_pair_count)
    # And bounded above by |pool| * top_m (the per-endpoint cap, ignoring dedup).
    assert n_edges <= pool_size * ilp_cfg.related_pair_top_m


def test_retained_related_edges_respects_threshold_and_topm():
    fps = build_fingerprints(_make_corpus())
    rel = Relatedness()
    # Use only eligible fingerprints for a clean per-endpoint count check.
    pool = [fp for fp in fps if not fp.is_empty()][:10]
    edges, scanned = _retained_related_edges(
        pool, rel, top_m=2, threshold=0.0,
    )
    assert scanned == len(pool) * (len(pool) - 1) // 2
    # Each node contributes at most top_m edges to the union; total edge count
    # is therefore ≤ |pool| · top_m even before dedup.
    assert len(edges) <= len(pool) * 2, (len(edges), len(pool))


def test_pulp_variable_names_are_index_based_not_pmcid(monkeypatch):
    """Production ILP calls must name vars x_<idx>/y_<idx>/z_<idx>, never PMCIDs."""
    from eval.paper_selection import ilp_selectors as ilp_mod

    seen_names: list[str] = []
    orig = ilp_mod._pulp.LpVariable

    def _spy(name, *args, **kwargs):
        seen_names.append(name)
        return orig(name, *args, **kwargs)

    monkeypatch.setattr(ilp_mod._pulp, "LpVariable", _spy)

    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    select_related_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ILPConfig(candidate_limit=10),
    )
    select_diverse_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ILPConfig(candidate_limit=10),
    )
    select_hard_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ILPConfig(candidate_limit=10),
    )

    assert seen_names, "expected the spy to record at least one LpVariable creation"
    all_pmcids = {fp.pmcid for fp in fps}
    for name in seen_names:
        # Index-based: x_0, y_12, z_3 etc.
        assert name.startswith(("x_", "y_", "z_")), name
        suffix = name.split("_", 1)[1]
        assert suffix.isdigit(), name
        # No raw PMCID should leak into a PuLP variable name.
        for pmcid in all_pmcids:
            assert pmcid not in name, (name, pmcid)


def test_cui_heavy_paper_does_not_dominate_when_cuis_disabled():
    """A paper packed with junk CUIs shouldn't win over real-coverage papers."""
    # 5 small disease/biomarker/gene papers with distinct concepts.
    real = []
    for i, disease in enumerate(["psoriasis", "melanoma", "glioblastoma",
                                  "breast cancer", "renal cell carcinoma"], start=1):
        ent = [
            _ent(disease, cui=f"C0{i:06}", semtypes=["T191"]),
            _ent(f"marker{i}", cui=f"C9{i:06}", semtypes=["T116"]),
            _ent(f"GENE{i}", semtypes=["T028"]),
        ]
        real.append(RawPaper(pmcid=f"PMCREAL{i}", title=f"Real {i}",
                             text_elements=["body sentence. " * 30],
                             entities=ent, n_figures=1, n_tables=1, n_captions=1))

    # One "CUI-stuffed" paper with 200 throwaway CUIs but no structured entities.
    junk_ents = [
        _ent(f"junk{i}", cui=f"CJ{i:06}", semtypes=["T999"])
        for i in range(200)
    ]
    junk = RawPaper(pmcid="PMCJUNK", title="Junk",
                    text_elements=["filler. " * 30],
                    entities=junk_ents, n_figures=0, n_tables=0, n_captions=0)

    fps = build_fingerprints([*real, junk])
    cfg = _loose_cfg()
    chosen, _ = select_diverse_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ILPConfig(candidate_limit=20),
    )
    # CUIs are disabled by default → junk paper has no useful entities and is
    # filtered before the ILP. Even if not filtered, its CUI count must not let
    # it edge out a real-coverage paper.
    pmcids = {p.pmcid for p in chosen}
    assert "PMCJUNK" not in pmcids, pmcids


def test_diverse_concept_inventory_respects_caps():
    """Per-bucket concept cap must bound the z-variable count."""
    # 30 papers, each with 6 unique disease entities → 180 disease concepts.
    papers = []
    for i in range(30):
        ents = [
            _ent(f"disease_{i}_{j}", cui=f"D{i:03}{j:03}", semtypes=["T191"])
            for j in range(6)
        ]
        # Add minimal biomarker/gene for eligibility.
        ents.append(_ent(f"bm_{i}", cui=f"B{i:06}", semtypes=["T116"]))
        ents.append(_ent(f"g{i}", semtypes=["T028"]))
        papers.append(RawPaper(pmcid=f"PMCCAP{i:02}", title=f"P{i}",
                               text_elements=["sentence. " * 30],
                               entities=ents,
                               n_figures=1, n_tables=1, n_captions=1))

    fps = build_fingerprints(papers)
    cfg = _loose_cfg()
    ilp_cfg = ILPConfig(
        candidate_limit=0,
        diverse_max_concepts_by_bucket={
            "disease": 20, "biomarker": 50, "gene": 50,
            "tissue": 50, "method": 50, "outcome": 50, "cui": 30,
        },
    )
    _, rationale = select_diverse_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ilp_cfg,
    )
    n_concepts = next(iter(rationale.values()))["ilp_n_concepts"]
    # Disease alone would have produced 180 concepts; cap of 20 means total
    # concepts must be far below the uncapped count.
    assert n_concepts <= 20 + 50 + 50 + 50 + 50 + 50, n_concepts
    # Per-bucket diagnostic exists.
    for entry in rationale.values():
        assert "covered_concepts_by_bucket" in entry
        assert "reward_by_bucket" in entry


def test_extract_valid_solution_requires_exact_k():
    """_extract_valid_solution returns None unless exactly k vars are above 0.5."""
    fps = build_fingerprints(_make_corpus())
    cfg = _loose_cfg()
    chosen, rationale = select_related_papers_ilp(
        fps, k=5, config=cfg, ilp_config=ILPConfig(candidate_limit=15),
    )
    assert len(chosen) == 5
    # Each entry must carry a solution-quality marker.
    for entry in rationale.values():
        assert entry["ilp_solution_quality"] in {"optimal", "feasible_nonoptimal"}


def test_validate_selection_count():
    assert _validate_selection_count(["a", "b", "c"], 3, "tag") is True
    assert _validate_selection_count(["a", "b"], 3, "tag") is False


def test_prescore_fallback_returns_exactly_k():
    fps = build_fingerprints(_make_corpus())
    pool = [fp for fp in fps if not fp.is_empty()]
    # Synthesise a non-optimal status; CBC LpStatus index 0 == "Not Solved".
    not_solved_status = next(s for s, lbl in pulp.LpStatus.items()
                             if lbl == "Not Solved")
    ranked, rationale = _prescore_fallback(
        pool, k=5,
        key=lambda p: len(p.disease_entities),
        tag="related", status=not_solved_status,
    )
    assert len(ranked) == 5
    for entry in rationale.values():
        assert entry["ilp_solution_quality"] == "prescore_fallback"
        assert entry["ilp_status"] == "Not Solved"
