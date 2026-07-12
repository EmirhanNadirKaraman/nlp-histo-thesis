"""Lightweight tests for the offline paper-selection module.

All tests use synthetic ``RawPaper`` payloads — no DB, no LLM, no I/O beyond
the temp directory used for the export test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


from eval.paper_selection.export import write_calibration_set
from eval.paper_selection.fingerprints import (
    build_fingerprint,
    build_fingerprints,
)
from eval.paper_selection.loaders import RawEntity, RawPaper
from eval.paper_selection.metrics import (
    Diversity,
    Hardness,
    HardnessConfig,
    Relatedness,
)
from eval.paper_selection.models import PaperFingerprint
from eval.paper_selection.selectors import SelectionConfig, select_calibration_set


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ent(text, *, label="ENTITY", cui=None, semtypes=None, score=0.95):
    return RawEntity(
        text=text, label=label, umls_cui=cui,
        canonical_name=text, semantic_types=list(semtypes or []),
        umls_score=score,
    )


def _make_lymphoma_paper(pmcid: str, n_paragraphs: int = 12,
                         extra_text: str = "") -> RawPaper:
    body = (
        "DLBCL with strong CD20 expression. CD30 was negative. "
        "BCL2 high (Ki-67 60%) prognostic for worse OS. "
    )
    paragraphs = [body + extra_text for _ in range(n_paragraphs)]
    entities = [
        _ent("DLBCL",                  cui="C0024301", semtypes=["T191"]),
        _ent("diffuse large b-cell lymphoma", cui="C0079744", semtypes=["T191"]),
        _ent("CD20",  cui="C0054945", semtypes=["T116"]),
        _ent("CD30",  cui="C0054916", semtypes=["T116"]),
        _ent("BCL2",  cui="C1366512", semtypes=["T028"]),
        _ent("lymph node", cui="C0024204", semtypes=["T023"]),
        _ent("immunohistochemistry", cui="C0021044", semtypes=["T060"]),
        _ent("overall survival", cui="C0282416", semtypes=["T201"]),
    ]
    return RawPaper(
        pmcid=pmcid, title=f"Paper {pmcid}",
        text_elements=paragraphs, entities=entities,
        n_figures=2, n_tables=1, n_captions=3,
    )


def _make_unrelated_paper(pmcid: str) -> RawPaper:
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
    return RawPaper(
        pmcid=pmcid, title=f"Paper {pmcid}",
        text_elements=paragraphs, entities=entities,
        n_figures=1, n_tables=0, n_captions=1,
    )


def _make_hard_paper(pmcid: str) -> RawPaper:
    """Many tables, dense entities, fragmented paragraphs."""
    short_paragraphs = [f"CD{i} positive." for i in range(3, 80)]
    long_paragraphs = ["x " * 1000]
    entities = [
        _ent(f"CD{i}",  cui=f"C00000{i:02}", semtypes=["T116"]) for i in range(3, 60)
    ] + [
        _ent("DLBCL", cui="C0024301", semtypes=["T191"]),
        _ent("Burkitt lymphoma", cui="C0006413", semtypes=["T191"]),
        _ent("MYC",  semtypes=["T028"]),
        _ent("BCL6", semtypes=["T028"]),
        _ent("immunohistochemistry", semtypes=["T060"]),
        _ent("FISH", semtypes=["T060"]),
        _ent("overall survival", semtypes=["T201"]),
    ]
    return RawPaper(
        pmcid=pmcid, title=f"Paper {pmcid}",
        text_elements=[*short_paragraphs, *long_paragraphs],
        entities=entities,
        n_figures=10, n_tables=12, n_captions=20,
    )


def _make_corpus() -> list[RawPaper]:
    """Build a varied 18-paper synthetic pool: 7 lymphoma, 6 unrelated, 5 hard."""
    papers: list[RawPaper] = []
    for i in range(1, 8):
        papers.append(_make_lymphoma_paper(f"PMC100{i}"))
    for i in range(1, 7):
        # Different unrelated diseases per paper to keep diverse bucket varied.
        p = _make_unrelated_paper(f"PMC200{i}")
        if i == 1:
            p.entities.append(_ent("psoriasis", cui="C0033860", semtypes=["T047"]))
        if i == 2:
            p.entities.append(_ent("melanoma", cui="C0025202", semtypes=["T191"]))
            p.text_elements = [t.replace("granuloma annulare", "melanoma") for t in p.text_elements]
        if i == 3:
            p.entities.append(_ent("hepatocellular carcinoma", cui="C2239176", semtypes=["T191"]))
        if i == 4:
            p.entities.append(_ent("breast cancer", cui="C0006142", semtypes=["T191"]))
        if i == 5:
            p.entities.append(_ent("renal cell carcinoma", cui="C0007134", semtypes=["T191"]))
        papers.append(p)
    for i in range(1, 6):
        papers.append(_make_hard_paper(f"PMC300{i}"))
    return papers


# ── Tests ────────────────────────────────────────────────────────────────────

def test_fingerprint_creation_from_synthetic_paper():
    raw = _make_lymphoma_paper("PMC1")
    fp = build_fingerprint(raw)
    assert isinstance(fp, PaperFingerprint)
    assert fp.pmcid == "PMC1"
    assert fp.n_text_elements == 12
    assert fp.n_sentences > 0
    assert fp.n_chunks >= 1
    # NER buckets populated
    assert {"dlbcl"}.issubset({d.lower() for d in fp.disease_entities})
    assert "CD20" in fp.biomarker_entities
    assert "BCL2" in {g.upper() for g in fp.gene_entities} or "bcl2" in fp.gene_entities
    # Generic noise stripped from `entities`
    assert "patient" not in fp.entities
    # UMLS provenance preserved
    assert "C0024301" in fp.umls_cuis
    assert "T191" in fp.semantic_types


def test_relatedness_higher_for_shared_disease_and_markers():
    a = build_fingerprint(_make_lymphoma_paper("PMC1"))
    b = build_fingerprint(_make_lymphoma_paper("PMC2"))
    c = build_fingerprint(_make_unrelated_paper("PMC3"))
    rel = Relatedness()
    related_score   = rel.score(a, b)
    unrelated_score = rel.score(a, c)
    assert related_score > unrelated_score, (
        f"two lymphoma papers should be more related than lymphoma vs GA: "
        f"{related_score:.3f} ≤ {unrelated_score:.3f}"
    )
    # Shared-disease pair must clear the no-disease-overlap penalty band
    assert related_score > 0.05


def test_hardness_increases_with_density_and_layout():
    easy = build_fingerprint(_make_lymphoma_paper("PMC_easy"))
    hard = build_fingerprint(_make_hard_paper("PMC_hard"))
    h = Hardness()
    easy_b = h.breakdown(easy)
    hard_b = h.breakdown(hard)
    assert hard_b.normalized_hardness > easy_b.normalized_hardness, (
        f"dense/table-heavy paper must outrank simple paper: "
        f"easy={easy_b.normalized_hardness:.3f} hard={hard_b.normalized_hardness:.3f}"
    )
    assert hard_b.absolute_hardness > easy_b.absolute_hardness
    assert hard_b.layout_complexity > easy_b.layout_complexity
    assert any("table" in r or "fragmentation" in r for r in hard_b.reasons)


def test_diversity_selector_lowers_overlap():
    fps = build_fingerprints(_make_corpus())
    div = Diversity()
    # Sanity: a balanced 5-paper diverse subset must score higher than the
    # all-lymphoma 5-paper subset.
    lymphomas = [fp for fp in fps if fp.pmcid.startswith("PMC100")][:5]
    mixed = (
        [fp for fp in fps if fp.pmcid.startswith("PMC100")][:1]
        + [fp for fp in fps if fp.pmcid.startswith("PMC200")][:3]
        + [fp for fp in fps if fp.pmcid.startswith("PMC300")][:1]
    )
    assert div.score(mixed) > div.score(lymphomas)


def test_calibration_selector_returns_15_unique_papers():
    fps = build_fingerprints(_make_corpus())
    cfg = SelectionConfig(
        max_sentences_related=10_000,   # synthetic papers are short
        max_sentences_diverse=10_000,
        min_sentences=1,
        min_useful_entities=1,
    )
    res = select_calibration_set(fps, config=cfg, allow_overlap=False)
    assert len(res.related) == 5
    assert len(res.diverse) == 5
    assert len(res.hard)    == 5
    assert len(set(res.all_pmcids())) == 15


def test_export_writes_yaml_json_csv(tmp_path: Path):
    fps = build_fingerprints(_make_corpus())
    cfg = SelectionConfig(
        max_sentences_related=10_000,
        max_sentences_diverse=10_000,
        min_sentences=1,
        min_useful_entities=1,
    )
    res = select_calibration_set(fps, config=cfg, allow_overlap=False)
    paths = write_calibration_set(res, fps, version="v_test", export_dir=tmp_path)

    assert paths.yaml_path.exists() and paths.yaml_path.read_text(encoding="utf-8")
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v_test"
    assert set(payload["buckets"]) == {"related", "diverse", "hard"}
    assert len(payload["papers"]) == 15

    with paths.csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 15
    assert {r["bucket"] for r in rows} == {"related", "diverse", "hard"}


def test_export_splits_roster_and_reports(tmp_path: Path):
    """``report_dir`` routes rationale/summary away from the roster directory."""
    fps = build_fingerprints(_make_corpus())
    cfg = SelectionConfig(
        max_sentences_related=10_000,
        max_sentences_diverse=10_000,
        min_sentences=1,
        min_useful_entities=1,
    )
    res = select_calibration_set(fps, config=cfg, allow_overlap=False)

    roster_dir = tmp_path / "configs" / "paper_selection"
    report_dir = tmp_path / "reports" / "paper_selection"
    paths = write_calibration_set(
        res, fps, version="v_split", export_dir=roster_dir, report_dir=report_dir,
    )

    # Both destinations are created on demand.
    assert roster_dir.is_dir() and report_dir.is_dir()

    # Returned paths point at the split destinations; filenames are unchanged.
    assert paths.yaml_path == roster_dir / "v_split.yaml"
    assert paths.json_path == report_dir / "v_split_rationale.json"
    assert paths.csv_path == report_dir / "v_split_summary.csv"
    assert paths.yaml_path.exists()
    assert paths.json_path.exists()
    assert paths.csv_path.exists()

    # The derived reports never land beside the roster.
    assert not (roster_dir / "v_split_rationale.json").exists()
    assert not (roster_dir / "v_split_summary.csv").exists()

    # Contents keep the existing format.
    assert paths.yaml_path.read_text(encoding="utf-8")
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["version"] == "v_split"
    assert set(payload["buckets"]) == {"related", "diverse", "hard"}
    assert len(payload["papers"]) == 15

    with paths.csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 15
    assert {r["bucket"] for r in rows} == {"related", "diverse", "hard"}


def test_relatedness_set_means_satisfy_validation():
    """related set should have higher mean pairwise relatedness than diverse."""
    fps = build_fingerprints(_make_corpus())
    cfg = SelectionConfig(
        max_sentences_related=10_000,
        max_sentences_diverse=10_000,
        min_sentences=1,
        min_useful_entities=1,
    )
    res = select_calibration_set(fps, config=cfg)
    rel = Relatedness()
    by = {fp.pmcid: fp for fp in fps}

    def _pairwise_mean(pmcids):
        ps = [by[p] for p in pmcids if p in by]
        if len(ps) < 2:
            return 0.0
        scores = [rel.score(ps[i], ps[j])
                  for i in range(len(ps)) for j in range(i + 1, len(ps))]
        return sum(scores) / len(scores)

    related_mean = _pairwise_mean(res.related)
    diverse_mean = _pairwise_mean(res.diverse)
    assert related_mean > diverse_mean, (
        f"related ({related_mean:.3f}) should be more related than diverse ({diverse_mean:.3f})"
    )


def test_hardness_set_mean_exceeds_other_buckets():
    fps = build_fingerprints(_make_corpus())
    cfg = SelectionConfig(
        max_sentences_related=10_000,
        max_sentences_diverse=10_000,
        min_sentences=1,
        min_useful_entities=1,
    )
    res = select_calibration_set(fps, config=cfg)
    h = Hardness(HardnessConfig())
    by = {fp.pmcid: fp for fp in fps}

    def _mean_h(pmcids):
        return sum(h.breakdown(by[p]).normalized_hardness for p in pmcids) / max(len(pmcids), 1)

    assert _mean_h(res.hard) > _mean_h(res.related)
    assert _mean_h(res.hard) > _mean_h(res.diverse)


def test_no_api_imports_in_paper_selection():
    """Static guarantee — no LLM/HTTP modules pulled in by the package."""
    import importlib
    forbidden = {"openai", "anthropic", "google.generativeai", "langchain_openai",
                 "langchain_anthropic", "requests", "httpx"}
    for mod in [
        "eval.paper_selection.models",
        "eval.paper_selection.fingerprints",
        "eval.paper_selection.metrics",
        "eval.paper_selection.selectors",
        "eval.paper_selection.export",
        "eval.paper_selection.loaders",
    ]:
        m = importlib.import_module(mod)
        for f in forbidden:
            assert f not in str(m.__dict__), f"{mod} unexpectedly references {f}"
