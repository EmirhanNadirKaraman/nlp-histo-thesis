"""YAML / JSON / CSV writers for the selection result.

Three deliverables per run, named after ``--output-version``, split across two
destinations:

  - ``{version}.yaml``           — minimal pmcid-only roster; written into the
                                   *roster* directory ``export_dir``
                                   (CLI ``--export-dir``, default
                                   ``configs/paper_selection/``)
  - ``{version}_rationale.json`` — full rationale + breakdowns
  - ``{version}_summary.csv``    — flat one-row-per-paper summary

The latter two are derived reports and are written into ``report_dir``
(CLI ``--report-dir``, default ``reports/paper_selection/``).

``report_dir=None`` preserves the previous colocated behaviour: all three files
are written into ``export_dir``.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .metrics import Hardness, Relatedness
from .models import PaperFingerprint, SelectionResult

logger = logging.getLogger(__name__)


@dataclass
class ExportPaths:
    yaml_path: Path
    json_path: Path
    csv_path:  Path


def export_paths(
    export_dir: Path,
    version: str,
    report_dir: Path | None = None,
) -> ExportPaths:
    """Build the three output paths.

    ``export_dir`` holds the YAML roster; ``report_dir`` holds the rationale
    JSON and summary CSV. ``report_dir=None`` colocates all three in
    ``export_dir`` (previous behaviour).
    """
    reports = export_dir if report_dir is None else report_dir
    return ExportPaths(
        yaml_path=export_dir / f"{version}.yaml",
        json_path=reports / f"{version}_rationale.json",
        csv_path=reports / f"{version}_summary.csv",
    )


def _to_yaml(obj, indent: int = 0) -> str:
    """Tiny YAML emitter — avoids a hard PyYAML dependency.

    Supports the shapes we actually emit: dict / list / scalar.
    """
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_to_yaml(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_to_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(obj)}"


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    if any(ch in s for ch in ":#[]{},&*!|>'\"%@`") or s.strip() != s or s == "":
        return json.dumps(s, ensure_ascii=False)
    return s


# ── Public exporter ──────────────────────────────────────────────────────────

def write_calibration_set(
    result: SelectionResult,
    fingerprints: list[PaperFingerprint],
    *,
    version: str,
    export_dir: Path,
    report_dir: Path | None = None,
) -> ExportPaths:
    """Write the roster YAML into ``export_dir`` and the derived rationale JSON
    + summary CSV into ``report_dir``.

    ``report_dir=None`` keeps the previous behaviour (all three colocated in
    ``export_dir``). Both directories are created if missing.
    """
    paths = export_paths(export_dir, version, report_dir)
    for directory in {paths.yaml_path.parent, paths.json_path.parent}:
        directory.mkdir(parents=True, exist_ok=True)
    by_pmcid = {fp.pmcid: fp for fp in fingerprints}

    # ── YAML: roster only ───────────────────────────────────────────────────
    roster = {
        version: {
            "related": list(result.related),
            "diverse": list(result.diverse),
            "hard":    list(result.hard),
        },
    }
    paths.yaml_path.write_text(_to_yaml(roster) + "\n", encoding="utf-8")

    # ── JSON: full rationale + breakdowns ───────────────────────────────────
    relatedness = Relatedness()
    hardness = Hardness()

    pairwise = {}
    if result.related and len(result.related) > 1:
        related_fps = [by_pmcid[p] for p in result.related if p in by_pmcid]
        for i, a in enumerate(related_fps):
            for b in related_fps[i + 1:]:
                pairwise[f"{a.pmcid}|{b.pmcid}"] = round(relatedness.score(a, b), 4)

    rationale_payload = {
        "version": version,
        "buckets": {
            "related": list(result.related),
            "diverse": list(result.diverse),
            "hard":    list(result.hard),
        },
        "papers": {},
        "related_pairwise_relatedness": pairwise,
    }

    for pmcid in result.all_pmcids():
        fp = by_pmcid.get(pmcid)
        bucket_info = result.rationale.get(pmcid, {})
        bucket = bucket_info.get("bucket")
        entry: dict = {
            "bucket": bucket,
            "title":  (fp.title if fp else None),
            "n_sentences": (fp.n_sentences if fp else None),
            "n_chunks":    (fp.n_chunks if fp else None),
            "n_text_elements": (fp.n_text_elements if fp else None),
            "n_tables":   (fp.n_tables if fp else None),
            "n_figures":  (fp.n_figures if fp else None),
            "top_disease_entities":  sorted(fp.disease_entities)[:10] if fp else [],
            "top_biomarker_entities": sorted(fp.biomarker_entities)[:10] if fp else [],
            "top_gene_entities":      sorted(fp.gene_entities)[:10] if fp else [],
            "top_method_entities":    sorted(fp.method_entities)[:10] if fp else [],
            "top_outcome_entities":   sorted(fp.outcome_entities)[:10] if fp else [],
            "selection_reason": bucket_info.get("selection_reason"),
        }
        if "mean_relatedness_to_set" in bucket_info:
            entry["mean_relatedness_to_set"] = bucket_info["mean_relatedness_to_set"]
        if "diversity_gain" in bucket_info:
            entry["diversity_gain"] = bucket_info["diversity_gain"]
        if "hardness_breakdown" in bucket_info:
            entry["hardness_breakdown"] = bucket_info["hardness_breakdown"]
        elif fp is not None and bucket == "hard":
            entry["hardness_breakdown"] = hardness.breakdown(fp).__dict__
        rationale_payload["papers"][pmcid] = entry

    paths.json_path.write_text(
        json.dumps(rationale_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ── CSV: one row per paper ──────────────────────────────────────────────
    columns = [
        "pmcid", "bucket", "title",
        "n_sentences", "n_chunks", "n_text_elements",
        "n_tables", "n_figures",
        "n_disease_entities", "n_biomarker_entities", "n_gene_entities",
        "n_method_entities", "n_outcome_entities",
        "normalized_hardness", "absolute_hardness",
        "selection_reason",
    ]
    with paths.csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for pmcid in result.all_pmcids():
            fp = by_pmcid.get(pmcid)
            info = result.rationale.get(pmcid, {})
            hb = info.get("hardness_breakdown") or {}
            writer.writerow({
                "pmcid": pmcid,
                "bucket": info.get("bucket"),
                "title": (fp.title if fp else None),
                "n_sentences": (fp.n_sentences if fp else None),
                "n_chunks": (fp.n_chunks if fp else None),
                "n_text_elements": (fp.n_text_elements if fp else None),
                "n_tables": (fp.n_tables if fp else None),
                "n_figures": (fp.n_figures if fp else None),
                "n_disease_entities":  (len(fp.disease_entities)   if fp else 0),
                "n_biomarker_entities": (len(fp.biomarker_entities) if fp else 0),
                "n_gene_entities":     (len(fp.gene_entities)      if fp else 0),
                "n_method_entities":   (len(fp.method_entities)    if fp else 0),
                "n_outcome_entities":  (len(fp.outcome_entities)   if fp else 0),
                "normalized_hardness": hb.get("normalized_hardness"),
                "absolute_hardness":   hb.get("absolute_hardness"),
                "selection_reason":    info.get("selection_reason"),
            })

    return paths
