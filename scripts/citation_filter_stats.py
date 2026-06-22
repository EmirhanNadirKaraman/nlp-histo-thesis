"""Thesis-ready citation-filter statistics for a voter cache.

Scans every voter finding (L1/L2/L3) in a primer voter_cache, runs the SAME
``ProvenanceValidator`` the production citation filter uses (B-080, with the
B-082 base-accession fix), and reports — by reason code, by voter, and by a
hallucination-vs-benign classification — how many findings carry an invalid /
foreign / unresolvable source citation.

Classification of a flagged finding (the citation the filter would drop on):
  malformed_citation    citation not parseable as S{n}|PMCID|te_id
  nonexistent_position  S{n} out of range for the chunk
  te_id_mismatch        right paper, wrong text_element_id
  cross_doc_in_source   cites a DIFFERENT paper whose id appears in the shown
                        source text — a reference the model latched onto
  cross_doc_external    cites a DIFFERENT paper NOT in the shown text — a
                        foreign/external source. This is the SYMPTOM only: it can
                        be a data-pipeline bug (e.g. B-081 contamination: real
                        findings mis-filed from another paper) OR a genuine model
                        hallucination. Distinguish via --db + the verbatim (does
                        the cited te_id resolve to a real paragraph whose text
                        matches the finding → foreign-but-real, not invented).

Writes a markdown + CSV under eval/reports/citation_filter_stats/ and prints a
summary + examples. Offline (no API, no scorer); optional --db resolves each
flagged te_id to its real paper.

NOTE: this is the *all-voter* citation-quality rate. What the citation filter
removes from the PRODUCTION FUNNEL is the selected-set number — measure that with
scripts/_replay_contamination.py after the cascade config is frozen.

Usage:
  python scripts/citation_filter_stats.py                                   # related15
  python scripts/citation_filter_stats.py eval/data/map_primer_heldout15/voter_cache.json --label heldout15
  python scripts/citation_filter_stats.py --db                             # also resolve te_ids in the DB
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.stages.summarization.helpers.citation_filter import (  # noqa: E402
    STRUCTURAL_CITATION_CODES,
)
from pipeline.stages.summarization.models import AuditableSummary  # noqa: E402
from pipeline.stages.summarization.routing.models import ReasonCode  # noqa: E402
from pipeline.stages.summarization.routing.provenance_validator import (  # noqa: E402
    ProvenanceValidator,
    SourceIndex,
)

_CITE = re.compile(r"^S(\d+)\|(PMC[\w\-]+)\|(\d+)$")
_BASE = re.compile(r"(PMC\d+)")
L1 = ["gemini-flash-lite", "gpt-4o-mini", "gpt-4.1-nano"]
L2 = ["gemini-flash", "gpt-4.1-mini", "claude-haiku"]


def _base(p):
    m = _BASE.match(p or "")
    return m.group(1) if m else (p or "")


def _voter_label(level, vi):
    if level == "l3":
        return "sonnet"
    tbl = L1 if level == "l1" else L2
    return tbl[vi] if vi < len(tbl) else f"{level}[{vi}]"


def _classify(finding, codes, doc_pmcid, source_text):
    """Bucket a flagged finding by why its citation is bad."""
    if ReasonCode.INVALID_SENTENCE_ID in codes:
        return "malformed_citation"
    if ReasonCode.NONEXISTENT_SOURCE in codes:
        return "nonexistent_position"
    if ReasonCode.CROSS_DOCUMENT_SOURCE_ERROR in codes:
        for ev in finding.evidence or []:
            m = _CITE.match(ev)
            if m and _base(m.group(2)) != _base(doc_pmcid):
                return "cross_doc_in_source" if m.group(2) in source_text else "cross_doc_external"
        return "cross_document_other"
    if ReasonCode.INVALID_TEXT_ELEMENT_ID in codes:
        return "te_id_mismatch"
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Citation-filter statistics for a voter cache.")
    ap.add_argument("cache", nargs="?", default="eval/data/map_primer/voter_cache.json")
    ap.add_argument("--label", default=None, help="output filename label (default: cache parent dir)")
    ap.add_argument("--db", action="store_true", help="resolve each flagged te_id to its real paper")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = _REPO / cache_path
    label = args.label or cache_path.parent.name
    cache = json.loads(cache_path.read_text())

    te_resolver = None
    if args.db:
        from dotenv import load_dotenv
        load_dotenv(str(_REPO / ".env"))
        from database import TextElement, get_db_connection
        _db = get_db_connection()

        def te_resolver(te_id):  # noqa: F811
            with _db.session_scope() as s:
                row = s.query(TextElement).filter(TextElement.id == te_id).first()
                if row and row.unique_path:
                    return row.unique_path.split("/")[0]
            return None

    n_findings = 0
    by_reason = Counter()
    by_class = Counter()
    by_voter_flagged = Counter()
    by_voter_total = Counter()
    examples = {}

    for entry in cache.values():
        doc = entry["pmcid"]
        for cid in entry.get("chunk_map", {}):
            chunk = entry["chunk_map"][cid]
            if not chunk:
                continue
            stext = entry.get("source_texts", {}).get(cid, "")
            validator = ProvenanceValidator(SourceIndex(chunk), doc,
                                            fabricated_threshold=0.0, weak_threshold=0.0)
            for level in ("l1", "l2", "l3"):
                raw = entry.get(level, {}).get(cid)
                if not raw:
                    continue
                outs = raw if isinstance(raw, list) else [raw]
                for vi, d in enumerate(outs):
                    if not d:
                        continue
                    voter = _voter_label(level, vi)
                    summ = AuditableSummary.model_validate(d)
                    vals = validator.validate(summ)
                    for f, val in zip(summ.findings, vals):
                        n_findings += 1
                        by_voter_total[voter] += 1
                        hard = {c for c in val.reason_codes if c in STRUCTURAL_CITATION_CODES}
                        if not hard:
                            continue
                        for c in hard:
                            by_reason[c.value] += 1
                        cls = _classify(f, hard, doc, stext)
                        by_class[cls] += 1
                        by_voter_flagged[voter] += 1
                        if cls not in examples:
                            cited = next((ev for ev in (f.evidence or []) if "|" in ev), "")
                            resolved = None
                            if te_resolver and cited.count("|") == 2:
                                try:
                                    resolved = te_resolver(int(cited.split("|")[2]))
                                except (ValueError, Exception):
                                    resolved = None
                            examples[cls] = {
                                "voter": voter, "doc": doc, "cited": cited,
                                "resolves_to": resolved,
                                "claim": (f.claim or "")[:70],
                                "verbatim": (f.verbatim_support or "")[:70],
                            }

    n_flagged = sum(by_class.values())
    pct = 100 * n_flagged / max(n_findings, 1)
    hallucinated = by_class.get("cross_doc_external", 0)

    # ── Markdown ──────────────────────────────────────────────────────────────
    lines = [
        f"# Citation-filter statistics — {label}",
        "",
        f"Source: `{cache_path.relative_to(_REPO)}` · all L1/L2/L3 voter findings · "
        f"validator = production citation filter (B-080 + B-082).",
        "",
        f"- **Findings scanned:** {n_findings}",
        f"- **Flagged (filter would drop):** {n_flagged} ({pct:.2f}%)",
        f"- **Foreign-source citations** (`cross_doc_external`, cites a paper not in "
        f"the shown text): {hallucinated} ({100*hallucinated/max(n_findings,1):.3f}%) "
        f"— SYMPTOM only; verify contamination-vs-hallucination via `--db` + verbatim.",
        "",
        "## By classification",
        "",
        "| class | count | what |",
        "|-------|-------|------|",
    ]
    _desc = {
        "malformed_citation": "citation not parseable",
        "nonexistent_position": "S{n} out of chunk range",
        "te_id_mismatch": "right paper, wrong te_id",
        "cross_doc_in_source": "different paper, id appears in shown text (in-text ref)",
        "cross_doc_external": "different paper NOT in shown text — foreign source (bug OR hallucination; verify)",
        "cross_document_other": "cross-document, unclassified",
        "other": "other",
    }
    for cls, n in by_class.most_common():
        lines.append(f"| {cls} | {n} | {_desc.get(cls, '')} |")
    lines += ["", "## By reason code", "", "| reason | count |", "|--------|-------|"]
    for r, n in by_reason.most_common():
        lines.append(f"| {r} | {n} |")
    lines += ["", "## By voter (flagged / total)", "", "| voter | flagged | total | rate |",
              "|-------|---------|-------|------|"]
    for voter in sorted(by_voter_total, key=lambda v: -by_voter_flagged.get(v, 0)):
        tot = by_voter_total[voter]
        fl = by_voter_flagged.get(voter, 0)
        lines.append(f"| {voter} | {fl} | {tot} | {100*fl/max(tot,1):.2f}% |")
    if examples:
        lines += ["", "## Examples (one per class)", ""]
        for cls, ex in examples.items():
            lines.append(f"- **{cls}** [{ex['voter']}] doc=`{ex['doc']}` cited=`{ex['cited']}`"
                         + (f" → resolves to `{ex['resolves_to']}`" if ex["resolves_to"] else ""))
            lines.append(f"  - claim: {ex['claim']!r}")
            lines.append(f"  - verbatim: {ex['verbatim']!r}")

    out_dir = _REPO / "eval/reports/citation_filter_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{label}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    csv_path = out_dir / f"{label}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["findings_scanned", n_findings])
        w.writerow(["flagged", n_flagged])
        w.writerow(["flagged_pct", round(pct, 4)])
        w.writerow(["hallucinated_external", hallucinated])
        for cls, n in by_class.most_common():
            w.writerow([f"class:{cls}", n])
        for r, n in by_reason.most_common():
            w.writerow([f"reason:{r}", n])

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nCitation-filter stats — {label}")
    print(f"  findings scanned: {n_findings}")
    print(f"  flagged:          {n_flagged} ({pct:.2f}%)")
    print(f"  foreign-source (cross_doc_external; bug OR hallucination — verify): {hallucinated}")
    print("  by class: " + (", ".join(f"{c}={n}" for c, n in by_class.most_common()) or "(none)"))
    print(f"\n→ {md_path.relative_to(_REPO)}\n→ {csv_path.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
