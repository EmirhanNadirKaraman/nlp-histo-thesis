"""Characterize the residual cross-document citation fails on the CLEAN cache.

For each cross-document finding (cited pmcid != document pmcid), record which
voter produced it and whether the cited pmcid appears in the source text shown
to the model (an in-text reference the model latched onto) vs nowhere (genuine
hallucination / external knowledge).
"""
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from pipeline.stages.knowledge_extraction.models import AuditableSummary  # noqa: E402
from pipeline.stages.knowledge_extraction.provenance.validator import (  # noqa: E402
    ProvenanceValidator,
    SourceIndex,
)

cache_path = sys.argv[1] if len(sys.argv) > 1 else "eval/data/map_primer/voter_cache.json"
cache = json.loads((_REPO / cache_path).read_text())

L1 = ["gemini-flash-lite", "gpt-4o-mini", "gpt-4.1-nano"]
L2 = ["gemini-flash", "gpt-4.1-mini", "claude-haiku"]


def label(level, vi):
    if level == "l3":
        return "sonnet(l3)"
    table = L1 if level == "l1" else L2
    return table[vi] if vi < len(table) else f"{level}[{vi}]"


by_voter = Counter()
in_text = Counter()
not_in_text = Counter()
examples = []

for entry in cache.values():
    epmcid = entry["pmcid"]
    for cid in entry.get("chunk_map", {}):
        chunk = entry["chunk_map"][cid]
        if not chunk:
            continue
        stext = entry.get("source_texts", {}).get(cid, "")
        validator = ProvenanceValidator(SourceIndex(chunk), epmcid,
                                        fabricated_threshold=0.0, weak_threshold=0.0)
        for level in ("l1", "l2", "l3"):
            raw = entry.get(level, {}).get(cid)
            if not raw:
                continue
            outs = raw if isinstance(raw, list) else [raw]
            for vi, d in enumerate(outs):
                if not d:
                    continue
                summ = AuditableSummary.model_validate(d)
                for f in summ.findings:
                    for ev in (f.evidence or []):
                        parts = ev.split("|")
                        if len(parts) == 3 and parts[1] != epmcid:
                            lbl = label(level, vi)
                            by_voter[lbl] += 1
                            present = parts[1] in stext
                            (in_text if present else not_in_text)[lbl] += 1
                            if len(examples) < 10:
                                examples.append(
                                    (lbl, epmcid, parts[1], parts[2], present,
                                     (f.claim or "")[:60], (f.verbatim_support or "")[:70])
                                )
                            break

total = sum(by_voter.values())
print(f"cross-document fails: {total}")
print(f"by voter:                {dict(by_voter)}")
print(f"cited pmcid IN shown text (in-text reference): {dict(in_text)}  "
      f"= {sum(in_text.values())}")
print(f"cited pmcid NOT in shown text (hallucinated/external): {dict(not_in_text)}  "
      f"= {sum(not_in_text.values())}")
print("\nexamples:")
for lbl, ep, cp, te, pres, claim, vb in examples:
    print(f"  [{lbl}] doc={ep}")
    print(f"      cited={cp}|{te}  in_source_text={pres}")
    print(f"      claim={claim!r}")
    print(f"      verbatim={vb!r}")
