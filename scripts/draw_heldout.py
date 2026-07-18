"""Draw a frozen, reproducible held-out knowledge-extraction evaluation set.

A random sample of N eligible papers, disjoint from the ILP-15 calibration
cluster and the 27-paper document-extraction set, written to
configs/paper_selection/heldout15.yaml. Free — DB reads + offline fingerprints,
no API calls. Re-running with the same SEED is deterministic.
"""
import sys
import re
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

import yaml
from eval.paper_selection.loaders import DBLoader
from eval.paper_selection.fingerprints import build_fingerprint
from eval.paper_selection.selectors import SelectionConfig, _eligible_short

SEED, N = 19, 15  # frozen seed → deterministic, reproducible held-out draw

# exclude the calibration cluster and the document-extraction set
ilp = set(re.findall(r"PMC\d+", (_REPO_ROOT / "configs/paper_selection/related15.yaml").read_text()))
doc = set(re.findall(r"PMC\d+", (_REPO_ROOT / "eval/pdfs_manifest.txt").read_text()))

cfg = SelectionConfig()
fps = [build_fingerprint(r) for r in DBLoader().iter_papers()]
# same eligibility as the ILP: 20-350 sentences, disease/biomarker/gene anchor, >=3 entities
eligible = _eligible_short(fps, cfg, cfg.max_sentences_related)
pool = [p.pmcid for p in eligible if p.pmcid.split("_")[0] not in (ilp | doc)]

chosen = sorted(random.Random(SEED).sample(pool, N))
out = _REPO_ROOT / "configs/paper_selection/heldout15.yaml"
# These papers are a RANDOM sample, not relatedness-selected. They are placed
# under `diverse:` purely because load_pmcids_from_selection only reads the
# related/diverse/hard buckets (it flattens all three) — the bucket name carries
# no selection meaning here. `related:` is left empty so the file does not imply
# the papers are related.
header = (
    "# Held-out knowledge-extraction evaluation set: a RANDOM, reproducible sample\n"
    f"# (seed={SEED}) of {N} eligible papers, disjoint from the ILP-15 calibration\n"
    "# cluster and the 27 document-extraction papers. NOT relatedness-selected --\n"
    "# the papers sit under `diverse:` only because load_pmcids_from_selection\n"
    "# reads the related/diverse/hard buckets; the bucket name is plumbing.\n"
)
body = yaml.safe_dump(
    {"heldout_random15": {"related": [], "diverse": chosen, "hard": []}},
    sort_keys=False,
)
out.write_text(header + body)

print(f"eligible pool (excl. calibration + doc): {len(pool)}")
print(f"wrote {len(chosen)} held-out papers (seed={SEED}) -> {out}")
print("\n".join(chosen))
