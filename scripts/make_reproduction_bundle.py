#!/usr/bin/env python3
"""Build the reproduction bundles for upload (TUM institutional storage).

A clean clone carries code, tests and the corpus definition — not the 1.5 GB of frozen
paid output the replay needs, nor the database. This assembles exactly what HOW_TO_RUN §0
tells a supervisor to ask for, with checksums, so what they receive can be verified rather
than trusted.

    python scripts/make_reproduction_bundle.py --out-dir /tmp/bundles            # replay
    python scripts/make_reproduction_bundle.py --out-dir /tmp/bundles --with-db  # + dump

Two bundles, deliberately separate:

* **replay** (~1.5 GB) — Path A. Reproduces the published tables. No database, no API key:
  `replay chapter9` never connects to PostgreSQL.
* **database** (~485 MB live, smaller compressed) — Path B. Only needed to query the
  corpus or re-run NER/knowledge extraction.

The PDFs are deliberately NOT bundled: §6 re-acquires them from NLM's AWS dataset using
`files/target_pmc_ids.txt`, which is in the clone. That saves 5.2 GB of upload and avoids
redistributing papers whose licences do not permit it (322 of 1093 carry no CC licence).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly what replay.REQUIRED_ARTIFACTS validates, minus what the clone already carries
# (scripts/eval/…, reports/stage6_PR.md are tracked).
REPLAY_MEMBERS = (
    "eval/data/embedding_cache_openai.sqlite",
    "eval/data/embedding_cache_gemini.sqlite",
    "eval/data/map_primer/voter_cache.json",
    "eval/data/silver_findings_related15.jsonl",
    "out/summaries/summaries",
    "out/summaries/cascade_decisions",
)
REPLAY_GLOBS = ("out/summaries/corpus_relations*.json",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(members, globs) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for rel in members:
        p = REPO_ROOT / rel
        (paths if p.exists() else missing).append(p if p.exists() else rel)
    for pattern in globs:
        hits = sorted(REPO_ROOT.glob(pattern))
        if not hits:
            missing.append(pattern)
        paths.extend(hits)
    if missing:
        # Shipping a bundle that is missing an artifact just moves the failure to the
        # recipient, where it is far more expensive to diagnose.
        raise SystemExit(
            "refusing to build an incomplete bundle — not found:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    return paths


def build_replay_bundle(out_dir: Path) -> Path:
    paths = _resolve(REPLAY_MEMBERS, REPLAY_GLOBS)
    archive = out_dir / "nlp-histo-replay-bundle.tar.gz"
    print(f"building {archive.name} …")
    with tarfile.open(archive, "w:gz") as tar:
        for p in paths:
            rel = p.relative_to(REPO_ROOT)
            print(f"  + {rel}")
            tar.add(p, arcname=str(rel))
    return archive


def build_db_dump(out_dir: Path) -> Path:
    """pg_dump the corpus. Custom format: compressed, and restorable with pg_restore."""
    from nlp_histo.database.db_connection import get_db_connection

    url = get_db_connection().engine.url
    dump = out_dir / "nlp-histo-corpus.dump"
    print(f"dumping {url.database} → {dump.name} …")
    env = dict(os.environ)
    if url.password:
        env["PGPASSWORD"] = str(url.password)
    cmd = [
        "pg_dump", "-Fc", "--no-owner", "--no-privileges",
        "-h", str(url.host), "-p", str(url.port), "-U", str(url.username),
        "-d", str(url.database), "-f", str(dump),
    ]
    # The password is passed via PGPASSWORD, never argv — argv is world-readable in `ps`.
    subprocess.run(cmd, check=True, env=env)
    return dump


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--with-db", action="store_true",
                    help="also pg_dump the corpus (Path B). Needs pg_dump on PATH.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [build_replay_bundle(out_dir)]
    if args.with_db:
        artifacts.append(build_db_dump(out_dir))

    sums = out_dir / "SHA256SUMS"
    lines = []
    for a in artifacts:
        digest = _sha256(a)
        size_mb = a.stat().st_size / 1024 / 1024
        print(f"  {a.name}: {size_mb:,.0f} MB  {digest[:16]}…")
        lines.append(f"{digest}  {a.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {sums}")
    print("Upload the archive(s) AND SHA256SUMS. Recipients verify with:")
    print("  shasum -a 256 -c SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
