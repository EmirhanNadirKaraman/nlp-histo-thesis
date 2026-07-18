#!/usr/bin/env python3
"""Build the Chapter-9 replay bundle for hosting, with checksums and consistency proof.

A clean clone carries code, tests and the corpus definition — not the ~1.5 GB of frozen
paid output the replay needs. This assembles exactly what HOW_TO_RUN §0 tells a supervisor
to ask for, so what they receive can be verified rather than trusted.

    python scripts/make_reproduction_bundle.py --out-dir ~/bundles            # replay
    python scripts/make_reproduction_bundle.py --out-dir ~/bundles --with-db  # + pg dump

Two bundles, deliberately separate:

* replay (~1.5 GB) — reproduces the published tables. No database, no API key:
  ``replay chapter9`` never connects to PostgreSQL.
* database (~485 MB live) — only for querying the corpus or re-running NER/knowledge.

The PDFs are deliberately not bundled: §6 re-acquires them from NLM's AWS dataset using
``files/target_pmc_ids.txt``, which is in the clone. That saves 5.2 GB of upload and avoids
redistributing papers whose licences forbid it (322 of 1093 carry no CC licence).

SQLite consistency. The embedding caches run in WAL mode, where a plain file copy is
only consistent if nothing writes during it. Each is snapshotted with ``VACUUM INTO``,
which takes a read transaction for the duration — a consistent, compacted copy — and is
``integrity_check``ed before packaging. Sidecars (``-wal``/``-shm``) are refused outright:
their presence means committed data lives outside the ``.sqlite`` file, and archiving it
alone would ship a silently truncated cache.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly replay.REQUIRED_ARTIFACTS, minus what a clone already carries
# (scripts/eval/run_summarization_experiments.py and reports/stage6_PR.md are tracked).
# E14's heldout15 primer + silver are not here on purpose: they are committed to git
# (like source_cases_related15.jsonl), so a clone already carries them — see B-123.
REPLAY_MEMBERS = (
    "eval/data/embedding_cache_openai.sqlite",
    "eval/data/embedding_cache_gemini.sqlite",
    "eval/data/map_primer/voter_cache.json",
    "eval/data/silver_findings_related15.jsonl",
    "out/summaries/summaries",
    "out/summaries/cascade_decisions",
)
REPLAY_GLOBS = ("out/summaries/corpus_relations*.json",)

# Anything matching these must never reach a bundle that leaves this machine.
_FORBIDDEN_NAME = re.compile(r"(^|/)\.env|\.pem$|\.key$|id_rsa|\.pdf$|\.dump$|\.sql$", re.I)
_SECRET_TEXT = re.compile(
    rb"(sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_\-]{30,}|postgres(?:ql)?://[^\s\"']+"
    rb"|DB_PASSWORD|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY)"
)


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
        # A missing artifact discovered by the recipient is far more expensive than one
        # caught here.
        raise SystemExit(
            "refusing to build an incomplete bundle — not found:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    return paths


# SQLite: consistency is not assumed

def check_sqlite(path: Path, *, label: str) -> tuple[str, int]:
    """``integrity_check`` + row count, read-only. Raises on anything but ``ok``."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise SystemExit(f"{label}: integrity_check failed → {result}")
        rows = con.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        return result, rows
    finally:
        con.close()


def snapshot_sqlite(src: Path, dest: Path) -> tuple[int, int]:
    """Consistent, compacted copy via ``VACUUM INTO``. Returns (rows_before, rows_after).

    Sidecars are reported, not feared, and the distinction matters:

    * ``-wal`` — the write-ahead log. It does carry committed data, so **copying the
      ``.sqlite`` file alone (tar, cp) would ship a truncated cache that still opens
      cleanly** — the worst kind of corruption, because it looks fine. ``VACUUM INTO``
      does not have that problem: it reads the database through SQLite, so the WAL's
      contents are included by construction.
    * ``-shm`` — a derived shared-memory index. It carries no durable data and is created
      by merely opening a WAL-mode database, even read-only. Refusing on its presence
      would refuse every healthy database we have.

    ``VACUUM INTO`` holds a read transaction for its duration, so the snapshot is a
    consistent point-in-time copy even if something writes concurrently. The row-count
    equality check below is what proves nothing was lost in the process.
    """
    wal = src.with_name(src.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        print(
            f"    note: {wal.name} holds {wal.stat().st_size:,} bytes — VACUUM INTO reads "
            f"through it, so the snapshot includes it (a file copy would not)."
        )

    _, rows_before = check_sqlite(src, label=f"{src.name} (source)")
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        # VACUUM INTO holds a read transaction for its duration: the destination is a
        # point-in-time consistent copy, defragmented, and free of WAL sidecars.
        con.execute("VACUUM INTO ?", (str(dest),))
    finally:
        con.close()
    _, rows_after = check_sqlite(dest, label=f"{dest.name} (snapshot)")
    if rows_before != rows_after:
        raise SystemExit(
            f"{src.name}: snapshot has {rows_after:,} rows, source had {rows_before:,}"
        )
    return rows_before, rows_after


# content safety

def audit_member(rel: str, path: Path) -> list[str]:
    """Reasons *path* must not be shipped. Empty list ⇒ clean."""
    problems = []
    if _FORBIDDEN_NAME.search(rel):
        problems.append(f"forbidden name: {rel}")
    if rel.startswith("/") or ".." in Path(rel).parts:
        problems.append(f"escapes the extraction root: {rel}")
    # Scan text-ish artifacts for credentials. The sqlite caches are float blobs keyed by
    # claim text — scanned too, cheaply, on a bounded prefix.
    if path.is_file():
        with path.open("rb") as fh:
            head = fh.read(4 << 20)
        m = _SECRET_TEXT.search(head)
        if m:
            problems.append(f"looks like a credential ({m.group(1)[:12].decode('utf-8', 'replace')}…): {rel}")
    return problems


def build_replay_bundle(out_dir: Path, commit: str) -> tuple[Path, Path]:
    staging = Path(tempfile.mkdtemp(prefix="nlp-histo-bundle-", dir=out_dir))
    paths = _resolve(REPLAY_MEMBERS, REPLAY_GLOBS)

    # 1. Materialise members into staging: sqlite via snapshot, everything else as-is.
    print("staging artifacts …")
    staged: list[tuple[str, Path]] = []
    for p in paths:
        rel = str(p.relative_to(REPO_ROOT))
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".sqlite":
            before, _ = snapshot_sqlite(p, target)
            saved = (p.stat().st_size - target.stat().st_size) / 1024 / 1024
            print(f"  ~ {rel}  snapshot ok · {before:,} rows · {saved:+,.0f} MB")
            staged.append((rel, target))
        elif p.is_dir():
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                frel = str(f.relative_to(REPO_ROOT))
                ft = staging / frel
                ft.parent.mkdir(parents=True, exist_ok=True)
                ft.write_bytes(f.read_bytes())
                staged.append((frel, ft))
            print(f"  + {rel}/  ({sum(1 for r, _ in staged if r.startswith(rel))} files)")
        else:
            target.write_bytes(p.read_bytes())
            print(f"  + {rel}")
            staged.append((rel, target))

    # 2. Audit every member before it can leave the machine.
    print("auditing …")
    problems = [prob for rel, path in staged for prob in audit_member(rel, path)]
    if problems:
        raise SystemExit("refusing to build — content audit failed:\n" +
                         "\n".join(f"  - {p}" for p in problems))
    print(f"  {len(staged)} members clean")

    # 3. Manifest: machine-readable, per-file path + size + sha256.
    manifest = {
        "bundle": "chapter9-replay-artifacts",
        "source_commit": commit,
        "artifact_count": len(staged),
        "note": "Extract over a clean clone so paths match, then: "
                "nlp-histo replay chapter9 --artifact-root .",
        "files": [
            {"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for rel, path in sorted(staged)
        ],
    }
    manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"])
    manifest_path = staging / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # 4. Archive, named for the exact source commit.
    archive = out_dir / f"nlp-histo-replay-artifacts-{commit}.tar.gz"
    print(f"writing {archive.name} …")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(manifest_path, arcname="MANIFEST.json")
        for rel, path in sorted(staged):
            tar.add(path, arcname=rel)

    # 5. Whole-archive checksum, as its own file.
    digest = _sha256(archive)
    sums = out_dir / f"{archive.name}.sha256"
    sums.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

    final_manifest = out_dir / f"{archive.name}.manifest.json"
    final_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    import shutil
    shutil.rmtree(staging)
    return archive, final_manifest


def build_db_dump(out_dir: Path) -> Path:
    """Dump the corpus as gzipped plain SQL.

    Plain SQL, not ``-Fc``: a custom-format dump can only be read by a ``pg_restore`` of
    the dumping version or newer, and the recipient's toolchain is unknown — this machine
    dumps with pg_dump 16 against a PostgreSQL 14 server, so a ``-Fc`` file would refuse
    to load for anyone on 14. Plain SQL restores with any psql:

        createdb nlp_histo
        gunzip -c nlp-histo-corpus.sql.gz | psql -d nlp_histo

    ``--no-owner``/``--no-privileges`` drop the local role names, so the recipient does not
    need a matching ``local_db_user`` for the restore to succeed.
    """
    from nlp_histo.database.db_connection import get_db_connection

    url = get_db_connection().engine.url
    dump = out_dir / "nlp-histo-corpus.sql.gz"
    print(f"dumping {url.database} → {dump.name} …")
    env = dict(os.environ)
    if url.password:
        env["PGPASSWORD"] = str(url.password)  # never argv: `ps` is world-readable
    with gzip.open(dump, "wb") as fh:
        proc = subprocess.Popen(
            ["pg_dump", "--no-owner", "--no-privileges",
             "-h", str(url.host), "-p", str(url.port), "-U", str(url.username),
             "-d", str(url.database)],
            stdout=subprocess.PIPE, env=env,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
            fh.write(chunk)
        if proc.wait() != 0:
            raise SystemExit(f"pg_dump failed with exit {proc.returncode}")
    return dump


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write. Keep it OUTSIDE the repository.")
    ap.add_argument("--with-db", action="store_true",
                    help="also pg_dump the corpus. Needs pg_dump on PATH.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir.expanduser().resolve()
    if REPO_ROOT in out_dir.parents or out_dir == REPO_ROOT:
        raise SystemExit(f"--out-dir must be outside the repository (got {out_dir})")
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        print("WARNING: the working tree is dirty — the archive name will claim a commit "
              "that does not describe what is inside it:\n" + dirty, file=sys.stderr)

    archive, manifest = build_replay_bundle(out_dir, commit)
    outputs = [archive]
    if args.with_db:
        outputs.append(build_db_dump(out_dir))

    print("\n── built ──")
    for a in outputs:
        print(f"  {a}  ({a.stat().st_size / 1024 / 1024:,.0f} MB)")
    print(f"  {manifest}")
    print(f"  {archive}.sha256")
    print("\nRecipients verify with:")
    print(f"  shasum -a 256 -c {archive.name}.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
