#!/usr/bin/env python3
"""Verify unpacked replay artifacts against the bundle's MANIFEST.json.

The archive ships a manifest listing every artifact with its byte size and
SHA-256. This checks the files on disk against it, so a supervisor can tell an
incomplete or corrupted unpack from a genuine result difference -- otherwise a
truncated cache surfaces much later as a confusing cache miss.

Usage::

    python scripts/verify_bundle.py                    # MANIFEST.json in cwd
    python scripts/verify_bundle.py --manifest path/to/MANIFEST.json
    python scripts/verify_bundle.py --fast             # size only, skip hashing

Exit codes: 0 all present and matching, 1 any missing/mismatched, 2 no manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CHUNK = 8 * 1024 * 1024

# The embedding caches are live SQLite databases: `matcher.py` does
# INSERT OR REPLACE into them, so they grow the first time an experiment runs a
# key the bundle did not already hold. Their manifest size and SHA-256 are exact
# only on a fresh unpack. Checking them strictly would report two failures on a
# perfectly good tree the moment anyone ran anything, so they are verified
# grow-only: present, and no smaller than shipped. A truncated or absent cache --
# the failure that actually matters -- is still caught.
GROW_ONLY = ("eval/data/embedding_cache_gemini.sqlite",
             "eval/data/embedding_cache_openai.sqlite")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=Path("MANIFEST.json"),
                    help="bundle manifest (default: MANIFEST.json in the cwd)")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="directory the archive was extracted over (default: cwd)")
    ap.add_argument("--fast", action="store_true",
                    help="check presence and size only; skip SHA-256")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"No manifest at {args.manifest}. It ships inside the artifact "
              f"archive -- extract that first.", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    total = manifest.get("total_bytes", sum(f["bytes"] for f in files))

    print(f"bundle        {manifest.get('bundle', '?')}")
    print(f"source commit {manifest.get('source_commit', '?')}")
    print(f"artifacts     {len(files)}  ({human(total)})")
    print(f"mode          {'size only (--fast)' if args.fast else 'size + SHA-256'}\n")

    missing, wrong_size, wrong_hash, grown = [], [], [], []
    for i, entry in enumerate(files, 1):
        path = args.root / entry["path"]
        label = entry["path"]
        tag = f"  [{i:>2}/{len(files)}]"
        if not path.is_file():
            missing.append(label)
            print(f"{tag} MISSING   {label}")
            continue
        size = path.stat().st_size

        if label in GROW_ONLY:
            if size < entry["bytes"]:
                wrong_size.append(label)
                print(f"{tag} TRUNCATED {label} "
                      f"({human(size)} on disk, shipped {human(entry['bytes'])})")
            elif size > entry["bytes"]:
                grown.append(label)
                print(f"{tag} ok, grown {label} "
                      f"(+{human(size - entry['bytes'])} of cached embeddings)")
            else:
                print(f"{tag} ok        {label} (unused since unpack)")
            continue

        if size != entry["bytes"]:
            wrong_size.append(label)
            print(f"{tag} SIZE      {label} "
                  f"({human(size)} on disk, expected {human(entry['bytes'])})")
            continue
        if args.fast:
            print(f"{tag} ok        {label}")
            continue
        got = sha256(path)
        if got != entry["sha256"]:
            wrong_hash.append(label)
            print(f"{tag} CHECKSUM  {label}\n"
                  f"            expected {entry['sha256']}\n"
                  f"            got      {got}")
        else:
            print(f"{tag} ok        {label}")

    bad = len(missing) + len(wrong_size) + len(wrong_hash)
    print()
    if not bad:
        scope = "present, correct size" if args.fast else "present and checksum-clean"
        print(f"All {len(files)} artifacts {scope}.")
        if grown:
            print(f"{len(grown)} embedding cache(s) larger than shipped, which is "
                  f"expected once an experiment has run: they are append-only "
                  f"SQLite stores, and a bigger cache means fewer future misses.")
        return 0
    print(f"{bad} of {len(files)} artifacts failed: "
          f"{len(missing)} missing, {len(wrong_size)} wrong size, "
          f"{len(wrong_hash)} checksum mismatch.")
    print("Re-extract the archive over a clean clone; a partial unpack surfaces "
          "later as a confusing cache miss rather than a clear error.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
