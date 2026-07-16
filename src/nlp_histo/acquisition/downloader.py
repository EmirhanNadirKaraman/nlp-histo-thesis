"""Download PMC Open-Access article packages (``.tar.gz``) for a list of PMCIDs.

Previously this module ran its download loop at *module scope*: importing it read
``target_pmc_ids.txt`` from the working directory, created ``histopathology_papers/``,
and started hitting the NCBI API. That cannot live in an installed package — an
import must never perform network I/O. The loop is now :func:`download_papers`, and
both paths are explicit arguments.

Network behaviour: same NCBI OA endpoint, same ``tgz`` link selection, same
``ftp://`` → ``https://`` scheme fix, same 0.5 s politeness delay, same skip-and-continue
handling for papers outside the OA subset.

⚠ **NCBI's OA API currently advertises paths it does not serve.** Every legacy FTP tree
was moved under ``/pub/pmc/deprecated/`` (NCBI readme, updated 2026-04-10) while
``oa.fcgi`` still returns the pre-move URLs, so the advertised link 404s for every paper.
:func:`candidate_urls` therefore tries the advertised URL first and the relocated one
second. **NCBI will delete the legacy files in August 2026**; after that this module
stops working and must move to the AWS OA service
(https://pmc.ncbi.nlm.nih.gov/tools/cloud/). See BUGS.md B-118.
"""
from __future__ import annotations

import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests

BASE_API_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Politeness delay between NCBI lookups, in seconds (unchanged).
REQUEST_DELAY_SEC = 0.5

# gzip magic. A 200 response carrying an HTML error page, or a truncated stream, is not
# an archive — and only looks like one if you never check (B-117).
_GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class DownloadReport:
    """Outcome of one ``download_papers`` run.

    The three categories are deliberately distinct: **skipped** is a legitimate,
    expected outcome (already on disk, or the paper is simply not in the OA subset),
    while **failed** means we asked for something that should have been there and did
    not get it. Only the latter makes the run unsuccessful.
    """

    requested: int
    succeeded: int
    failed: int
    skipped: int

    @property
    def ok(self) -> bool:
        """True when nothing we asked for was denied. Partial failure is NOT ok."""
        return self.failed == 0

    def summary(self) -> str:
        return (
            f"{self.succeeded} succeeded · {self.failed} failed · {self.skipped} skipped "
            f"(of {self.requested} requested)"
        )


def is_valid_archive(path: Path) -> bool:
    """True when *path* is a non-empty, readable ``.tar.gz``.

    A 404 is loud, but a 200 carrying a zero-byte body or an HTML error page is not: it
    lands on disk looking like a download, is counted as success, and only fails later in
    `unpack` — or silently produces an empty corpus. So the archive is verified here.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as fh:
            if fh.read(2) != _GZIP_MAGIC:
                return False
        return tarfile.is_tarfile(path)
    except Exception:  # noqa: BLE001 — unreadable is invalid, whatever the reason
        return False


def load_pmc_ids(file_path: str | Path) -> List[str]:
    """Load PMC IDs from a text file (one per line, blanks ignored)."""
    input_file = Path(file_path)

    if not input_file.exists():
        raise FileNotFoundError(f"PMC ID file not found: {input_file}")

    with open(input_file, 'r') as f:
        pmc_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(pmc_ids)} PMC IDs from {input_file}")
    return pmc_ids


def get_download_link(pmcid: str) -> str | None:
    """Ask NCBI for the article-package (``tgz``) link, or None if not in the OA subset."""
    try:
        response = requests.get(f"{BASE_API_URL}?id={pmcid}")
        root = ET.fromstring(response.content)
        for link in root.findall(".//link"):
            if link.attrib.get('format') == 'tgz':
                return link.attrib.get('href')
        return None  # No OA package found
    except Exception as e:
        print(f"Error querying {pmcid}: {e}")
        return None


def candidate_urls(advertised: str) -> List[str]:
    """The URLs worth trying for an OA package the API advertises, in order.

    NCBI moved every legacy FTP tree under ``/pub/pmc/deprecated/`` (their readme, updated
    2026-04-10) but ``oa.fcgi`` still advertises the *pre-move* paths, so the advertised
    URL 404s for every paper — verified 0/5 across 2010–2025 publications, with the same
    file present at 7 556 375 bytes under ``deprecated/`` (B-118).

    The advertised URL is tried **first** so this repairs itself the day NCBI updates its
    API, and the relocation is a fallback rather than a hard-coded assumption.

    ⚠ **The fallback has an expiry date.** NCBI states the legacy files "will be removed
    in August 2026". After that both candidates 404 and acquisition fails loudly — which
    is correct, and is the signal to migrate to the AWS OA service
    (https://pmc.ncbi.nlm.nih.gov/tools/cloud/). See THESIS.md.
    """
    https = advertised.replace("ftp://", "https://", 1) if advertised.startswith("ftp://") else advertised
    urls = [https]
    if "/pub/pmc/" in https and "/pub/pmc/deprecated/" not in https:
        urls.append(https.replace("/pub/pmc/", "/pub/pmc/deprecated/", 1))
    return urls


def download_package(advertised_url: str, target: Path) -> bool:
    """Fetch one OA package, trying each candidate URL. True when a valid archive landed.

    Only the final attempt reports its failure: an intermediate miss is expected while
    NCBI's API points at the old tree, and printing it for all 1093 papers would bury the
    signal.
    """
    candidates = candidate_urls(advertised_url)
    for index, url in enumerate(candidates):
        is_last = index == len(candidates) - 1
        if download_file(url, target, quiet=not is_last):
            if index > 0:
                print(f"   ↪ via NCBI's relocated legacy tree ({url}) — temporary, see B-118")
            return True
    return False


def download_file(url: str, filename: str | Path, *, quiet: bool = False) -> bool:
    """Stream a single file to disk. Returns True only if a valid archive landed.

    A download that produced nothing usable is removed: leaving a zero-byte or corrupt
    file behind would make the next run's ``target.exists()`` check *skip* it, so one bad
    fetch would poison every retry. Successful archives are never touched.

    ``quiet`` suppresses only the failure message — for callers trying several candidate
    URLs, where an early miss is expected rather than newsworthy.
    """
    filename = Path(filename)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        if not quiet:
            print(f"❌ Failed to download {url}: {e}")
        filename.unlink(missing_ok=True)  # a partial write is not a resumable result
        return False

    if not is_valid_archive(filename):
        size = filename.stat().st_size if filename.is_file() else 0
        if not quiet:
            print(
                f"❌ Downloaded {filename.name} but it is not a usable .tar.gz "
                f"({size} bytes) — discarding. The server returned 200 with a body that "
                f"is empty, truncated, or not an archive (an error page, perhaps)."
            )
        filename.unlink(missing_ok=True)
        return False

    print(f"✅ Downloaded: {filename}")
    return True


def download_papers(
    pmcid_file: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> DownloadReport:
    """Download the OA package for every PMCID in *pmcid_file* into *output_dir*.

    Returns a :class:`DownloadReport`. Validates its inputs before creating anything: a
    missing PMCID file raises ``FileNotFoundError`` with the offending path, and no
    output directory is created until the input has been read.

    ``overwrite=False`` (the default) skips PMCIDs whose tarball already exists, so a
    re-run resumes rather than re-downloading.

    Outcomes are counted in three buckets, and the distinction is the point (B-117):

    * **succeeded** — a valid archive is on disk;
    * **skipped** — already present, or the paper is not in the OA subset. Both are
      expected and neither is an error;
    * **failed** — NCBI advertised a package we could not get, or what arrived was not a
      usable archive. Any of these makes the run unsuccessful, even alongside successes.
    """
    pmcid_file = Path(pmcid_file)
    output_dir = Path(output_dir)

    pmc_ids = load_pmc_ids(pmcid_file)  # raises before any mutation

    if not pmc_ids:
        # Vacuous success, but say so — an empty run and a successful one must not look
        # alike.
        print(f"Nothing requested — {pmcid_file} contains no PMC IDs.")
        return DownloadReport(requested=0, succeeded=0, failed=0, skipped=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(pmc_ids)} papers into {output_dir}...")
    succeeded = failed = skipped = 0

    for pmcid in pmc_ids:
        target = output_dir / f"{pmcid}.tar.gz"
        if target.exists() and not overwrite:
            print(f"↷ {pmcid} already downloaded — skipping (use --overwrite to force)")
            skipped += 1
            continue

        print(f"Looking up {pmcid}...")
        ftp_link = get_download_link(pmcid)

        if ftp_link:
            # The advertised URL first, then NCBI's relocated legacy tree — see
            # candidate_urls(). The scheme fix lives there too.
            if download_package(ftp_link, target):
                succeeded += 1
            else:
                failed += 1
        else:
            # Not every paper is in the OA subset. That is NCBI's answer, not a fault.
            print(f"⚠️ {pmcid} is not in the Open Access Subset (No Tarball available).")
            skipped += 1

        time.sleep(REQUEST_DELAY_SEC)

    report = DownloadReport(
        requested=len(pmc_ids), succeeded=succeeded, failed=failed, skipped=skipped
    )
    print(f"\nDone — {report.summary()}. Tarballs in {output_dir}.")
    return report
