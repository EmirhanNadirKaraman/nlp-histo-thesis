"""Download PMC Open-Access article packages (``.tar.gz``) for a list of PMCIDs.

Previously this module ran its download loop at *module scope*: importing it read
``target_pmc_ids.txt`` from the working directory, created ``histopathology_papers/``,
and started hitting the NCBI API. That cannot live in an installed package — an
import must never perform network I/O. The loop is now :func:`download_papers`, and
both paths are explicit arguments.

Network behaviour is unchanged: same NCBI OA endpoint, same ``tgz`` link selection,
same ``ftp://`` → ``https://`` scheme fix, same 0.5 s politeness delay, same
skip-and-continue handling for papers outside the OA subset.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import requests

BASE_API_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Politeness delay between NCBI lookups, in seconds (unchanged).
REQUEST_DELAY_SEC = 0.5


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


def download_file(url: str, filename: str | Path) -> bool:
    """Stream a single file to disk. Returns True on success."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def download_papers(
    pmcid_file: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Download the OA package for every PMCID in *pmcid_file* into *output_dir*.

    Returns the number of tarballs successfully downloaded. Validates its inputs
    before creating anything: a missing PMCID file raises ``FileNotFoundError``
    with the offending path, and no output directory is created until the input
    has been read.

    ``overwrite=False`` (the default) skips PMCIDs whose tarball already exists,
    so a re-run resumes rather than re-downloading.
    """
    pmcid_file = Path(pmcid_file)
    output_dir = Path(output_dir)

    pmc_ids = load_pmc_ids(pmcid_file)  # raises before any mutation

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(pmc_ids)} papers into {output_dir}...")
    downloaded = 0

    for pmcid in pmc_ids:
        target = output_dir / f"{pmcid}.tar.gz"
        if target.exists() and not overwrite:
            print(f"↷ {pmcid} already downloaded — skipping (use --overwrite to force)")
            continue

        print(f"Looking up {pmcid}...")
        ftp_link = get_download_link(pmcid)

        if ftp_link:
            # NCBI advertises ftp:// links; requests speaks HTTPS, and the hosts
            # serve the same paths over TLS.
            if ftp_link.startswith("ftp://"):
                ftp_link = ftp_link.replace("ftp://", "https://")

            if download_file(ftp_link, target):
                downloaded += 1
        else:
            print(f"⚠️ {pmcid} is not in the Open Access Subset (No Tarball available).")

        time.sleep(REQUEST_DELAY_SEC)

    print(f"\nDone — {downloaded} tarball(s) in {output_dir}.")
    return downloaded
