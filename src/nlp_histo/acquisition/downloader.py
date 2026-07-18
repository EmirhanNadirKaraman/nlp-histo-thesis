"""Download PMC Open-Access article packages (``.tar.gz``) for a list of PMCIDs.

Previously this module ran its download loop at *module scope*: importing it read
``target_pmc_ids.txt`` from the working directory, created ``histopathology_papers/``,
and started hitting the NCBI API. That cannot live in an installed package — an
import must never perform network I/O. The loop is now :func:`download_papers`, and
both paths are explicit arguments.

Network behaviour: same NCBI OA endpoint, same ``tgz`` link selection, same
``ftp://`` → ``https://`` scheme fix, same 0.5 s politeness delay, same skip-and-continue
handling for papers outside the OA subset.

NCBI's OA API currently advertises paths it does not serve. Every legacy FTP tree
was moved under ``/pub/pmc/deprecated/`` (NCBI readme, updated 2026-04-10) while
``oa.fcgi`` still returns the pre-move URLs, so the advertised link 404s for every paper.
:func:`candidate_urls` therefore tries the advertised URL first and the relocated one
second. NCBI will delete the legacy files in August 2026; after that this module
stops working and must move to the AWS OA service
(https://pmc.ncbi.nlm.nih.gov/tools/cloud/). See BUGS.md B-118.
"""
from __future__ import annotations

import json
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

    The three categories are deliberately distinct: skipped is a legitimate,
    expected outcome (already on disk, or the paper is simply not in the OA subset),
    while failed means we asked for something that should have been there and did
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


# AWS Open-Access dataset (the successor to the FTP tarballs)
#
# NLM publishes the OA Subset on AWS, free and without login
# (https://pmc.ncbi.nlm.nih.gov/tools/cloud/, https://registry.opendata.aws/ncbi-pmc).
# It is the durable route: the FTP tree it replaces is deleted in August 2026 (B-118).
#
# Layout differs from FTP in two ways that matter:
#   * no tarball — the PDF, XML, text and figures are individual objects;
#   * objects live under a VERSIONED prefix, "PMC8395919.1/", so a PMCID alone does not
#     determine the URL. The version is discovered by listing the prefix.
AWS_OA_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

_PDF_MAGIC = b"%PDF"


class UnresolvablePdfName(RuntimeError):
    """The article's JATS XML does not authoritatively name its PDF.

    Raised rather than guessed at: inventing a filename would mint a document ID that
    silently differs from the FTP-derived one for the same paper, which is a duplicate
    corpus entry rather than an error anyone would notice (B-119).
    """


def publisher_pdf_filename(xml_bytes: bytes) -> str:
    """The publisher's PDF filename, from the JATS ``self-uri`` element.

    This is the authoritative mapping — the same name the FTP tarball carried — so an AWS
    download reaches the same document ID as an FTP one. Example::

        <self-uri content-type="pmc-pdf" xlink:href="dermatopathology-08-00036.pdf"/>

    The value is untrusted: it is publisher-supplied text that becomes a path we
    write to. Everything below is a rejection, not a repair — a surprising value means we
    do not understand the article, and guessing is how you get a wrong identifier or a
    write outside the corpus directory.
    """
    XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise UnresolvablePdfName(f"JATS XML is unparsable: {e}") from e

    hrefs = [
        el.get(XLINK_HREF)
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1] == "self-uri" and el.get("content-type") == "pmc-pdf"
    ]
    hrefs = [h for h in hrefs if h]

    if not hrefs:
        raise UnresolvablePdfName(
            "no <self-uri content-type='pmc-pdf'> in the JATS XML — the article does not "
            "name its own PDF, so no authoritative filename exists"
        )
    if len(set(hrefs)) > 1:
        raise UnresolvablePdfName(
            f"ambiguous: {len(set(hrefs))} different pmc-pdf self-uri values {sorted(set(hrefs))!r}"
        )

    href = hrefs[0].strip()
    if not href:
        raise UnresolvablePdfName("pmc-pdf self-uri is empty")
    if "://" in href or href.startswith("//"):
        raise UnresolvablePdfName(f"pmc-pdf self-uri is a URL, not a filename: {href!r}")
    if href.startswith("/") or (len(href) > 1 and href[1] == ":"):
        raise UnresolvablePdfName(f"pmc-pdf self-uri is an absolute path: {href!r}")
    if ".." in href.split("/"):
        raise UnresolvablePdfName(f"pmc-pdf self-uri traverses upward: {href!r}")

    # A nested reference ("supplementary/paper.pdf") reduces to its filename: the tarball
    # flattened these too, and by this point the value is known to be relative and
    # traversal-free, so taking the last component cannot escape the paper directory.
    name = href.rsplit("/", 1)[-1]

    if not name.lower().endswith(".pdf"):
        raise UnresolvablePdfName(f"pmc-pdf self-uri is not a PDF: {href!r}")
    if name in (".", "..") or "\x00" in name:
        raise UnresolvablePdfName(f"pmc-pdf self-uri is not a usable filename: {href!r}")
    return name


def _object_version(key: str) -> int:
    """Article version from an S3 key, e.g. ``PMC8395919.1/…`` → 1. ``-1`` if unparsable."""
    head = key.split("/", 1)[0]
    try:
        return int(head.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return -1


def aws_object_keys(pmcid: str) -> List[str]:
    """Keys for the newest version of *pmcid*, or ``[]`` when it is not in the dataset.

    The trailing dot in the prefix matters: ``PMC839591.`` must not match ``PMC8395919…``.

    Version selection is numeric, so v10 beats v9 — a lexical ``max`` would pick
    ``.9`` and quietly serve a superseded article.

    Pagination is followed to exhaustion. S3 caps a response at 1000 keys and signals
    more with ``IsTruncated``; ignoring it would silently drop objects — for a
    figure-heavy article, possibly the PDF itself — and the result would look like a
    complete listing.
    """
    keys: List[str] = []
    token: str | None = None
    while True:
        url = f"{AWS_OA_BUCKET}?list-type=2&prefix={pmcid}.&max-keys=1000"
        if token:
            url += f"&continuation-token={requests.utils.quote(token, safe='')}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys.extend(
            c.find("s3:Key", _S3_NS).text for c in root.findall(".//s3:Contents", _S3_NS)
        )
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS) or "").lower()
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        if truncated != "true" or not token:
            break

    if not keys:
        return []
    newest = max(_object_version(k) for k in keys)  # int comparison: v10 > v9
    return [k for k in keys if _object_version(k) == newest]


def download_paper_aws(pmcid: str, corpus_dir: Path, *, overwrite: bool = False) -> str:
    """Fetch one paper's PDF + XML from AWS into ``corpus_dir/<pmcid>/``.

    Reproduces exactly what ``acquire unpack`` produces from a tarball — the XML as
    ``<PMCID>.nxml``, the PDF under the publisher's filename — so ``acquire organize``
    consumes it unchanged and both sources reach the *same* document ID. AWS names its
    objects ``PMC8395919.1.pdf``; using that would mint ``PMC8395919.1`` as a new document
    ID and duplicate a paper the corpus already holds (B-119).

    The XML is fetched first, because it is what names the PDF.

    Returns ``"succeeded"``, ``"failed"`` or ``"skipped"`` — the same three-way outcome
    the FTP path reports (B-117), so neither source can turn a miss into a success.
    """
    dest = corpus_dir / pmcid
    if dest.is_dir() and any(dest.glob("*.pdf")) and not overwrite:
        print(f"↷ {pmcid} already downloaded — skipping (use --overwrite to force)")
        return "skipped"

    try:
        keys = aws_object_keys(pmcid)
    except Exception as e:  # noqa: BLE001 — a lookup failure is a failure, not a skip
        print(f"❌ {pmcid}: AWS listing failed: {e}")
        return "failed"

    if not keys:
        print(f"⚠️ {pmcid} is not in the AWS Open Access dataset.")
        return "skipped"

    pdf_keys = [k for k in keys if k.lower().endswith(".pdf")]
    xml_keys = [k for k in keys if k.lower().endswith(".xml")]
    if not pdf_keys:
        # In the dataset but carrying no PDF: a real gap for a PDF pipeline, not a success.
        print(f"❌ {pmcid}: present in the dataset but has no PDF ({len(keys)} objects).")
        return "failed"
    if not xml_keys:
        print(f"❌ {pmcid}: no JATS XML, so the publisher's PDF filename cannot be resolved.")
        return "failed"

    # 1. XML first — it is the authority on the PDF's name.
    try:
        r = requests.get(f"{AWS_OA_BUCKET}/{xml_keys[0]}", timeout=120)
        r.raise_for_status()
        xml_bytes = r.content
        pdf_name = publisher_pdf_filename(xml_bytes)
    except UnresolvablePdfName as e:
        # Refuse rather than invent: a fabricated name yields a document ID that silently
        # disagrees with the FTP-derived one for the same paper.
        print(f"❌ {pmcid}: {e}")
        return "failed"
    except Exception as e:  # noqa: BLE001
        print(f"❌ {pmcid}: could not fetch the JATS XML: {e}")
        return "failed"

    # 2. Write the unpack-equivalent layout.
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{pmcid}.nxml").write_bytes(xml_bytes)
    if not download_object(f"{AWS_OA_BUCKET}/{pdf_keys[0]}", dest / pdf_name, magic=_PDF_MAGIC):
        return "failed"

    # 3. Provenance: the AWS key and the publisher name are not recoverable from the
    # renamed files, and `organize` ignores non-pdf/xml files, so this rides along safely.
    (dest / "_source.json").write_text(
        json.dumps(
            {
                "source": "aws",
                "bucket": AWS_OA_BUCKET,
                "version": _object_version(pdf_keys[0]),
                "aws_pdf_key": pdf_keys[0],
                "aws_xml_key": xml_keys[0],
                "publisher_pdf_filename": pdf_name,
                "document_id": f"{pmcid}_{Path(pdf_name).stem}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ Downloaded: {pmcid} → {dest} (pdf={pdf_name})")
    return "succeeded"


def download_object(url: str, target: Path, *, magic: bytes | None = None) -> bool:
    """Stream one object to disk; verify it is non-empty and starts with *magic*.

    Same reasoning as the archive check: a 200 carrying an error page is not a PDF, and
    only looks like one if nobody reads the first four bytes. Unusable files are removed
    so a re-run refetches instead of skipping.
    """
    target = Path(target)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Failed to download {url}: {e}")
        target.unlink(missing_ok=True)
        return False

    if target.stat().st_size == 0 or (magic and target.open("rb").read(len(magic)) != magic):
        print(f"❌ {target.name}: not the expected content ({target.stat().st_size} bytes) — discarding.")
        target.unlink(missing_ok=True)
        return False
    return True


# legacy FTP tarballs (works until NCBI deletes them, August 2026)

def candidate_urls(advertised: str) -> List[str]:
    """The URLs worth trying for an OA package the API advertises, in order.

    NCBI moved every legacy FTP tree under ``/pub/pmc/deprecated/`` (their readme, updated
    2026-04-10) but ``oa.fcgi`` still advertises the *pre-move* paths, so the advertised
    URL 404s for every paper — verified 0/5 across 2010–2025 publications, with the same
    file present at 7 556 375 bytes under ``deprecated/`` (B-118).

    The advertised URL is tried first so this repairs itself the day NCBI updates its
    API, and the relocation is a fallback rather than a hard-coded assumption.

    The fallback has an expiry date. NCBI states the legacy files "will be removed
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
    source: str = "aws",
) -> DownloadReport:
    """Download every PMCID in *pmcid_file* into *output_dir*.

    ``source="aws"`` (default) pulls individual PDF/XML objects from NLM's AWS Open-Access
    dataset into ``output_dir/<pmcid>/`` — the layout ``acquire unpack`` produces, so
    ``organize`` follows directly and no unpack step is needed.

    ``source="ftp"`` fetches the legacy ``.tar.gz`` packages into ``output_dir`` and still
    requires ``acquire unpack``. NCBI deletes that tree in August 2026 (B-118); it is
    kept only so a mid-migration corpus can still be resumed.

    Returns a :class:`DownloadReport`. Validates its inputs before creating anything: a
    missing PMCID file raises ``FileNotFoundError`` with the offending path, and no
    output directory is created until the input has been read.

    ``overwrite=False`` (the default) skips PMCIDs whose tarball already exists, so a
    re-run resumes rather than re-downloading.

    Outcomes are counted in three buckets, and the distinction is the point (B-117):

    * succeeded — a valid archive is on disk;
    * skipped — already present, or the paper is not in the OA subset. Both are
      expected and neither is an error;
    * failed — NCBI advertised a package we could not get, or what arrived was not a
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

    if source not in ("aws", "ftp"):
        raise ValueError(f"unknown source {source!r}; expected 'aws' or 'ftp'")

    print(f"Processing {len(pmc_ids)} papers into {output_dir} (source={source})...")
    if source == "ftp":
        print(
            "⚠ The FTP tarball route is deprecated upstream: NCBI deletes these files in "
            "August 2026 (B-118). Prefer --source aws."
        )
    succeeded = failed = skipped = 0

    for pmcid in pmc_ids:
        if source == "aws":
            outcome = download_paper_aws(pmcid, output_dir, overwrite=overwrite)
            succeeded += outcome == "succeeded"
            failed += outcome == "failed"
            skipped += outcome == "skipped"
            time.sleep(REQUEST_DELAY_SEC)
            continue

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
