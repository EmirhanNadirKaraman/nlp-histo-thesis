"""B-119 — the same paper, fetched either way, must become the same document.

This is the test that matters: AWS is now the default, so resuming an FTP-built corpus
with AWS is a normal thing to do. If the two routes disagree about identity, the corpus
silently grows a second copy of a paper it already has — as a duplicate row, not an
error.

Both routes are driven end-to-end through the public commands, offline, with the *same*
PDF bytes served by each, and the resulting document IDs compared.
"""
from __future__ import annotations

import io
import tarfile

import pytest

from nlp_histo.acquisition import downloader
from nlp_histo.acquisition.organizer import organize_pdfs
from nlp_histo.acquisition.tarballs import unpack_tarballs
from nlp_histo.document_id import canonical_document_id

PMCID = "PMC8395919"
PUBLISHER_STEM = "dermatopathology-08-00036"
PDF_BYTES = b"%PDF-1.4 the identical article bytes"
XML_BYTES = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
    f'<self-uri content-type="pmc-pdf" xlink:href="{PUBLISHER_STEM}.pdf"/>'
    "</article>"
).encode()

# What organize must produce from EITHER route.
EXPECTED_PDF_NAME = f"{PMCID}_{PUBLISHER_STEM}.pdf"
EXPECTED_DOCUMENT_ID = f"{PMCID}_{PUBLISHER_STEM}"


class _Resp:
    def __init__(self, body): self.content, self.status_code = body, 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def raise_for_status(self): pass
    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


def _s3_listing(*keys: str) -> bytes:
    body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return ('<?xml version="1.0"?><ListBucketResult '
            'xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + body + "<IsTruncated>false</IsTruncated></ListBucketResult>").encode()


def _tarball_bytes() -> bytes:
    """A real tarball shaped like NCBI's: publisher-named PDF + .nxml under <PMCID>/."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (
            (f"{PMCID}/{PUBLISHER_STEM}.pdf", PDF_BYTES),
            (f"{PMCID}/{PUBLISHER_STEM}.nxml", XML_BYTES),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(downloader.time, "sleep", lambda *_a: None)


def _corpus_via_aws(tmp_path, monkeypatch):
    """acquire download --source aws → corpus/"""
    listing = _s3_listing(
        f"{PMCID}.1/{PMCID}.1.pdf", f"{PMCID}.1/{PMCID}.1.xml", f"{PMCID}.1/fig1.jpg",
    )

    def _get(url, **k):
        if "list-type=2" in url:
            return _Resp(listing)
        return _Resp(PDF_BYTES if url.endswith(".pdf") else XML_BYTES)

    monkeypatch.setattr(downloader.requests, "get", _get)
    corpus = tmp_path / "aws" / "corpus"
    ids = tmp_path / "aws_ids.txt"
    ids.write_text(f"{PMCID}\n", encoding="utf-8")
    report = downloader.download_papers(ids, corpus, source="aws")
    assert report.ok and report.succeeded == 1
    return corpus


def _corpus_via_ftp(tmp_path, monkeypatch):
    """acquire download --source ftp → tarballs → acquire unpack → corpus/"""
    monkeypatch.setattr(
        downloader, "get_download_link",
        lambda p: f"ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/e5/a1/{p}.tar.gz",
    )
    monkeypatch.setattr(downloader.requests, "get", lambda *a, **k: _Resp(_tarball_bytes()))
    tarballs = tmp_path / "ftp" / "tarballs"
    ids = tmp_path / "ftp_ids.txt"
    ids.write_text(f"{PMCID}\n", encoding="utf-8")
    report = downloader.download_papers(ids, tarballs, source="ftp")
    assert report.ok and report.succeeded == 1

    corpus = tmp_path / "ftp" / "corpus"
    unpack_tarballs(tarballs, corpus)
    return corpus


# ── the claim ─────────────────────────────────────────────────────────────────

def test_both_sources_produce_the_same_document_id(tmp_path, monkeypatch) -> None:
    aws_corpus = _corpus_via_aws(tmp_path, monkeypatch)
    ftp_corpus = _corpus_via_ftp(tmp_path, monkeypatch)

    # 1. identical source PDF bytes (the premise — otherwise nothing else means anything)
    aws_pdf = next(aws_corpus.rglob("*.pdf"))
    ftp_pdf = next(ftp_corpus.rglob("*.pdf"))
    assert aws_pdf.read_bytes() == ftp_pdf.read_bytes() == PDF_BYTES

    # 2. equivalent pre-organize layout: publisher-named PDF + <PMCID>.nxml
    assert aws_pdf.name == ftp_pdf.name == f"{PUBLISHER_STEM}.pdf"
    assert (aws_corpus / PMCID / f"{PMCID}.nxml").is_file()
    assert (ftp_corpus / PMCID / f"{PMCID}.nxml").is_file()

    # 3. identical organized filename
    org = {}
    for label, corpus in (("aws", aws_corpus), ("ftp", ftp_corpus)):
        pdf_dir, xml_dir = tmp_path / label / "opdf", tmp_path / label / "oxml"
        organize_pdfs(corpus, pdf_dir, xml_dir)
        org[label] = sorted(p.name for p in pdf_dir.glob("*.pdf"))
    assert org["aws"] == org["ftp"] == [EXPECTED_PDF_NAME]

    # 4. identical document ID — what ingest derives and documents.pmcid stores
    ids = {label: canonical_document_id(names[0].removesuffix(".pdf"))
           for label, names in org.items()}
    assert ids["aws"] == ids["ftp"] == EXPECTED_DOCUMENT_ID
    assert ".1" not in ids["aws"], "a version suffix must never reach the document ID"


def test_resume_an_ftp_corpus_with_aws_does_not_duplicate(tmp_path, monkeypatch) -> None:
    """The normal case now that AWS is the default: a corpus built over FTP, resumed
    with AWS. The paper must be recognised as already present, not re-added under a
    versioned identifier."""
    ftp_corpus = _corpus_via_ftp(tmp_path, monkeypatch)

    # Now point AWS at the SAME corpus directory.
    listing = _s3_listing(f"{PMCID}.1/{PMCID}.1.pdf", f"{PMCID}.1/{PMCID}.1.xml")

    def _get(url, **k):
        if "list-type=2" in url:
            return _Resp(listing)
        return _Resp(PDF_BYTES if url.endswith(".pdf") else XML_BYTES)

    monkeypatch.setattr(downloader.requests, "get", _get)
    ids = tmp_path / "resume_ids.txt"
    ids.write_text(f"{PMCID}\n", encoding="utf-8")
    report = downloader.download_papers(ids, ftp_corpus, source="aws")

    assert report.skipped == 1 and report.succeeded == 0, "already present → skip"
    assert report.ok

    # one paper, one PDF — no second copy under a versioned name
    pdfs = sorted(p.name for p in (ftp_corpus / PMCID).glob("*.pdf"))
    assert pdfs == [f"{PUBLISHER_STEM}.pdf"]
    assert not (ftp_corpus / PMCID / f"{PMCID}.1.pdf").exists()


def test_aws_refuses_rather_than_inventing_an_identifier(tmp_path, monkeypatch) -> None:
    """No self-uri → the article fails. The alternative is a document ID that silently
    disagrees with the FTP-derived one, which reads as a duplicate paper."""
    listing = _s3_listing(f"{PMCID}.1/{PMCID}.1.pdf", f"{PMCID}.1/{PMCID}.1.xml")

    def _get(url, **k):
        if "list-type=2" in url:
            return _Resp(listing)
        return _Resp(PDF_BYTES if url.endswith(".pdf") else b"<article/>")  # no self-uri

    monkeypatch.setattr(downloader.requests, "get", _get)
    assert downloader.download_paper_aws(PMCID, tmp_path / "corpus") == "failed"
    assert not (tmp_path / "corpus" / PMCID).glob("*.pdf") or \
        not list((tmp_path / "corpus" / PMCID).glob("*.pdf"))


def test_aws_records_provenance(tmp_path, monkeypatch) -> None:
    """The AWS key and publisher name are unrecoverable from the renamed files."""
    import json

    listing = _s3_listing(f"{PMCID}.1/{PMCID}.1.pdf", f"{PMCID}.1/{PMCID}.1.xml")

    def _get(url, **k):
        if "list-type=2" in url:
            return _Resp(listing)
        return _Resp(PDF_BYTES if url.endswith(".pdf") else XML_BYTES)

    monkeypatch.setattr(downloader.requests, "get", _get)
    corpus = tmp_path / "corpus"
    assert downloader.download_paper_aws(PMCID, corpus) == "succeeded"

    prov = json.loads((corpus / PMCID / "_source.json").read_text())
    assert prov["source"] == "aws"
    assert prov["aws_pdf_key"] == f"{PMCID}.1/{PMCID}.1.pdf"
    assert prov["publisher_pdf_filename"] == f"{PUBLISHER_STEM}.pdf"
    assert prov["version"] == 1
    assert prov["document_id"] == EXPECTED_DOCUMENT_ID


def test_provenance_file_does_not_disturb_organize(tmp_path, monkeypatch) -> None:
    """_source.json rides in the corpus dir; organize must ignore it."""
    corpus = _corpus_via_aws(tmp_path, monkeypatch)
    pdf_dir, xml_dir = tmp_path / "opdf", tmp_path / "oxml"
    organize_pdfs(corpus, pdf_dir, xml_dir)
    assert sorted(p.name for p in pdf_dir.iterdir()) == [EXPECTED_PDF_NAME]
    assert not any(p.suffix == ".json" for p in pdf_dir.iterdir())
    assert not any(p.suffix == ".json" for p in xml_dir.iterdir())
