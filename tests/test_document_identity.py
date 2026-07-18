"""B-119 — AWS and FTP must reach the same document ID for the same paper.

Two names, kept apart on purpose:

* **PMC accession** — the bare NLM value, ``PMC8395919``.
* **document ID** — this project's composite identifier,
  ``PMC8395919_dermatopathology-08-00036``, which ``documents.pmcid`` stores and which
  977 corpus rows, the frozen replay artifacts and the silver labels are all keyed on.

The hazard: AWS names its objects ``PMC8395919.1.pdf`` (``.1`` = article version). Taken
naively as a stem, that mints ``PMC8395919.1`` — a *second* document for a paper the
corpus already has, appearing only as a duplicate row rather than an error. So the AWS
route names the PDF from the JATS ``self-uri`` (the same name the tarball carried), and a
shared parser strips any version suffix that still slips through.
"""
from __future__ import annotations

import pytest

from nlp_histo.acquisition import downloader
from nlp_histo.acquisition.downloader import (
    UnresolvablePdfName,
    publisher_pdf_filename,
)
from nlp_histo.document_id import canonical_document_id, pmc_accession


# the shared normalisation safety net

@pytest.mark.parametrize("raw,expected", [
    # the AWS shape: a version suffix must go
    ("PMC8395919.1", "PMC8395919"),
    ("PMC8395919.12", "PMC8395919"),
    # version + publisher component: strip the version, keep the publisher
    ("PMC8395919.1_dermatopathology-08-00036", "PMC8395919_dermatopathology-08-00036"),
    # the established shape: untouched
    ("PMC8395919_dermatopathology-08-00036", "PMC8395919_dermatopathology-08-00036"),
    ("PMC10047158_dermatopathology-10-00017", "PMC10047158_dermatopathology-10-00017"),
    ("PMC8395919", "PMC8395919"),
    # publisher stems containing dots must survive — the whole point of the lookahead
    ("PMC8395919_paper.v2.final", "PMC8395919_paper.v2.final"),
    ("PMC7810815_main", "PMC7810815_main"),
    # not a shape NLM produces: refuse to guess rather than truncate blindly
    ("PMC8395919.1.2", "PMC8395919.1.2"),
    # nothing that isn't an accession is touched
    ("not-a-pmcid.1", "not-a-pmcid.1"),
    ("PMCX.1", "PMCX.1"),
    ("", ""),
])
def test_canonical_document_id(raw: str, expected: str) -> None:
    assert canonical_document_id(raw) == expected


def test_canonicalisation_is_idempotent() -> None:
    once = canonical_document_id("PMC8395919.1_derm-08-00036")
    assert canonical_document_id(once) == once


def test_pmc_accession_extracts_the_bare_value() -> None:
    assert pmc_accession("PMC8395919_dermatopathology-08-00036") == "PMC8395919"
    assert pmc_accession("PMC8395919") == "PMC8395919"
    assert pmc_accession("not-a-pmcid") is None


# self-uri: authoritative, and untrusted

def _jats(href: str | None, *, extra: str = "", content_type: str = "pmc-pdf") -> bytes:
    uri = (
        f'<self-uri xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'content-type="{content_type}" xlink:href="{href}"/>' if href is not None else ""
    )
    return f"<article>{uri}{extra}</article>".encode()


def test_resolves_the_publisher_filename() -> None:
    assert publisher_pdf_filename(_jats("dermatopathology-08-00036.pdf")) == \
        "dermatopathology-08-00036.pdf"


def test_missing_self_uri_is_rejected() -> None:
    """No authority → fail the article. Inventing a name mints a mismatched ID."""
    with pytest.raises(UnresolvablePdfName, match="no <self-uri"):
        publisher_pdf_filename(_jats(None))


def test_self_uri_of_another_content_type_does_not_count() -> None:
    with pytest.raises(UnresolvablePdfName, match="no <self-uri"):
        publisher_pdf_filename(_jats("paper.xml", content_type="pmc-xml"))


def test_multiple_differing_self_uris_are_rejected_as_ambiguous() -> None:
    extra = ('<self-uri xmlns:xlink="http://www.w3.org/1999/xlink" '
             'content-type="pmc-pdf" xlink:href="other.pdf"/>')
    with pytest.raises(UnresolvablePdfName, match="ambiguous"):
        publisher_pdf_filename(_jats("paper.pdf", extra=extra))


def test_multiple_identical_self_uris_are_fine() -> None:
    extra = ('<self-uri xmlns:xlink="http://www.w3.org/1999/xlink" '
             'content-type="pmc-pdf" xlink:href="paper.pdf"/>')
    assert publisher_pdf_filename(_jats("paper.pdf", extra=extra)) == "paper.pdf"


@pytest.mark.parametrize("href,reason", [
    ("", "empty"),
    ("   ", "empty"),
    ("/etc/passwd.pdf", "absolute"),
    ("C:/windows/x.pdf", "absolute"),
    ("//evil.example/x.pdf", "URL"),
    ("https://evil.example/x.pdf", "URL"),
    ("../../../etc/passwd.pdf", "traverses"),
    ("a/../../b.pdf", "traverses"),
    ("paper.xml", "not a PDF"),
    ("paper", "not a PDF"),
])
def test_hostile_self_uri_values_are_rejected(href: str, reason: str) -> None:
    """The href is publisher-supplied text that becomes a path we write to."""
    with pytest.raises(UnresolvablePdfName):
        publisher_pdf_filename(_jats(href))


def test_nested_reference_reduces_to_its_filename() -> None:
    """Relative and traversal-free, so the last component cannot escape the directory —
    and the tarball flattened these too."""
    assert publisher_pdf_filename(_jats("supplementary/paper.pdf")) == "paper.pdf"


def test_unparsable_xml_is_rejected() -> None:
    with pytest.raises(UnresolvablePdfName, match="unparsable"):
        publisher_pdf_filename(b"<article>truncated")


# version selection is numeric

def _listing(*keys: str, truncated: bool = False, token: str = "") -> bytes:
    body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    extra = (f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}"
             f"</NextContinuationToken>") if truncated else "<IsTruncated>false</IsTruncated>"
    return ('<?xml version="1.0"?><ListBucketResult '
            'xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + body + extra + "</ListBucketResult>").encode()


class _Resp:
    def __init__(self, content): self.content = content
    def raise_for_status(self): pass


def test_version_10_beats_version_9_numerically(monkeypatch) -> None:
    """A lexical max would pick '.9' and silently serve a superseded article."""
    monkeypatch.setattr(
        downloader.requests, "get",
        lambda *a, **k: _Resp(_listing("PMC1.9/PMC1.9.pdf", "PMC1.10/PMC1.10.pdf")),
    )
    assert downloader.aws_object_keys("PMC1") == ["PMC1.10/PMC1.10.pdf"]


def test_version_2_beats_version_1(monkeypatch) -> None:
    monkeypatch.setattr(
        downloader.requests, "get",
        lambda *a, **k: _Resp(_listing(
            "PMC1.1/PMC1.1.pdf", "PMC1.1/PMC1.1.xml",
            "PMC1.2/PMC1.2.pdf", "PMC1.2/PMC1.2.xml",
        )),
    )
    assert downloader.aws_object_keys("PMC1") == ["PMC1.2/PMC1.2.pdf", "PMC1.2/PMC1.2.xml"]


# pagination cannot silently omit objects

def test_listing_follows_pagination(monkeypatch) -> None:
    """S3 caps a page at 1000 keys; a figure-heavy article can exceed it, and the missing
    object could be the PDF. Ignoring IsTruncated would look like a complete listing."""
    pages = [
        _listing("PMC1.1/a.jpg", truncated=True, token="TOKEN-1"),
        _listing("PMC1.1/PMC1.1.pdf", "PMC1.1/PMC1.1.xml"),
    ]
    seen_urls = []

    def _get(url, **k):
        seen_urls.append(url)
        return _Resp(pages[len(seen_urls) - 1])

    monkeypatch.setattr(downloader.requests, "get", _get)
    keys = downloader.aws_object_keys("PMC1")
    assert "PMC1.1/PMC1.1.pdf" in keys, "the PDF was on page 2 and must not be dropped"
    assert len(seen_urls) == 2
    assert "continuation-token=TOKEN-1" in seen_urls[1]


def test_listing_stops_when_not_truncated(monkeypatch) -> None:
    calls = []

    def _get(url, **k):
        calls.append(url)
        return _Resp(_listing("PMC1.1/PMC1.1.pdf"))

    monkeypatch.setattr(downloader.requests, "get", _get)
    downloader.aws_object_keys("PMC1")
    assert len(calls) == 1
