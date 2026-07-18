"""B-117 — a download that did not deliver must not report success.

Originally `download_papers`' return value was discarded and the CLI returned 0
unconditionally: a run in which every paper 404'd printed "Done — 0 tarball(s)" and
exited 0. Observed live against NCBI.

The contract these tests pin:

* failed > 0 → non-zero, even alongside successes. A corpus quietly missing papers
  NCBI said it had is worse than a red exit.
* skipped is not failed. Outside the OA subset, or already on disk, are expected
  answers — not denied requests.
* nothing requested → 0, but said out loud. An empty run and a successful one must
  not look alike.
* a 200 is not an archive. Zero bytes, an HTML error page, or a truncated stream all
  count as failure — and are removed, because leaving them makes the next run's
  `target.exists()` skip them.
"""
from __future__ import annotations

import io
import tarfile

import pytest

from nlp_histo.acquisition import downloader
from nlp_histo.acquisition.downloader import (
    DownloadReport,
    candidate_urls,
    is_valid_archive,
)
from nlp_histo.cli.main import main


# helpers

def _pmcid_file(tmp_path, *ids):
    p = tmp_path / "ids.txt"
    p.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return p


def _real_targz_bytes() -> bytes:
    """A genuine, minimal .tar.gz — the fixture must be real or it proves nothing."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"%PDF-1.4 fake"
        info = tarfile.TarInfo("PMC1/paper.pdf")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body, self.status_code = body, status
        # `.content` for the non-streamed XML fetch; `iter_content` for streamed objects.
        self.content = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests_HTTPError(f"{self.status_code} Client Error")

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


def requests_HTTPError(msg):  # noqa: N802 — mirrors requests' name at the call site
    import requests

    return requests.exceptions.HTTPError(msg)


@pytest.fixture(autouse=True)
def _no_politeness_sleep(monkeypatch):
    monkeypatch.setattr(downloader.time, "sleep", lambda *_a: None)


def _serve(monkeypatch, body: bytes, status: int = 200):
    monkeypatch.setattr(downloader, "get_download_link", lambda p: f"https://x/{p}.tar.gz")
    monkeypatch.setattr(
        downloader.requests, "get", lambda *a, **k: _FakeResponse(body, status)
    )


# archive validation

def test_a_real_targz_is_valid(tmp_path) -> None:
    p = tmp_path / "ok.tar.gz"
    p.write_bytes(_real_targz_bytes())
    assert is_valid_archive(p)


def test_zero_byte_file_is_not_valid(tmp_path) -> None:
    p = tmp_path / "empty.tar.gz"
    p.write_bytes(b"")
    assert not is_valid_archive(p)


def test_html_error_page_is_not_valid(tmp_path) -> None:
    """A 200 carrying an error page is the dangerous case: it looks like a download."""
    p = tmp_path / "nope.tar.gz"
    p.write_bytes(b"<!DOCTYPE html><html><body>404 Not Found</body></html>")
    assert not is_valid_archive(p)


def test_truncated_gzip_is_not_valid(tmp_path) -> None:
    """Right magic bytes, unusable content — the header alone proves nothing."""
    p = tmp_path / "trunc.tar.gz"
    p.write_bytes(_real_targz_bytes()[:20])
    assert not is_valid_archive(p)


# per-file download outcomes

def test_zero_byte_response_counts_as_failure_and_is_removed(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, b"")
    target = tmp_path / "PMC1.tar.gz"
    assert downloader.download_file("https://x/PMC1.tar.gz", target) is False
    assert not target.exists(), "an unusable file must not survive to be skipped on re-run"


def test_non_archive_response_counts_as_failure_and_is_removed(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, b"<html>error</html>")
    target = tmp_path / "PMC1.tar.gz"
    assert downloader.download_file("https://x/PMC1.tar.gz", target) is False
    assert not target.exists()


def test_http_error_leaves_no_partial_file(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, b"", status=404)
    target = tmp_path / "PMC1.tar.gz"
    assert downloader.download_file("https://x/PMC1.tar.gz", target) is False
    assert not target.exists()


def test_valid_archive_is_kept(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, _real_targz_bytes())
    target = tmp_path / "PMC1.tar.gz"
    assert downloader.download_file("https://x/PMC1.tar.gz", target) is True
    assert target.is_file() and target.stat().st_size > 0


# B-118: NCBI moved the packages; the API still advertises the old path

_ADVERTISED = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz"
_RELOCATED = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_package/e5/a1/PMC8395919.tar.gz"


def test_candidates_are_advertised_first_then_relocated() -> None:
    """Order matters: the advertised URL is tried first so this self-heals the day NCBI
    fixes its API, rather than pinning us to a directory they intend to delete."""
    assert candidate_urls(_ADVERTISED) == [
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/e5/a1/PMC8395919.tar.gz",
        _RELOCATED,
    ]


def test_ftp_scheme_is_rewritten_to_https() -> None:
    assert all(u.startswith("https://") for u in candidate_urls(_ADVERTISED))


def test_an_already_relocated_url_is_not_double_rewritten() -> None:
    """Guard the string surgery: inserting deprecated/ twice would 404 on both tries."""
    assert candidate_urls(_RELOCATED) == [_RELOCATED]


def test_a_non_pmc_url_gets_no_fallback() -> None:
    """The relocation is a PMC fact, not a general one."""
    assert candidate_urls("https://example.org/thing.tar.gz") == ["https://example.org/thing.tar.gz"]


def test_download_package_falls_back_when_the_advertised_url_404s(tmp_path, monkeypatch) -> None:
    """Today's reality: advertised 404s, relocated serves the archive."""
    good = _real_targz_bytes()
    seen = []

    def _get(url, **k):
        seen.append(url)
        return _FakeResponse(good) if "/deprecated/" in url else _FakeResponse(b"", 404)

    monkeypatch.setattr(downloader.requests, "get", _get)
    target = tmp_path / "PMC8395919.tar.gz"
    assert downloader.download_package(_ADVERTISED, target) is True
    assert target.is_file()
    assert len(seen) == 2 and "/deprecated/" in seen[1], "must try advertised before relocated"


def test_download_package_does_not_fall_back_when_advertised_works(tmp_path, monkeypatch) -> None:
    """When NCBI repairs its API, the fallback must go quiet on its own."""
    seen = []

    def _get(url, **k):
        seen.append(url)
        return _FakeResponse(_real_targz_bytes())

    monkeypatch.setattr(downloader.requests, "get", _get)
    assert downloader.download_package(_ADVERTISED, tmp_path / "x.tar.gz") is True
    assert len(seen) == 1, "the advertised URL worked; nothing else should be requested"


def test_download_package_fails_when_both_candidates_fail(tmp_path, monkeypatch, capsys) -> None:
    """After August 2026 NCBI deletes the legacy tree — this must fail loudly, not
    silently produce an empty corpus."""
    monkeypatch.setattr(downloader.requests, "get", lambda url, **k: _FakeResponse(b"", 404))
    assert downloader.download_package(_ADVERTISED, tmp_path / "x.tar.gz") is False
    assert "Failed to download" in capsys.readouterr().out


# AWS backend (B-118's durable successor)

def _s3_listing(*keys: str) -> bytes:
    body = "".join(f"<Contents><Key>{k}</Key><Size>1</Size></Contents>" for k in keys)
    return (
        '<?xml version="1.0"?><ListBucketResult '
        'xmlns="http://s3.amazonaws.com/doc/2006-03-01/">' + body + "</ListBucketResult>"
    ).encode()


class _XmlResponse:
    def __init__(self, content): self.content, self.status_code = content, 200
    def raise_for_status(self): pass


def test_aws_picks_the_newest_version_only(monkeypatch) -> None:
    """A paper with v1 and v2 must yield v2's objects, never a mixture."""
    monkeypatch.setattr(
        downloader.requests, "get",
        lambda *a, **k: _XmlResponse(_s3_listing(
            "PMC8395919.1/PMC8395919.1.pdf", "PMC8395919.1/PMC8395919.1.xml",
            "PMC8395919.2/PMC8395919.2.pdf", "PMC8395919.2/PMC8395919.2.xml",
        )),
    )
    keys = downloader.aws_object_keys("PMC8395919")
    assert keys == ["PMC8395919.2/PMC8395919.2.pdf", "PMC8395919.2/PMC8395919.2.xml"]


def test_aws_absent_paper_is_skipped_not_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(downloader.requests, "get", lambda *a, **k: _XmlResponse(_s3_listing()))
    assert downloader.download_paper_aws("PMC404", tmp_path) == "skipped"


def test_aws_listing_error_is_a_failure_not_a_skip(tmp_path, monkeypatch) -> None:
    """A lookup that errored tells us nothing — it must not be read as 'not in the set'."""
    def _boom(*a, **k):
        raise ConnectionError("network down")
    monkeypatch.setattr(downloader.requests, "get", _boom)
    assert downloader.download_paper_aws("PMC1", tmp_path) == "failed"


def test_aws_paper_without_a_pdf_is_a_failure(tmp_path, monkeypatch) -> None:
    """A PDF pipeline getting no PDF is a gap, not a success."""
    monkeypatch.setattr(
        downloader.requests, "get",
        lambda *a, **k: _XmlResponse(_s3_listing("PMC1.1/PMC1.1.xml", "PMC1.1/PMC1.1.txt")),
    )
    assert downloader.download_paper_aws("PMC1", tmp_path) == "failed"


_JATS = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
    '<self-uri content-type="pmc-pdf" xlink:href="paper-08-00036.pdf"/></article>'
).encode()


def test_aws_writes_the_layout_organize_expects(tmp_path, monkeypatch) -> None:
    """corpus/<PMCID>/ with a publisher-named PDF and <PMCID>.nxml — exactly what unpack
    produces, so organize consumes it unchanged and both sources reach the same document
    ID. The AWS object name (PMC1.1.pdf) must not survive: it would mint PMC1.1 (B-119).
    """
    listing = _s3_listing("PMC1.1/PMC1.1.pdf", "PMC1.1/PMC1.1.xml", "PMC1.1/fig1.jpg")

    def _get(url, **k):
        if "list-type=2" in url:
            return _XmlResponse(listing)
        return _FakeResponse(b"%PDF-1.4 body" if url.endswith(".pdf") else _JATS)

    monkeypatch.setattr(downloader.requests, "get", _get)
    assert downloader.download_paper_aws("PMC1", tmp_path) == "succeeded"
    assert (tmp_path / "PMC1" / "paper-08-00036.pdf").is_file(), "named from the JATS self-uri"
    assert (tmp_path / "PMC1" / "PMC1.nxml").is_file(), "unpack names the XML <PMCID>.nxml"
    assert not (tmp_path / "PMC1" / "PMC1.1.pdf").exists(), "the versioned name must not survive"
    assert not (tmp_path / "PMC1" / "fig1.jpg").exists(), "only PDF+XML are needed"


def test_aws_rejects_a_pdf_that_is_not_a_pdf(tmp_path, monkeypatch) -> None:
    """A 200 carrying an error page must not land as a PDF (same lesson as B-117)."""
    listing = _s3_listing("PMC1.1/PMC1.1.pdf", "PMC1.1/PMC1.1.xml")

    def _get(url, **k):
        if "list-type=2" in url:
            return _XmlResponse(listing)
        return _FakeResponse(_JATS) if url.endswith(".xml") else _FakeResponse(b"<html>nope</html>")

    monkeypatch.setattr(downloader.requests, "get", _get)
    assert downloader.download_paper_aws("PMC1", tmp_path) == "failed"
    assert not (tmp_path / "PMC1" / "paper-08-00036.pdf").exists()


def test_aws_already_downloaded_is_skipped(tmp_path, monkeypatch) -> None:
    (tmp_path / "PMC1").mkdir()
    (tmp_path / "PMC1" / "PMC1.1.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(
        downloader.requests, "get",
        lambda *a, **k: pytest.fail("must not hit the network for an existing paper"),
    )
    assert downloader.download_paper_aws("PMC1", tmp_path) == "skipped"


def test_unknown_source_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        downloader.download_papers(_pmcid_file(tmp_path, "PMC1"), tmp_path / "o", source="carrier-pigeon")


# run-level reports (FTP backend; the AWS equivalents are above)

def test_total_success(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, _real_targz_bytes())
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), tmp_path / "out", source="ftp")
    assert (r.requested, r.succeeded, r.failed, r.skipped) == (2, 2, 0, 0)
    assert r.ok


def test_total_failure(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, b"", status=404)
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), tmp_path / "out", source="ftp")
    assert (r.succeeded, r.failed) == (0, 2)
    assert not r.ok


def test_partial_failure_is_not_ok_and_keeps_the_good_file(tmp_path, monkeypatch) -> None:
    """The case the first fix got wrong: one success must not excuse one failure."""
    good = _real_targz_bytes()
    monkeypatch.setattr(downloader, "get_download_link", lambda p: f"https://x/{p}.tar.gz")
    monkeypatch.setattr(
        downloader.requests, "get",
        lambda url, **k: _FakeResponse(good) if "PMC1." in url else _FakeResponse(b"", 404),
    )
    out = tmp_path / "out"
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), out, source="ftp")
    assert (r.succeeded, r.failed) == (1, 1)
    assert not r.ok, "a partial failure must not report success"
    assert (out / "PMC1.tar.gz").is_file(), "the successful archive must be preserved"


def test_not_in_oa_subset_is_skipped_not_failed(tmp_path, monkeypatch) -> None:
    """NCBI answering 'no package' is an answer, not a denial."""
    monkeypatch.setattr(downloader, "get_download_link", lambda p: None)
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), tmp_path / "out", source="ftp")
    assert (r.succeeded, r.failed, r.skipped) == (0, 0, 2)
    assert r.ok


def test_already_present_is_skipped(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "PMC1.tar.gz").write_bytes(_real_targz_bytes())
    monkeypatch.setattr(
        downloader, "get_download_link",
        lambda p: pytest.fail("must not look up an already-downloaded paper"),
    )
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1"), out, source="ftp")
    assert (r.succeeded, r.failed, r.skipped) == (0, 0, 1)
    assert r.ok


def test_zero_requested_is_reported_explicitly(tmp_path, capsys) -> None:
    r = downloader.download_papers(_pmcid_file(tmp_path), tmp_path / "out")
    assert (r.requested, r.succeeded, r.failed, r.skipped) == (0, 0, 0, 0)
    assert r.ok
    assert "Nothing requested" in capsys.readouterr().out


def test_report_summary_states_all_three_counts() -> None:
    s = DownloadReport(requested=5, succeeded=2, failed=1, skipped=2).summary()
    assert "2 succeeded" in s and "1 failed" in s and "2 skipped" in s and "5 requested" in s


# CLI exit codes

def _cli(tmp_path, ids):
    return main(["acquire", "download", "--pmcid-file", str(ids), "--output-dir", str(tmp_path / "o")])


def test_cli_exits_nonzero_on_partial_failure(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        downloader, "download_papers",
        lambda *a, **k: DownloadReport(requested=2, succeeded=1, failed=1, skipped=0),
    )
    assert _cli(tmp_path, _pmcid_file(tmp_path, "PMC1", "PMC2")) == 1
    err = capsys.readouterr().err
    assert "1 of 2" in err and "kept" in err


def test_cli_exits_zero_on_total_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        downloader, "download_papers",
        lambda *a, **k: DownloadReport(requested=2, succeeded=2, failed=0, skipped=0),
    )
    assert _cli(tmp_path, _pmcid_file(tmp_path, "PMC1", "PMC2")) == 0


def test_cli_exits_zero_when_only_skips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        downloader, "download_papers",
        lambda *a, **k: DownloadReport(requested=2, succeeded=0, failed=0, skipped=2),
    )
    assert _cli(tmp_path, _pmcid_file(tmp_path, "PMC1", "PMC2")) == 0


def test_cli_exits_zero_when_nothing_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        downloader, "download_papers",
        lambda *a, **k: DownloadReport(requested=0, succeeded=0, failed=0, skipped=0),
    )
    assert _cli(tmp_path, _pmcid_file(tmp_path)) == 0
