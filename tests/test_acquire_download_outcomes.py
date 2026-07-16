"""B-117 — a download that did not deliver must not report success.

Originally `download_papers`' return value was discarded and the CLI returned 0
unconditionally: a run in which every paper 404'd printed "Done — 0 tarball(s)" and
exited 0. Observed live against NCBI.

The contract these tests pin:

* **failed > 0 → non-zero**, even alongside successes. A corpus quietly missing papers
  NCBI said it had is worse than a red exit.
* **skipped is not failed.** Outside the OA subset, or already on disk, are expected
  answers — not denied requests.
* **nothing requested → 0, but said out loud.** An empty run and a successful one must
  not look alike.
* **a 200 is not an archive.** Zero bytes, an HTML error page, or a truncated stream all
  count as failure — and are removed, because leaving them makes the next run's
  `target.exists()` skip them.
"""
from __future__ import annotations

import io
import tarfile

import pytest

from nlp_histo.acquisition import downloader
from nlp_histo.acquisition.downloader import DownloadReport, is_valid_archive
from nlp_histo.cli.main import main


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── archive validation ────────────────────────────────────────────────────────

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


# ── per-file download outcomes ────────────────────────────────────────────────

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


# ── run-level reports ─────────────────────────────────────────────────────────

def test_total_success(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, _real_targz_bytes())
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), tmp_path / "out")
    assert (r.requested, r.succeeded, r.failed, r.skipped) == (2, 2, 0, 0)
    assert r.ok


def test_total_failure(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, b"", status=404)
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), tmp_path / "out")
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
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), out)
    assert (r.succeeded, r.failed) == (1, 1)
    assert not r.ok, "a partial failure must not report success"
    assert (out / "PMC1.tar.gz").is_file(), "the successful archive must be preserved"


def test_not_in_oa_subset_is_skipped_not_failed(tmp_path, monkeypatch) -> None:
    """NCBI answering 'no package' is an answer, not a denial."""
    monkeypatch.setattr(downloader, "get_download_link", lambda p: None)
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1", "PMC2"), tmp_path / "out")
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
    r = downloader.download_papers(_pmcid_file(tmp_path, "PMC1"), out)
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


# ── CLI exit codes ────────────────────────────────────────────────────────────

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
