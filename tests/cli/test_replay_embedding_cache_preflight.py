"""B-112 — a command advertised as API-free must not be able to reach a paid provider.

The bug: the replay preflight required ``embedding_cache_openai.sqlite`` but not
``embedding_cache_gemini.sqlite``, while analyses 06/10/12 load the map context with the
gemini embedder. A tree carrying only the OpenAI cache passed validation, then missed on
every claim and embedded ~15k of them against a live Gemini client. The gemini path was
also resolved from ``eval.paths.REPO_ROOT`` rather than ``--artifact-root``, so the flag
that is supposed to say where the data lives did not govern it.

Two layers are tested here, because either alone is insufficient:

* preflight — enumerable incompleteness is caught up front, non-zero, nothing written;
* cache-only — any unexpected miss (race, malformed row, a preflight that did not
  enumerate some text) raises rather than billing.

No test constructs a real provider or performs any network call.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from nlp_histo.workflows import replay
from nlp_histo.workflows.replay import EmbeddingCacheIncompleteError


# fixture: a minimal but real artifact tree

def _write_voter_cache(root, claims: list[str]) -> None:
    """A voter cache whose findings carry exactly ``claims``.

    Field-for-field the shape of the real `eval/data/map_primer/voter_cache.json`:
    l1/l2 map a voter id to a list of outputs, l3 to a single output, and every
    Finding carries the full required set. This must stay valid — `_required_claim_texts`
    silently skips rows that fail `AuditableSummary.model_validate`, exactly as the
    production pre-warm does, so an under-specified fixture would yield zero claims and
    test nothing.
    """
    def _finding(claim: str) -> dict:
        # Copied field-for-field from a real row of eval/data/map_primer/voter_cache.json
        # (evidence is a list of sentence ids; confidence is an enum, not a float;
        # grounding_score is nullable at MAP time).
        return {
            "category": "demographic",
            "claim": claim,
            "evidence": ["S1|PMC_TEST|1"],
            "confidence": "high",
            "verbatim_support": "verbatim sentence supporting the claim.",
            "subject_entity": "subject",
            "outcome_entity": "outcome",
            "relation_type": "demographic",
            "direction": "positive",
            "scope": {
                "disease_subtype": None, "cohort_n": None, "assay_method": None,
                "biomarker_cutoff": None, "tissue_site": None,
                "treatment_context": None, "endpoint": None,
                "study_design": "cohort", "scope_parsed": True,
            },
            "grounding_score": None,
        }

    def _summary(claim_texts: list[str]) -> dict:
        return {
            "chunk_id": "c1",
            "findings": [_finding(c) for c in claim_texts],
            "summary_text": "s",
            "audit_metadata": {
                "sentences_analyzed": 1,
                "sentences_cited": ["S1|PMC_TEST|1"],
                "pmcids_referenced": ["PMC_TEST"],
                "uncited_sentences": [],
            },
        }

    (root / "eval" / "data" / "map_primer").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "data" / "map_primer" / "voter_cache.json").write_text(
        json.dumps({"case1": {"l1": {"voter_a": [_summary(claims)]}, "l2": {}, "l3": {}}}),
        encoding="utf-8",
    )


def _write_cache(path, model: str, texts: list[str]) -> None:
    """Create a real SQLite embedding cache holding ``texts`` under ``model``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from nlp_histo.evaluation.matching.matcher import make_embedding_cache

    cache = make_embedding_cache(path, model)
    for t in texts:
        cache.set(t, [0.1, 0.2, 0.3])
    cache.save()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A non-default --artifact-root with both caches complete."""
    from nlp_histo.evaluation.matching.matcher import EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL

    claims = ["alpha claim", "beta claim"]
    root = tmp_path / "artifacts"
    _write_voter_cache(root, claims)
    _write_cache(root / replay.FROZEN_EMBEDDING_CACHES["openai"], EMBEDDING_MODEL, claims)
    _write_cache(root / replay.FROZEN_EMBEDDING_CACHES["gemini"], GEMINI_EMBEDDING_MODEL, claims)
    monkeypatch.setattr(replay, "_REPO_ROOT", root)
    return root, claims


# preflight

def test_complete_caches_pass(tree) -> None:
    root, _ = tree
    replay.validate_embedding_cache_entries(root)  # must not raise


def test_missing_openai_cache_is_rejected(tree) -> None:
    root, _ = tree
    (root / replay.FROZEN_EMBEDDING_CACHES["openai"]).unlink()
    with pytest.raises(EmbeddingCacheIncompleteError) as exc:
        replay.validate_embedding_cache_entries(root)
    assert "openai" in str(exc.value)


def test_missing_gemini_cache_is_rejected(tree) -> None:
    """The regression: this used to pass validation and then spend money."""
    root, _ = tree
    (root / replay.FROZEN_EMBEDDING_CACHES["gemini"]).unlink()
    with pytest.raises(EmbeddingCacheIncompleteError) as exc:
        replay.validate_embedding_cache_entries(root)
    assert "gemini" in str(exc.value)


def test_incomplete_openai_cache_is_rejected(tree) -> None:
    """A present-but-partial cache is a valid SQLite file — existence proves nothing."""
    from nlp_histo.evaluation.matching.matcher import EMBEDDING_MODEL

    root, claims = tree
    path = root / replay.FROZEN_EMBEDDING_CACHES["openai"]
    path.unlink()
    _write_cache(path, EMBEDDING_MODEL, claims[:1])  # one of two
    with pytest.raises(EmbeddingCacheIncompleteError) as exc:
        replay.validate_embedding_cache_entries(root)
    msg = str(exc.value)
    assert "openai" in msg and "1 of 2 required entries missing" in msg


def test_incomplete_gemini_cache_is_rejected(tree) -> None:
    from nlp_histo.evaluation.matching.matcher import GEMINI_EMBEDDING_MODEL

    root, claims = tree
    path = root / replay.FROZEN_EMBEDDING_CACHES["gemini"]
    path.unlink()
    _write_cache(path, GEMINI_EMBEDDING_MODEL, claims[:1])
    with pytest.raises(EmbeddingCacheIncompleteError) as exc:
        replay.validate_embedding_cache_entries(root)
    assert "gemini" in str(exc.value)


def test_empty_cache_is_rejected(tree) -> None:
    """An empty database is the shape `make_embedding_cache` creates on a missing file."""
    from nlp_histo.evaluation.matching.matcher import GEMINI_EMBEDDING_MODEL

    root, _ = tree
    path = root / replay.FROZEN_EMBEDDING_CACHES["gemini"]
    path.unlink()
    _write_cache(path, GEMINI_EMBEDDING_MODEL, [])
    with pytest.raises(EmbeddingCacheIncompleteError):
        replay.validate_embedding_cache_entries(root)


def test_corrupt_cache_is_rejected_not_ignored(tree) -> None:
    """A cache that cannot be read must fail closed, never be treated as complete.

    The sidecars must go too: SQLite in WAL mode keeps committed rows in
    ``<db>-wal`` until a checkpoint, so overwriting only the main file lets the next
    connection recover the data and report a healthy cache — the corruption would be
    silently undone and this test would prove nothing.
    """
    root, _ = tree
    path = root / replay.FROZEN_EMBEDDING_CACHES["gemini"]
    for sidecar in (path, path.with_suffix(path.suffix + "-wal"),
                    path.with_suffix(path.suffix + "-shm"),
                    path.with_suffix(path.suffix + "-journal")):
        if sidecar.exists():
            sidecar.unlink()
    path.write_bytes(b"this is definitively not a SQLite database")
    with pytest.raises(EmbeddingCacheIncompleteError):
        replay.validate_embedding_cache_entries(root)


def test_error_reports_counts_and_paths_but_not_claim_texts(tree) -> None:
    """Requirement: counts + paths, no secrets, no dumping thousands of keys."""
    root, claims = tree
    (root / replay.FROZEN_EMBEDDING_CACHES["gemini"]).unlink()
    with pytest.raises(EmbeddingCacheIncompleteError) as exc:
        replay.validate_embedding_cache_entries(root)
    msg = str(exc.value)
    assert "embedding_cache_gemini.sqlite" in msg     # path
    assert "required entries missing" in msg          # count
    for claim in claims:
        assert claim not in msg, "claim text must not be echoed into the error"


# the artifact root is authoritative

def test_caches_resolve_from_artifact_root_not_the_repository(tree, monkeypatch) -> None:
    """--artifact-root governs. A non-default root is what these tests use throughout;
    assert the resolver actually derives from it rather than any repo-anchored constant."""
    root, _ = tree
    assert replay.frozen_embedding_cache("openai") == root / replay.FROZEN_EMBEDDING_CACHES["openai"]
    assert replay.frozen_embedding_cache("gemini") == root / replay.FROZEN_EMBEDDING_CACHES["gemini"]
    for kind in ("openai", "gemini"):
        assert str(replay.frozen_embedding_cache(kind)).startswith(str(root))


def test_unknown_embedder_kind_is_a_loud_error(tree) -> None:
    with pytest.raises(ValueError):
        replay.frozen_embedding_cache("not-an-embedder")


# main() exit code + nothing written

def test_incomplete_cache_exits_4_and_writes_nothing(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(replay, "validate_artifacts", lambda root: [])
    monkeypatch.setattr(replay, "_require_umls_or_refuse", lambda: None)

    def _incomplete(root):
        raise EmbeddingCacheIncompleteError("gemini: 15273 of 15273 required entries missing")

    monkeypatch.setattr(replay, "validate_embedding_cache_entries", _incomplete)
    monkeypatch.setattr(replay, "_run_replay", lambda: pytest.fail("must not run"))

    out_dir = tmp_path / "out"
    rc = replay.main(["--artifact-root", str(tmp_path), "--output-dir", str(out_dir)])
    assert rc == 4
    assert "required entries missing" in capsys.readouterr().err
    assert not out_dir.exists(), "a refused run must leave no output directory"


def test_cache_gate_runs_before_any_output_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(replay, "validate_artifacts", lambda root: [])
    monkeypatch.setattr(replay, "_require_umls_or_refuse", lambda: None)
    out_dir = tmp_path / "out"
    seen: dict[str, bool] = {}
    monkeypatch.setattr(
        replay, "validate_embedding_cache_entries",
        lambda root: seen.__setitem__("existed", out_dir.exists()),
    )
    monkeypatch.setattr(replay, "_run_replay", lambda: None)

    assert replay.main(["--artifact-root", str(tmp_path), "--output-dir", str(out_dir)]) == 0
    assert seen["existed"] is False


# cache-only: no provider can be reached at runtime

def test_strict_cache_only_never_constructs_a_provider(tree, monkeypatch) -> None:
    """Proof that no paid client is built and no key is consulted in cache-only mode."""
    from eval.silver.analysis import map_context

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import nlp_histo.evaluation.matching.embedders as embedders
    import nlp_histo.pipeline.stages.knowledge_extraction.agreement.providers as providers

    for mod, name in (
        (embedders, "GeminiEmbedder"), (embedders, "OpenAIEmbedder"),
        (providers, "GeminiEmbedder"), (providers, "OpenAIEmbedder"),
    ):
        monkeypatch.setattr(
            mod, name,
            lambda *a, **k: pytest.fail("a provider was constructed in cache-only mode"),
        )

    root, claims = tree
    ctx = map_context._load_map_context(
        "gemini",
        embed_cache_path=str(root / replay.FROZEN_EMBEDDING_CACHES["gemini"]),
        cache_path=root / "eval" / "data" / "map_primer" / "voter_cache.json",
        silver_path=_empty_silver(root),
        strict_cache_only=True,
    )
    # Cached text still resolves — the cache, not the provider, answers.
    # approx on the inner vector: the SQLite cache stores float32, so the round-trip is
    # not bit-exact, and pytest.approx does not descend into nested sequences.
    vectors = ctx.agreement_embed_fn([claims[0]])
    assert len(vectors) == 1
    assert vectors[0] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)


def test_unexpected_runtime_miss_raises_instead_of_billing(tree) -> None:
    """The backstop: even if the preflight missed something, a miss must not bill."""
    from eval.silver.analysis.map_context import CacheOnlyViolation, _load_map_context

    root, _ = tree
    ctx = _load_map_context(
        "gemini",
        embed_cache_path=str(root / replay.FROZEN_EMBEDDING_CACHES["gemini"]),
        cache_path=root / "eval" / "data" / "map_primer" / "voter_cache.json",
        silver_path=_empty_silver(root),
        strict_cache_only=True,
    )
    with pytest.raises(CacheOnlyViolation) as exc:
        ctx.agreement_embed_fn(["a text that was never embedded"])
    msg = str(exc.value)
    assert "PAID" in msg
    assert "a text that was never embedded" not in msg, "must not echo the text"


def _empty_silver(root):
    p = root / "eval" / "data" / "silver_empty.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def test_sqlite_fixture_is_a_real_database(tree) -> None:
    """Guard the fixture: if these stopped being real SQLite caches, the suite is theatre."""
    root, _ = tree
    for kind in ("openai", "gemini"):
        path = root / replay.FROZEN_EMBEDDING_CACHES[kind]
        with sqlite3.connect(path) as con:
            assert con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] > 0
