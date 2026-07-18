"""Tests for the SQLite embedding-cache backend + JSON→SQLite import (B-064)."""
from __future__ import annotations

import json
import sqlite3

import pytest

from nlp_histo.evaluation.matching.matcher import (
    EmbeddingCache,
    SQLiteEmbeddingCache,
    import_json_cache_to_sqlite,
    make_embedding_cache,
)


# SQLite backend round-trip

def test_sqlite_miss_set_save_hit(tmp_path):
    db = tmp_path / "c.sqlite"
    cache = SQLiteEmbeddingCache(db, "model-x")
    assert cache.get("hello world") is None          # miss
    cache.set("hello world", [0.1, 0.2, 0.3])
    cache.save()
    got = cache.get("hello world")                    # hit
    assert got == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert len(cache) == 1


def test_sqlite_float32_roundtrip(tmp_path):
    cache = SQLiteEmbeddingCache(tmp_path / "c.sqlite", "m")
    vec = [0.123456789, -0.987654321, 0.5, 0.0]
    cache.set("v", vec)
    cache.save()
    got = cache.get("v")
    assert got == pytest.approx(vec, abs=1e-6)        # float32 tolerance


def test_keys_do_not_collide_text(tmp_path):
    cache = SQLiteEmbeddingCache(tmp_path / "c.sqlite", "m")
    cache.set("text one", [1.0, 2.0])
    cache.set("text two", [3.0, 4.0])
    cache.save()
    assert cache.get("text one") == pytest.approx([1.0, 2.0])
    assert cache.get("text two") == pytest.approx([3.0, 4.0])
    assert cache.get("text three") is None


def test_keys_do_not_collide_model(tmp_path):
    db = tmp_path / "c.sqlite"
    a = SQLiteEmbeddingCache(db, "model-a")
    a.set("same text", [1.0])
    a.save()
    # Same DB file, same text, different model → different key → miss.
    b = SQLiteEmbeddingCache(db, "model-b")
    assert b.get("same text") is None
    assert a.get("same text") == pytest.approx([1.0])


def test_normalisation_matches_json_backend(tmp_path):
    # Key uses text.lower().strip(); both backends must agree.
    j = EmbeddingCache(tmp_path / "c.json", "m")
    s = SQLiteEmbeddingCache(tmp_path / "c.sqlite", "m")
    assert j._make_key("  Hello ") == s._make_key("hello")


# JSON → SQLite import

def _write_json_cache(path, model, entries):
    path.write_text(json.dumps({"embedding_model": model, "entries": entries}), encoding="utf-8")


def test_import_preserves_keys_and_count(tmp_path):
    jp = tmp_path / "cache.json"
    sp = tmp_path / "cache.sqlite"
    entries = {"k1": [0.1, 0.2, 0.3], "k2": [0.4, 0.5, 0.6]}
    _write_json_cache(jp, "model-x", entries)

    stats = import_json_cache_to_sqlite(jp, sp)
    assert stats["imported"] == 2
    assert stats["skipped"] == 0
    assert stats["total"] == 2
    assert stats["dims"] == {3: 2}

    conn = sqlite3.connect(str(sp))
    keys = {r[0] for r in conn.execute("SELECT key FROM embeddings")}
    conn.close()
    assert keys == {"k1", "k2"}                        # hash keys copied verbatim
    assert jp.exists()                                 # JSON untouched


def test_reimport_is_idempotent(tmp_path):
    jp = tmp_path / "cache.json"
    sp = tmp_path / "cache.sqlite"
    _write_json_cache(jp, "m", {"k1": [1.0], "k2": [2.0]})
    import_json_cache_to_sqlite(jp, sp)
    stats2 = import_json_cache_to_sqlite(jp, sp)        # second run
    assert stats2["imported"] == 0
    assert stats2["skipped"] == 2
    assert stats2["total"] == 2


def test_import_default_sqlite_path(tmp_path):
    jp = tmp_path / "embedding_cache_gemini.json"
    _write_json_cache(jp, "gemini-embedding-001", {"k": [0.1, 0.2]})
    stats = import_json_cache_to_sqlite(jp)             # no sqlite path → derived
    assert stats["dest"].endswith("embedding_cache_gemini.sqlite")


# Factory / backend switch

def test_factory_defaults_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("NLP_HISTO_EMBEDDING_CACHE_BACKEND", raising=False)
    cache = make_embedding_cache(tmp_path / "c.json", "m")
    assert isinstance(cache, SQLiteEmbeddingCache)
    assert str(cache.path).endswith("c.sqlite")        # .json → .sqlite


def test_factory_json_backend_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_HISTO_EMBEDDING_CACHE_BACKEND", "json")
    cache = make_embedding_cache(tmp_path / "c.json", "m")
    assert isinstance(cache, EmbeddingCache)
    assert not isinstance(cache, SQLiteEmbeddingCache)
    cache.set("x", [1.0, 2.0])
    cache.save()
    assert cache.get("x") == pytest.approx([1.0, 2.0])
