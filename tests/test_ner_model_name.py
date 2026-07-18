"""B-115 — `ner merge` / `ner export` must filter on the model name that is stored.

The bug: `ner.py` persists spaCy's ``nlp.meta["name"]`` — ``core_sci_lg`` — because the
``en_`` in ``en_core_sci_lg`` is ``meta["lang"]``, not part of ``meta["name"]``. Both
consumers defaulted ``--model`` to the *package* name ``en_core_sci_lg``, which matches no
row. So the documented §8 sequence exited 0 having silently produced nothing, against a
corpus holding 1 792 440 entities.
"""
from __future__ import annotations

import pytest

from nlp_histo.ner.enums import DEFAULT_MODEL_NAME


def test_default_model_name_is_the_spacy_meta_name_not_the_package_name() -> None:
    """The constant must be the stored value. Guarding the exact confusion that caused
    B-115: the package is `en_core_sci_lg`; what lands in entities.model_name is not."""
    assert DEFAULT_MODEL_NAME == "core_sci_lg"
    assert DEFAULT_MODEL_NAME != "en_core_sci_lg", (
        "the package name is not what ner.py stores — filtering on it matches zero rows"
    )


@pytest.mark.parametrize("module_path", [
    "nlp_histo.ner.merge_entities_by_umls",
    "nlp_histo.ner.export_disease_entities",
])
def test_consumer_cli_defaults_to_the_stored_model_name(module_path: str) -> None:
    """Both consumers must default to the value ner.py writes, or they no-op silently."""
    import importlib

    mod = importlib.import_module(module_path)
    parser = None
    for attr in ("build_parser", "_build_parser"):
        if hasattr(mod, attr):
            parser = getattr(mod, attr)()
            break
    if parser is None:
        # These modules build their parser inside main(); assert on the source instead —
        # crude, but it pins the regression rather than skipping it.
        import inspect

        src = inspect.getsource(mod)
        assert "default='en_core_sci_lg'" not in src, (
            f"{module_path} defaults --model to the package name; it matches no stored row"
        )
        assert "DEFAULT_MODEL_NAME" in src, (
            f"{module_path} should take its --model default from the shared constant"
        )
        return
    default = parser.get_default("model")
    assert default == DEFAULT_MODEL_NAME


def test_ner_writes_and_filters_the_same_model_name() -> None:
    """ner.py's skip-if-already-processed check and the consumers must agree.

    They disagreed for the whole life of the corpus: ner.py hardcoded "core_sci_lg"
    (correct) while the consumers defaulted to "en_core_sci_lg" (never matching).
    """
    import inspect

    from nlp_histo.ner import ner

    src = inspect.getsource(ner)
    assert 'Entity.model_name == DEFAULT_MODEL_NAME' in src, (
        "ner.py should filter on the shared constant rather than a literal"
    )


# the whole documented path must agree on one identifier

def test_extract_merge_export_share_one_stored_identifier() -> None:
    """extract writes it, merge and export filter on it — one constant, no literals.

    The bug was precisely a disagreement between the writer and the two readers, so the
    regression worth pinning is that all three now derive from the same source.
    """
    import inspect

    from nlp_histo.ner import export_disease_entities, merge_entities_by_umls, ner

    for mod in (ner, merge_entities_by_umls, export_disease_entities):
        src = inspect.getsource(mod)
        assert "DEFAULT_MODEL_NAME" in src, f"{mod.__name__} should use the shared constant"
        assert "'en_core_sci_lg'" not in src.replace(
            "package name 'en_core_sci_lg'", ""
        ).replace("'en_core_sci_lg' package", ""), (
            f"{mod.__name__} still refers to the package name as a value"
        )


# an empty result must not be reported as success when it is a mismatch

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def distinct(self):
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    """Stands in for a session holding entities under the given model names."""

    def __init__(self, model_names):
        self._rows = [(m,) for m in model_names]

    def query(self, *_a, **_kw):
        return _FakeQuery(self._rows)


def test_mismatched_filter_raises_rather_than_reporting_empty() -> None:
    from nlp_histo.ner.model_filter import NoMatchingEntitiesError, check_model_filter

    session = _FakeSession(["core_sci_lg"])          # what the corpus actually holds
    with pytest.raises(NoMatchingEntitiesError) as exc:
        check_model_filter(session, "en_core_sci_lg")  # what the old default asked for
    msg = str(exc.value)
    assert "core_sci_lg" in msg              # names what IS available
    assert "--model" in msg                  # actionable
    assert "meta['name']" in msg             # explains the root cause


def test_matching_filter_is_silent() -> None:
    from nlp_histo.ner.model_filter import check_model_filter

    check_model_filter(_FakeSession(["core_sci_lg"]), "core_sci_lg")


def test_genuinely_empty_corpus_is_not_an_error() -> None:
    """No entities at all is an honest empty result — a fresh database, not a mismatch."""
    from nlp_histo.ner.model_filter import check_model_filter

    check_model_filter(_FakeSession([]), "core_sci_lg")


def test_no_filter_requested_is_not_an_error() -> None:
    from nlp_histo.ner.model_filter import check_model_filter

    check_model_filter(_FakeSession(["core_sci_lg"]), None)


def test_explicit_model_override_is_preserved() -> None:
    """--model must still select a non-default model that exists."""
    from nlp_histo.ner.model_filter import check_model_filter

    check_model_filter(_FakeSession(["core_sci_lg", "core_sci_sm"]), "core_sci_sm")


@pytest.mark.slow
def test_spacy_meta_name_matches_the_constant() -> None:
    """The ground truth, if the model is installed: spaCy reports lang='en',
    name='core_sci_lg' — and ner.py stores meta['name']."""
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load(
            "en_core_sci_lg",
            disable=["parser", "tagger", "ner", "lemmatizer", "attribute_ruler"],
        )
    except OSError:
        pytest.skip("en_core_sci_lg not installed")
    assert nlp.meta["name"] == DEFAULT_MODEL_NAME
    assert nlp.meta["lang"] == "en"
