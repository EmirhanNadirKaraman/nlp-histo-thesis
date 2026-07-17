"""The packaged runtime resource must be reachable the way an installed wheel reaches it.

``synonyms.yaml`` is the single source of truth for NORMALIZE's synonym dictionary
and is read *at import time*. It is the only non-Python file the installed package
loads at runtime, so it must be located through ``importlib.resources`` — an
installed wheel has no repository tree, and a ``Path(__file__).parents[n]`` walk
would resolve to the wrong directory (or outside the package entirely).

These tests fail if the package-data declaration is dropped from pyproject.toml, or
if the resource lookup regresses to a filesystem-depth calculation.
"""
from __future__ import annotations

from importlib import resources

import pytest

ENTITIES_PKG = "nlp_histo.pipeline.stages.knowledge_extraction.entities"


def test_synonyms_resource_is_reachable_via_importlib_resources() -> None:
    resource = resources.files(ENTITIES_PKG).joinpath("synonyms.yaml")
    assert resource.is_file(), f"synonyms.yaml not found as package data in {ENTITIES_PKG}"
    assert resource.read_text(encoding="utf-8").strip(), "synonyms.yaml is empty"


def test_synonyms_resource_parses_as_a_mapping() -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(
        resources.files(ENTITIES_PKG).joinpath("synonyms.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(data, dict) and data, "synonyms.yaml must be a non-empty mapping"


def test_normalize_stage_loaded_the_synonyms_at_import() -> None:
    """The module-level dictionary is populated — i.e. the resource resolved for real."""
    from nlp_histo.pipeline.stages.knowledge_extraction.stages import normalize_stage

    assert normalize_stage._SYNONYMS, (
        "NORMALIZE loaded an empty synonym map — synonyms.yaml was not found through "
        "importlib.resources (check the package-data rule in pyproject.toml)."
    )


def test_normalize_stage_does_not_use_filesystem_depth_for_the_resource() -> None:
    """Guard the regression: no `__file__.parents[...]` walk may creep back in.

    Parses the AST rather than grepping the text, so prose in a comment or
    docstring that merely *mentions* ``parents[...]`` cannot fail (or pass) this.
    """
    import ast
    from pathlib import Path

    from nlp_histo.pipeline.stages.knowledge_extraction.stages import normalize_stage

    tree = ast.parse(Path(normalize_stage.__file__).read_text(encoding="utf-8"))
    depth_walks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "parents"
    ]
    assert not depth_walks, (
        "normalize_stage must not locate packaged data by filesystem depth "
        f"(found {len(depth_walks)} `.parents` access) — that breaks in an installed wheel."
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "files"
        for node in ast.walk(tree)
    ), "the synonyms lookup must go through importlib.resources.files()"
