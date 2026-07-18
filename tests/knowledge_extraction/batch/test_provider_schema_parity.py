"""Issue D — strict-mode parity audit across all batch providers.

Pins the observed behaviour of each provider's `submit()`:

* OpenAI direct, Azure (Foundry), and Vertex Gemini all consume the
  `openai_tool` argument and pass the AuditableSummary schema to the
  provider's batch API. The API rejects malformed outputs before they
  reach `parse_result()`.
* Anthropic consumes `openai_tool` and converts it to the Anthropic
  `input_schema` shape — also API-side enforcement.
* Direct Gemini (`google.genai`) discards `openai_tool` and submits with
  `response_mime_type='application/json'` only. No API-side schema
  enforcement. Malformed outputs are caught downstream by
  `parse_result()` (Pydantic validation → `logs/bad_findings.jsonl`).

This test asserts those expectations using AST-level inspection — no
provider credentials required, no real API calls.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from nlp_histo.pipeline.stages.knowledge_extraction import batch

# Anchored to the installed package rather than a repository-relative path, so
# this keeps reading the real batch sources wherever the package lives.
_BATCH_DIR = Path(batch.__file__).parent


def _read_source(filename: str) -> str:
    return (_BATCH_DIR / filename).read_text(encoding="utf-8")


def _submit_function_source(module_filename: str, class_name: str) -> str:
    """Extract the textual body of `<class>.submit` from a provider module."""
    src = _read_source(module_filename)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "submit":
                    return ast.unparse(item)
    raise AssertionError(f"{class_name}.submit() not found in {module_filename}")


def test_openai_submit_uses_tool_schema():
    """OpenAI batch submits with the strict-mode tool schema."""
    src = _submit_function_source("openai_batch.py", "OpenAIBatchProvider")
    # The submit() body must reference openai_tool — it's the schema source.
    assert "openai_tool" in src
    assert "tools" in src and "tool_choice" in src, (
        "OpenAI submit must include tools + tool_choice for strict-mode enforcement"
    )


def test_azure_submit_uses_tool_schema():
    """Azure (Foundry) batch follows the OpenAI Chat Completions shape with the same tool."""
    src = _submit_function_source("azure_batch.py", "AzureBatchProvider")
    assert "openai_tool" in src
    assert "tools" in src and "tool_choice" in src


def test_anthropic_submit_converts_tool_schema():
    """Anthropic batch converts the OpenAI tool to its `input_schema` shape and
    forces tool use via tool_choice."""
    src = _submit_function_source("claude_batch.py", "AnthropicBatchProvider")
    assert "openai_tool" in src
    assert "input_schema" in src, (
        "Anthropic submit must build input_schema from openai_tool['function']['parameters']"
    )
    assert "tool_choice" in src, "Anthropic submit must force the function via tool_choice"


def test_vertex_gemini_submit_uses_function_declarations():
    """Vertex Gemini consumes the tool and registers it as a functionDeclaration
    with toolConfig.functionCallingConfig.mode='any'."""
    src = _submit_function_source("vertex_batch.py", "VertexGeminiBatchProvider")
    assert "openai_tool" in src
    assert "functionDeclarations" in src
    assert "toolConfig" in src or "tool_config" in src


def test_direct_gemini_submit_documents_schema_gap():
    """ISSUE D — direct Gemini batch does not pass openai_tool to the API.

    The body must:
      (a) accept openai_tool to keep the cross-provider signature stable,
      (b) carry an explicit comment marking the parity gap,
      (c) emit an INFO log so operators see the gap at submit time.

    If a future change adds API-side schema enforcement (response_schema /
    response_json_schema with the anyOf stripped), this test should be
    flipped to assert the schema IS passed.
    """
    src = _read_source("gemini_batch.py")
    submit_src = _submit_function_source("gemini_batch.py", "GeminiBatchProvider")
    # Sanity: function signature still accepts the argument.
    assert "openai_tool" in submit_src
    # Schema-gap acknowledgement is present in the file (comment + log).
    assert "Issue D" in src, (
        "gemini_batch.py must reference Issue D audit so a future reader "
        "knows the gap is deliberate and observability-only"
    )
    # The INFO log explicitly tells operators the schema is not API-enforced.
    assert "without API-side schema" in src
    # `response_mime_type` is the only safety net at the API level.
    assert "response_mime_type" in src
    assert '"application/json"' in src or "'application/json'" in src


def test_dispatch_passes_openai_tool_to_every_provider():
    """`dispatch.submit_level` must call `provider.submit(reqs, OPENAI_MAP_TOOL)`
    for every provider — even Gemini-direct, which discards it. This keeps the
    cross-provider signature stable so re-enabling the schema in gemini_batch
    later doesn't require touching dispatch."""
    from nlp_histo.pipeline.stages.knowledge_extraction.batch import dispatch
    src = inspect.getsource(dispatch.submit_level)
    assert "OPENAI_MAP_TOOL" in src
    # The submit call shape — every provider gets the same two args.
    assert ".submit(reqs, OPENAI_MAP_TOOL)" in src.replace("\n", " ").replace("  ", " ")


def test_openai_strict_schema_validates_at_import():
    """Sanity check that the OPENAI_MAP_TOOL building doesn't raise — guards
    against schema features that drift out of strict-mode compliance.
    `_build_openai_tool` raises ValueError on any strict-mode violation."""
    # Re-import via fresh module load is overkill; the constant is already
    # built at module load time. The test passes by virtue of the import.
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.dispatch import OPENAI_MAP_TOOL
    assert OPENAI_MAP_TOOL["function"]["strict"] is True
    params = OPENAI_MAP_TOOL["function"]["parameters"]
    assert params["additionalProperties"] is False
    # Every property must be listed as required (strict-mode invariant).
    assert set(params["required"]) == set(params["properties"].keys())
