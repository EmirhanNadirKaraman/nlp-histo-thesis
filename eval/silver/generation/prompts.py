"""Versioned Opus prompts for silver-label generation.

The silver ``SYSTEM_PROMPT`` is *derived* from the voter MAP prompt
(``pipeline.stages.knowledge_extraction.llm.prompts._MAP_SYSTEM``) so silver labels are
extracted under the same instructions as the cascade voters — only the
model tier differs. Two voter-only pieces are stripped, because silver inputs
are raw paragraphs with no sentence IDs and the ``extract_findings`` tool has no
``evidence`` field (and neither piece affects the matched-strict-F1):

  * the per-sentence CITATION rule, and
  * the ``<OutputFormat>`` block (evidence / summary_text / audit_metadata).

Deriving (rather than copying) keeps the two prompts from drifting; the guards
in ``_derive_silver_system_prompt`` fail loudly if the voter prompt is
restructured so the two are resynced deliberately.
"""
from __future__ import annotations

from nlp_histo.pipeline.stages.knowledge_extraction.llm.prompts import _MAP_SYSTEM

PROMPT_VERSION = "v4"


def _derive_silver_system_prompt(map_system: str) -> str:
    """Voter MAP system prompt minus the provenance scaffolding silver can't use."""
    out_marker = "\n\n<OutputFormat>"
    if out_marker not in map_system:
        raise RuntimeError(
            "voter _MAP_SYSTEM shape changed (no <OutputFormat>); resync silver prompt"
        )
    body = map_system.split(out_marker, 1)[0].rstrip()

    citation_rule = (
        "5. CITATION: Every claim MUST cite its Sentence ID using the format: "
        "[S1|PMC123456|789].\n"
    )
    if citation_rule not in body:
        raise RuntimeError("voter CITATION rule text changed; resync silver prompt")
    body = body.replace(citation_rule, "")

    return body + (
        "\n\nIf the paragraph contains no medical findings, "
        "return an empty findings array."
    )


SYSTEM_PROMPT = _derive_silver_system_prompt(_MAP_SYSTEM)


def make_user_prompt(text: str, path_string: str) -> str:
    return (
        f"<Section>{path_string}</Section>\n\n"
        f"<Paragraph>\n{text}\n</Paragraph>\n\n"
        "Extract all findings."
    )


EXTRACT_FINDINGS_TOOL = {
    "name": "extract_findings",
    "description": "Return all atomic medical findings extracted from the paragraph.",
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"],
                        },
                        "claim":            {"type": "string"},
                        "subject_entity":   {"type": ["string", "null"]},
                        "outcome_entity":   {"type": ["string", "null"]},
                        "relation_type": {
                            "type": "string",
                            "enum": ["has_feature", "expression", "prognostic", "comparative", "demographic", "treatment_response", "unclear"],
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["positive", "negative", "absent", "partial", "unclear", "no_direction"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "verbatim_support": {"type": "string"},
                        "scope": {
                            "type": "object",
                            "properties": {
                                "disease_subtype":   {"type": ["string", "null"]},
                                "cohort_n":          {"type": ["integer", "null"]},
                                "assay_method":      {"type": ["string", "null"]},
                                "biomarker_cutoff":  {"type": ["string", "null"]},
                                "tissue_site":       {"type": ["string", "null"]},
                                "treatment_context": {"type": ["string", "null"]},
                                "endpoint":          {"type": ["string", "null"]},
                                "study_design":      {"type": ["string", "null"]},
                                "scope_parsed":      {"type": "boolean"},
                            },
                        },
                    },
                    "required": ["category", "claim", "relation_type", "confidence", "verbatim_support"],
                },
            }
        },
        "required": ["findings"],
    },
}
