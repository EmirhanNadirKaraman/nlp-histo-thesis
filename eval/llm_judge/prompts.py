"""
eval.llm_judge.prompts — System prompts, user templates, and tool schemas.

All judge evaluation prompts, controlled enums, and structured output
schemas for Q1, Q2, Q3, Q5.

Labels produced here are silver / proxy labels — not clinical ground truth.
"""
from __future__ import annotations

from typing import Any

# Shared filter rules

FILTER_RULES = """\
<FilterRules>
SKIP: Author names, journal metadata, acknowledgments, funding, and generic boilerplate.
SKIP: Patient-specific narratives — age/sex of individual patients, presenting symptoms,
      clinical history of a single case (e.g. "46-year-old man presented with a nodule").
      These are case vignettes, NOT generalizable facts.
ONLY extract: generalizable medical facts — findings that apply to a disease entity,
      population, biomarker, diagnosis, prognosis, morphology, IHC, molecular genetics,
      staging, or treatment response across cases. A valid claim is true for the entity
      in general, not just for one patient in one visit.
ONLY extract from: methods, results, discussion, and case descriptions
      (but only the generalizable conclusions drawn from cases, not the case narrative itself).
</FilterRules>"""

# Controlled enums (matching pipeline models)

RELATION_TYPES = [
    "has_feature", "expression", "prognostic", "comparative",
    "demographic", "treatment_response", "unclear",
]
DIRECTIONS = ["positive", "negative", "absent", "partial", "unclear"]
CATEGORIES = [
    "morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis",
]
RELATION_LABELS = ["SUPPORT", "CONTRADICT", "SCOPE_QUALIFY", "UNRELATED"]


# Q1 — MAP finding precision / field correctness

Q1_SYSTEM = """\
You are an expert histopathologist verifying AI-extracted medical findings.
Your outputs are silver / proxy labels — not clinical ground truth.

You will be given a verbatim sentence from a histopathology paper and an
AI-extracted finding. Assess whether the claim is directly entailed by the
verbatim text, and provide the correct values for each metadata field.

Judge STRICTLY against the verbatim text only — do not use external medical
knowledge to fill in fields that are absent from the text. If the verbatim
is ambiguous, prefer "unclear". Keep explanations short (1-2 sentences)."""

Q1_USER = """\
VERBATIM SOURCE TEXT:
{verbatim}

EXTRACTED FINDING:
- claim: {claim}
- subject_entity: {subject}
- outcome_entity: {outcome}
- relation_type: {relation_type}
- direction: {direction}
- category: {category}
- scope_disease_subtype: {scope_disease_subtype}
- scope_assay_method: {scope_assay_method}

Assess grounding and provide the correct field values."""

Q1_TOOL_NAME = "q1_precision_judgment"

Q1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_grounded": {
            "type": "boolean",
            "description": "True if the claim is entailed by the verbatim text",
        },
        "correct_relation_type": {"type": "string", "enum": RELATION_TYPES},
        "correct_direction": {
            "anyOf": [{"type": "string", "enum": DIRECTIONS}, {"type": "null"}],
            "description": "Correct direction, or null if not applicable",
        },
        "correct_category": {"type": "string", "enum": CATEGORIES},
        "correct_subject_entity": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Correct subject_entity from the verbatim, or null if none",
        },
        "correct_outcome_entity": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Correct outcome_entity from the verbatim, or null if none",
        },
        "scope_errors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Scope field names where the pipeline value is wrong or missing (e.g. 'disease_subtype', 'assay_method')",
        },
        "explanation": {
            "type": "string",
            "description": "Brief explanation of the judgment (1-2 sentences)",
        },
    },
    "required": [
        "is_grounded", "correct_relation_type", "correct_direction",
        "correct_category", "correct_subject_entity", "correct_outcome_entity",
        "scope_errors", "explanation",
    ],
}


# Q2 — RELATE label accuracy (blind judging by default)

Q2_SYSTEM = """\
You are an expert reviewing logical relationships between medical claims
from histopathology literature. Your outputs are silver / proxy labels.

Classify the relation between two claims using exactly one of:

SUPPORT       — Both claims assert the same relationship with the same polarity.
CONTRADICT    — One claim asserts the opposite polarity of the other for the
                same entity/outcome.
SCOPE_QUALIFY — One claim is a narrower version of the other — same finding but
                restricted to a disease subtype, assay method, patient cohort,
                or treatment context.
UNRELATED     — Different subject or outcome entities; no logical relationship.

Apply these definitions strictly. Classify SUPPORT only when both claims
explicitly assert compatible facts — do not infer compatibility from
background knowledge. Keep explanations short (1-2 sentences)."""

Q2_USER_BLIND = """\
CLAIM A: {predicate_a}
  subject: {subject_a} | outcome: {outcome_a} | direction: {direction_a} | category: {category_a}

CLAIM B: {predicate_b}
  subject: {subject_b} | outcome: {outcome_b} | direction: {direction_b} | category: {category_b}

What is the correct relation type?"""

Q2_USER_WITH_LABEL = """\
CLAIM A: {predicate_a}
  subject: {subject_a} | outcome: {outcome_a} | direction: {direction_a} | category: {category_a}

CLAIM B: {predicate_b}
  subject: {subject_b} | outcome: {outcome_b} | direction: {direction_b} | category: {category_b}

PIPELINE LABEL: {pipeline_label}

What is the correct relation type?"""

Q2_TOOL_NAME = "q2_relation_judgment"

Q2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct_label": {"type": "string", "enum": RELATION_LABELS},
        "explanation": {
            "type": "string",
            "description": "Brief explanation (1-2 sentences)",
        },
    },
    "required": ["correct_label", "explanation"],
}


# Q3 — Recall gap check

Q3_SYSTEM = f"""\
You are an expert histopathologist reviewing a paragraph from a
histopathology paper for extraction completeness. Your outputs are
silver / proxy labels.

You will be given a paragraph and a list of medical claims already extracted
from it (may be empty). Your task: identify any generalizable medical
findings present in the paragraph that are NOT captured by the extracted
claims.

Apply the same filter rules as the extraction pipeline:
{FILTER_RULES}

Report ONLY claims that are:
- Clearly stated in the paragraph text
- Generalizable (not patient-specific)
- Not already captured by any extracted claim (semantically — do not flag
  rephrased duplicates)

If all findings are captured (or the paragraph contains no extractable
findings), return an empty missing_findings list."""

Q3_USER = """\
PARAGRAPH:
{paragraph}

ALREADY EXTRACTED CLAIMS:
{claims}

List any generalizable medical findings in the paragraph that are missing
from the extracted claims."""

Q3_TOOL_NAME = "q3_recall_judgment"

Q3_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "missing_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The missing finding"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "subject_entity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "outcome_entity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["claim"],
            },
            "description": "Generalizable claims present in paragraph but absent from extracted list",
        },
        "has_gaps": {"type": "boolean"},
        "explanation": {
            "type": "string",
            "description": "Brief explanation (1-2 sentences)",
        },
    },
    "required": ["missing_findings", "has_gaps", "explanation"],
}


# Q5 — Paragraph-level extraction F1

Q5_SYSTEM = f"""\
You are an expert histopathologist performing medical fact extraction from
histopathology papers. Your outputs are silver / proxy labels.

You will be given a paragraph and a list of pipeline-extracted findings.
Your task:
1. Extract ALL generalizable medical findings from the paragraph using the
   same schema as the pipeline.
2. Align each of your extracted findings to the pipeline findings:
   - "match" if a pipeline finding captures the same claim
   - "partial" if a pipeline finding is close but has wrong fields
   - "unmatched" if no pipeline finding corresponds

Apply the same filter rules:
{FILTER_RULES}

Do NOT use external medical knowledge — extract only what is explicitly
stated in the paragraph. Keep your findings atomic (one fact per finding)."""

Q5_USER = """\
PARAGRAPH:
{paragraph}

PIPELINE FINDINGS:
{pipeline_findings}

Extract all generalizable medical findings from this paragraph and align
them against the pipeline findings."""

Q5_TOOL_NAME = "q5_extraction_judgment"

Q5_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "silver_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "subject_entity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "outcome_entity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "relation_type": {"type": "string", "enum": RELATION_TYPES},
                    "direction": {"anyOf": [{"type": "string", "enum": DIRECTIONS}, {"type": "null"}]},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
                "required": ["claim"],
            },
            "description": "Complete silver findings extracted from the paragraph",
        },
        "alignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "silver_index": {
                        "type": "integer",
                        "description": "0-based index into silver_findings",
                    },
                    "pipeline_index": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "description": "0-based index into pipeline findings, or null if unmatched",
                    },
                    "match_type": {
                        "type": "string",
                        "enum": ["match", "partial", "unmatched"],
                    },
                },
                "required": ["silver_index", "pipeline_index", "match_type"],
            },
            "description": "Alignment of each silver finding to pipeline findings",
        },
        "explanation": {
            "type": "string",
            "description": "Brief explanation (1-2 sentences)",
        },
    },
    "required": ["silver_findings", "alignments", "explanation"],
}
