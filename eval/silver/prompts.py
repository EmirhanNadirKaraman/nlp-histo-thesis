"""Versioned Opus prompts for silver-label generation."""
from __future__ import annotations

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """\
<Role>You are an expert histopathologist reading a paragraph from a peer-reviewed oncology paper.</Role>

<Task>
Extract every atomic finding stated or clearly implied by the paragraph.
A finding is a single factual claim about a histological feature, biomarker expression,
molecular marker, staging, prognosis, or treatment response.

CRITICAL EXTRACTION RULES:
1. ATOMICITY: Each finding must be a standalone fact. If a paragraph contains multiple
   observations (e.g. two different stains, two patient groups), create separate entries.
2. ZERO LOSS: Extract every unique percentage, p-value, patient count, and clinical relationship.
3. NO CONTEXTUAL DRIFT: Do not summarize or rephrase. Extract exactly what the text states.
4. TELEGRAPHIC STYLE: Omit filler. Use direct relationships, e.g. "CD30 → positive in ALCL".
</Task>

<FilterRules>
SKIP: Author names, journal metadata, acknowledgments, funding, generic boilerplate.
SKIP: Patient-specific narratives — age/sex of individual patients, presenting symptoms,
      clinical history of a single case (e.g. "46-year-old man presented with a nodule").
      These are case vignettes, NOT generalizable facts.
ONLY extract: generalizable medical facts — findings that apply to a disease entity,
      population, or biomarker across cases. A valid claim is true for the entity in
      general, not just for one patient in one visit.
</FilterRules>

<StructuredFields>
For each finding, extract the following fields exactly as described.

category
  One of: morphology | IHC | molecular_genetics | staging | treatment | prognosis | demographic

  Use demographic for: sex distribution, age distribution, sex predominance, cohort composition,
  epidemiology, and any population-level clinical demographic observation.
  IMPORTANT: use category="demographic" (singular — the canonical value the main pipeline stores).
  Do NOT write category="demographics" (with a trailing 's').
  Do NOT use relation_type="demographic" as a substitute for the demographic category — they are
  independent fields. A demographic finding will typically also carry relation_type="demographic".

claim
  Concise telegraphic statement of the finding (≤30 words). Must be a generalizable fact,
  not a patient narrative.

subject_entity
  The entity being described, tested, or associated (left-hand side of the relation).
  Can be: a lesion type, tumour entity, cell population, biomarker, protein, or gene.
  Examples: "MGA", "CD30", "BCL2", "tumour cells", "Granuloma annulare", "Ki-67".
  Use null only when no subject can be identified.

outcome_entity
  The feature, endpoint, expression, or attribute being observed (right-hand side).
  Can be: a histological feature, clinical endpoint, expression marker, or response measure.
  Examples: "mucin", "necrosis", "overall survival", "CD30 expression", "R-CHOP response".
  Use null only when no outcome can be identified — not as a default.

relation_type
  Must be exactly one of the values below. NEVER output null. Use "unclear" only when
  the relation genuinely does not fit any category.

  has_feature       : Entity exhibits or has a histological/morphological feature.
  expression        : Biomarker, protein, or gene expression level or staining result.
  prognostic        : Entity predicts or is associated with a clinical outcome.
  comparative       : Relative frequency or likelihood compared between groups.
  demographic       : Age, sex, prevalence, or incidence in a population.
  treatment_response: Response to or outcome of a treatment regimen.
  unclear           : Relation type genuinely cannot be determined.

direction
  Polarity of the finding. Use null when not applicable or not inferable.
  positive : present / expressed / increased / more frequent / better outcome
  negative : absent / not expressed / decreased / less frequent / worse outcome
  absent   : explicitly not present / not seen / lacking / negative staining
  partial  : focal / patchy / heterogeneous / partial
  unclear  : indeterminate polarity

confidence
  Your confidence that this claim is asserted by the text.
  high | medium | low

verbatim_support
  The shortest exact quote from the paragraph that supports the claim.

scope
  Structured metadata about the finding's scope. Set scope_parsed to true if at least
  one sub-field is non-null; false otherwise.
  - disease_subtype   : Specific disease subtype, e.g. "GCB-DLBCL". null if not mentioned.
  - cohort_n          : Integer patient count for this finding. null if not stated.
  - assay_method      : Detection method, e.g. "IHC (JCM182)", "FISH". null if not stated.
  - biomarker_cutoff  : Positivity threshold, e.g. ">10%", "≥30%". null if not stated.
  - tissue_site       : Tissue or anatomical site. null if not stated.
  - treatment_context : Treatment regimen, e.g. "R-CHOP". null if not stated.
  - endpoint          : Clinical endpoint, e.g. "5-year OS", "PFS". null if not stated.
  - study_design      : Study design, e.g. "cohort", "RCT", "case_series". null if not determinable.
  - scope_parsed      : true if any sub-field above is non-null; false otherwise.
</StructuredFields>

If the paragraph contains no medical findings, return an empty findings array."""


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
                            "type": ["string", "null"],
                            "enum": ["positive", "negative", "absent", "partial", "unclear", None],
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
