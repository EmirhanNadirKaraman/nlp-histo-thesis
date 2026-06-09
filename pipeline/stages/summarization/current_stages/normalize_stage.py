"""
NORMALIZE stage: Finding[] → NormalFinding[]

Two operations:
1. Entity normalization — resolve surface-form variants to canonical names.
   Resolution order:
     a. Hand-curated synonym dictionary (_SYNONYMS) — covers rare entities
        that UMLS does not link (CEAN, MGA, …).
     b. UMLS linker via scispaCy (en_core_sci_lg, threshold 0.85) —
        process-wide singleton from ``..umls_resources``; per-text results
        cached in ``_UMLS_CACHE``.  Skipped if scispaCy unavailable.
     c. Identity fallback — stripped input returned unchanged.

2. Conditional dedup — findings that share the same
   (text_element_id, subject_entity, outcome_entity, relation_type) after
   normalization are merged into one NormalFinding (provenance union).
   Findings that differ on any of those keys stay separate even if they
   came from the same text element.  Findings where subject_entity or
   outcome_entity is None, or relation_type is unclear, are kept as
   individual NormalFindings without attempting dedup.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

from ..models import DirectionEnum, Finding, FindingScope, NormalFinding, RelationTypeEnum, SourceSpan
from ..umls_utils import best_cui as _best_cui

logger = logging.getLogger(__name__)

# ── Synonym dictionary ─────────────────────────────────────────────────────────
# synonyms.yaml is the SINGLE source of truth — there is no hardcoded fallback.
# It lives at the summarization package root, one level above current_stages/
# (B-066: the prior Path(__file__).parent pointed inside current_stages/ where no
# file exists. With the fallback removed, a missing or unreadable YAML now yields
# an empty synonym map and a loud error instead of silently masking the problem
# with a stale hardcoded dict.)
_SYNONYMS_YAML = Path(__file__).resolve().parents[1] / "synonyms.yaml"


def _load_synonyms() -> dict[str, str]:
    """Load the synonym dictionary from synonyms.yaml — the single source of truth.

    There is no hardcoded fallback: if the file is missing, unreadable, or not a
    mapping, return an empty dict and log an error (synonyms come *only* from the
    YAML). PyYAML is a hard dependency, so an import failure is surfaced too.
    """
    try:
        import yaml  # type: ignore
        with open(_SYNONYMS_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("NORMALIZE: synonyms.yaml not found at %s — no synonyms loaded", _SYNONYMS_YAML)
        return {}
    except Exception as exc:
        logger.error("NORMALIZE: failed to load synonyms.yaml (%s) — no synonyms loaded", exc)
        return {}
    if not isinstance(data, dict):
        logger.error("NORMALIZE: synonyms.yaml is not a mapping — no synonyms loaded")
        return {}
    loaded = {str(k).lower(): str(v) for k, v in data.items()}
    logger.debug("NORMALIZE: loaded %d synonyms from %s", len(loaded), _SYNONYMS_YAML)
    return loaded


_SYNONYMS: dict[str, str] = _load_synonyms()


# ── UMLS entity resolver ───────────────────────────────────────────────────────
# Module-level cache: entity surface text → (canonical_name, cui) pair.
# Shared across all NormalizeStage instances in the same process.
_UMLS_CACHE: dict[str, tuple[str | None, str | None]] = {}


def _umls_canonical_with_cui(text: str) -> tuple[str | None, str | None]:
    """
    Return (canonical_name, cui) for *text*, or (None, None) if no confident link.

    Uses the process-wide scispaCy singleton from ``umls_resources``.  Results
    are cached per-process in ``_UMLS_CACHE``.
    """
    if text in _UMLS_CACHE:
        return _UMLS_CACHE[text]

    from ..umls_resources import get_nlp, get_linker  # local import: avoid cycles
    nlp = get_nlp()
    if nlp is None:
        _UMLS_CACHE[text] = (None, None)
        return (None, None)
    linker = get_linker()

    try:
        doc = nlp(text)
        spans = doc.ents if doc.ents else [doc]
        kb_ents_by_span = [getattr(span._, "kb_ents", []) for span in spans]
        cui = _best_cui(kb_ents_by_span, linker.kb, text)

        canonical: str | None = None
        if cui:
            concept = linker.kb.cui_to_entity.get(cui)
            if concept:
                canonical = concept.canonical_name

        result = (canonical, cui)
        _UMLS_CACHE[text] = result
        return result
    except Exception as exc:
        logger.debug("NORMALIZE: UMLS lookup failed for %r: %s", text, exc)
        result = (None, None)
        _UMLS_CACHE[text] = result
        return result


def _umls_canonical(text: str) -> str | None:
    """Return just the canonical name. Backward-compat wrapper around _umls_canonical_with_cui."""
    canonical, _ = _umls_canonical_with_cui(text)
    return canonical


def _resolve_entity(name: str | None, synonyms: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Canonical resolution core shared by `normalize_entity` and `NormalizeStage._norm`.

    Returns (canonical_name, cui | None).

    Resolution order:
      1. synonyms dict — domain knowledge takes priority; covers acronyms the
         UMLS linker gets wrong (e.g. CEAN→Cetacea).  After resolving the
         surface form, a follow-up UMLS lookup on the canonical name fills in
         the CUI so grouped findings can be keyed on CUI even when the synonym
         dict fires.
      2. UMLS linker (scispaCy singleton) — returns canonical name + CUI.
      3. Identity — stripped input returned unchanged, CUI is None.
    """
    if name is None:
        return None, None
    stripped = name.strip()
    from_dict = synonyms.get(stripped.lower())
    if from_dict is not None:
        # Synonym dict resolved the name; look up CUI on the canonical form.
        _, cui = _umls_canonical_with_cui(from_dict)
        return from_dict, cui
    canonical, cui = _umls_canonical_with_cui(stripped)
    if canonical:
        return canonical, cui
    return stripped, None


def normalize_entity(name: str | None) -> str | None:
    """Return the canonical form of an entity name using the built-in synonym dict."""
    canonical, _ = _resolve_entity(name, _SYNONYMS)
    return canonical


def normalize_entity_with_cui(name: str | None) -> tuple[str | None, str | None]:
    """Return (canonical_name, cui) pair using the built-in synonym dict + UMLS."""
    return _resolve_entity(name, _SYNONYMS)


# ── Assertion status inference ─────────────────────────────────────────────────

# Ordered: negative checked first so "not expressed" beats "expressed"
_NEGATIVE_TRIGGERS = (
    "not immunoreactive",
    "not reactive",
    "non-reactive",
    "not identified",
    "not detected",
    "not expressed",
    "no recurrence",
    "no metastasis",
    "no pleomorphism",
    "no cytologic atypia",
    "no atypia",
    "no mitoses",
    "not seen",
    "not observed",
    "non-",
    "absent",
    "negative",
    "lacking",
    "without",
    "no ",
    "not ",
)

_POSITIVE_TRIGGERS = (
    "present",
    "positive",
    "expressed",
    "immunoreactive",
    "reactive",
    "identified",
    "detected",
    "seen",
    "observed",
    "exhibits",
    "shows",
    "found",
)


def infer_direction(claim: str) -> DirectionEnum:
    """
    Infer direction from claim text using ordered keyword matching.

    Negative triggers are checked before positive triggers so that phrases
    like "not expressed" correctly return negative rather than positive.

    Returns DirectionEnum.unclear when no trigger matches.
    """
    lower = claim.lower()
    for trigger in _NEGATIVE_TRIGGERS:
        if trigger in lower:
            return DirectionEnum.negative
    for trigger in _POSITIVE_TRIGGERS:
        if trigger in lower:
            return DirectionEnum.positive
    return DirectionEnum.unclear



# ── Dedup key ──────────────────────────────────────────────────────────────────

def _dedup_key(
    text_element_id: int | None,
    subject: str | None,
    outcome: str | None,
    relation_type: RelationTypeEnum,
    direction: DirectionEnum | None,
    category: str | None,
) -> str | None:
    """
    Return a grouping key for conditional dedup, or None if any field that
    would make the key meaningful is absent (→ finding kept as-is, not merged).

    text_element_id=None means the evidence string was missing or malformed;
    treat as ungroupable to avoid merging unrelated findings under a shared
    sentinel value.

    direction is part of the key so opposite-polarity findings extracted from
    the same sentence stay separate — otherwise a positive and negative claim
    on the same te_id collapse silently and CONTRADICT can never surface at
    RELATE.

    category is part of the key so findings differing only on category (e.g.
    "morphology" vs "IHC") stay separate. Pre-this-change the dedup pretended
    these were the same claim and silently picked the best-grounded member's
    category — losing the discarded category entirely. Downstream GROUP keys
    on category, so a cross-category merge would precommit a single category
    before GROUP even saw the data.
    """
    if (
        subject is None
        or outcome is None
        or relation_type is RelationTypeEnum.unclear
        or text_element_id is None
        or category is None
    ):
        return None
    dir_key = direction.value if direction is not None else "none"
    return f"{text_element_id}|{subject}|{outcome}|{relation_type.value}|{dir_key}|{category}"


# ── Source span extraction ─────────────────────────────────────────────────────

def _spans_from_finding(f: Finding) -> list[SourceSpan]:
    """
    Convert a Finding's evidence list to SourceSpan objects.

    Evidence strings have the format "S{i}|{pmcid}|{te_id}".
    Malformed entries are skipped with a debug log.
    """
    spans = []
    for ev in f.evidence:
        parts = ev.split("|")
        if len(parts) != 3:
            logger.debug("Skipping malformed evidence string: %r", ev)
            continue
        sentence_id, pmcid, te_id_str = parts
        try:
            te_id = int(te_id_str)
        except ValueError:
            logger.debug("Non-integer text_element_id in evidence: %r", ev)
            continue
        spans.append(SourceSpan(
            sentence_id=sentence_id,
            pmcid=pmcid,
            text_element_id=te_id,
            verbatim=f.verbatim_support,
        ))
    return spans


def _normal_id(pmcid: str, subject: str | None, outcome: str | None,
               relation_type: RelationTypeEnum, te_ids: list[int]) -> str:
    raw = f"{subject}|{outcome}|{relation_type.value}|{'_'.join(map(str, sorted(te_ids)))}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"NF_{pmcid}_{digest}"


def _best_scope(findings: list[Finding]) -> FindingScope:
    """
    Return the scope from the finding with the highest grounding_score,
    falling back to the first finding if none have scores.
    """
    scored = [(f.grounding_score or 0.0, f) for f in findings]
    return max(scored, key=lambda x: x[0])[1].scope or FindingScope()


def _mean_score(findings: list[Finding]) -> float | None:
    scores = [f.grounding_score for f in findings if f.grounding_score is not None]
    return sum(scores) / len(scores) if scores else None


def _collect_source_ids(findings: list[Finding]) -> list[str]:
    """Stable, de-duplicated list of source finding_ids from MAP.

    Skips findings whose finding_id was never assigned (e.g. legacy cached
    payloads consumed in isolation, or unit tests that bypass the runner).
    Order follows first-seen order across `findings` so the result is
    reproducible across runs.
    """
    seen: set[str] = set()
    out: list[str] = []
    for f in findings:
        fid = getattr(f, "finding_id", None)
        if fid and fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


# ── Public API ─────────────────────────────────────────────────────────────────

class NormalizeStage:
    """
    NORMALIZE stage: entity normalization + conditional dedup.

    Parameters
    ----------
    extra_synonyms:
        Additional synonym mappings to merge with the built-in dictionary.
        Keys must be lowercase surface forms.
    """

    def __init__(self, extra_synonyms: dict[str, str] | None = None) -> None:
        self._synonyms = dict(_SYNONYMS)
        if extra_synonyms:
            self._synonyms.update(extra_synonyms)

    def normalize(self, findings: list[Finding], pmcid: str) -> list[NormalFinding]:
        """
        Normalize and deduplicate findings for one paper.

        Returns one NormalFinding per distinct (te_id, subject, outcome,
        direction) group, plus one per ungroupable finding (missing fields).
        """
        if not findings:
            return []

        # Step 1: normalize entity names in-place on a working copy
        normalized = self._normalize_entities(findings)

        # Step 2: partition into groupable and ungroupable
        groupable: dict[str, list[tuple[Finding, str | None, str | None]]] = defaultdict(list)
        ungroupable: list[tuple[Finding, str | None, str | None]] = []

        for f, te_id, subj_cui, out_cui in normalized:
            key = _dedup_key(
                te_id, f.subject_entity, f.outcome_entity,
                f.relation_type, f.direction, f.category,
            )
            if key is None:
                ungroupable.append((f, subj_cui, out_cui))
            else:
                groupable[key].append((f, subj_cui, out_cui))

        results: list[NormalFinding] = []

        # Step 3: merge each groupable cluster
        for key, group in groupable.items():
            results.append(self._merge(group, pmcid))

        # Step 4: wrap each ungroupable finding individually
        for f, subj_cui, out_cui in ungroupable:
            results.append(self._wrap_single(f, pmcid, subj_cui, out_cui))

        logger.info(
            "[%s] NORMALIZE: %d findings → %d NormalFindings "
            "(%d grouped clusters, %d ungroupable)",
            pmcid, len(findings), len(results),
            len(groupable), len(ungroupable),
        )
        return results

    # ── Internals ──────────────────────────────────────────────────────────────

    def _normalize_entities(
        self, findings: list[Finding]
    ) -> list[tuple[Finding, int | None, str | None, str | None]]:
        """
        Return (finding_with_normalized_entities, text_element_id, subject_cui, outcome_cui) tuples.
        text_element_id is extracted from the first evidence string; None if absent/malformed.
        """
        result = []
        for f in findings:
            # Extract te_id from first evidence string for grouping key.
            # None means evidence was absent or malformed → treated as
            # ungroupable so unrelated findings are never merged.
            te_id: int | None = None
            if f.evidence:
                parts = f.evidence[0].split("|")
                if len(parts) == 3:
                    try:
                        te_id = int(parts[2])
                    except ValueError:
                        pass

            # Apply the keyword heuristic when the LLM could not determine
            # direction — it may recover a clear polarity the LLM missed.
            # When the LLM supplied a concrete direction, preserve it.
            direction = (
                infer_direction(f.claim)
                if (f.direction is None or f.direction == DirectionEnum.unclear)
                else f.direction
            )
            subj_name, subj_cui = self._norm(f.subject_entity)
            out_name, out_cui = self._norm(f.outcome_entity)
            normed = f.model_copy(update={
                "subject_entity": subj_name,
                "outcome_entity": out_name,
                "direction": direction,
            })
            result.append((normed, te_id, subj_cui, out_cui))
        return result

    def _norm(self, name: str | None) -> tuple[str | None, str | None]:
        """Resolve an entity name; returns (canonical_name, cui | None)."""
        return _resolve_entity(name, self._synonyms)

    def _merge(
        self,
        group: list[tuple[Finding, str | None, str | None]],
        pmcid: str,
    ) -> NormalFinding:
        findings = [f for f, _, _ in group]
        # Representative finding: highest grounding_score
        rep = max(findings, key=lambda f: f.grounding_score or 0.0)
        rep_subj_cui = next((sc for f, sc, _ in group if f is rep), None)
        rep_out_cui = next((oc for f, _, oc in group if f is rep), None)

        # Deduplicated spans (by sentence_id + te_id)
        seen: set[tuple[str, int]] = set()
        spans: list[SourceSpan] = []
        te_ids: list[int] = []
        for f in findings:
            for span in _spans_from_finding(f):
                key = (span.sentence_id, span.text_element_id)
                if key not in seen:
                    seen.add(key)
                    spans.append(span)
                if span.text_element_id not in te_ids:
                    te_ids.append(span.text_element_id)

        unique_pmcids = sorted({s.pmcid for s in spans}) or [pmcid]

        direction = (
            infer_direction(rep.claim)
            if (rep.direction is None or rep.direction == DirectionEnum.unclear)
            else rep.direction
        )
        return NormalFinding(
            normal_id=_normal_id(pmcid, rep.subject_entity, rep.outcome_entity,
                                 rep.relation_type, te_ids),
            subject_entity=rep.subject_entity,
            outcome_entity=rep.outcome_entity,
            subject_cui=rep_subj_cui,
            outcome_cui=rep_out_cui,
            relation_type=rep.relation_type,
            direction=direction,
            category=rep.category,
            predicate_text=rep.claim,
            scope=rep.scope or FindingScope(),
            source_finding_ids=_collect_source_ids(findings),
            evidence=spans,
            pmcids=unique_pmcids,
            mean_grounding_score=_mean_score(findings),
        )

    def _wrap_single(
        self,
        f: Finding,
        pmcid: str,
        subject_cui: str | None = None,
        outcome_cui: str | None = None,
    ) -> NormalFinding:
        spans = _spans_from_finding(f)
        te_ids = [s.text_element_id for s in spans]
        unique_pmcids = sorted({s.pmcid for s in spans}) or [pmcid]
        direction = (
            infer_direction(f.claim)
            if (f.direction is None or f.direction == DirectionEnum.unclear)
            else f.direction
        )
        return NormalFinding(
            normal_id=_normal_id(pmcid, f.subject_entity, f.outcome_entity,
                                 f.relation_type, te_ids),
            subject_entity=f.subject_entity,
            outcome_entity=f.outcome_entity,
            subject_cui=subject_cui,
            outcome_cui=outcome_cui,
            relation_type=f.relation_type,
            direction=direction,
            category=f.category,
            predicate_text=f.claim,
            scope=f.scope or FindingScope(),
            source_finding_ids=_collect_source_ids([f]),
            evidence=spans,
            pmcids=unique_pmcids,
            mean_grounding_score=f.grounding_score,
        )
