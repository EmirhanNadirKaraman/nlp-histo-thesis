"""Fingerprint construction.

A :class:`PaperFingerprint` is built from a :class:`RawPaper` (loaders.py)
using three independent signals:

  1. NER/UMLS entities (with semantic-type filtering).
  2. Regex extraction of common histopathology markers (CD/Ki-67/genes).
  3. Keyword dictionaries (disease, tissue, method, outcome).

A configurable generic-term filter prevents broad words like "patient" or
"cell" from dominating downstream relatedness.

Sentence count is approximated locally (no NLTK / spaCy required) by counting
``.``/``!``/``?`` followed by whitespace; chunk count uses the same chunk size
as the summarisation MAP stage so workload numbers line up.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field

from .loaders import RawPaper
from .models import PaperFingerprint

logger = logging.getLogger(__name__)

_PROGRESS_EVERY = 100

# UMLS semantic-type buckets
# Keys are TUI codes; semantic types not listed here fall through to the
# generic ``entities`` set only.

DISEASE_TUIS: set[str] = {
    "T047",  # Disease or Syndrome
    "T191",  # Neoplastic Process
    "T048",  # Mental or Behavioral Dysfunction
    "T046",  # Pathologic Function
    "T020",  # Congenital Abnormality
    "T037",  # Injury or Poisoning
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T019",  # Congenital Abnormality (alt)
    "T190",  # Anatomical Abnormality
}
GENE_TUIS: set[str] = {
    "T028",  # Gene or Genome
    "T123",  # Biologically Active Substance
    "T126",  # Enzyme
    "T116",  # Amino Acid, Peptide, or Protein
    "T044",  # Molecular Function
}
TISSUE_TUIS: set[str] = {
    "T024",  # Tissue
    "T023",  # Body Part, Organ, or Organ Component
    "T025",  # Cell
    "T029",  # Body Location or Region
    "T017",  # Anatomical Structure
}
METHOD_TUIS: set[str] = {
    "T059",  # Laboratory Procedure
    "T060",  # Diagnostic Procedure
    "T063",  # Molecular Biology Research Technique
    "T034",  # Laboratory or Test Result
}
OUTCOME_TUIS: set[str] = {
    "T201",  # Clinical Attribute
    "T169",  # Functional Concept (broad — used as soft signal only)
}


# Regex patterns

_CD_MARKER_RE = re.compile(r"\bCD[- ]?\d+[a-z]?\b", re.IGNORECASE)
_KI67_RE      = re.compile(r"\bKi[- ]?67\b", re.IGNORECASE)
# Gene-like uppercase symbol: 2–6 letters/digits, must contain a letter and
# look gene-like (not a generic acronym like USA / DNA / PDF). Word boundaries.
_GENE_LIKE_RE = re.compile(r"\b(?=[A-Z]*\d|[A-Z]{2,})[A-Z][A-Z0-9]{1,5}\b")

# Curated gene/protein allow-list — extends regex hits and survives the
# generic-acronym filter.
_GENE_ALLOWLIST: set[str] = {
    "MYC", "BCL2", "BCL6", "TP53", "TP63", "ALK", "EGFR", "ROS1", "KRAS",
    "NRAS", "HRAS", "BRAF", "MEK", "PIK3CA", "AKT", "PTEN", "RB1", "MDM2",
    "PDL1", "PD1", "CTLA4", "VEGFA", "VEGFR", "HER2", "ERBB2", "ERBB3",
    "ERG", "ETV4", "ETV5", "TMPRSS2", "FLI1", "MET", "RET", "NTRK1",
    "NTRK2", "NTRK3", "FGFR1", "FGFR2", "FGFR3", "IDH1", "IDH2", "ATRX",
    "TERT", "BRCA1", "BRCA2", "ATM", "CHEK2", "MLH1", "MSH2", "MSH6",
    "PMS2", "APC", "SMAD4", "DPC4", "WT1", "PAX8", "PAX5", "GATA3",
    "FOXP3", "MITF", "SOX2", "SOX10", "S100", "MUM1", "IRF4", "MUC1",
    "MUC2", "MUC4", "MUC5AC", "MUC6", "EBV", "HHV8", "HPV",
}

# Gene-like-but-actually-generic acronyms.
_GENE_BLOCKLIST: set[str] = {
    "DNA", "RNA", "MRNA", "CDNA", "PCR", "FISH", "IHC", "USA", "UK", "NHS",
    "WHO", "DOI", "JPG", "JPEG", "PNG", "PDF", "URL", "XML", "JSON",
    "FFPE", "HE", "HES", "AB", "ABS", "IGG", "IGM", "IGA", "IGE", "IGD",
    "TNM", "AJCC", "UICC", "WHO", "OS", "PFS", "DFS", "RFS", "DSS",
    "CR", "PR", "CT", "MRI", "PET", "BMI", "ECG", "EEG", "VS", "CI",
}


# Keyword dictionaries

DISEASE_KEYWORDS: set[str] = {
    "carcinoma", "lymphoma", "melanoma", "sarcoma", "leukemia", "leukaemia",
    "adenoma", "adenocarcinoma", "papilloma", "neoplasm", "tumor", "tumour",
    "hodgkin", "non-hodgkin", "dlbcl", "follicular lymphoma",
    "burkitt", "myeloma", "glioma", "glioblastoma", "neuroblastoma",
    "mesothelioma", "teratoma", "hepatocellular", "renal cell",
    "breast cancer", "prostate cancer", "lung cancer", "colorectal",
    "gastric", "pancreatic", "endometrial", "cervical", "ovarian",
    "thyroid carcinoma", "squamous cell", "basal cell",
    "granuloma", "granuloma annulare", "psoriasis", "dermatitis",
    "fibrosis", "cirrhosis", "metaplasia", "dysplasia",
}
TISSUE_KEYWORDS: set[str] = {
    "lymph node", "bone marrow", "spleen", "liver", "kidney", "lung",
    "breast", "prostate", "colon", "stomach", "pancreas", "thyroid",
    "skin", "dermis", "epidermis", "subcutaneous", "ovary", "uterus",
    "endometrium", "cervix", "testis", "bladder", "brain", "spinal cord",
    "intestine", "jejunum", "ileum", "duodenum", "gallbladder",
    "tonsil", "salivary gland", "thymus",
}
METHOD_KEYWORDS: set[str] = {
    "ihc", "immunohistochemistry", "immunohistochemical",
    "fish", "fluorescence in situ hybridization",
    "pcr", "rt-pcr", "qpcr", "rna-seq", "rnaseq",
    "wgs", "wes", "ngs", "sanger sequencing",
    "h&e", "hematoxylin", "eosin", "ngs panel",
    "microarray", "flow cytometry", "western blot",
    "elisa", "mass spectrometry", "ihc panel",
    "core needle biopsy", "fine needle aspiration",
    "tma", "tissue microarray",
}
OUTCOME_KEYWORDS: set[str] = {
    "overall survival", "progression-free survival", "recurrence-free",
    "disease-free", "relapse", "metastasis", "metastases",
    "5-year survival", "five-year survival",
    "complete remission", "partial response",
    "treatment response", "refractory", "resistant",
    "median survival", "hazard ratio", "p-value",
    "mortality", "morbidity", "prognosis",
}

# Words to strip from the generic ``entities`` bucket so they don't dominate
# overlap-based relatedness. Disease-family terms (carcinoma/lymphoma) stay in
# disease_entities — see ``_keyword_hits``.
GENERIC_NOISE: set[str] = {
    "patient", "patients", "case", "cases", "tumor", "tumors", "tumour",
    "tumours", "cell", "cells", "expression", "staining", "positive",
    "negative", "disease", "study", "analysis", "result", "results",
    "method", "methods", "finding", "findings", "outcome", "outcomes",
    "report", "reports", "section", "specimen", "specimens", "biopsy",
    "biopsies", "tissue", "tissues",
}


# Config

@dataclass
class FingerprintConfig:
    """Tunable knobs for fingerprint construction.

    ``chunk_size`` mirrors ``MapConfig.chunk_size`` so workload numbers used by
    the hardness metric line up with what the MAP stage will actually see.
    """
    chunk_size: int = 10
    min_umls_score: float = 0.0
    keep_disease_family_terms: bool = True
    extra_disease_keywords: set[str] = field(default_factory=set)
    extra_method_keywords:  set[str] = field(default_factory=set)
    extra_outcome_keywords: set[str] = field(default_factory=set)
    extra_tissue_keywords:  set[str] = field(default_factory=set)


# Helpers

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ABBREV_RE = re.compile(r"\b[A-Z]{2,5}\b")
_TOKEN_RE  = re.compile(r"\w+")


def _count_sentences(text: str) -> int:
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return max(len(parts), 1) if text.strip() else 0


def _is_gene_like(token: str) -> bool:
    if token in _GENE_BLOCKLIST:
        return False
    if token in _GENE_ALLOWLIST:
        return True
    return bool(_GENE_LIKE_RE.fullmatch(token))


def _norm(text: str) -> str:
    return text.strip().lower()


def _keyword_hits(text_lower: str, keywords: set[str]) -> set[str]:
    return {kw for kw in keywords if kw in text_lower}


def _bucket_from_semantic_types(semtypes: list[str]) -> set[str]:
    """Return human-readable bucket labels for an entity's TUI list."""
    buckets: set[str] = set()
    for tui in semtypes:
        if tui in DISEASE_TUIS:
            buckets.add("disease")
        if tui in GENE_TUIS:
            buckets.add("gene")
        if tui in TISSUE_TUIS:
            buckets.add("tissue")
        if tui in METHOD_TUIS:
            buckets.add("method")
        if tui in OUTCOME_TUIS:
            buckets.add("outcome")
    return buckets


# Public API

def build_fingerprint(paper: RawPaper, config: FingerprintConfig | None = None) -> PaperFingerprint:
    """Convert a :class:`RawPaper` into a :class:`PaperFingerprint`.

    Pure function; deterministic for a given input. Logs nothing.
    """
    cfg = config or FingerprintConfig()

    fp = PaperFingerprint(pmcid=paper.pmcid, title=paper.title)

    # Workload counters
    fp.n_text_elements = len(paper.text_elements)
    fp.n_paragraphs    = len(paper.text_elements)
    fp.n_sentences     = sum(_count_sentences(t) for t in paper.text_elements)
    fp.n_chunks        = max(1, math.ceil(fp.n_sentences / max(cfg.chunk_size, 1))) if fp.n_sentences else 0
    fp.n_tables        = paper.n_tables
    fp.n_figures       = paper.n_figures
    fp.n_captions      = paper.n_captions

    # Token approximation: word-count proxy
    n_tokens = sum(len(_TOKEN_RE.findall(t)) for t in paper.text_elements)
    fp.n_tokens_est = n_tokens or None

    # Entity buckets (NER + UMLS)
    counts: dict[str, int] = {}
    for ent in paper.entities:
        if cfg.min_umls_score and ent.umls_score is not None and ent.umls_score < cfg.min_umls_score:
            continue
        text = _norm(ent.text)
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
        if ent.umls_cui:
            fp.umls_cuis.add(ent.umls_cui)
        for tui in ent.semantic_types or []:
            fp.semantic_types.add(tui)

        buckets = _bucket_from_semantic_types(ent.semantic_types or [])
        if "disease" in buckets:
            fp.disease_entities.add(text)
        if "gene" in buckets:
            fp.gene_entities.add(text)
        if "tissue" in buckets:
            fp.tissue_entities.add(text)
        if "method" in buckets:
            fp.method_entities.add(text)
        if "outcome" in buckets:
            fp.outcome_entities.add(text)
        if not buckets:
            fp.entities.add(text)

    # Drop generic noise from the catch-all bucket only — we deliberately keep
    # generic words inside disease_/method_/outcome_ buckets when they came
    # from a real semantic-type hit.
    fp.entities -= GENERIC_NOISE

    # Regex markers + keyword dictionaries
    full_text = "\n".join(paper.text_elements)
    full_lower = full_text.lower()

    cd_hits = {m.group(0).upper().replace(" ", "").replace("-", "")
               for m in _CD_MARKER_RE.finditer(full_text)}
    fp.biomarker_entities |= cd_hits
    if _KI67_RE.search(full_text):
        fp.biomarker_entities.add("KI67")

    # Gene-like uppercase tokens, filtered against blocklist/allowlist
    candidates = set(_GENE_LIKE_RE.findall(full_text))
    candidates |= set(re.findall(r"\b[A-Z][A-Z0-9]{1,6}\b", full_text))
    gene_hits: set[str] = {tok for tok in candidates if _is_gene_like(tok)}
    fp.gene_entities |= {g.lower() for g in gene_hits}

    # Keyword hits use lowercase substring matching
    disease_kw = (DISEASE_KEYWORDS | cfg.extra_disease_keywords)
    tissue_kw  = (TISSUE_KEYWORDS  | cfg.extra_tissue_keywords)
    method_kw  = (METHOD_KEYWORDS  | cfg.extra_method_keywords)
    outcome_kw = (OUTCOME_KEYWORDS | cfg.extra_outcome_keywords)

    fp.disease_entities |= _keyword_hits(full_lower, disease_kw)
    fp.tissue_entities  |= _keyword_hits(full_lower, tissue_kw)
    fp.method_entities  |= _keyword_hits(full_lower, method_kw)
    fp.outcome_entities |= _keyword_hits(full_lower, outcome_kw)

    fp.entity_counts = counts

    # Layout / quality stats used by hardness
    para_lens = [len(t) for t in paper.text_elements] or [0]
    very_short = sum(1 for n in para_lens if 0 < n < 60)
    very_long  = sum(1 for n in para_lens if n > 1500)
    abbr_count = len(_ABBREV_RE.findall(full_text))

    fp.source_stats = {
        "mean_paragraph_chars": float(sum(para_lens) / max(len(para_lens), 1)),
        "very_short_paragraphs": very_short,
        "very_long_paragraphs":  very_long,
        "abbrev_count":          abbr_count,
        "total_chars":           sum(para_lens),
    }

    return fp


def build_fingerprints(papers: list[RawPaper],
                       config: FingerprintConfig | None = None) -> list[PaperFingerprint]:
    cfg = config or FingerprintConfig()
    logger.info("build_fingerprints: starting on %d papers (chunk_size=%d)",
                len(papers), cfg.chunk_size)
    t0 = time.perf_counter()
    out: list[PaperFingerprint] = []
    for i, paper in enumerate(papers, start=1):
        fp = build_fingerprint(paper, cfg)
        out.append(fp)
        logger.debug("build_fingerprints[%s]: n_sentences=%d n_chunks=%d "
                     "n_useful_entities=%d",
                     fp.pmcid, fp.n_sentences, fp.n_chunks, len(fp.entity_counts))
        if i % _PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(papers) - i) / rate if rate > 0 else 0.0
            logger.info("build_fingerprints: %d/%d (%.1f papers/s, ETA %.0fs)",
                        i, len(papers), rate, eta)

    n_empty = sum(1 for fp in out if fp.is_empty())
    n_useful = sum(1 for fp in out if fp.has_useful_entities())
    n_sentences = sum(fp.n_sentences for fp in out)
    logger.info(
        "build_fingerprints: done in %.1fs — %d papers, %d empty, %d with useful "
        "entities, %d sentences total (mean %.0f/paper)",
        time.perf_counter() - t0, len(out), n_empty, n_useful, n_sentences,
        n_sentences / max(len(out), 1),
    )
    return out
