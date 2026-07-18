"""
Resilient Evaluator for Medical Summarization Pipeline.
Features: 
- ID Provenance Verification (Database)
- Numeric Integrity Check (Regex)
- Negation-Aware UMLS Concept Matching (scispaCy + Negex)
- Memory-Efficient Streaming for 45M+ token runs
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator

import spacy

# Database imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_db_connection
from database.models import TextElement, Document

# =============================================================================
# GLOBAL INITIALIZATION (Load once, use everywhere)
# =============================================================================
print("Initializing Medical Logic Engine...")
nlp = spacy.load("en_core_sci_lg", disable=["parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe("sentencizer")
nlp.add_pipe("scispacy_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "umls",
})
# Add Negation detection - Critical for avoiding "No history of X" hallucinations
# nlp.add_pipe("negex", config={"ent_types": ["DISEASE", "CHEMICAL"]})
nlp.add_pipe("negex")

print("Medical Engine Ready.")

@dataclass
class CitationResult:
    citation_raw: str
    pmcid: str
    text_element_id: int
    exists: bool
    numeric_ok: bool
    concept_score: float
    hallucinated_entities: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        # Strict medical threshold: Numbers must match, Concepts must be 70%+
        return self.exists and self.numeric_ok and self.concept_score >= 0.7

@dataclass
class EvaluationResult:
    cui: str
    concept_name: str
    citation_results: List[CitationResult] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        if not self.citation_results:
            return 1.0
        return sum(1 for r in self.citation_results if r.is_valid) / len(self.citation_results)

# =============================================================================
# CORE MEDICAL LOGIC
# =============================================================================

def get_clinical_concepts(text: str, threshold: float = 0.65) -> List[List[str]]:
    """
    Extracts UMLS candidate CUIs per entity, ignoring negated terms.

    Returns:
        List of candidate lists - each entity maps to multiple possible CUIs.
        Example: [['C001', 'C002'], ['C003']] for 2 entities
    """
    doc = nlp(text)
    concepts = []
    for ent in doc.ents:
        # Only count entities that are not negated in the text
        if ent._.kb_ents and not ent._.negex:
            # Get all candidates above threshold
            candidates = [cui for cui, score in ent._.kb_ents if score >= threshold]
            if candidates:
                concepts.append(candidates)
    return concepts


def compare_medical_entities(claim: str, source: str) -> Tuple[bool, float, List[str]]:
    """
    Compare medical concepts between claim and source text.

    Uses lenient matching: if ANY candidate CUI from claim matches ANY candidate
    from source, it's considered a match. Only flags hallucination when there's
    zero overlap between candidate sets.

    Returns:
        (is_safe, precision_score, hallucinated_cuis)
    """
    claim_concepts = get_clinical_concepts(claim)
    source_concepts = get_clinical_concepts(source)

    # Flatten source into one big set of all possible CUIs
    source_concept_set = set()
    for candidates in source_concepts:
        source_concept_set.update(candidates)

    if not claim_concepts:
        return True, 1.0, []

    matched_count = 0
    hallucinations = []

    for concept_candidates in claim_concepts:
        # Check if ANY candidate matches the source
        found = any(candidate in source_concept_set for candidate in concept_candidates)

        if found:
            matched_count += 1
        else:
            # Store the top candidate (most likely CUI) for reporting
            hallucinations.append(concept_candidates[0])

    precision = matched_count / len(claim_concepts)
    is_safe = len(hallucinations) == 0

    return is_safe, round(precision, 2), hallucinations


def verify_semantic_fidelity(claim: str, source: str) -> Tuple[bool, float, List[str]]:
    """Deterministic check for numbers and medical concepts."""
    # 1. Numeric Check
    claim_nums = set(re.findall(r'\d+', claim))
    source_nums = set(re.findall(r'\d+', source))
    numeric_ok = claim_nums.issubset(source_nums) if claim_nums else True

    # 2. Concept Check
    is_safe, concept_score, hallucinations = compare_medical_entities(claim, source)

    return numeric_ok, concept_score, hallucinations
    

# =============================================================================
# PARSING & DATABASE UTILS
# =============================================================================

def parse_citations_with_context(text: str) -> Iterator[Tuple[str, str]]:
    """Streams (claim, citation) pairs to save memory."""
    # Space-agnostic regex to handle LLM formatting drift
    pattern = r'(\[\s*(?:C\d+:)?S\d+\s*\|\s*[^\]]+\s*\|\s*\d+\s*\])'
    parts = re.split(pattern, text)
    
    current_claim = ""
    for part in parts:
        if re.match(pattern, part):
            yield (current_claim.strip(), part)
            current_claim = ""
        else:
            current_claim += part

def parse_citation_components(raw: str) -> dict:
    """Extracts components from varied citation strings."""
    # Supports [S1|PMC123|456] and [Chunk:C1, Evidence:S2|PMC123|te456]
    clean = re.sub(r'[\[\]\s]', '', raw)
    # Target: S2|PMC123|456
    parts = clean.split('|')
    if len(parts) >= 3:
        try:
            return {
                "pmcid": parts[-2],
                "te_id": int(re.search(r'\d+', parts[-1]).group())
            }
        except Exception: 
            return None
    return None

# =============================================================================
# EXECUTION PIPELINE
# =============================================================================

def evaluate_single_file(file_path: Path, session) -> EvaluationResult:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    eval_res = EvaluationResult(cui=data.get('cui', 'N/A'), concept_name=data.get('concept_name', 'N/A'))
    
    # Process main summary and narrative
    content = data.get('summary', "") + " " + data.get('audit_trail', {}).get('master_summary', {}).get('narrative_summary', "")
    
    for claim, raw_cite in parse_citations_with_context(content):
        comp = parse_citation_components(raw_cite)
        if not comp: 
            continue
        
        # Database verification
        te = session.query(TextElement).filter(TextElement.id == comp['te_id']).first()
        doc = session.query(Document).filter(Document.id == te.document_id).first() if te else None
        
        exists = te is not None and doc.pmcid == comp['pmcid']
        
        if exists:
            num_ok, concept_score, hallucinations = verify_semantic_fidelity(claim, te.text_content)
            eval_res.citation_results.append(CitationResult(
                citation_raw=raw_cite, pmcid=comp['pmcid'], text_element_id=comp['te_id'],
                exists=True, numeric_ok=num_ok, concept_score=concept_score, hallucinated_entities=hallucinations
            ))
        else:
            eval_res.citation_results.append(CitationResult(
                citation_raw=raw_cite, pmcid=comp['pmcid'], text_element_id=comp['te_id'],
                exists=False, numeric_ok=False, concept_score=0.0
            ))
            
    return eval_res

def main():
    results_dir = Path("langchain-summarization/summarization_results/summaries")
    db = get_db_connection()
    all_evals = []

    print(f"\n{'='*60}\nSTARTING CLINICAL AUDIT (45M TOKEN MODE)\n{'='*60}")

    with db.session_scope() as session:
        for json_file in results_dir.glob("*.json"):
            res = evaluate_single_file(json_file, session)
            all_evals.append(res)
            
            # Real-time Status
            color = "\033[92m" if res.score > 0.9 else "\033[93m" if res.score > 0.7 else "\033[91m"
            print(f"{color}[{res.score:.0%}]\033[0m {res.concept_name[:40]}")
            
            for c in res.citation_results:
                if not c.is_valid:
                    print(f"   └─ FAIL: {c.citation_raw} | Hallucinations: {c.hallucinated_entities}")

    # Global Stats
    avg_score = sum(e.score for e in all_evals) / len(all_evals) if all_evals else 0
    print(f"\n{'='*60}\nFINAL SYSTEM FIDELITY: {avg_score:.2%}\n{'='*60}")

if __name__ == "__main__":
    main()