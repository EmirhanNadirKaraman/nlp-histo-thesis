"""
Export Disease Entities by UMLS CUI

Combines functionality of merge_entities_by_umls.py and copy_relevant_files.py.
Directly queries disease-related entities and exports them grouped by UMLS CUI.

This is more efficient than:
1. Generating ALL entity files
2. Copying only disease-related ones

Instead, it:
1. Queries only disease-related entities (filtered by semantic types)
2. Groups by UMLS CUI
3. Exports JSON and TXT files for each concept

Usage:
    Run from inside `named_entity_recognition/`: the default output directory is
    CWD-relative, so this keeps the files where they have always been written
    (`named_entity_recognition/disease_entities_lg/`). The `cd` is not needed for
    imports — the editable install provides those.

    cd named_entity_recognition
    python -m nlp_histo.ner.export_disease_entities
    python -m nlp_histo.ner.export_disease_entities --output-dir disease_entities
    python -m nlp_histo.ner.export_disease_entities --min-occurrences 5
    python -m nlp_histo.ner.export_disease_entities --semantic-types T047 T191
"""

import json
import sys
import argparse
from collections.abc import Sequence
from pathlib import Path
from collections import defaultdict

from sqlalchemy.dialects.postgresql import array
from nlp_histo.database import get_db_connection, Entity, TextElement, Document
from nlp_histo.ner.enums import DEFAULT_MODEL_NAME, UMLS_DISEASE_TYPES
from nlp_histo.ner.model_filter import NoMatchingEntitiesError, check_model_filter


def export_disease_entities(
    output_dir: str = None,
    min_occurrences: int = 1,
    semantic_types: list = None,
    limit: int = None,
    pmcid: str = None,
    model_name: str = None
):
    """
    Query disease-related entities and export grouped by UMLS CUI.

    Args:
        output_dir: Output directory for JSON/TXT files. If None, auto-generates based on model_name.
        min_occurrences: Minimum occurrences to include a CUI
        semantic_types: List of TUI codes to filter (default: all disease types)
        limit: Limit to first N documents
        pmcid: Filter to specific PMCID
        model_name: Optional model name to filter entities. This is spaCy's
                    ``nlp.meta["name"]`` as stored in entities.model_name —
                    'core_sci_lg', not the package name 'en_core_sci_lg' (B-115).

    Returns:
        dict: Exported entity data grouped by UMLS CUI
    """
    # Auto-generate output directory based on model name if not specified
    if output_dir is None:
        if model_name:
            if model_name.endswith('_lg'):
                output_dir = "disease_entities_lg"
            elif model_name.endswith('_sm'):
                output_dir = "disease_entities_sm"
            else:
                output_dir = f"disease_entities_{model_name}"
        else:
            output_dir = "disease_entities"
    print("=" * 80)
    print("Disease Entity Exporter")
    print("=" * 80)

    if model_name:
        print(f"Model filter: {model_name}")

    # Use default disease types if not specified
    if semantic_types is None:
        semantic_types = list(UMLS_DISEASE_TYPES.keys())

    # Show which semantic types we're filtering
    print(f"Semantic types filter ({len(semantic_types)}):")
    for tui in semantic_types:
        name = UMLS_DISEASE_TYPES.get(tui, "Unknown")
        print(f"  {tui}: {name}")

    print(f"\nMinimum occurrences: {min_occurrences}")
    print(f"Output directory: {output_dir}/")

    if pmcid:
        print(f"Filter: PMCID {pmcid}")
    elif limit:
        print(f"Filter: First {limit} documents")
    else:
        print("Filter: All documents")

    print("=" * 80 + "\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db = get_db_connection()

    with db.session_scope() as session:
        # Build query for disease entities with their text context
        query = (
            session.query(
                Entity.umls_cui,
                Entity.canonical_name,
                Entity.entity_label,
                Entity.entity_text,
                Entity.start_char,
                Entity.end_char,
                Entity.umls_score,
                Entity.semantic_types,
                TextElement.id.label('text_element_id'),
                TextElement.text_content,
                TextElement.path_string,
                Document.pmcid
            )
            .join(TextElement, Entity.text_element_id == TextElement.id)
            .join(Document, TextElement.document_id == Document.id)
            .filter(Entity.umls_cui.isnot(None))
            # Filter by semantic types using array overlap operator
            .filter(Entity.semantic_types.op('&&')(array(semantic_types)))
        )

        # Filter by model name if specified
        if model_name:
            query = query.filter(Entity.model_name == model_name)

        # Apply optional filters
        if pmcid:
            query = query.filter(Document.pmcid == pmcid)
        elif limit:
            # Get document IDs to filter by
            doc_ids = [
                doc.id for doc in session.query(Document.id).limit(limit).all()
            ]
            query = query.filter(Document.id.in_(doc_ids))
            print(f"Processing entities from {len(doc_ids)} documents\n")

        # Order by CUI for consistent output
        query = query.order_by(Entity.umls_cui)

        print("Querying database (streaming results)...")

        # Group by UMLS CUI
        umls_groups = defaultdict(lambda: {
            "canonical_name": None,
            "entity_label": None,
            "semantic_types": None,
            "total_occurrences": 0,
            "unique_entity_texts": set(),
            "sentences": [],
            "seen_sentences": set()
        })

        row_count = 0
        for row in query.yield_per(1000):
            row_count += 1
            cui = row.umls_cui

            # Set metadata (first occurrence)
            if umls_groups[cui]["canonical_name"] is None:
                umls_groups[cui]["canonical_name"] = row.canonical_name
                umls_groups[cui]["entity_label"] = row.entity_label
                umls_groups[cui]["semantic_types"] = row.semantic_types

            # Add entity text variant
            umls_groups[cui]["unique_entity_texts"].add(row.entity_text)
            umls_groups[cui]["total_occurrences"] += 1

            # Add sentence occurrence (only if unique)
            sentence_text = row.text_content
            if sentence_text not in umls_groups[cui]["seen_sentences"]:
                umls_groups[cui]["seen_sentences"].add(sentence_text)
                umls_groups[cui]["sentences"].append({
                    "pmcid": row.pmcid,
                    "text_element_id": row.text_element_id,
                    "sentence": sentence_text,
                    "section": row.path_string,
                    "entity_text": row.entity_text,
                    "start_char": row.start_char,
                    "end_char": row.end_char,
                    "umls_score": float(row.umls_score) if row.umls_score else None
                })

            # Progress indicator
            if row_count % 10000 == 0:
                print(f"  Processed {row_count} rows, {len(umls_groups)} unique CUIs...")

        if row_count == 0:
            # See B-115: only report emptiness when the filter matches existing data.
            check_model_filter(session, model_name)
            print("No disease entities found matching the criteria.")
            return {}

        print(f"\nProcessed {row_count} entity occurrences")
        print(f"Found {len(umls_groups)} unique disease CUIs\n")

        # Filter by minimum occurrences and convert sets to lists
        print(f"Filtering CUIs with at least {min_occurrences} occurrences...")
        filtered_data = {}

        for cui, data in umls_groups.items():
            if data["total_occurrences"] >= min_occurrences:
                sorted_sentences = sorted(
                    data["sentences"], key=lambda x: x["text_element_id"]
                )

                filtered_data[cui] = {
                    "canonical_name": data["canonical_name"],
                    "entity_label": data["entity_label"],
                    "semantic_types": data["semantic_types"],
                    "total_occurrences": data["total_occurrences"],
                    "unique_entity_texts": sorted(list(data["unique_entity_texts"])),
                    "sentences": sorted_sentences
                }

        print(f"Retained {len(filtered_data)} CUIs after filtering\n")

        # Save each UMLS CUI to separate files
        print(f"Writing {len(filtered_data)} JSON and TXT files to {output_dir}/...")

        for cui, data in filtered_data.items():
            # Create safe filename
            safe_filename = cui.replace('/', '_').replace('\\', '_')

            if data["canonical_name"]:
                canonical_safe = data["canonical_name"][:50]
                canonical_safe = "".join(
                    c if c.isalnum() or c in (' ', '-', '_') else '_'
                    for c in canonical_safe
                )
                canonical_safe = canonical_safe.replace(' ', '_')
                base_filename = f"{safe_filename}_{canonical_safe}"
            else:
                base_filename = safe_filename

            # Save JSON file
            json_path = output_path / f"{base_filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "umls_cui": cui,
                    "canonical_name": data["canonical_name"],
                    "entity_label": data["entity_label"],
                    "semantic_types": data["semantic_types"],
                    "total_occurrences": data["total_occurrences"],
                    "unique_entity_texts": data["unique_entity_texts"],
                    "sentences": data["sentences"]
                }, f, indent=2, ensure_ascii=False)

            # Save TXT file
            txt_path = output_path / f"{base_filename}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"UMLS CUI: {cui}\n")
                f.write(f"Canonical Name: {data['canonical_name']}\n")
                f.write(f"Entity Label: {data['entity_label']}\n")
                f.write(f"Semantic Types: {data['semantic_types']}\n")
                f.write(f"Total Occurrences: {data['total_occurrences']}\n")
                f.write(f"Unique Sentences: {len(data['sentences'])}\n")
                f.write(f"Text Variants: {', '.join(data['unique_entity_texts'])}\n")
                f.write("=" * 80 + "\n\n")

                for sentence_data in data["sentences"]:
                    f.write(sentence_data["sentence"])
                    f.write("\n\n")

        print(f"✓ Saved {len(filtered_data) * 2} files to {output_dir}/\n")

        # Print summary
        print("=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(f"Unique disease CUIs: {len(filtered_data)}")

        if filtered_data:
            total_occurrences = sum(d["total_occurrences"] for d in filtered_data.values())
            total_sentences = sum(len(d["sentences"]) for d in filtered_data.values())
            print(f"Total entity occurrences: {total_occurrences}")
            print(f"Total unique sentences: {total_sentences}")

            # Top 10 most frequent
            top_concepts = sorted(
                filtered_data.items(),
                key=lambda x: x[1]["total_occurrences"],
                reverse=True
            )[:10]

            print("\nTop 10 Most Frequent Disease Concepts:")
            print("-" * 80)
            for i, (cui, data) in enumerate(top_concepts, 1):
                canonical = data["canonical_name"] or "Unknown"
                label = data["entity_label"]
                count = data["total_occurrences"]
                unique_sents = len(data["sentences"])
                sem_types = data["semantic_types"]
                print(f"{i:2d}. {canonical[:40]:<40} [{label}]")
                print(f"    CUI: {cui} | Types: {sem_types}")
                print(f"    Occurrences: {count} | Unique sentences: {unique_sents}")
                print()

        print("=" * 80 + "\n")

        return filtered_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export disease-related entities grouped by UMLS CUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run from inside named_entity_recognition/ — the default output directory is
CWD-relative, so this writes to named_entity_recognition/disease_entities_lg/ as
before.  (The cd is only for the output location, not for imports.)

  cd named_entity_recognition

Examples:
  # Export all disease entities
  python -m nlp_histo.ner.export_disease_entities

  # Filter by model (auto-sets output dir to disease_entities_lg)
  python -m nlp_histo.ner.export_disease_entities --model core_sci_lg

  # Custom output directory
  python -m nlp_histo.ner.export_disease_entities --output-dir results/diseases

  # Filter by minimum occurrences
  python -m nlp_histo.ner.export_disease_entities --min-occurrences 10

  # Specific semantic types only (diseases and neoplasms)
  python -m nlp_histo.ner.export_disease_entities --semantic-types T047 T191

  # Process specific document
  python -m nlp_histo.ner.export_disease_entities --pmcid PMC1448691

  # Limit to first N documents
  python -m nlp_histo.ner.export_disease_entities --limit 50

Available semantic types:
  T047 - Disease or Syndrome
  T191 - Neoplastic Process
  T048 - Mental or Behavioral Dysfunction
  T037 - Injury or Poisoning
  T046 - Pathologic Function
  T020 - Congenital Abnormality
  T184 - Sign or Symptom
  T033 - Finding
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for JSON/TXT files (default: auto-based on model)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f'Filter by model name (default: {DEFAULT_MODEL_NAME}). Output dir auto-set '
             'if not specified.'
    )
    parser.add_argument(
        '--min-occurrences',
        type=int,
        default=1,
        help='Minimum occurrences to include a CUI (default: 1)'
    )
    parser.add_argument(
        '--semantic-types',
        nargs='+',
        type=str,
        help='TUI codes to filter (default: all disease types). Example: T047 T191'
    )
    parser.add_argument(
        '--pmcid',
        type=str,
        help='Process only this PMCID'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit to first N documents'
    )

    args = parser.parse_args(argv)

    try:
        export_disease_entities(
            output_dir=args.output_dir,
            min_occurrences=args.min_occurrences,
            semantic_types=args.semantic_types,
            limit=args.limit,
            pmcid=args.pmcid,
            model_name=args.model
        )
    except NoMatchingEntitiesError as e:
        # A filter that matches nothing while data exists is an error, not an
        # empty result (B-115).
        print(f"\n❌ {e}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
