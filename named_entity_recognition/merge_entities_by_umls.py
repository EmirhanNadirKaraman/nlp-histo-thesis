"""
Merge Entities by UMLS CUI

Groups all text elements (sentences) by UMLS Concept Unique Identifier (CUI).
For each UMLS concept, creates a separate JSON file with all sentences where that concept appears.

Output: Separate JSON files (one per CUI) with structure:
{
  "umls_cui": "C0004057",
  "canonical_name": "Aspirin",
  "entity_label": "CHEMICAL",
  "total_occurrences": 42,
  "unique_entity_texts": ["aspirin", "Aspirin", "ASA"],
  "sentences": [
    {
      "pmcid": "PMC1234567",
      "text_element_id": 123,
      "sentence": "Patients were treated with aspirin daily.",
      "section": "Methods > Treatment",
      "entity_text": "aspirin",
      "start_char": 28,
      "end_char": 35,
      "umls_score": 0.98
    },
    ...
  ]
}

Files are saved as: {output_dir}/{CUI}_{canonical_name}.json

Usage:
    python named_entity_recognition/merge_entities_by_umls.py
    python named_entity_recognition/merge_entities_by_umls.py --output-dir results/umls
    python named_entity_recognition/merge_entities_by_umls.py --pmcid PMC1234567
    python named_entity_recognition/merge_entities_by_umls.py --min-occurrences 5
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

from database import get_db_connection, Entity, TextElement, Document


def merge_entities_by_umls(pmcid=None, min_occurrences=1, output_dir=None, limit=None, model_name=None):
    """
    Merge all sentences where each UMLS entry occurs.
    Creates separate JSON and TXT files for each UMLS CUI.

    Args:
        pmcid: Optional PMCID to filter results (processes all documents if None)
        min_occurrences: Minimum number of occurrences to include a CUI
        output_dir: Output directory for output files (one file per CUI).
                    If None, auto-generates based on model_name.
        limit: Optional limit to process only first N documents
        model_name: Optional model name to filter entities (e.g., 'en_core_sci_lg')

    Returns:
        dict: Merged entity data grouped by UMLS CUI
    """
    # Auto-generate output directory based on model name if not specified
    if output_dir is None:
        if model_name:
            # Extract short name: en_core_sci_lg -> lg, en_core_sci_sm -> sm
            if model_name.endswith('_lg'):
                output_dir = "umls_entities_lg"
            elif model_name.endswith('_sm'):
                output_dir = "umls_entities_sm"
            else:
                output_dir = f"umls_entities_{model_name}"
        else:
            output_dir = "umls_entities"

    print("="*80)
    print("UMLS Entity Merger - Separate JSON Files")
    print("="*80)
    if model_name:
        print(f"Model filter: {model_name}")
    if limit:
        print(f"Filter: First {limit} documents")
    elif pmcid:
        print(f"Filter: PMCID {pmcid}")
    else:
        print("Filter: All documents")
    print(f"Minimum occurrences: {min_occurrences}")
    print(f"Output directory: {output_dir}/")
    print("="*80 + "\n")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db = get_db_connection()

    with db.session_scope() as session:
        # Get document IDs to filter by (if limit is specified)
        doc_ids = None
        if limit:
            doc_ids = [doc.id for doc in session.query(Document.id).limit(limit).all()]
            print(f"Processing entities from {len(doc_ids)} documents\n")

        # Build query
        query = (
            session.query(
                Entity.umls_cui,
                Entity.canonical_name,
                Entity.entity_label,
                Entity.entity_text,
                Entity.start_char,
                Entity.end_char,
                Entity.umls_score,
                TextElement.id.label('text_element_id'),
                TextElement.text_content,
                TextElement.path_string,
                Document.pmcid
            )
            .join(TextElement, Entity.text_element_id == TextElement.id)
            .join(Document, TextElement.document_id == Document.id)
            .filter(Entity.umls_cui.isnot(None))  # Only entities with UMLS mapping
        )

        # Filter by model name if specified
        if model_name:
            query = query.filter(Entity.model_name == model_name)

        # Filter by document IDs if limit specified
        if doc_ids:
            query = query.filter(Document.id.in_(doc_ids))
        # Filter by PMCID if specified
        elif pmcid:
            query = query.filter(Document.pmcid == pmcid)

        # Order by CUI for consistent output
        query = query.order_by(Entity.umls_cui)

        print("Querying database (streaming results)...")

        # Group by UMLS CUI
        umls_groups = defaultdict(lambda: {
            "canonical_name": None,
            "entity_label": None,
            "total_occurrences": 0,
            "unique_entity_texts": set(),
            "sentences": [],
            "seen_sentences": set()  # Track unique sentence texts
        })

        row_count = 0
        for row in query.yield_per(1000):
            row_count += 1
            cui = row.umls_cui

            # Set metadata (first occurrence)
            if umls_groups[cui]["canonical_name"] is None:
                umls_groups[cui]["canonical_name"] = row.canonical_name
                umls_groups[cui]["entity_label"] = row.entity_label

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

        if row_count == 0:
            print("No entities with UMLS mappings found.")
            if pmcid:
                print(f"\nMake sure document {pmcid} has been processed with NER:")
                print(f"  python named-entity-recognition/ner.py {pmcid} --save")
            return {}

        print(f"Processed {row_count} entity occurrences with UMLS mappings\n")

        # Filter by minimum occurrences and convert sets to lists
        print(f"Filtering CUIs with at least {min_occurrences} occurrences...")
        filtered_data = {}

        for cui, data in umls_groups.items():
            if data["total_occurrences"] >= min_occurrences:
                # Sort sentences by text_element_id
                sorted_sentences = sorted(data["sentences"], key=lambda x: x["text_element_id"])

                filtered_data[cui] = {
                    "canonical_name": data["canonical_name"],
                    "entity_label": data["entity_label"],
                    "total_occurrences": data["total_occurrences"],
                    "unique_entity_texts": sorted(list(data["unique_entity_texts"])),
                    "sentences": sorted_sentences
                }

        print(f"Retained {len(filtered_data)} unique UMLS concepts\n")

        # Save each UMLS CUI to separate file(s)
        print(f"Writing {len(filtered_data)} JSON and TXT files to {output_dir}/...")

        for cui, data in filtered_data.items():
            # Create safe filename from CUI
            safe_filename = cui.replace('/', '_').replace('\\', '_')

            # Use canonical name in filename if available
            if data["canonical_name"]:
                canonical_safe = data["canonical_name"][:50]  # Limit length
                canonical_safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in canonical_safe)
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
                    "total_occurrences": data["total_occurrences"],
                    "unique_entity_texts": data["unique_entity_texts"],
                    "sentences": data["sentences"]
                }, f, indent=2, ensure_ascii=False)

            # Save TXT file (combined sentences with blank lines)
            txt_path = output_path / f"{base_filename}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"UMLS CUI: {cui}\n")
                f.write(f"Canonical Name: {data['canonical_name']}\n")
                f.write(f"Entity Label: {data['entity_label']}\n")
                f.write(f"Total Occurrences: {data['total_occurrences']}\n")
                f.write(f"Unique Sentences: {len(data['sentences'])}\n")
                f.write(f"Text Variants: {', '.join(data['unique_entity_texts'])}\n")
                f.write("=" * 80 + "\n\n")

                # Write all unique sentences with blank lines between them
                for sentence_data in data["sentences"]:
                    f.write(sentence_data["sentence"])
                    f.write("\n\n")

        files_saved = len(filtered_data) * 2
        print(f"✓ Saved {files_saved} files to {output_dir}/\n")

        # Print summary statistics
        print("="*80)
        print("Summary Statistics")
        print("="*80)
        print(f"Unique UMLS concepts: {len(filtered_data)}")

        if filtered_data:
            total_occurrences = sum(d["total_occurrences"] for d in filtered_data.values())
            print(f"Total entity occurrences: {total_occurrences}")

            # Top 10 most frequent concepts
            top_concepts = sorted(
                filtered_data.items(),
                key=lambda x: x[1]["total_occurrences"],
                reverse=True
            )[:10]

            print("\nTop 10 Most Frequent UMLS Concepts:")
            print("-" * 80)
            for i, (cui, data) in enumerate(top_concepts, 1):
                canonical = data["canonical_name"] or "Unknown"
                label = data["entity_label"]
                count = data["total_occurrences"]
                unique_sents = len(data["sentences"])
                variants = len(data["unique_entity_texts"])
                print(f"{i:2d}. {canonical[:40]:<40} [{label}]")
                print(f"    CUI: {cui}")
                print(f"    Total occurrences: {count} | Unique sentences: {unique_sents} | Text variants: {variants}")
                print(f"    Variants: {', '.join(data['unique_entity_texts'][:5])}")
                if variants > 5:
                    print(f"              ... and {variants - 5} more")
                print()

        print("="*80 + "\n")

        return filtered_data


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge entities by UMLS CUI and generate JSON and TXT output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents (creates both JSON and TXT files)
  python named_entity_recognition/merge_entities_by_umls.py

  # Filter by model (auto-sets output dir to umls_entities_lg)
  python named_entity_recognition/merge_entities_by_umls.py --model en_core_sci_lg

  # Process specific document
  python named_entity_recognition/merge_entities_by_umls.py --pmcid PMC1448691

  # Filter by minimum occurrences (only CUIs with 10+ mentions)
  python named_entity_recognition/merge_entities_by_umls.py --min-occurrences 10

  # Custom output directory
  python named_entity_recognition/merge_entities_by_umls.py --output-dir results/umls_concepts
        """
    )

    parser.add_argument(
        '--pmcid',
        type=str,
        help='Process only this PMCID (default: all documents)'
    )
    parser.add_argument(
        '--min-occurrences',
        type=int,
        default=1,
        help='Minimum occurrences to include a UMLS concept (default: 1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for output files (default: auto-based on model)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process only first N documents (default: all documents)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='en_core_sci_lg',
        help='Filter by model name (default: en_core_sci_lg). Output dir auto-set if not specified.'
    )

    args = parser.parse_args()

    try:
        merge_entities_by_umls(
            pmcid=args.pmcid,
            min_occurrences=args.min_occurrences,
            output_dir=args.output_dir,
            limit=args.limit,
            model_name=args.model
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("="*80)
        print("Troubleshooting:")
        print("="*80)
        print("\n  1. Make sure you have entities in the database:")
        print("     python named-entity-recognition/ner.py PMC1234567 --save")
        print("\n  2. Check database connection:")
        print("     python database/query_examples.py")
        print("\n  3. Verify UMLS data exists:")
        print("     python named-entity-recognition/query_entities.py")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
