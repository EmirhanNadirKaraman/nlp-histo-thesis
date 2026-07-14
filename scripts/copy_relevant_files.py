import os
import shutil
import argparse
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nlp_histo.database import get_db_connection, Entity
from sqlalchemy.dialects.postgresql import array 
from named_entity_recognition.enums import UMLS_DISEASE_TYPES

def copy_disease_files(source_dir="umls_entities/", dest_dir="relevant_json/"):
    """
    Copy disease-related files from source to destination directory.

    Args:
        source_dir: Directory containing all UMLS entity files
        dest_dir: Directory where disease-related files will be copied
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    db = get_db_connection()
    RELEVANT_TUIS = list(UMLS_DISEASE_TYPES.keys())

    with db.session_scope() as session:
        # Query the DB for unique CUIs that have the relevant semantic types
        # query_filters = [Entity.semantic_types.contains(tui) for tui in RELEVANT_TUIS]
        disease_entities = session.query(Entity).filter(
            Entity.semantic_types.op('&&')(array(RELEVANT_TUIS)) # '&&' checks if the arrays have any common elements
        ).distinct().all()

        disease_cuis = {entity.umls_cui for entity in disease_entities if entity.umls_cui}
        print(f"Found {len(disease_cuis)} unique disease CUIs in the database.")

        # Iterate through the files and copy if the CUI matches
        copied_count = 0
        all_files = os.listdir(source_dir)

        for filename in all_files:
            if filename.endswith(".json"):
                # Extract CUI from filename (e.g., 'C0000737_Abdominal_Pain.json')
                cui = filename.split('_')[0]

                if cui in disease_cuis:
                    src_path = os.path.join(source_dir, filename)
                    dst_path = os.path.join(dest_dir, filename)

                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {filename}: {e}")

        print("\n--- Process Complete ---")
        print(f"Files scanned: {len(all_files)}")
        print(f"Files copied to '{dest_dir}': {copied_count}")

        return copied_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy disease-related UMLS entity files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default directories
  python scripts/copy_relevant_files.py

  # Custom directories
  python scripts/copy_relevant_files.py --source test_umls/ --dest test_relevant/

  # Use with test results
  python scripts/copy_relevant_files.py --source test_results_50_docs/umls_entities --dest test_results_50_docs/relevant_texts
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        default='umls_entities/',
        help='Source directory containing UMLS entity files (default: umls_entities/)'
    )
    parser.add_argument(
        '--dest',
        type=str,
        default='relevant_texts/',
        help='Destination directory for disease-related files (default: relevant_texts/)'
    )

    args = parser.parse_args()
    copy_disease_files(source_dir=args.source, dest_dir=args.dest)