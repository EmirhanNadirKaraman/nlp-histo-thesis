import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func
from database import get_db_connection, Document, TextElement
from named_entity_recognition.ner import run_ner_on_db, load_ner_model, load_linker_model

# Cache file path
CACHE_FILE = Path(__file__).parent / "entity_linking_cache.json"


def load_entity_cache() -> dict:
    """Load entity linking cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"✓ Loaded {len(cache)} cached entity mappings from {CACHE_FILE.name}")
            return cache
        except Exception as e:
            print(f"⚠ Could not load cache: {e}")
            return {}
    return {}


def save_entity_cache(cache: dict):
    """Save entity linking cache to disk."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"✓ Saved {len(cache)} entity mappings to {CACHE_FILE.name}")
    except Exception as e:
        print(f"⚠ Could not save cache: {e}")


def process_document_worker(args):
    """
    Worker function for processing a single document's NER in a thread.

    Args:
        args: Tuple of (pmcid, index, total_count, nlp, linker_nlp, entity_cache, cache_lock, min_chars, force, stats, stats_lock)

    Returns:
        Dict with processing result and metadata
    """
    pmcid, index, total_count, nlp, linker_nlp, entity_cache, cache_lock, min_chars, force, stats, stats_lock = args
    thread_name = threading.current_thread().name

    print(f"[{index}/{total_count}] [{thread_name}] Processing {pmcid}...")

    try:
        # Create a thread-local copy of cache for reading (avoid lock contention)
        with cache_lock:
            local_cache = dict(entity_cache)

        # Call the optimized run_ner_on_db function
        results = run_ner_on_db(
            pmcid,
            min_chars,
            save_to_db=True,
            force=force,
            nlp=nlp,
            linker_nlp=linker_nlp,
            entity_cache=local_cache
        )

        # Update shared cache with any new mappings
        with cache_lock:
            new_entries = {k: v for k, v in local_cache.items() if k not in entity_cache}
            if new_entries:
                entity_cache.update(new_entries)

        if results:
            with stats_lock:
                stats['processed'] += 1
            return {
                'status': 'processed',
                'pmcid': pmcid,
                'entities': len(results),
                'new_cache_entries': len(new_entries) if 'new_entries' in dir() else 0
            }
        else:
            with stats_lock:
                stats['skipped'] += 1
            return {
                'status': 'skipped',
                'pmcid': pmcid,
                'reason': 'no_entities_or_already_exists'
            }

    except Exception as e:
        with stats_lock:
            stats['errors'] += 1
        print(f"[{thread_name}] ✗ Error processing {pmcid}: {e}")
        return {
            'status': 'error',
            'pmcid': pmcid,
            'error': str(e)
        }


def batch_process_all_documents(min_chars: int = 50, force: bool = False, limit: int = None, order_by: str = None, save_interval: int = 50):
    db = get_db_connection()

    # 1. LOAD PERSISTENT CACHE
    print("="*80)
    print("Loading Entity Linking Cache...")
    print("="*80)
    entity_cache = load_entity_cache()
    cache_lock = threading.Lock()

    # 2. PRE-LOAD MODELS
    print("\n" + "="*80)
    print("Loading NLP Models...")
    print("="*80)

    print("Loading fast NER model (no linker)...")
    nlp = load_ner_model()
    print("✓ Fast NER model loaded")

    print("Loading UMLS linker model...")
    linker_nlp = load_linker_model()
    print("✓ UMLS linker model loaded")

    with db.session_scope() as session:
        if order_by in ('sentences', 'sentences-asc', 'sentences-desc'):
            # Order by sentence count (number of text elements per document)
            query = session.query(
                Document.pmcid,
                func.count(TextElement.id).label('sentence_count')
            ).join(TextElement).group_by(Document.pmcid)

            if order_by == 'sentences-desc':
                query = query.order_by(func.count(TextElement.id).desc())
            else:
                query = query.order_by(func.count(TextElement.id).asc())

            documents = query.all()
            pmcids = [doc.pmcid for doc in documents]
            print(f"Ordered by sentence count ({'descending' if order_by == 'sentences-desc' else 'ascending'})")
        else:
            documents = session.query(Document.pmcid).all()
            pmcids = [doc.pmcid for doc in documents]

    if not pmcids:
        print("No documents found in database.")
        return

    # Apply limit if specified
    if limit:
        pmcids = pmcids[:limit]
        print(f"Limiting to first {limit} documents")

    print("\n" + "="*80)
    print(f"Batch NER Processing Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Documents to process: {len(pmcids)}")
    print(f"Cached entity texts: {len(entity_cache)}")
    print(f"Cache save interval: {'disabled' if save_interval == 0 else f'every {save_interval} new entities'}")

    # Thread-safe stats tracking
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    stats_lock = threading.Lock()

    # Auto-detect worker count: CPU cores / 2, minimum 1
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, cpu_count // 2)

    print(f"\nUsing {max_workers} worker threads (CPU count: {cpu_count})")
    print("="*80)

    # Prepare work items
    work_items = [
        (pmcid, idx + 1, len(pmcids), nlp, linker_nlp, entity_cache, cache_lock, min_chars, force, stats, stats_lock)
        for idx, pmcid in enumerate(pmcids)
    ]

    # Process with ThreadPoolExecutor
    results = []
    initial_cache_size = len(entity_cache)
    interrupted = False
    last_save_size = initial_cache_size

    try:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="NERWorker"
        ) as executor:

            # Submit all tasks
            future_to_pmcid = {
                executor.submit(process_document_worker, item): item[0]  # item[0] is pmcid
                for item in work_items
            }

            # Collect results as they complete
            with tqdm(total=len(future_to_pmcid), unit="doc", desc="NER") as pbar:
                for future in as_completed(future_to_pmcid):
                    pmcid = future_to_pmcid[future]

                    try:
                        result = future.result()
                        results.append(result)

                        # Periodic cache save
                        if save_interval > 0:
                            new_since_save = len(entity_cache) - last_save_size
                            if new_since_save >= save_interval:
                                with cache_lock:
                                    save_entity_cache(entity_cache)
                                    last_save_size = len(entity_cache)

                    except Exception as e:
                        tqdm.write(f"✗ Future exception for {pmcid}: {e}")
                        results.append({'status': 'error', 'pmcid': pmcid, 'error': str(e)})

                    ok = sum(1 for r in results if r.get('status') == 'processed')
                    skip = sum(1 for r in results if r.get('status') == 'skipped')
                    fail = sum(1 for r in results if r.get('status') == 'error')
                    pbar.update(1)
                    pbar.set_postfix(ok=ok, skip=skip, fail=fail, cache=len(entity_cache))

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⚠️  Interrupt received!")
        print("Cancelling pending tasks and saving cache...")

        # Cancel pending futures (won't stop running ones, but prevents new ones)
        for future in future_to_pmcid:
            future.cancel()

        print("Waiting for running tasks to finish (Ctrl+C again to force quit)...")

    # 3. SAVE CACHE TO DISK
    print("\n" + "="*80)
    print("Saving Entity Linking Cache...")
    print("="*80)
    new_cache_entries = len(entity_cache) - initial_cache_size
    print(f"New cache entries this run: {new_cache_entries}")
    save_entity_cache(entity_cache)

    # Print detailed result breakdown
    if results:
        print("\n" + "="*80)
        print("Processing Results by Status")
        print("="*80)

        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        for status in sorted(status_counts.keys()):
            count = status_counts[status]
            print(f"  {status:15s}: {count:4d}")

    print("\n" + "="*80)
    if interrupted:
        print("⚠️  RUN INTERRUPTED - partial results below")
    print(f"Summary: {stats['processed']} Processed | {stats['skipped']} Skipped | {stats['errors']} Errors")
    print(f"Cache: {initial_cache_size} -> {len(entity_cache)} entries (+{new_cache_entries} new)")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process NER for all documents')
    parser.add_argument('--min-chars', type=int, default=50, help='Minimum characters for entity text (default: 50)')
    parser.add_argument('--force', '-f', action='store_true', help='Force reprocess documents that already have entities')
    parser.add_argument('--limit', type=int, help='Limit processing to first N documents')
    parser.add_argument('--order-by', type=str, choices=['sentences', 'sentences-asc', 'sentences-desc'],
                        help='Order documents by sentence count: sentences/sentences-asc (smallest first), sentences-desc (largest first)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save cache every N new entity mappings (default: 50, set to 0 to disable)')
    args = parser.parse_args()

    batch_process_all_documents(
        min_chars=args.min_chars,
        force=args.force,
        limit=args.limit,
        order_by=args.order_by,
        save_interval=args.save_interval
    )
