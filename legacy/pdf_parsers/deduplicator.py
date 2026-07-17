"""
Medical Paper Deduplication

Removes running headers/footers while preserving section headings and medical terminology.

Rules:
1. Delete: Line occurs >2 times AND is in Top/Bottom 10% → Running header/footer
2. Keep as Title: Line occurs 1 time AND has Unique Font/Size → Section heading
3. Keep: Line occurs >2 times BUT is in Middle 80% → Common medical term
"""

from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MedicalPaperDeduplicator:
    """
    Removes duplicate headers/footers from medical papers while preserving content.
    """

    def __init__(
        self,
        header_footer_threshold: float = 0.10,  # Top/bottom 10%
        min_occurrences_for_duplicate: int = 2
    ):
        """
        Initialize deduplicator.

        Args:
            header_footer_threshold: Fraction of pages to consider as header/footer zone
            min_occurrences_for_duplicate: Minimum occurrences to consider as duplicate
        """
        self.header_footer_threshold = header_footer_threshold
        self.min_occurrences = min_occurrences_for_duplicate

    def deduplicate(
        self,
        text_elements: List[Dict],
        page_info: List[Dict] = None
    ) -> List[Dict]:
        """
        Remove duplicate headers/footers from text elements.

        Args:
            text_elements: List of text elements with 'text', 'page', 'y_position' etc.
            page_info: Optional page routing information

        Returns:
            Deduplicated list of text elements
        """
        if not text_elements:
            return []

        logger.info(f"Deduplicating {len(text_elements)} text elements")

        # Step 1: Analyze text occurrences and positions
        text_occurrences = defaultdict(list)
        total_pages = 0

        for i, elem in enumerate(text_elements):
            text = elem.get('text', '').strip()
            if not text:
                continue

            # Track where this text appears
            page_num = elem.get('page', 0)
            total_pages = max(total_pages, page_num + 1)

            text_occurrences[text].append({
                'index': i,
                'page': page_num,
                'element': elem
            })

        # Step 2: Classify each unique text
        to_delete = set()
        to_mark_as_title = set()

        for text, occurrences in text_occurrences.items():
            count = len(occurrences)

            # Get positions across pages
            positions = self._analyze_positions(occurrences, total_pages)

            # Rule 1: Delete if occurs >2 times AND in top/bottom 10%
            if count > self.min_occurrences and positions['is_header_footer']:
                for occ in occurrences:
                    to_delete.add(occ['index'])
                logger.debug(f"Deleting header/footer: '{text[:50]}...' ({count} occurrences)")

            # Rule 2: Keep as Title if occurs 1 time AND unique font
            elif count == 1 and positions['has_unique_font']:
                for occ in occurrences:
                    to_mark_as_title.add(occ['index'])
                logger.debug(f"Marking as title: '{text[:50]}...'")

            # Rule 3: Keep if occurs >2 times BUT in middle 80%
            # (These are kept automatically, no action needed)

        # Step 3: Build deduplicated list
        deduplicated = []
        deleted_count = 0
        title_count = 0

        for i, elem in enumerate(text_elements):
            if i in to_delete:
                deleted_count += 1
                continue

            # Mark as title if needed
            if i in to_mark_as_title:
                elem = elem.copy()
                elem['is_section_title'] = True
                title_count += 1

            deduplicated.append(elem)

        logger.info(
            f"Deduplication complete: "
            f"Removed {deleted_count} duplicates, "
            f"Marked {title_count} titles, "
            f"Kept {len(deduplicated)} elements"
        )

        return deduplicated

    def _analyze_positions(
        self,
        occurrences: List[Dict],
        total_pages: int
    ) -> Dict:
        """
        Analyze where text occurs to determine if it's a header/footer.

        Args:
            occurrences: List of occurrences with page/position info
            total_pages: Total number of pages

        Returns:
            Dict with analysis results
        """
        # Check if all occurrences are in top/bottom 10% of pages
        page_nums = [occ['page'] for occ in occurrences]

        top_threshold = total_pages * self.header_footer_threshold
        bottom_threshold = total_pages * (1 - self.header_footer_threshold)

        all_in_header_footer = all(
            (page_num < top_threshold or page_num >= bottom_threshold)
            for page_num in page_nums
        )

        # Check for unique font (if font info available)
        # For now, assume unique if occurs only once
        has_unique_font = len(occurrences) == 1

        return {
            'is_header_footer': all_in_header_footer,
            'has_unique_font': has_unique_font,
            'page_distribution': page_nums
        }


def deduplicate_text_elements(
    text_elements: List[Dict],
    page_info: List[Dict] = None
) -> List[Dict]:
    """
    Convenience function to deduplicate text elements.

    Args:
        text_elements: List of text elements
        page_info: Optional page routing information

    Returns:
        Deduplicated text elements
    """
    deduplicator = MedicalPaperDeduplicator()
    return deduplicator.deduplicate(text_elements, page_info)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example text elements
    sample_elements = [
        {'text': 'Page 1', 'page': 0},
        {'text': 'Introduction', 'page': 0},
        {'text': 'Cancer is...', 'page': 0},
        {'text': 'Page 2', 'page': 1},
        {'text': 'Methods', 'page': 1},
        {'text': 'We used...', 'page': 1},
        {'text': 'Page 3', 'page': 2},
        {'text': 'lymphoma', 'page': 0},
        {'text': 'lymphoma', 'page': 1},
        {'text': 'lymphoma', 'page': 2},
    ]

    deduplicated = deduplicate_text_elements(sample_elements)

    print("\nOriginal:", len(sample_elements))
    print("Deduplicated:", len(deduplicated))
    print("\nKept elements:")
    for elem in deduplicated:
        print(f"  - {elem['text']}")
