#!/usr/bin/env python3
"""
Visualize full Docling layout extraction results.
Shows all detected elements (text, titles, lists, tables, figures) with color-coded boxes.

Usage:
    python visualize_docling_full.py
    python visualize_docling_full.py --types TABLE FIGURE PICTURE
    python visualize_docling_full.py --json out/docling_full/custom_layout.json
"""

import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

# Comprehensive color scheme for all Docling element types
ELEMENT_COLORS = {
    'TABLE': (0, 0.8, 0),           # Green
    'FIGURE': (1, 0, 0.5),          # Pink/Magenta
    'PICTURE': (0.5, 0, 1),         # Purple
    'PARAGRAPH': (0.7, 0.7, 0.7),   # Light Gray
    'TITLE': (1, 0.5, 0),           # Orange
    'SECTION_HEADER': (1, 0.5, 0),  # Orange
    'LIST': (0, 0.5, 1),            # Blue
    'LIST_ITEM': (0.3, 0.7, 1),     # Light Blue
    'CAPTION': (1, 0.8, 0),         # Yellow
    'PAGE_HEADER': (0.5, 0.5, 0.5), # Gray
    'PAGE_FOOTER': (0.5, 0.5, 0.5), # Gray
    'FOOTNOTE': (0.8, 0.4, 0.2),    # Brown
    'FORMULA': (0.9, 0.1, 0.5),     # Dark Pink
    'CODE': (0.2, 0.6, 0.4),        # Teal
    'RECONSTRUCTED_TABLE': (0, 1, 0.5),  # Bright Green (for reconstructed tables)
    'UNKNOWN': (0.5, 0.5, 0.5),     # Gray
}


def load_full_layout_json(json_path: Path):
    """Load Docling full layout JSON."""
    if not json_path.exists():
        print(f"❌ JSON not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        elements = data.get('elements', [])
        metadata = data.get('metadata', {})

        print(f"Loaded: {metadata.get('total_elements_found', len(elements))} elements")
        return data
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return None


def get_rect(bbox, page_height):
    """Convert bbox to PyMuPDF rect."""
    try:
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']

        # Determine top/bottom based on PDF standard (bottom-up)
        pdf_y_top = max(y1, y2)
        pdf_y_bottom = min(y1, y2)

        # Convert to PyMuPDF (top-down)
        return fitz.Rect(x1, page_height - pdf_y_top, x2, page_height - pdf_y_bottom)
    except (KeyError, TypeError):
        return None


def reconstruct_tables_from_lists(json_path, threshold_multiplier=1.2):
    """
    Groups elements into a table based on adaptive vertical proximity.
    Uses actual spacing between first two table rows to set threshold.

    Args:
        json_path: Path to Docling JSON file
        threshold_multiplier: Multiplier for determining max allowed gap between rows
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    elements = data.get("elements", [])
    reconstructed = []

    i = 0
    while i < len(elements):
        el = elements[i]

        # Identify the start of a table via its caption
        if el.get("type") == "CAPTION" and "table" in (el.get("text") or "").lower():
            reconstructed.append(el)

            table_group = {
                "type": "RECONSTRUCTED_TABLE",
                "caption": el.get("text"),
                "page": el.get("page"),
                "sub_elements": []
            }

            # Initialize with a safe default, then refine based on actual spacing
            max_allowed_gap = 20
            last_y2 = el["bbox"]["y2"]
            i += 1

            while i < len(elements):
                next_el = elements[i]
                if next_el.get("type") not in ["TEXT", "LIST_ITEM"]:
                    break

                current_y1 = next_el["bbox"]["y1"]
                vertical_gap = abs(current_y1 - last_y2)

                # REFINEMENT: After capturing the first item, use the gap between
                # item 1 and item 2 as the true baseline gutter spacing
                if len(table_group["sub_elements"]) == 1:
                    first_item_y2 = table_group["sub_elements"][0]["bbox"]["y2"]
                    true_gutter = abs(current_y1 - first_item_y2)
                    max_allowed_gap = true_gutter * threshold_multiplier

                if vertical_gap < max_allowed_gap:
                    table_group["sub_elements"].append(next_el)
                    last_y2 = next_el["bbox"]["y2"]
                    i += 1
                else:
                    break

            # Calculate a bounding box that covers all grouped items
            if table_group["sub_elements"]:
                all_bboxes = [e["bbox"] for e in table_group["sub_elements"]]
                # Preserve existing min/max logic
                table_group["bbox"] = {
                    "x1": min(b["x1"] for b in all_bboxes),
                    "y1": max(b["y1"] for b in all_bboxes),
                    "x2": max(b["x2"] for b in all_bboxes),
                    "y2": min(b["y2"] for b in all_bboxes)
                }
                reconstructed.append(table_group)
        else:
            # Not a table caption, keep the element as is
            reconstructed.append(el)
            i += 1

    return reconstructed

def _visualize_elements(pdf_file: Path, elements: list, metadata: dict, output_path: str,
                        element_types: list = None):
    """
    Internal function to visualize elements on PDF.

    Args:
        pdf_file: Path object to PDF
        elements: List of elements to visualize
        metadata: Metadata dict
        element_types: Optional filter for element types
        output_suffix: Suffix to add to output filename
        reconstruct: Whether this is the reconstructed version
    """
    # Filter elements by type if specified
    if element_types:
        element_types_upper = [t.upper() for t in element_types]
        elements = [e for e in elements if e.get('type', '').upper() in element_types_upper]
        print(f"Filtering to types: {element_types_upper}")
        print(f"Matched elements: {len(elements)}")

    # Open PDF
    doc = fitz.open(str(pdf_file))

    print(f"\nPDF: {pdf_file.name}")
    print(f"Pages: {len(doc)}\n")

    # Group elements by type for summary
    type_counts = {}
    total_drawn = 0

    # Draw all elements
    for element in elements:
        page_no = element.get('page')
        elem_type = element.get('type', 'UNKNOWN')
        bbox = element.get('bbox')

        if not bbox or not page_no:
            continue

        if page_no < 1 or page_no > len(doc):
            continue

        page = doc[page_no - 1]  # Convert to 0-indexed
        page_height = page.rect.height

        rect = get_rect(bbox, page_height)
        if not rect:
            continue

        # Get color for this element type
        color = ELEMENT_COLORS.get(elem_type, ELEMENT_COLORS['UNKNOWN'])

        # Draw box with dashed style
        page.draw_rect(rect, color=color, width=1, dashes="[2] 0")

        # Add small label (abbreviated type)
        label = elem_type[:3].upper()
        page.insert_text(
            (rect.x0 + 1, rect.y0 + 8),
            label,
            fontsize=6,
            color=color
        )

        # Update counts
        type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        total_drawn += 1

    # Add legend on first page
    if len(doc) > 0 and type_counts:
        first_page = doc[0]
        lx, ly = 20, 20

        # Calculate legend size based on types found
        legend_height = 35 + len(type_counts) * 10

        # Background
        first_page.draw_rect(
            fitz.Rect(lx-5, ly-5, lx+150, ly+legend_height),
            color=(0,0,0),
            fill=(1,1,1),
            width=0.5
        )

        # Title
        title = "Docling Elements:"
        first_page.insert_text((lx, ly+10), title,
                              fontsize=9, color=(0,0,0))

        # Legend items (sorted by count, descending)
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (elem_type, count) in enumerate(sorted_types):
            y_pos = ly + 25 + (i * 10)
            color = ELEMENT_COLORS.get(elem_type, ELEMENT_COLORS['UNKNOWN'])

            # Draw sample line
            first_page.draw_line((lx, y_pos-3), (lx+12, y_pos-3),
                                color=color, width=1, dashes="[2] 0")

            # Type and count
            label = f"{elem_type} ({count})"
            first_page.insert_text((lx+15, y_pos), label,
                                  fontsize=7, color=(0,0,0))

    # Save
    output_dir = Path("out/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / output_path
    print(output_file)

    doc.save(str(output_file))
    doc.close()

    print(f"\nTotal elements drawn: {total_drawn}\n")
    print("Element type breakdown:")
    for elem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {elem_type:20s}: {count:4d}")

    print(f"\n✅ Visualization saved to: {output_file}\n")
    return output_file


def visualize_full_layout(pdf_path: str, json_path: str, output_path: str, 
                          element_types: list = None):
    """
    Create visualization of all Docling detected elements.

    Args:
        pdf_path: Path to original PDF
        json_path: Path to Docling full layout JSON
        output_path: Where to save visualization
        element_types: List of element types to visualize (None = all)
        save_both: If True, saves both with and without reconstructed tables
    """
    pdf_file = Path(pdf_path)
    json_file = Path(json_path)

    if not pdf_file.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    # Load layout data
    layout_data = load_full_layout_json(json_file)
    if not layout_data:
        return

    elements_original = layout_data.get('elements', [])
    metadata = layout_data.get('metadata', {})
    _visualize_elements(pdf_file, elements_original, metadata, output_path, element_types)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Docling full layout extraction"
    )
    parser.add_argument(
        "pdf_path",
        nargs='?',
        default="files/organized_pdfs/PMC1448691_his_2369.pdf",
        help="Path to PDF file (default: PMC1448691_his_2369.pdf)"
    )
    parser.add_argument(
        "--json",
        help="Path to Docling full layout JSON (default: auto-detect)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output PDF path"
    )
    parser.add_argument(
        "--types", "-t",
        nargs='+',
        help="Filter to specific element types (e.g., --types TABLE FIGURE PICTURE)"
    )
    parser.add_argument(
        "--save-both",
        action='store_true',
        default=True,
        help="Save both original and reconstructed versions (default: True)"
    )
    parser.add_argument(
        "--no-save-both",
        dest='save_both',
        action='store_false',
        help="Only save reconstructed version"
    )

    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)

    # Auto-detect JSON if not specified
    if args.json:
        json_path = args.json
    else:
        json_path = Path("out/docling_full") / f"{pdf_file.stem}_full_layout.json"
        if not json_path.exists():
            print(f"❌ Auto-detect failed. JSON not found at: {json_path}")
            print("   Run: python scripts/extract_figures_docling.py")
            print("   Or specify JSON with: --json /path/to/file.json")
            sys.exit(1)
        print(f"Auto-detected JSON: {json_path}")

    visualize_full_layout(args.pdf_path, json_path, args.output, args.types, args.save_both)


if __name__ == "__main__":
    main()
