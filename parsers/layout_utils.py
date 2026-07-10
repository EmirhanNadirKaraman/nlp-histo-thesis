"""
Shared layout processing utilities.

The active consumer is ``pipeline/stages/pdf_text_extraction/`` (its text
assembler + artifact-filter components delegate here).  The legacy demo
notebook ``legacy/PDF_Processing_Pipeline.ipynb`` also imports from here.

Covers:
  - Element-type constants   (DOCLING_MASK_TYPES, TEXT_ELEMENT_TYPES, …)
  - Character normalisation  (fix_ligatures)
  - Bounding-box helpers     (merge_rects, bbox_overlaps, centroid_inside,
                               union_bbox, build_table_bboxes, build_picture_pages)
  - Caption matching         (nearest_caption, parse_caption_num, FIG_NUM_RE, TAB_NUM_RE)
  - Figure panel detection   (count_panels)
  - Artifact filtering       (filter_artifacts)
  - Paragraph relevance      (is_relevant_para)
  - Text extraction          (extract_text)
  - Text saving              (save_text)
"""

import re
from collections import defaultdict

# ── Element type sets ─────────────────────────────────────────────────────────
DOCLING_MASK_TYPES = {'PICTURE', 'CAPTION', 'TABLE', 'RECONSTRUCTED_TABLE'}
TEXT_ELEMENT_TYPES = {'TEXT', 'PARAGRAPH', 'SECTION_HEADER', 'TITLE', 'LIST', 'LIST_ITEM'}
SKIP_TYPES         = {'TABLE', 'FIGURE', 'PICTURE', 'CAPTION', 'RECONSTRUCTED_TABLE', 'FOOTNOTE'}
CAPTION_PATTERN    = re.compile(r'^(Table|Figure)\s+\d+', re.IGNORECASE)

# ── Character normalisation ───────────────────────────────────────────────────
# PDFs sometimes encode fi/fl/ff etc. as Unicode ligature codepoints (U+FB00–FB06).
# Docling/poppler may emit these as literal Unicode chars OR as PostScript glyph
# name strings like "/uniFB01" (surrounded by spaces: "in /uniFB02 uenza").
_LIGATURE_CHARS = {
    '\uFB00': 'ff',  '\uFB01': 'fi',  '\uFB02': 'fl',
    '\uFB03': 'ffi', '\uFB04': 'ffl', '\uFB05': 'st', '\uFB06': 'st',
}
_LIGATURE_NAME_MAP = {
    'FB00': 'ff',  'FB01': 'fi',  'FB02': 'fl',
    'FB03': 'ffi', 'FB04': 'ffl', 'FB05': 'st', 'FB06': 'st',
}
_LIGATURE_NAME_RE = re.compile(r'\s*/uni(FB0[0-6])\s*', re.IGNORECASE)

# Some PDFs extract accented characters as isolated tokens with surrounding spaces,
# e.g. "Mart í n" instead of "Martín".
_ACCENTED       = 'àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ'
_PRE_ACCENT_RE  = re.compile(rf'([A-Za-z{_ACCENTED}]) ([{_ACCENTED}])')
_POST_ACCENT_RE = re.compile(rf'([{_ACCENTED}]) ([A-Za-z{_ACCENTED}])')


def fix_ligatures(text: str) -> str:
    """Resolve ligature codepoints and remove spurious spaces around accented chars."""
    for char, repl in _LIGATURE_CHARS.items():
        text = text.replace(char, repl)
    text = _LIGATURE_NAME_RE.sub(lambda m: _LIGATURE_NAME_MAP[m.group(1).upper()], text)
    text = _PRE_ACCENT_RE.sub(r'\1\2', text)
    text = _POST_ACCENT_RE.sub(r'\1\2', text)
    return text


# ── Bounding-box helpers ──────────────────────────────────────────────────────
def bbox_overlaps(a, b) -> bool:
    """Return True if Docling bbox dicts a and b overlap."""
    return not (a['x1'] >= b['x2'] or a['x2'] <= b['x1'] or
                a['y2'] >= b['y1'] or a['y1'] <= b['y2'])


def centroid_inside(a, b) -> bool:
    """Return True if the centroid of bbox a falls inside bbox b."""
    cx = (a['x1'] + a['x2']) / 2
    cy = (a['y1'] + a['y2']) / 2
    return (b['x1'] < cx < b['x2'] and
            min(b['y1'], b['y2']) < cy < max(b['y1'], b['y2']))


def union_bbox(a, b) -> dict:
    """Return the smallest bbox that covers both a and b (Docling coords)."""
    return {
        'x1': min(a['x1'], b['x1']),
        'x2': max(a['x2'], b['x2']),
        'y1': max(max(a['y1'], a['y2']), max(b['y1'], b['y2'])),  # top
        'y2': min(min(a['y1'], a['y2']), min(b['y1'], b['y2'])),  # bottom
    }


def build_table_bboxes(elements, types=('TABLE', 'RECONSTRUCTED_TABLE')) -> dict:
    """Return a {page: [bbox, …]} index for the given element types."""
    index = defaultdict(list)
    for el in elements:
        if el.get('type') in types:
            page, bbox = el.get('page'), el.get('bbox')
            if page and bbox:
                index[page].append(bbox)
    return index


def build_picture_pages(elements) -> set:
    """Return the set of page numbers that contain at least one PICTURE element."""
    return {el['page'] for el in elements
            if el.get('type') == 'PICTURE' and el.get('page')}


def merge_rects(rects) -> list:
    """Iteratively union any-intersecting fitz.Rect objects into minimal covering rectangles.

    Merges on `Rect.intersects` (boolean overlap), not on IOU threshold. Any
    non-zero overlap absorbs the smaller rect into the union.
    """
    if not rects:
        return []
    merged = [r for r in rects if not r.is_empty]
    changed = True
    while changed:
        changed = False
        result, used = [], [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = merged[i]
            for j in range(i + 1, len(merged)):
                if not used[j] and cur.intersects(merged[j]):
                    cur = cur | merged[j]
                    used[j] = True
                    changed = True
            result.append(cur)
        merged = result
    return merged


# ── Caption-matching helpers ──────────────────────────────────────────────────
FIG_NUM_RE = re.compile(r'^figure\s+(\d+)', re.IGNORECASE)
TAB_NUM_RE = re.compile(r'^table\s+(\d+)',  re.IGNORECASE)


def parse_caption_num(text, pattern):
    """Extract the integer from a caption string, e.g. 'Figure 3' → 3."""
    m = pattern.match(text or '')
    return int(m.group(1)) if m else None


def _deduplicate_caption(text: str) -> str:
    """
    Remove exact repetitions that Docling sometimes produces when a caption is
    encoded multiple times in the PDF (visible text + alt-text concatenated).

    Handles 2×, 3×, 4×, ... repetitions with or without a space separator.
    Example: 'Figure 1. Foo. Figure 1. Foo. Figure 1. Foo.' → 'Figure 1. Foo.'
    """
    if not text:
        return text
    stripped = text.strip()
    n = len(stripped)

    for reps in range(2, 6):
        for sep in (' ', ''):
            sep_len = len(sep)
            remainder = n - (reps - 1) * sep_len
            if remainder <= 0 or remainder % reps != 0:
                continue
            unit_len = remainder // reps
            unit = stripped[:unit_len]
            if sep.join([unit] * reps) == stripped:
                return unit.rstrip()

    return stripped


def nearest_caption(el, captions):
    """
    Return the nearest CAPTION element on the same page using edge-to-edge
    vertical gap rather than centroid distance.

    Captions sit directly above or below figures, so the gap between the
    figure's bottom edge and the caption's top edge (or vice versa) is a
    much more reliable signal than centroid-to-centroid distance.
    A small horizontal offset term breaks ties when two captions are
    equidistant vertically.
    """
    page = el.get('page')
    b    = el.get('bbox')
    if not page or not b:
        return None
    same_page = [c for c in captions if c.get('page') == page]
    if not same_page:
        return None

    fig_top = max(b['y1'], b['y2'])   # Docling coords: y=0 at bottom, so top > bottom
    fig_bot = min(b['y1'], b['y2'])
    fig_cx  = (b['x1'] + b['x2']) / 2

    def _edge_distance(cap):
        cb      = cap.get('bbox', {})
        cap_top = max(cb.get('y1', 0), cb.get('y2', 0))
        cap_bot = min(cb.get('y1', 0), cb.get('y2', 0))
        cap_cx  = (cb.get('x1', 0) + cb.get('x2', 0)) / 2

        if cap_bot >= fig_top:          # caption is entirely above the figure
            vert = cap_bot - fig_top
        elif cap_top <= fig_bot:        # caption is entirely below the figure
            vert = fig_bot - cap_top
        else:                           # overlapping vertically (rare)
            vert = 0

        horiz = abs(fig_cx - cap_cx) * 0.1   # small tiebreaker
        return vert + horiz

    best = min(same_page, key=_edge_distance)
    if best and best.get('text'):
        best = dict(best)
        best['text'] = _deduplicate_caption(fix_ligatures(best['text']))
    return best


# ── Figure panel detection ────────────────────────────────────────────────────
def count_panels(page, bbox, page_height, min_gap_px=8, white_threshold=248) -> int:
    """
    Detect how many side-by-side panels a figure contains by finding white
    vertical gaps in the rendered pixels.  Returns 1 when no split is detected.

    Requires fitz (PyMuPDF) and numpy.
    """
    import numpy as np
    import fitz

    b    = bbox
    rect = fitz.Rect(
        b['x1'], page_height - max(b['y1'], b['y2']),
        b['x2'], page_height - min(b['y1'], b['y2']),
    )
    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(1, 1))
    if pix.width == 0 or pix.height == 0:
        return 1

    img       = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    col_means = (img[..., :3].mean(axis=(0, 2)) if pix.n >= 3
                 else img[..., 0].mean(axis=0).astype(float))

    is_white = col_means > white_threshold
    gaps, in_gap, gap_start = 0, False, 0
    for i, w in enumerate(is_white):
        if w and not in_gap:
            in_gap, gap_start = True, i
        elif not w and in_gap:
            in_gap = False
            if i - gap_start >= min_gap_px:
                gaps += 1
    return gaps + 1


# ── Artifact filtering ────────────────────────────────────────────────────────
MIN_ANCHOR_H = 15  # pts — minimum bbox height to qualify as a substantial block

_NON_BIO_ENT_LABELS = frozenset({
    'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
    'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
    'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL',
})


def has_bio_entity(doc) -> bool:
    """Return True if the spaCy Doc contains at least one biomedical entity."""
    return any(ent.label_ not in _NON_BIO_ENT_LABELS for ent in doc.ents)


def nlp_is_meaningful(text: str, nlp) -> bool:
    """Return True if a short fragment is real scientific content (verb, bio-entity, or ≥5 words)."""
    words = text.split()
    if len(words) < 2:
        return False
    doc = nlp(text)
    return any(tok.pos_ == 'VERB' for tok in doc) or has_bio_entity(doc) or len(words) >= 5


_SIDEBAR_METADATA = re.compile(
    r'^(Received|Revised|Accepted|Published|Citation|Academic\s+Editor'
    r'|Handling\s+Editor|Copyright|Licensee)\s*[:\s]',
    re.IGNORECASE,
)


def filter_artifacts(elements, nlp=None) -> list:
    """
    Remove common layout artifacts from Docling re-extraction output.

    1. Elements with no alphabetic characters.
    2. Short sidebar metadata lines (Received/Revised/Accepted/Published/etc.).
    3. Single-line TEXT elements outside the y-range of anchor blocks on the same page.
    4. On pages with no anchors, single-line TEXT elements that fail NER scoring.
    """
    page_bounds: dict = {}
    for el in elements:
        page = el.get('page')
        if page is None:
            continue
        b = el.get('bbox', {})
        h = abs(b.get('y1', 0) - b.get('y2', 0))
        if h >= MIN_ANCHOR_H:
            y1, y2 = b['y1'], b['y2']
            if page in page_bounds:
                top, bot = page_bounds[page]
                page_bounds[page] = (max(top, y1), min(bot, y2))
            else:
                page_bounds[page] = (y1, y2)

    kept = []
    for el in elements:
        text  = (el.get('text') or '').strip()
        etype = el.get('type', '')
        bbox  = el.get('bbox', {})
        page  = el.get('page')

        if not text:
            continue
        if not re.search(r'[a-zA-Z]', text):
            continue
        if _SIDEBAR_METADATA.match(text):
            continue

        if etype == 'TEXT':
            y1 = bbox.get('y1', 0)
            y2 = bbox.get('y2', 0)
            h  = abs(y1 - y2)
            if h < MIN_ANCHOR_H:
                if page in page_bounds:
                    top_bound, bot_bound = page_bounds[page]
                    if y1 > top_bound or y2 < bot_bound:
                        continue
                elif nlp is not None:
                    if not nlp_is_meaningful(text, nlp):
                        continue

        kept.append(el)
    return kept


# ── Paragraph relevance filtering ─────────────────────────────────────────────
_BOILERPLATE_STARTS = re.compile(
    r'^(Keywords?[:\s]|Author Contributions?:|Funding:|'
    r'Institutional Review Board|Informed Consent|'
    r'Data Availability|Conflicts?\s+of\s+Interest|'
    r'Disclaimer|Publisher.{1,3}s\s+Note|Supplementary Materials?|'
    r'Acknowledgements?:|Abbreviations?:)',
    re.IGNORECASE
)

# Author byline: comma/and-separated "First Last" pairs with optional affiliation
# markers (*, †, ‡, superscript digits). Requires ≥2 authors so single proper
# names aren't caught. e.g. "Anna Colagrande, Giuseppe Ingravallo * and Gerardo Cazzato *"
_AUTHOR_BYLINE_RE = re.compile(
    r'^[A-Z][a-z\u00C0-\u017E]+(?:-[A-Z][a-z\u00C0-\u017E]+)?'
    r'\s+[A-Z][a-zA-Z\u00C0-\u017E-]+[\s\d*†‡]*'
    r'(?:(?:[,;]\s*|\s+and\s+)'
    r'[A-Z][a-z\u00C0-\u017E]+(?:-[A-Z][a-z\u00C0-\u017E]+)?'
    r'\s+[A-Z][a-zA-Z\u00C0-\u017E-]+[\s\d*†‡]*'
    r')+'
    r'$'
)


def is_relevant_para(text: str, nlp=None) -> bool:
    """
    Two-stage relevance filter for extracted paragraphs.

    Stage 1 — fast heuristic rejection:
      • Boilerplate admin prefixes (Keywords, Funding, Author Contributions, …)
      • Numbered reference entries
      • NLM-style author-first references with [CrossRef]/[PubMed] or year + journal
      • Paragraphs shorter than 4 words

    Stage 2 — NER/POS scoring (requires nlp model):
      • ≥ 20 words → auto-pass
      • 4–19 words → must have a biomedical entity OR a verb
    """
    from parsers.text_processing import is_reference_entry
    t = text.strip()
    if not t:
        return False

    if _BOILERPLATE_STARTS.match(t):
        return False
    if _AUTHOR_BYLINE_RE.match(t):
        return False
    if is_reference_entry(t):
        return False
    if re.match(r'^[A-Z][\w\u00C0-\u017E-]+,\s+[A-Z][\w.]*', t):
        if re.search(r'\[(CrossRef|PubMed)\]', t):
            return False
        if (re.search(r'\b(19|20)\d{2}\b', t) and
                re.search(r'\b(doi|DOI|J\.|Am\.|Int\.|Eur\.|Press:|Eds?\.)\b', t)):
            return False

    words = t.split()
    if len(words) < 4:
        return False
    if nlp is None:
        return True
    if len(words) >= 20:
        return True
    doc = nlp(t)
    return any(tok.pos_ == 'VERB' for tok in doc) or has_bio_entity(doc)


# ── Text extraction ───────────────────────────────────────────────────────────
def _reference_list_skip_set(elements) -> set:
    """
    Return a set of indices for LIST/LIST_ITEM elements that belong to a
    bibliography group.

    A consecutive run of LIST/LIST_ITEM elements is considered a bibliography
    group — and all its indices are returned — when at least 2 of its items
    match ``is_reference_entry``.  This catches reference lists that Docling
    extracts as list elements rather than plain TEXT paragraphs.
    """
    from parsers.text_processing import is_reference_entry

    skip: set = set()
    i, n = 0, len(elements)
    while i < n:
        if elements[i].get('type', '') in ('LIST', 'LIST_ITEM'):
            run_start = i
            while i < n and elements[i].get('type', '') in ('LIST', 'LIST_ITEM'):
                i += 1
            run = range(run_start, i)
            ref_count = sum(
                1 for idx in run
                if is_reference_entry(
                    fix_ligatures((elements[idx].get('text') or '').strip())
                )
            )
            if ref_count >= 2:
                skip.update(run)
        else:
            i += 1
    return skip


def extract_text(
    elements,
    nlp=None,
    table_bboxes=None,
    use_centroid=False,
    pre_filter=True,
    with_sources=False,
):
    """
    Walk Docling elements in document order and return
    (rows, n_skipped) where rows is a list of
    (path_string, path_list, depth, text) tuples, or
    (path_string, path_list, depth, text, source_chunks) 5-tuples when
    ``with_sources=True``.

    Args:
        elements:      List of element dicts (type, page, level, bbox, text).
        nlp:           Optional scispaCy model for relevance filtering
                       (only used when pre_filter=True).
        table_bboxes:  Optional {page: [bbox, …]} — TEXT elements overlapping
                       these bboxes are skipped (useful for unmasked extraction).
        use_centroid:  If True, use centroid_inside instead of bbox_overlaps for
                       the table overlap check (less aggressive).
        pre_filter:    If True (default), apply is_relevant_para before stitching
                       (original behaviour).  If False, all non-empty text reaches
                       the stitcher; nothing is dropped before merge.
        with_sources:  If True, rows are 5-tuples that include source_chunks — the
                       list of pre-stitch texts merged into each output paragraph.
    """
    from parsers.text_processing import remove_citations, ContextAwareStitcher

    overlap_fn    = centroid_inside if use_centroid else bbox_overlaps
    picture_pages = build_picture_pages(elements)
    ref_list_skip = _reference_list_skip_set(elements)

    hierarchy  = {}
    # Per-path set of already-seen texts used to drop Docling duplicates and
    # same-named-section collisions before the stitcher ever sees them.
    path_seen: dict = defaultdict(set)
    # Arrival-order log of (path_str, text). Output rows are emitted by walking
    # this list and stitching only contiguous same-path runs — re-grouping by
    # path globally would reorder a section's second half to follow its first
    # (B-040: sub-section returns to parent get teleported into one block).
    ordered: list = []
    n_skipped  = 0
    n_deduped  = 0

    for idx, el in enumerate(elements):
        if idx in ref_list_skip:
            n_skipped += 1
            continue
        etype = el.get('type', '')
        text  = fix_ligatures((el.get('text') or '').strip())
        if not text:
            continue

        if etype == 'SECTION_HEADER':
            level = el.get('level', 0)
            hierarchy[level] = text
            hierarchy = {k: v for k, v in hierarchy.items() if k <= level}
        elif etype not in SKIP_TYPES:
            if len(text) == 1 and el.get('page') in picture_pages:
                n_skipped += 1
                continue
            if table_bboxes:
                page = el.get('page')
                bbox = el.get('bbox')
                if page and bbox and any(
                    overlap_fn(bbox, tb) for tb in table_bboxes.get(page, [])
                ):
                    n_skipped += 1
                    continue
            path_parts = [hierarchy[k] for k in sorted(hierarchy) if hierarchy.get(k)]
            path_str   = ' > '.join(path_parts) or 'Root'
            if text in path_seen[path_str]:
                n_deduped += 1
                continue
            path_seen[path_str].add(text)
            ordered.append((path_str, text))

    stitcher = ContextAwareStitcher()
    rows = []
    i = 0
    while i < len(ordered):
        path_str = ordered[i][0]
        j = i
        texts: list = []
        while j < len(ordered) and ordered[j][0] == path_str:
            texts.append(ordered[j][1])
            j += 1
        path_list = [] if path_str == 'Root' else [p.strip() for p in path_str.split(' > ')]
        depth     = len(path_list)
        filtered  = [t for t in texts if is_relevant_para(t, nlp)] if pre_filter else texts
        inputs    = [remove_citations(t) for t in filtered]
        if with_sources:
            for para, sources in stitcher.reconstruct_with_sources(inputs):
                if para.strip():
                    rows.append((path_str, path_list, depth, para, sources))
        else:
            for para in stitcher.reconstruct_paragraphs(inputs):
                if para.strip():
                    rows.append((path_str, path_list, depth, para))
        i = j
    return rows, n_skipped + n_deduped


# ── Text saving ───────────────────────────────────────────────────────────────
def save_text(rows, out_path, pmcid: str, label: str = '', ordered: bool = True):
    """
    Write extracted text rows to a .txt file.

    Args:
        rows:     List of (path_str, path_list, depth, text) as returned by extract_text.
        out_path: Destination file path.
        pmcid:    Document ID shown in the header.
        label:    Short name for the extraction variant (e.g. 'combined | masked').
        ordered:  True → document order (default); False → alphabetical by section path.
    """
    order_label = 'document order' if ordered else 'alphabetical order'
    if not ordered:
        rows = sorted(rows, key=lambda r: r[0])

    with open(out_path, 'w') as f:
        f.write(f'Document: {pmcid}  |  {label}  |  {order_label}\n{"="*80}\n\n')
        current_path = None
        for path_str, _, _, text in rows:
            if path_str != current_path:
                f.write(f'[{path_str}]\n{"-"*80}\n')
                current_path = path_str
            f.write(f'{text}\n\n')
