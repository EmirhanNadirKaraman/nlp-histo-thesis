#!/usr/bin/env python3
"""
DEPRECATED — early experimental pipeline, superseded by
``pipeline/stages/pdf_text_extraction/runner.py`` (PipelineRunner). Kept for
reference only: not on the production path and not imported by any live module.

Combined Masked Pipeline (Docling + TATR)

For each PDF:
  1. Extract layout with Docling (original PDF)
  2. Detect tables with TATR
  3. Merge Docling + TATR bounding boxes and mask the PDF
  4. Re-extract layout from masked PDF with Docling
  5. Stitch text into hierarchical sections (document order)
  6. Save ordered .txt and ingest to database

Usage:
    python scripts/combined_pipeline.py --pdf files/organized_pdfs/PMC123.pdf
    python scripts/combined_pipeline.py --pdf-dir files/organized_pdfs --workers 4
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from parsers.layout_utils import (  # noqa: E402
    DOCLING_MASK_TYPES, TEXT_ELEMENT_TYPES, CAPTION_PATTERN,
    fix_ligatures, merge_rects,
    FIG_NUM_RE, TAB_NUM_RE, parse_caption_num, union_bbox, nearest_caption,
    MIN_ANCHOR_H, nlp_is_meaningful, filter_artifacts,
    is_relevant_para, extract_text, save_text,
)

TATR_DPI       = 150
TATR_THRESHOLD = 0.99
SCALE          = 72 / TATR_DPI   # pixels -> PDF points


# ── Core processor ─────────────────────────────────────────────────────────────
class CombinedPipelineProcessor:
    def __init__(
        self,
        masked_pdf_dir : str = 'out/masked_pdfs',
        docling_out_dir: str = 'out/docling_full',
        text_dir       : str = 'out/text',
        figures_dir    : str = 'out/figures',
        tables_dir     : str = 'out/tables',
        vis_dir        : str = 'out/visualization',
        blacklist_file : str = 'out/failed_pdfs_blacklist.json',
        tatr_threshold : float = TATR_THRESHOLD,
        db_ingest      : bool = True,
        visualize      : bool = True,   # set False to skip visualization PDFs
    ):
        self.masked_pdf_dir  = Path(masked_pdf_dir)
        self.docling_out_dir = Path(docling_out_dir)
        self.text_dir        = Path(text_dir)
        self.figures_dir     = Path(figures_dir)
        self.tables_dir      = Path(tables_dir)
        self.vis_dir         = Path(vis_dir)
        self.blacklist_file  = Path(blacklist_file)
        self.tatr_threshold  = tatr_threshold
        self.db_ingest       = db_ingest
        self.visualize       = visualize

        for d in [self.masked_pdf_dir, self.docling_out_dir,
                  self.text_dir, self.figures_dir, self.tables_dir, self.vis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.blacklist_file.parent.mkdir(parents=True, exist_ok=True)

        self._converter   = None   # lazy Docling
        self._tatr_proc   = None   # lazy TATR processor
        self._tatr_model  = None   # lazy TATR model
        self._db          = None   # lazy DB connection
        self._nlp         = None   # lazy scispaCy model

        self.blacklist = self._load_blacklist()

    # ── Lazy initialisers ──────────────────────────────────────────────────────
    @property
    def converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            opts = PdfPipelineOptions()
            opts.do_table_structure = False
            opts.do_ocr = True
            opts.images_scale = 2.0
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        return self._converter

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            logger.info('Loading scispaCy model (en_core_sci_sm)...')
            self._nlp = spacy.load('en_core_sci_sm')
        return self._nlp

    @property
    def tatr(self):
        if self._tatr_proc is None:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            logger.info('Loading TATR model...')
            self._tatr_proc  = AutoImageProcessor.from_pretrained(
                'microsoft/table-transformer-detection')
            self._tatr_model = AutoModelForObjectDetection.from_pretrained(
                'microsoft/table-transformer-detection')
            self._tatr_model.eval()
            logger.info('TATR model loaded.')
        return self._tatr_proc, self._tatr_model

    @property
    def db(self):
        if self._db is None and self.db_ingest:
            from database import get_db_connection
            self._db = get_db_connection()
        return self._db

    # ── Blacklist ──────────────────────────────────────────────────────────────
    def _load_blacklist(self):
        if self.blacklist_file.exists():
            return set(json.loads(self.blacklist_file.read_text()).get('blacklisted', []))
        return set()

    def _add_to_blacklist(self, pmcid, reason):
        self.blacklist.add(pmcid)
        data = {'blacklisted': list(self.blacklist), 'last_updated': datetime.now().isoformat()}
        self.blacklist_file.write_text(json.dumps(data, indent=2))
        logger.warning(f'Blacklisted {pmcid}: {reason}')

    # ── Step 1: Docling extraction on original PDF ─────────────────────────────
    def _docling_extract(self, pdf_path: Path):
        result = self.converter.convert(str(pdf_path))
        doc    = result.document
        elements = []
        for element, level in doc.iterate_items():
            label = str(getattr(element, 'label', 'UNKNOWN')).split('.')[-1].upper()
            if not (hasattr(element, 'prov') and element.prov):
                continue
            prov = element.prov[0]
            bbox = prov.bbox
            text = ''
            if hasattr(element, 'text'):
                text = element.text or ''
            elif hasattr(element, 'caption') and element.caption:
                text = element.caption.text or ''
            elements.append({
                'type' : label,
                'page' : prov.page_no,
                'level': level,
                'bbox' : {'x1': bbox.l, 'y1': bbox.t, 'x2': bbox.r, 'y2': bbox.b},
                'text' : text.strip() or None,
            })
        # Reclassify text elements that look like captions
        for el in elements:
            if el['type'] == 'TEXT' and CAPTION_PATTERN.match(el.get('text') or ''):
                el['type'] = 'CAPTION'
        page_dims = {no: {'width': p.size.width, 'height': p.size.height}
                     for no, p in doc.pages.items()}
        return elements, page_dims

    # ── Step 2: TATR table detection ───────────────────────────────────────────
    def _tatr_detect(self, pdf_path: Path):
        import fitz
        import torch
        from PIL import Image as PILImage
        tatr_proc, tatr_model = self.tatr

        doc_fitz    = fitz.open(str(pdf_path))
        detections  = []
        for page_num in range(len(doc_fitz)):
            page = doc_fitz[page_num]
            mat  = fitz.Matrix(TATR_DPI / 72, TATR_DPI / 72)
            pix  = page.get_pixmap(matrix=mat)
            img  = PILImage.frombytes('RGB', [pix.width, pix.height], pix.samples)

            inputs = tatr_proc(images=img, return_tensors='pt')
            with torch.no_grad():
                outputs = tatr_model(**inputs)
            results = tatr_proc.post_process_object_detection(
                outputs,
                threshold=self.tatr_threshold,
                target_sizes=[(img.height, img.width)]
            )[0]

            for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    'page' : page_num + 1,
                    'rect' : fitz.Rect(x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE),
                    'score': round(score.item(), 3),
                    'label': tatr_model.config.id2label[label.item()],
                })
        doc_fitz.close()
        return detections

    # ── Step 2b: Detect header/footer/sidebar regions from original extraction ──
    def _detect_header_footer_elements(self, docling_elements, page_dims):
        """
        Return elements to mask before Docling re-extracts text, covering three
        categories of artifact:

        Sidebar elements (left column metadata like Citation, Copyright):
          Detected by finding the largest x1 gap across all multi-line elements —
          this gap separates the narrow left annotation column from the main text
          column.  Elements with x2 < x_left_bound_main are sidebar-only and are
          masked individually (not as a full-width strip, to avoid clipping
          full-width elements like the title that also start at the left margin).
          Sidebar elements are also excluded from the anchor pool so they don't
          distort the y-bounds.

        Header / footer strips (running headers, page numbers, overflow fragments):
          Per-page y-bounds are derived from non-sidebar multi-line anchor blocks.
          Full-page-width strips are created above the topmost anchor and below
          the bottommost anchor on each page.

        Figure-page fallback (pages with no anchor blocks):
          NER via nlp_is_meaningful filters individual single-line TEXT elements.
        """
        # ── Detect sidebar boundaries from gaps in anchor x1 values ─────────
        SIDEBAR_MAX_W  = 150   # pts — annotation columns are narrow by definition
        COLUMN_GAP_MIN = 50    # pts — minimum x1 gap to count as a column boundary

        anchor_x1s = sorted(
            el['bbox']['x1'] for el in docling_elements
            if abs(el['bbox'].get('y1', 0) - el['bbox'].get('y2', 0)) >= MIN_ANCHOR_H
        )
        sig_gaps = []   # [(gap_size, left_x1, right_x1), ...]
        for i in range(len(anchor_x1s) - 1):
            gap = anchor_x1s[i + 1] - anchor_x1s[i]
            if gap > COLUMN_GAP_MIN:
                sig_gaps.append((gap, anchor_x1s[i], anchor_x1s[i + 1]))

        x_left_bound_main   = sig_gaps[0][2]  if sig_gaps           else None
        x_right_bound_start = sig_gaps[-1][1] if len(sig_gaps) >= 2 else None

        def _is_sidebar(el):
            b = el.get('bbox', {})
            x1, x2 = b.get('x1', 0), b.get('x2', 0)
            width = x2 - x1
            if width >= SIDEBAR_MAX_W:
                return False
            if x_left_bound_main  is not None and x2 < x_left_bound_main:
                return True
            if x_right_bound_start is not None and x1 > x_right_bound_start:
                return True
            return False

        # ── Pass 1: per-page anchor bounds, excluding sidebar elements ────────
        page_bounds: dict = {}
        for el in docling_elements:
            if _is_sidebar(el):
                continue
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

        # ── Pass 2: build mask list ───────────────────────────────────────────
        mask_elements = []

        def _dims(page):
            d = page_dims.get(page) or page_dims.get(str(page)) or {}
            return d.get('width', 595.0), d.get('height', 842.0)

        # Category 1: sidebar elements (mask by individual bbox)
        n_sidebar = 0
        for el in docling_elements:
            if _is_sidebar(el) and (el.get('text') or '').strip():
                mask_elements.append(el)
                n_sidebar += 1

        # Category 2: header/footer strips (full-width, per page)
        n_strips = 0
        for page, (top_bound, bot_bound) in page_bounds.items():
            pw, ph = _dims(page)
            if top_bound < ph:
                mask_elements.append({
                    'page': page,
                    'bbox': {'x1': 0, 'y1': ph, 'x2': pw, 'y2': top_bound},
                })
                n_strips += 1
            if bot_bound > 0:
                mask_elements.append({
                    'page': page,
                    'bbox': {'x1': 0, 'y1': bot_bound, 'x2': pw, 'y2': 0},
                })
                n_strips += 1

        # Category 3: NER fallback for pages with no anchors
        n_ner = 0
        pages_with_anchors = set(page_bounds.keys())
        for el in docling_elements:
            if el.get('type') != 'TEXT' or _is_sidebar(el):
                continue
            page = el.get('page')
            if page in pages_with_anchors:
                continue
            text = (el.get('text') or '').strip()
            b = el.get('bbox', {})
            h = abs(b.get('y1', 0) - b.get('y2', 0))
            if h < MIN_ANCHOR_H and not nlp_is_meaningful(text, self.nlp):
                mask_elements.append(el)
                n_ner += 1

        logger.info(f'  Header/footer/sidebar: {n_sidebar} sidebar, '
                    f'{n_strips} strip region(s), {n_ner} NER-filtered')
        return mask_elements

    # ── Step 3: Merge bboxes and create combined masked PDF ────────────────────
    def _create_masked_pdf(self, pdf_path: Path, docling_elements, tatr_detections,
                           output_path: Path, hf_elements=None):
        import fitz
        doc_fitz   = fitz.open(str(pdf_path))
        page_rects = {}
        hf_elements = hf_elements or []

        for page_num in range(len(doc_fitz)):
            page_no = page_num + 1
            h       = doc_fitz[page_num].rect.height
            rects   = []

            for d in tatr_detections:
                if d['page'] == page_no:
                    rects.append(d['rect'])

            for el in docling_elements:
                if el.get('type') in DOCLING_MASK_TYPES and el.get('page') == page_no:
                    b = el['bbox']
                    rects.append(fitz.Rect(b['x1'], h - b['y1'], b['x2'], h - b['y2']))

            for el in hf_elements:
                if el.get('page') == page_no:
                    b = el['bbox']
                    rects.append(fitz.Rect(b['x1'], h - b['y1'], b['x2'], h - b['y2']))

            if rects:
                page_rects[page_no] = merge_rects(rects)

        for page_num in range(len(doc_fitz)):
            page_no = page_num + 1
            for rect in page_rects.get(page_no, []):
                doc_fitz[page_num].add_redact_annot(rect, fill=(1, 1, 1))
            doc_fitz[page_num].apply_redactions()

        doc_fitz.save(str(output_path))
        doc_fitz.close()

        n_raw    = sum(
            len([d for d in tatr_detections if d['page'] == p]) +
            len([el for el in docling_elements
                 if el.get('type') in DOCLING_MASK_TYPES and el.get('page') == p])
            for p in page_rects
        )
        n_merged = sum(len(v) for v in page_rects.values())
        logger.info(f'  Masked: {n_raw} raw rects → {n_merged} merged regions')
        return page_rects

    # ── Step 4: Re-extract from masked PDF ────────────────────────────────────
    def _docling_extract_masked(self, masked_pdf_path: Path):
        result   = self.converter.convert(str(masked_pdf_path))
        doc      = result.document
        elements = []
        for element, level in doc.iterate_items():
            label = str(getattr(element, 'label', 'UNKNOWN')).split('.')[-1].upper()
            if not (hasattr(element, 'prov') and element.prov):
                continue
            prov = element.prov[0]
            bbox = prov.bbox
            text = (getattr(element, 'text', '') or '').strip()
            elements.append({
                'type' : label,
                'page' : prov.page_no,
                'level': level,
                'bbox' : {'x1': bbox.l, 'y1': bbox.t, 'x2': bbox.r, 'y2': bbox.b},
                'text' : text or None,
            })
        return elements

    # ── Step 5: Save ordered txt ───────────────────────────────────────────────
    def _save_txt(self, rows, pmcid: str):
        out_path = self.text_dir / f'{pmcid}_combined_masked_ordered.txt'
        save_text(rows, out_path, pmcid, label='combined | masked', ordered=True)
        logger.info(f'  Saved text: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)')
        return out_path

    # ── Step 6: Database ingestion ─────────────────────────────────────────────
    def _ingest_db(self, rows, pmcid: str, pdf_path: Path):
        from database import Document, TextElement
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self.db.session_scope() as session:
            existing = session.query(Document).filter_by(pmcid=pmcid).first()
            if existing:
                logger.warning(f'  {pmcid} already in database — skipping ingestion')
                return 0

            doc = Document(
                pmcid=pmcid,
                filename=pdf_path.name,
                file_path=str(pdf_path.absolute()),
                title=f'Document {pmcid}',
                text_source='pdf',
            )
            session.add(doc)
            session.flush()

            path_counts = defaultdict(int)
            inserted = 0
            for path_str, path_list, depth, text in rows:
                pos_in_sec  = path_counts[path_str]
                path_counts[path_str] += 1
                unique_path = (f'{pmcid}/{path_str}/{pos_in_sec}'
                               if path_str else f'{pmcid}/(Root)/{pos_in_sec}')
                stmt = pg_insert(TextElement).values(
                    unique_path        = unique_path,
                    document_id        = doc.id,
                    path_list          = path_list,
                    path_string        = path_str,
                    depth              = depth,
                    text_content       = text,
                    position_in_section= pos_in_sec,
                    references         = {},
                ).on_conflict_do_nothing(index_elements=['unique_path'])
                result = session.execute(stmt)
                if result.rowcount:
                    inserted += 1

            session.flush()
            logger.info(f'  Ingested {inserted} text elements for {pmcid} (doc id={doc.id})')
            return inserted

    # ── Step 6a: Visualize TATR detections ────────────────────────────────────
    def _visualize_tatr(self, pdf_path: Path, tatr_detections, stem: str):
        import fitz
        TATR_COLORS = {
            'table':         (1.0, 0.35, 0.0),
            'table rotated': (0.8, 0.0,  0.8),
        }
        DEFAULT_COLOR = (1.0, 0.0, 0.0)

        out_path = self.vis_dir / f'{stem}_tatr_detections.pdf'
        doc = fitz.open(str(pdf_path))
        for d in tatr_detections:
            page  = doc[d['page'] - 1]
            color = TATR_COLORS.get(d['label'].lower(), DEFAULT_COLOR)
            rect  = d['rect']
            page.draw_rect(rect, color=color, width=2)
            page.insert_text(
                (rect.x0 + 2, max(rect.y0 - 2, 8)),
                f"{d['label']} {d['score']:.2f}",
                fontsize=7, color=color,
            )

        fp = doc[0]
        lx, ly = 20, 20
        label_counts = Counter(d['label'] for d in tatr_detections)
        legend_h = 28 + len(label_counts) * 11
        fp.draw_rect(fitz.Rect(lx-5, ly-5, lx+175, ly+legend_h),
                     color=(0,0,0), fill=(1,1,1), width=0.5)
        fp.insert_text((lx, ly+10), 'TATR Detections:', fontsize=9, color=(0,0,0))
        for row, (lbl, cnt) in enumerate(sorted(label_counts.items())):
            y = ly + 22 + row * 11
            c = TATR_COLORS.get(lbl.lower(), DEFAULT_COLOR)
            fp.draw_line((lx, y), (lx+12, y), color=c, width=2)
            fp.insert_text((lx+16, y+3), f"{lbl} ({cnt})", fontsize=7, color=(0,0,0))

        doc.save(str(out_path))
        doc.close()
        logger.info(f'  Saved TATR visualization: {out_path}')

    # ── Step 6b: Visualize combined (Docling + TATR) merged detections ─────────
    def _visualize_combined(self, pdf_path: Path, docling_elements, tatr_detections,
                            page_rects, stem: str):
        import fitz
        COLOR_TATR    = (1.0, 0.45, 0.0)
        COLOR_DOCLING = (0.1, 0.45, 0.9)
        COLOR_MERGED  = (0.0, 0.70, 0.2)
        COLOR_TEXT    = (0.6, 0.0, 0.8)

        out_path = self.vis_dir / f'{stem}_combined_detections.pdf'
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page_no = page_num + 1
            page    = doc[page_num]
            h       = page.rect.height

            for d in tatr_detections:
                if d['page'] == page_no:
                    page.draw_rect(d['rect'], color=COLOR_TATR, width=1.5)
                    page.insert_text(
                        (d['rect'].x0 + 2, max(d['rect'].y0 - 2, 8)),
                        f"TATR {d['score']:.2f}", fontsize=6, color=COLOR_TATR,
                    )

            for el in docling_elements:
                if el.get('page') != page_no:
                    continue
                b = el['bbox']
                r = fitz.Rect(b['x1'], h - b['y1'], b['x2'], h - b['y2'])
                if el.get('type') in DOCLING_MASK_TYPES:
                    page.draw_rect(r, color=COLOR_DOCLING, width=1.5)
                    page.insert_text(
                        (r.x0 + 2, max(r.y0 - 2, 8)),
                        el.get('type', ''), fontsize=6, color=COLOR_DOCLING,
                    )
                elif el.get('type') in TEXT_ELEMENT_TYPES:
                    page.draw_rect(r, color=COLOR_TEXT, width=0.8)
                    page.insert_text(
                        (r.x0 + 2, max(r.y0 - 2, 8)),
                        el.get('type', ''), fontsize=5, color=COLOR_TEXT,
                    )

            for rect in page_rects.get(page_no, []):
                page.draw_rect(rect, color=COLOR_MERGED, width=2.5)

        fp = doc[0]
        lx, ly = 20, 20
        legend_items = [
            ('TATR detection',    COLOR_TATR,    1.5),
            ('Docling detection', COLOR_DOCLING, 1.5),
            ('Merged (masked)',   COLOR_MERGED,  2.5),
            ('Text element',      COLOR_TEXT,    0.8),
        ]
        legend_h = 28 + len(legend_items) * 12
        fp.draw_rect(fitz.Rect(lx-5, ly-5, lx+185, ly+legend_h),
                     color=(0,0,0), fill=(1,1,1), width=0.5)
        fp.insert_text((lx, ly+10), 'Combined Detections:', fontsize=9, color=(0,0,0))
        for row, (lbl, color, lw) in enumerate(legend_items):
            y = ly + 23 + row * 12
            fp.draw_line((lx, y), (lx+14, y), color=color, width=lw)
            fp.insert_text((lx+18, y+3), lbl, fontsize=7, color=(0,0,0))

        doc.save(str(out_path))
        doc.close()
        logger.info(f'  Saved combined visualization: {out_path}')

    # ── Step 6c: Crop and save figure/table images ─────────────────────────────
    def _crop_and_save(self, pdf_path: Path, pmcid: str,
                       docling_elements, tatr_detections):
        import fitz
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(2, 2)  # 2x scale for quality

        all_captions = [el for el in docling_elements if el.get('type') == 'CAPTION']

        # ── Figures ───────────────────────────────────────────────────────────
        # Match each PICTURE to its nearest caption, merge duplicates sharing
        # the same figure number, fall back to a sequential id when no caption.
        merged_figures = {}   # key: str figure number or fallback id
        for el in docling_elements:
            if el.get('type') != 'PICTURE':
                continue
            cap_el  = nearest_caption(el, all_captions)
            caption = cap_el.get('text', '') if cap_el else ''
            num     = str(parse_caption_num(caption, FIG_NUM_RE) or (len(merged_figures) + 1))
            if num not in merged_figures:
                merged_figures[num] = {
                    'figure_id': num,
                    'caption'  : caption or f'Figure {num}',
                    'page'     : el.get('page'),
                    'bbox'     : el.get('bbox'),
                }
            else:
                existing = merged_figures[num]
                if existing['bbox'] and el.get('bbox'):
                    existing['bbox'] = union_bbox(existing['bbox'], el['bbox'])
                if len(caption) > len(existing['caption'] or ''):
                    existing['caption'] = caption
                logger.debug(f'  Merged duplicate Figure {num}')

        for fig in merged_figures.values():
            page_no = fig['page']
            b       = fig['bbox']
            if page_no is None or b is None:
                continue
            page = doc[page_no - 1]
            h    = page.rect.height
            rect = fitz.Rect(b['x1'], h - b['y1'], b['x2'], h - b['y2'])
            pix  = page.get_pixmap(clip=rect, matrix=mat)
            pix.save(str(self.figures_dir / f'{pmcid}_figure_{fig["figure_id"]}.png'))

        # ── Tables ────────────────────────────────────────────────────────────
        # TATR detections → nearest caption for table number, same merge logic.
        merged_tables = {}
        for d in tatr_detections:
            # Convert TATR fitz.Rect → pseudo-element for nearest_caption
            r  = d['rect']
            ph = doc[d['page'] - 1].rect.height
            el = {'page': d['page'],
                  'bbox': {'x1': r.x0, 'y1': ph - r.y0,
                           'x2': r.x1, 'y2': ph - r.y1}}
            cap_el  = nearest_caption(el, all_captions)
            caption = cap_el.get('text', '') if cap_el else ''
            num     = str(parse_caption_num(caption, TAB_NUM_RE) or (len(merged_tables) + 1))
            if num not in merged_tables:
                merged_tables[num] = {
                    'table_id': num,
                    'caption' : caption or f'Table {num}',
                    'rect'    : d['rect'],
                    'page'    : d['page'],
                }
            else:
                existing = merged_tables[num]
                existing['rect'] = existing['rect'] | d['rect']   # fitz union
                if len(caption) > len(existing['caption'] or ''):
                    existing['caption'] = caption
                logger.debug(f'  Merged duplicate Table {num}')

        for tbl in merged_tables.values():
            page = doc[tbl['page'] - 1]
            pix  = page.get_pixmap(clip=tbl['rect'], matrix=mat)
            pix.save(str(self.tables_dir / f'{pmcid}_table_{tbl["table_id"]}.png'))

        doc.close()
        logger.info(f'  Saved {len(merged_figures)} figure(s) and {len(merged_tables)} table(s)')

    # ── Main entry point ───────────────────────────────────────────────────────
    def process(self, pdf_path: Path, pmcid: str = None) -> bool:
        pmcid = pmcid or pdf_path.stem.split('_')[0]

        if pmcid in self.blacklist:
            logger.info(f'Skipping blacklisted {pmcid}')
            return False

        logger.info(f'Processing {pmcid} ({pdf_path.name})')
        try:
            # 1. Original layout
            logger.info('  Step 1/5: Docling extraction (original PDF)...')
            docling_els, page_dims = self._docling_extract(pdf_path)
            logger.info(f'  {len(docling_els)} elements detected')

            # 2. TATR detection
            logger.info('  Step 2/5: TATR table detection...')
            tatr_dets = self._tatr_detect(pdf_path)
            logger.info(f'  TATR: {len(tatr_dets)} table(s) detected '
                        f'(threshold={self.tatr_threshold})')

            # 2b. Detect header/footer regions from original extraction
            logger.info('  Step 2b: Detecting header/footer regions...')
            hf_els = self._detect_header_footer_elements(docling_els, page_dims)

            # 3. Combined masked PDF (tables + figures + headers/footers)
            logger.info('  Step 3/5: Creating combined masked PDF...')
            masked_path = self.masked_pdf_dir / f'{pdf_path.stem}_combined_masked.pdf'
            page_rects  = self._create_masked_pdf(pdf_path, docling_els, tatr_dets, masked_path,
                                                  hf_elements=hf_els)

            # Save layout JSON for the masked PDF
            masked_els = self._docling_extract_masked(masked_path)
            masked_els = filter_artifacts(masked_els, nlp=self.nlp)
            # Apply NER relevance filter to TEXT elements only
            # (SECTION_HEADER, TITLE, LIST etc. are kept unconditionally)
            before = len(masked_els)
            masked_els = [
                el for el in masked_els
                if el.get('type') != 'TEXT'
                or is_relevant_para(el.get('text') or '', self.nlp)
            ]
            logger.info(f'  NER relevance filter: {before - len(masked_els)} TEXT element(s) removed')
            # Normalise text in every element before saving (fixes ligatures and
            # spurious spaces around accented characters from PDF font encoding)
            for el in masked_els:
                if el.get('text'):
                    el['text'] = fix_ligatures(el['text'])
            json_path = self.docling_out_dir / f'{masked_path.stem}_layout.json'
            json_path.write_text(json.dumps({
                'metadata': {
                    'pmcid': pmcid,
                    'pdf_path': str(masked_path),
                    'source': 'combined_docling_tatr',
                    'tatr_threshold': self.tatr_threshold,
                    'extraction_date': datetime.now().isoformat(),
                },
                'page_dimensions': page_dims,
                'elements': masked_els,
            }, indent=2, ensure_ascii=False))

            # 4. Visualizations (optional)
            logger.info('  Step 4/6: Visualizations...' if self.visualize else '  Step 4/6: Visualizations skipped.')
            if self.visualize:
                stem = pdf_path.stem
                if tatr_dets:
                    self._visualize_tatr(pdf_path, tatr_dets, stem)
                self._visualize_combined(pdf_path, docling_els, tatr_dets, page_rects, stem)

            # 5. Crop and save figures/tables
            logger.info('  Step 5/6: Cropping and saving figures/tables...')
            self._crop_and_save(pdf_path, pmcid, docling_els, tatr_dets)

            # 6. Extract text (document order)
            logger.info('  Step 6/6: Extracting and stitching text...')
            text_els = [el for el in masked_els if el.get('type') in TEXT_ELEMENT_TYPES]
            rows, n_skipped = extract_text(text_els, nlp=self.nlp)
            logger.info(f'  {len(rows)} paragraphs across '
                        f'{len(set(r[0] for r in rows))} sections'
                        + (f' ({n_skipped} elements skipped)' if n_skipped else ''))

            self._save_txt(rows, pmcid)

            # DB ingestion
            if self.db_ingest and self.db:
                self._ingest_db(rows, pmcid, pdf_path)

            logger.info(f'✅ Done: {pmcid}')
            return True

        except Exception as e:
            logger.error(f'❌ Failed {pmcid}: {e}', exc_info=True)
            self._add_to_blacklist(pmcid, str(e))
            return False


if __name__ == '__main__':
    PDF_DIR = Path('files/organized_pdfs')
    pdfs    = sorted(PDF_DIR.glob('*.pdf'))[:5]

    processor = CombinedPipelineProcessor(db_ingest=True, visualize=True)
    for pdf in pdfs:
        pmcid = pdf.stem.split('_')[0]
        processor.process(pdf, pmcid)
