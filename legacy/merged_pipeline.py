#!/usr/bin/env python3
"""
DEPRECATED — experimental pipeline, superseded by
``pipeline/stages/pdf_text_extraction/runner.py`` (PipelineRunner). Kept for
provenance: its header/footer detection was ported to
``pipeline/stages/pdf_text_extraction/components/region_masker.py``
(``_detect_header_footer_elements``). Not on the production path; not imported
by any live module.

Merged Pipeline (Docling + TATR)

Combines functionality from deneme.py and combined_pipeline.py:
  - Production-ready batch processing with blacklist support
  - Lazy model loading (Docling, TATR, scispaCy, DB)
  - Header/footer/sidebar detection and masking
  - NER-based artifact filtering
  - Optional table reconstruction from Docling sub-elements
  - Figure/table cropping with panel counting
  - Full database ingestion (text elements, figures, tables)
  - Optional multi-combination text extraction and comparison

For each PDF:
  1.  Extract layout with Docling (original PDF)
  2.  Optionally reconstruct tables from sub-elements
  3.  Detect tables with TATR
  4.  Detect header/footer/sidebar regions
  5.  Merge all bounding boxes and create masked PDF
  6.  Re-extract layout from masked PDF with Docling
  7.  Filter artifacts and irrelevant text via NER
  8.  Visualize detections (optional)
  9.  Crop and save figure/table images (with panel counting)
  10. Stitch text into hierarchical sections (document order)
  11. Ingest to database (text elements, figures, tables)

Usage:
    python scripts/merged_pipeline.py --pdf files/organized_pdfs/PMC123.pdf
    python scripts/merged_pipeline.py --pdf-dir files/organized_pdfs
    python scripts/merged_pipeline.py --pdf-dir files/organized_pdfs --reconstruct --no-vis
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
    SKIP_TYPES, fix_ligatures, merge_rects,
    FIG_NUM_RE, TAB_NUM_RE, parse_caption_num, union_bbox, nearest_caption,
    MIN_ANCHOR_H, nlp_is_meaningful, filter_artifacts,
    is_relevant_para, extract_text, save_text,
    build_table_bboxes, build_picture_pages, bbox_overlaps, centroid_inside,
    count_panels,
)
from parsers.text_processing import ContextAwareStitcher, remove_citations  # noqa: E402

TATR_DPI       = 150
TATR_THRESHOLD = 0.99
SCALE          = 72 / TATR_DPI   # pixels -> PDF points


class MergedPipelineProcessor:
    def __init__(
        self,
        masked_pdf_dir : str   = 'out/masked_pdfs',
        docling_out_dir: str   = 'out/docling_full',
        text_dir       : str   = 'out/text',
        figures_dir    : str   = 'files/figures',
        tables_dir     : str   = 'files/tables',
        vis_dir        : str   = 'out/visualization',
        blacklist_file : str   = 'out/failed_pdfs_blacklist.json',
        tatr_threshold : float = TATR_THRESHOLD,
        db_ingest      : bool  = True,
        visualize      : bool  = True,
        reconstruct    : bool  = False,  # reconstruct tables from Docling sub-elements
        baseline       : str   = 'masked',
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
        self.reconstruct     = reconstruct
        self.baseline        = baseline

        for d in [self.masked_pdf_dir, self.docling_out_dir,
                  self.text_dir, self.figures_dir, self.tables_dir, self.vis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.blacklist_file.parent.mkdir(parents=True, exist_ok=True)

        self._converter  = None   # lazy Docling
        self._tatr_proc  = None   # lazy TATR processor
        self._tatr_model = None   # lazy TATR model
        self._db         = None   # lazy DB connection
        self._nlp        = None   # lazy scispaCy model

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
        # Reclassify TEXT elements that look like captions
        reclassified = 0
        for el in elements:
            if el['type'] == 'TEXT' and CAPTION_PATTERN.match(el.get('text') or ''):
                el['type'] = 'CAPTION'
                reclassified += 1
        if reclassified:
            logger.info(f'  Reclassified {reclassified} TEXT elements as CAPTION')
        page_dims = {no: {'width': p.size.width, 'height': p.size.height}
                     for no, p in doc.pages.items()}
        return elements, page_dims

    # ── Step 1b: Reconstruct tables from Docling sub-elements (optional) ───────
    def _reconstruct_tables(self, pdf_path: Path, elements):
        """
        Merge Docling TABLE sub-elements into RECONSTRUCTED_TABLE entries.
        Requires scripts/visualize_docling_full.py to be present.
        Returns (elements_raw, reconstructed_elements).
        """
        import importlib.util
        vis_path = Path(__file__).parent / 'visualize_docling_full.py'
        if not vis_path.exists():
            logger.warning(f'visualize_docling_full.py not found at {vis_path}, skipping')
            return elements, elements

        spec   = importlib.util.spec_from_file_location('visualize', vis_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Write a temporary JSON so reconstruct_tables_from_lists can read it
        tmp_json = self.docling_out_dir / f'{pdf_path.stem}_tmp_layout.json'
        tmp_json.write_text(json.dumps({'elements': elements}, indent=2))

        elements_raw    = list(elements)
        reconstructed   = module.reconstruct_tables_from_lists(str(tmp_json))
        n_reconstructed = len([e for e in reconstructed if e.get('type') == 'RECONSTRUCTED_TABLE'])
        logger.info(f'  Reconstruction: {len(elements_raw)} → {len(reconstructed)} elements '
                    f'({n_reconstructed} RECONSTRUCTED_TABLE)')
        tmp_json.unlink(missing_ok=True)
        return elements_raw, reconstructed

    # ── Step 2: TATR table detection ───────────────────────────────────────────
    def _tatr_detect(self, pdf_path: Path):
        import fitz
        import torch
        from PIL import Image as PILImage
        tatr_proc, tatr_model = self.tatr

        doc_fitz   = fitz.open(str(pdf_path))
        detections = []
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

    # ── Step 2b: Detect header/footer/sidebar regions ─────────────────────────
    def _detect_header_footer_elements(self, docling_elements, page_dims):
        """
        Detects sidebar, header, and footer elements for masking.

        - Sidebar: narrow left/right annotation columns, detected by x1 gap analysis.
        - Header/footer: full-width strips outside anchor block y-bounds per page.
        - NER fallback: single-line TEXT elements on pages with no anchor blocks.
        """
        SIDEBAR_MAX_W  = 150   # pts — narrow annotation columns
        COLUMN_GAP_MIN = 50    # pts — minimum x1 gap to detect a column boundary

        anchor_x1s = sorted(
            el['bbox']['x1'] for el in docling_elements
            if abs(el['bbox'].get('y1', 0) - el['bbox'].get('y2', 0)) >= MIN_ANCHOR_H
        )
        sig_gaps = []
        for i in range(len(anchor_x1s) - 1):
            gap = anchor_x1s[i + 1] - anchor_x1s[i]
            if gap > COLUMN_GAP_MIN:
                sig_gaps.append((gap, anchor_x1s[i], anchor_x1s[i + 1]))

        x_left_bound_main   = sig_gaps[0][2]  if sig_gaps           else None
        x_right_bound_start = sig_gaps[-1][1] if len(sig_gaps) >= 2 else None

        def _is_sidebar(el):
            b = el.get('bbox', {})
            x1, x2 = b.get('x1', 0), b.get('x2', 0)
            if (x2 - x1) >= SIDEBAR_MAX_W:
                return False
            if x_left_bound_main  is not None and x2 < x_left_bound_main:
                return True
            if x_right_bound_start is not None and x1 > x_right_bound_start:
                return True
            return False

        # Pass 1: per-page anchor y-bounds (excluding sidebar elements)
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

        # Pass 2: build mask list
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
                mask_elements.append({'page': page,
                                      'bbox': {'x1': 0, 'y1': ph, 'x2': pw, 'y2': top_bound}})
                n_strips += 1
            if bot_bound > 0:
                mask_elements.append({'page': page,
                                      'bbox': {'x1': 0, 'y1': bot_bound, 'x2': pw, 'y2': 0}})
                n_strips += 1

        # Category 3: NER fallback for pages with no anchor blocks
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
        doc_fitz    = fitz.open(str(pdf_path))
        page_rects  = {}
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

        n_raw = sum(
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

    # ── Step 6: Database ingestion (text elements + figures + tables) ──────────
    def _ingest_db(self, rows, pmcid: str, pdf_path: Path,
                   figure_data=None, table_data=None):
        from database import Document, TextElement, Figure, Table
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self.db.session_scope() as session:
            existing = session.query(Document).filter_by(pmcid=pmcid).first()
            if existing:
                logger.warning(f'  {pmcid} already in database — skipping ingestion')
                return 0

            doc = Document(
                pmcid      = pmcid,
                filename   = pdf_path.name,
                file_path  = str(pdf_path.absolute()),
                title      = f'Document {pmcid}',
                text_source= 'pdf',
            )
            session.add(doc)
            session.flush()

            # ── Text elements ─────────────────────────────────────────────────
            path_counts = defaultdict(int)
            inserted    = 0
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
            logger.info(f'  Ingested {inserted} text elements (doc id={doc.id})')

            # ── Figures ───────────────────────────────────────────────────────
            fig_inserted = 0
            for fig in (figure_data or []):
                image_path     = fig.get('image_path')
                image_filename = Path(image_path).name if image_path else None
                stmt = pg_insert(Figure).values(
                    document_id   = doc.id,
                    figure_id     = fig['figure_id'],
                    figure_label  = f"Figure {fig['figure_id']}",
                    figure_number = fig['figure_id'],
                    caption_text  = fig.get('caption'),
                    image_filename= image_filename,
                    image_path    = image_path,
                ).on_conflict_do_nothing()
                result = session.execute(stmt)
                if result.rowcount:
                    fig_inserted += 1
            if figure_data:
                session.flush()
                logger.info(f'  Ingested {fig_inserted}/{len(figure_data)} figures')

            # ── Tables ────────────────────────────────────────────────────────
            tbl_inserted = 0
            for tbl in (table_data or []):
                image_path     = tbl.get('image_path')
                image_filename = Path(image_path).name if image_path else None
                stmt = pg_insert(Table).values(
                    document_id   = doc.id,
                    table_id      = tbl['table_id'],
                    table_label   = f"Table {tbl['table_id']}",
                    table_number  = tbl['table_id'],
                    caption_text  = tbl.get('caption'),
                    image_filename= image_filename,
                    image_path    = image_path,
                ).on_conflict_do_nothing()
                result = session.execute(stmt)
                if result.rowcount:
                    tbl_inserted += 1
            if table_data:
                session.flush()
                logger.info(f'  Ingested {tbl_inserted}/{len(table_data)} tables')

            return inserted

    # ── Visualization: TATR detections ────────────────────────────────────────
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

    # ── Visualization: combined (Docling + TATR) merged detections ─────────────
    def _visualize_combined(self, pdf_path: Path, docling_elements, tatr_detections,
                            page_rects, stem: str):
        import fitz
        COLOR_TATR    = (1.0, 0.45, 0.0)
        COLOR_DOCLING = (0.1, 0.45, 0.9)
        COLOR_MERGED  = (0.0, 0.70, 0.2)
        COLOR_TEXT    = (0.6, 0.0,  0.8)

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

    # ── Step 7: Crop and save figure/table images ──────────────────────────────
    def _crop_and_save(self, pdf_path: Path, pmcid: str,
                       docling_elements, tatr_detections):
        """
        Crop and save PNG images for each figure and table from the original PDF.

        Figures: sourced from PICTURE/FIGURE Docling elements, merged by caption number.
                 Panel count is detected and stored in metadata.
        Tables:  sourced from TATR detections (primary) and TABLE/RECONSTRUCTED_TABLE
                 Docling elements (supplementary), merged by caption number.

        Returns (figure_data, table_data) — lists of dicts with image_path set.
        """
        import fitz
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(2, 2)  # 2x scale for quality

        all_captions = [el for el in docling_elements if el.get('type') == 'CAPTION']

        # ── Figures ───────────────────────────────────────────────────────────
        merged_figures = {}
        for el in docling_elements:
            if el.get('type') not in ('PICTURE', 'FIGURE'):
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

        figure_data = []
        for fig in merged_figures.values():
            page_no = fig['page']
            b       = fig['bbox']
            if page_no is None or b is None:
                continue
            page     = doc[page_no - 1]
            h        = page.rect.height
            rect     = fitz.Rect(b['x1'], h - max(b['y1'], b['y2']),
                                 b['x2'], h - min(b['y1'], b['y2']))
            n_panels = count_panels(page, b, h)
            pix      = page.get_pixmap(clip=rect, matrix=mat)
            img_path = str(self.figures_dir / f'{pmcid}_figure_{fig["figure_id"]}.png')
            pix.save(img_path)
            figure_data.append({**fig, 'image_path': img_path, 'panels': n_panels})

        # ── Tables ────────────────────────────────────────────────────────────
        # Primary source: TATR detections (already in PDF top-left coords as fitz.Rect)
        merged_tables = {}
        for d in tatr_detections:
            r  = d['rect']
            ph = doc[d['page'] - 1].rect.height
            pseudo_el = {'page': d['page'],
                         'bbox': {'x1': r.x0, 'y1': ph - r.y0,
                                  'x2': r.x1, 'y2': ph - r.y1}}
            cap_el  = nearest_caption(pseudo_el, all_captions)
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
                merged_tables[num]['rect'] = merged_tables[num]['rect'] | d['rect']
                if len(caption) > len(merged_tables[num]['caption'] or ''):
                    merged_tables[num]['caption'] = caption
                logger.debug(f'  Merged duplicate TATR Table {num}')

        # Supplementary source: TABLE / RECONSTRUCTED_TABLE Docling elements
        for el in docling_elements:
            if el.get('type') not in ('TABLE', 'RECONSTRUCTED_TABLE'):
                continue
            page_no = el.get('page')
            b       = el.get('bbox') or {}
            if page_no is None or not b:
                continue
            cap_el  = nearest_caption(el, all_captions)
            caption = cap_el.get('text', '') if cap_el else el.get('caption') or ''
            num     = str(parse_caption_num(caption, TAB_NUM_RE) or (len(merged_tables) + 1))
            ph      = doc[page_no - 1].rect.height
            rect    = fitz.Rect(b['x1'], ph - b['y1'], b['x2'], ph - b['y2'])
            if num not in merged_tables:
                merged_tables[num] = {
                    'table_id': num,
                    'caption' : caption or f'Table {num}',
                    'rect'    : rect,
                    'page'    : page_no,
                }
            else:
                merged_tables[num]['rect'] = merged_tables[num]['rect'] | rect
                if len(caption) > len(merged_tables[num]['caption'] or ''):
                    merged_tables[num]['caption'] = caption
                logger.debug(f'  Merged Docling Table {num} into existing entry')

        table_data = []
        for tbl in merged_tables.values():
            page     = doc[tbl['page'] - 1]
            pix      = page.get_pixmap(clip=tbl['rect'], matrix=mat)
            img_path = str(self.tables_dir / f'{pmcid}_table_{tbl["table_id"]}.png')
            pix.save(img_path)
            table_data.append({**tbl, 'image_path': img_path})

        doc.close()
        logger.info(f'  Saved {len(figure_data)} figure(s) and {len(table_data)} table(s)')
        return figure_data, table_data

    # ── Optional: hierarchical text extraction with bbox/centroid filtering ────
    def _extract_text_hierarchical(self, elements, table_bboxes=None, use_centroid=False):
        """
        Extract and stitch hierarchical text from elements.

        Args:
            elements:      List of element dicts (type, page, level, bbox, text).
            table_bboxes:  Optional {page: [bbox, ...]} to skip overlapping elements.
            use_centroid:  If True, skip only when the element centroid is inside a
                           table/figure bbox (less aggressive than full-overlap check).

        Returns:
            (stitched_by_path, n_skipped)
        """
        overlap_fn    = centroid_inside if use_centroid else bbox_overlaps
        picture_pages = build_picture_pages(elements)

        hierarchy = {}
        by_path   = defaultdict(list)
        skipped   = 0

        for el in elements:
            etype = el.get('type', '')
            level = el.get('level', 0)
            text  = fix_ligatures((el.get('text') or '').strip())
            if not text:
                continue

            if etype == 'SECTION_HEADER':
                hierarchy[level] = text
                hierarchy = {k: v for k, v in hierarchy.items() if k <= level}
            elif etype not in SKIP_TYPES:
                # Drop single-char tokens on figure pages (likely panel labels)
                if len(text) == 1 and el.get('page') in picture_pages:
                    skipped += 1
                    continue
                if table_bboxes:
                    page = el.get('page')
                    bbox = el.get('bbox')
                    if (page and bbox and
                            any(overlap_fn(bbox, tb) for tb in table_bboxes.get(page, []))):
                        skipped += 1
                        continue
                path_parts = [hierarchy[k] for k in sorted(hierarchy) if hierarchy.get(k)]
                by_path[' > '.join(path_parts) or 'Root'].append(text)

        stitcher = ContextAwareStitcher()
        stitched = {
            path: stitcher.reconstruct_paragraphs([remove_citations(t) for t in texts])
            for path, texts in by_path.items()
        }
        return stitched, skipped

    # ── Optional: run multiple extraction combinations and compare ─────────────
    def run_all_combinations(self, elements_raw, masked_elements):
        """
        Run text extraction with multiple strategies and return a results dict.
        Useful for comparing 'direct | raw' vs 'masked' vs 'combined | masked'.

        Args:
            elements_raw:     Original (unmasked) Docling elements.
            masked_elements:  Elements extracted from the combined-masked PDF.

        Returns:
            dict mapping combination name → stitched_by_path dict.
        """
        combinations = {
            'direct | raw': (
                elements_raw,
                build_table_bboxes(elements_raw, types=('TABLE', 'PICTURE'))
            ),
            'masked': (
                [el for el in masked_elements if el.get('type') in TEXT_ELEMENT_TYPES],
                None
            ),
        }

        results = {}
        logger.info('Running text extraction combinations:')
        for name, (elements, bboxes) in combinations.items():
            stitched, n_skipped = self._extract_text_hierarchical(elements, bboxes)
            results[name] = stitched
            n_paths = len(stitched)
            n_paras = sum(len(v) for v in stitched.values())
            logger.info(f'  {name:<26s}: {n_paths} paths, {n_paras} paras, {n_skipped} skipped')
        return results

    def compare_results(self, results):
        """Print paragraph-count table and unified diffs across all combinations."""
        import difflib
        names     = list(results.keys())
        others    = [n for n in names if n != self.baseline]
        all_paths = sorted(set().union(*[set(results[n]) for n in names]))

        header = f"{'Path':<55s}" + ''.join(f" {n[:10]:>10s}" for n in names)
        logger.info(header)
        logger.info('-' * len(header))
        for path in all_paths:
            counts = [len(results[n].get(path, [])) for n in names]
            marker = ' *' if len(set(counts)) > 1 else ''
            logger.info(f"{path[:53]:<55s}" + ''.join(f" {c:>10d}" for c in counts) + marker)

        for other in others:
            logger.info(f"\n{'='*80}\nDIFF  {self.baseline!r}  vs  {other!r}\n{'='*80}")
            any_diff = False
            for path in all_paths:
                a = results[self.baseline].get(path, [])
                b = results[other].get(path, [])
                if a == b:
                    continue
                any_diff = True
                logger.info(f'\n  [{path}]')
                for line in difflib.unified_diff(
                        a, b, lineterm='', fromfile=self.baseline, tofile=other):
                    if line.startswith('+') and not line.startswith('+++'):
                        logger.info(f'    + {line[1:][:120]}')
                    elif line.startswith('-') and not line.startswith('---'):
                        logger.info(f'    - {line[1:][:120]}')
            if not any_diff:
                logger.info('  (identical)')

    # ── Main entry point ───────────────────────────────────────────────────────
    def process(self, pdf_path: Path, pmcid: str = None) -> bool:
        pmcid = pmcid or pdf_path.stem.split('_')[0]

        if pmcid in self.blacklist:
            logger.info(f'Skipping blacklisted {pmcid}')
            return False

        logger.info(f'Processing {pmcid} ({pdf_path.name})')
        try:
            # Step 1: Docling extraction on original PDF
            logger.info('  Step 1: Docling extraction (original PDF)...')
            docling_els, page_dims = self._docling_extract(pdf_path)
            logger.info(f'  {len(docling_els)} elements detected')

            # Step 1b: Optional table reconstruction
            if self.reconstruct:
                logger.info('  Step 1b: Reconstructing tables from sub-elements...')
                _, docling_els = self._reconstruct_tables(pdf_path, docling_els)

            # Step 2: TATR table detection
            logger.info('  Step 2: TATR table detection...')
            tatr_dets = self._tatr_detect(pdf_path)
            logger.info(f'  TATR: {len(tatr_dets)} table(s) detected '
                        f'(threshold={self.tatr_threshold})')

            # Step 2b: Header/footer/sidebar detection
            logger.info('  Step 2b: Detecting header/footer/sidebar regions...')
            hf_els = self._detect_header_footer_elements(docling_els, page_dims)

            # Step 3: Create combined masked PDF
            logger.info('  Step 3: Creating combined masked PDF...')
            masked_path = self.masked_pdf_dir / f'{pdf_path.stem}_combined_masked.pdf'
            page_rects  = self._create_masked_pdf(
                pdf_path, docling_els, tatr_dets, masked_path, hf_elements=hf_els)

            # Step 4: Re-extract from masked PDF, filter artifacts via NER
            logger.info('  Step 4: Re-extracting from masked PDF...')
            masked_els = self._docling_extract_masked(masked_path)
            masked_els = filter_artifacts(masked_els, nlp=self.nlp)
            before     = len(masked_els)
            masked_els = [
                el for el in masked_els
                if el.get('type') != 'TEXT'
                or is_relevant_para(el.get('text') or '', self.nlp)
            ]
            logger.info(f'  NER filter: {before - len(masked_els)} TEXT element(s) removed')
            for el in masked_els:
                if el.get('text'):
                    el['text'] = fix_ligatures(el['text'])

            # Save layout JSON for the masked extraction
            json_path = self.docling_out_dir / f'{masked_path.stem}_layout.json'
            json_path.write_text(json.dumps({
                'metadata': {
                    'pmcid'          : pmcid,
                    'pdf_path'       : str(masked_path),
                    'source'         : 'combined_docling_tatr',
                    'tatr_threshold' : self.tatr_threshold,
                    'extraction_date': datetime.now().isoformat(),
                },
                'page_dimensions': page_dims,
                'elements'       : masked_els,
            }, indent=2, ensure_ascii=False))

            # Step 5: Visualizations (optional)
            if self.visualize:
                logger.info('  Step 5: Generating visualizations...')
                stem = pdf_path.stem
                if tatr_dets:
                    self._visualize_tatr(pdf_path, tatr_dets, stem)
                self._visualize_combined(pdf_path, docling_els, tatr_dets, page_rects, stem)
            else:
                logger.info('  Step 5: Visualizations skipped.')

            # Step 6: Crop and save figures/tables
            logger.info('  Step 6: Cropping and saving figures/tables...')
            figure_data, table_data = self._crop_and_save(
                pdf_path, pmcid, docling_els, tatr_dets)

            # Step 7: Extract and stitch text (document order)
            logger.info('  Step 7: Extracting and stitching text...')
            text_els = [el for el in masked_els if el.get('type') in TEXT_ELEMENT_TYPES]
            rows, n_skipped = extract_text(text_els, nlp=self.nlp)
            logger.info(f'  {len(rows)} paragraphs across '
                        f'{len(set(r[0] for r in rows))} sections'
                        + (f' ({n_skipped} elements skipped)' if n_skipped else ''))

            self._save_txt(rows, pmcid)

            # Step 8: Database ingestion
            if self.db_ingest and self.db:
                self._ingest_db(rows, pmcid, pdf_path,
                                figure_data=figure_data, table_data=table_data)

            logger.info(f'Done: {pmcid}')
            return True

        except Exception as e:
            logger.error(f'Failed {pmcid}: {e}', exc_info=True)
            self._add_to_blacklist(pmcid, str(e))
            return False


if __name__ == '__main__':
    PDF_DIR = Path('files/organized_pdfs')
    pdfs    = sorted(PDF_DIR.glob('*.pdf'))

    processor = MergedPipelineProcessor(db_ingest=True, visualize=True)
    for pdf in pdfs:
        pmcid = pdf.stem.split('_')[0]
        processor.process(pdf, pmcid)
