from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional



class TableDetectorType(str, Enum):
    TATR = "tatr"
    DOCLING = "docling"
    HYBRID = "hybrid"
    VLM = "vlm"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class OcrEngine(str, Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    RAPIDOCR = "rapidocr"


@dataclass(slots=True)
class PathConfig:
    project_root: Path = Path(".")
    output_root: Path = Path("out")
    files_root: Path = Path("files")

    # Core outputs
    masked_pdf_dir: Path = Path("out/masked_pdfs")
    docling_full_dir: Path = Path("out/docling_full")
    docling_masked_dir: Path = Path("out/docling_masked")
    text_dir: Path = Path("out/text")
    text_raw_dir: Path = Path("out/text_raw")
    json_dir: Path = Path("out/json")
    vis_dir: Path = Path("out/visualization")

    # Crops / media
    figures_dir: Path = Path("out/figures")
    tables_dir: Path = Path("out/tables")

    # Metadata / bookkeeping
    blacklist_file: Path = Path("out/failed_pdfs_blacklist.json")
    completed_file: Optional[Path] = None  # if set, fully-processed PMCIDs are saved here
    run_metadata_dir: Path = Path("out/run_metadata")

    def ensure_dirs(self) -> None:
        dirs = [
            self.output_root,
            self.files_root,
            self.masked_pdf_dir,
            self.docling_full_dir,
            self.docling_masked_dir,
            self.text_dir,
            self.text_raw_dir,
            self.json_dir,
            self.vis_dir,
            self.figures_dir,
            self.tables_dir,
            self.run_metadata_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DoclingConfig:
    enabled: bool = True
    do_table_structure: bool = True
    do_ocr: bool = False
    force_full_page_ocr: bool = False  # force OCR even on text-native PDFs
    ocr_engine: OcrEngine = OcrEngine.EASYOCR  # OCR engine when do_ocr=True
    images_scale: float = 2.0  # image resolution multiplier (higher = better OCR, slower)
    accelerator_device: str = "cpu"  # "cpu", "cuda", "mps"
    reconstruct_tables_from_lists: bool = False
    export_intermediate_json: bool = True
    timeout_sec: int = 300

    def content_key(self) -> str:
        """
        Short string that uniquely identifies the content-affecting settings.
        Used as a suffix in cache filenames so that changing options does not
        silently serve stale cached results.
        """
        return (
            f"tbl{int(self.do_table_structure)}"
            f"_ocr{int(self.do_ocr)}"
            f"_fp{int(self.force_full_page_ocr)}"
            f"_eng{self.ocr_engine.value if self.do_ocr else 'na'}"
            f"_sc{self.images_scale}"
        )


@dataclass(slots=True)
class TATRConfig:
    threshold: float = 0.99
    device: str = "cpu"  # "cpu", "cuda", "mps"
    model_name: str = "microsoft/table-transformer-detection"
    render_dpi: int = 150
    """DPI used when rasterising PDF pages for the TATR detector. Recall is
    DPI-sensitive — bump for small / faint tables, drop for speed."""


@dataclass(slots=True)
class MaskingConfig:
    enabled: bool = True
    mask_tables: bool = True
    mask_figures: bool = True
    mask_header_footer_sidebar: bool = True
    merge_overlapping_boxes: bool = True
    expand_box_px: int = 2  # small padding to avoid glyph remnants
    # Drop table-detector regions whose bbox is mostly inside any FIGURE/
    # PICTURE element (Docling layout).  Eliminates "table-inside-figure"
    # false positives that the table detector produces when figures contain
    # screenshots / printed tables / spreadsheet images.
    # Threshold: a table is dropped if ≥80 % of its area lies inside some
    # figure bbox on the same page.
    # Flipped from False → True on 2026-05-18 after empirical observation
    # of table-in-figure FPs during labelling.  Pre-flip sweeps live in
    # out/sweeps/ (baseline, baseline_evalcfg, …) and represent the
    # *disabled* state; re-run them with --no-drop-tables-inside-figures
    # to reproduce the historical behaviour.
    drop_tables_inside_figures: bool = True

    # Experimental: white out PICTURE/FIGURE bboxes BEFORE running Step 2
    # table detection so the detector never sees table-grid pixels embedded
    # in figures (printed-table screenshots, spreadsheet images).
    # Independent of ``drop_tables_inside_figures`` — the two can be
    # combined for belt-and-suspenders coverage, or compared head-to-head.
    # Only affects the detector input; final media cropping still uses the
    # original PDF. Has no effect on DoclingTableDetector (which reads
    # layout, not pixels) — only TATR / Hybrid see the change.
    mask_figures_before_table_detection: bool = False


@dataclass(slots=True)
class FilteringConfig:
    enabled: bool = True
    apply_ner_filtering: bool = True
    apply_paragraph_relevance_filtering: bool = True


@dataclass(slots=True)
class CroppingConfig:
    enabled: bool = True
    save_figure_crops: bool = True
    save_table_crops: bool = True
    image_format: str = "png"
    dpi: int = 200
    min_figure_pts: int = 50        # minimum width AND height in PDF points; smaller figures are skipped
    merge_figures_by_caption: bool = False  # merge PICTURE elements sharing the same caption number
    merge_tables_by_caption: bool = False   # merge TABLE/detection regions sharing the same caption number
    subfigure_proximity_pts: int = 20       # max edge-to-edge gap to treat adjacent figures as subfigure panels
    expand_tables_with_footnotes: bool = False   # absorb nearby TEXT/LIST_ITEM/FOOTNOTE elements below each table
    footnote_proximity_pts: float = 20.0         # max gap for FOOTNOTE / LIST_ITEM elements (adaptive)
    text_footnote_proximity_pts: float = 8.0     # fixed max gap for TEXT elements (non-adaptive)
    footnote_threshold_multiplier: float = 1.2   # after first absorption: max_gap = first_gap * this


@dataclass(slots=True)
class TextAssemblyConfig:
    write_raw_text: bool = False  # dump pre-assembly elements to out/text_raw/
    pre_filter_relevance: bool = True  # False → skip is_relevant_para; use post-stitch boilerplate filter instead


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    save_tatr_visualization: bool = True
    save_combined_visualization: bool = True
    max_pages: Optional[int] = None


@dataclass(slots=True)
class DatabaseConfig:
    enabled: bool = False
    db_url: Optional[str] = None


@dataclass(slots=True)
class RuntimeConfig:
    log_level: LogLevel = LogLevel.INFO
    fail_fast: bool = False
    skip_blacklisted: bool = True
    skip_existing_in_db: bool = True   # skip documents already in the database
    update_blacklist_on_failure: bool = True
    blacklist_if_rows_exceed: Optional[int] = None  # blacklist after success if row count exceeds this
    skip_existing_media_json: bool = False  # skip documents whose media JSON already exists
    skip_existing_outputs: bool = False     # skip individual stages whose output files already exist
    multi_source_crops: bool = False        # produce three media JSONs (docling / docling_recon / full)
    save_error_traces: bool = True
    seed: int | None = 42
    """Seed used by `PipelineRunner._seed_pipeline()` for `random`, `numpy`,
    and (when installed) `torch` / `torch.cuda`. Set to ``None`` to opt out
    of seeding entirely. Does not guarantee deterministic output from
    external libraries (Docling, TATR, OCR, scispaCy)."""
    num_workers: int = 1


@dataclass(slots=True)
class TwoPassConfig:
    """
    Configuration for the two-pass invisible-text detection and header-masking
    pipeline (TwoPassTextExtractor).

    Set ``enabled=True`` in PipelineConfig.two_pass to activate two-pass mode
    in PipelineRunner, replacing the standard Steps 1 / 3 / 4 with a single
    TwoPassTextExtractor call.
    """

    enabled: bool = True
    """
    When True, PipelineRunner uses TwoPassTextExtractor instead of the standard
    layout-extract → mask → re-extract sequence (Steps 1, 3, 4).
    Steps 2, 5–8 are unaffected.

    Default flipped to True after empirical verification (May 2026): the
    pixel-render path in PyMuPDFEvidenceGatherer catches Docling phantom
    elements (text content at bboxes that do not render), PDF render-mode-3
    invisible text, and ExtGState fill_opacity=0 ghost layers.  See
    scripts/verify_ghost_text_detection.py and scripts/thesis_demo_ghost_text.py.
    """

    # ── Rendering ─────────────────────────────────────────────────────────────
    render_dpi: int = 150
    """DPI for rendering PDF pages to pixel arrays (higher = slower but more accurate)."""

    # ── Blank-space detection (Rule R1) ───────────────────────────────────────
    blank_brightness_threshold: float = 245.0
    """
    Mean pixel luminance (0–255) above which a region is considered visually
    blank.  255 = pure white; 245 allows for JPEG compression artefacts.
    """

    blank_dark_pixel_max_fraction: float = 0.02
    """
    Maximum fraction of pixels below the ink-darkness threshold that still
    counts as a blank region.  0.02 = at most 2% dark pixels → blank.
    """

    # ── Word-count fallback (Rule R2, render_skipped only) ────────────────────
    min_char_coverage_threshold: float = 0.05
    """
    char_coverage below this value triggers Rule R2 when rendering is skipped.
    char_coverage = fitz_word_chars / len(docling_text); 0.05 means fewer than
    5% of Docling's text characters were found by fitz.
    """

    min_text_chars_for_word_check: int = 8
    """
    Minimum length of the Docling text string (after strip) to apply Rule R2.
    Very short strings (< 8 chars) are ignored to avoid false positives on
    single-word decorative elements.
    """

    # ── Header-zone hint (affects rejection message wording only) ────────────
    max_top_fraction_header: float = 0.15
    """
    Top ``max_top_fraction_header`` fraction of page height is considered the
    "header zone".  Rejected nodes in this zone get a more descriptive reason
    string (e.g. "in header zone").  Does not change the keep/drop outcome.
    """

    # ── Pass-1 figure / table masking ────────────────────────────────────────
    mask_figures: bool = True
    """
    When True, PICTURE/FIGURE bboxes detected in Pass 1 are added to the mask
    before Pass 2 runs.  This prevents figure-interior text (axis labels,
    callouts) from leaking into Pass-2 text elements — mirroring what the
    standard pipeline's region masker does in Step 3.
    """

    mask_tables: bool = True
    """
    When True, TABLE/RECONSTRUCTED_TABLE bboxes detected in Pass 1 are masked
    before Pass 2.  Prevents table cell text from being flattened into body
    paragraphs by Pass-2 Docling.
    """

    # ── White-text ghost layer (Rule R-color) ────────────────────────────────
    max_white_char_fraction: float = 1.0
    """
    Fraction of near-white-colored characters in an element's bbox above which
    the element is classified as a white-text ghost layer and rejected (Rule
    R-color).  Near-white means all RGB channels >= 240.

    Default disabled (1.0) after empirical observation (May 2026): the
    color signal alone produces false positives on legitimate inverted
    headers — e.g. white "Key Points" SECTION_HEADERs printed on a coloured
    banner.  Spans report color=0xffffff (R-color trips) but the rendered
    bbox is full of dark pixels (dark_pixel_fraction ≈ 0.73), so Rule R1
    (visually_blank) correctly keeps them.  R1 is the trustworthy signal;
    leave R-color off unless you have a corpus where R1 is unavailable.
    Set to a value < 1.0 to re-enable.
    """

    # ── Dense-text heuristic (Rule R3) ───────────────────────────────────────
    max_chars_per_bbox_pt: float = 15.0
    """
    Maximum ratio of text character count to bbox height in PDF points.
    Hidden text layers are extremely dense: a 180-char paragraph squeezed into
    a ~10pt-tall bbox yields ~18 chars/pt, far above any real body text
    (typically 3–6 chars/pt).  Elements exceeding this threshold are rejected
    regardless of pixel evidence.
    Set to 0 to disable.
    """

    # ── Body-anchor finding ───────────────────────────────────────────────────
    min_anchor_word_count: int = 5
    """
    Minimum word count for an accepted TEXT/LIST_ITEM element to be eligible
    as the body anchor.  Prevents short running headers that passed scoring
    (real ink, but ≤ 4 words) from becoming the anchor.
    """

    # ── Header-mask geometry ──────────────────────────────────────────────────
    header_mask_margin_pt: float = 3.0
    """
    Gap in PDF points left between the bottom edge of the header mask and the
    top edge of the body anchor.  Prevents the mask from clipping the anchor's
    own ascenders.
    """


@dataclass(slots=True)
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    # docling is used for Step 1 (full layout extraction / table-figure detection).
    # docling_text, if set, overrides docling for Step 4 (masked re-extraction / text assembly).
    # Leave docling_text=None to use the same settings for both steps.
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    docling_text: Optional[DoclingConfig] = None
    tatr: TATRConfig = field(default_factory=TATRConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    cropping: CroppingConfig = field(default_factory=CroppingConfig)
    text: TextAssemblyConfig = field(default_factory=TextAssemblyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    table_detector: TableDetectorType = TableDetectorType.HYBRID
    two_pass: TwoPassConfig = field(default_factory=TwoPassConfig)

    def validate(self) -> None:
        if self.tatr.threshold < 0.0 or self.tatr.threshold > 1.0:
            raise ValueError(f"tatr.threshold must be in [0, 1], got {self.tatr.threshold}")

        if self.cropping.dpi <= 0:
            raise ValueError(f"cropping.dpi must be > 0, got {self.cropping.dpi}")

        if self.runtime.num_workers < 1:
            raise ValueError(f"runtime.num_workers must be >= 1, got {self.runtime.num_workers}")

        if self.database.enabled and not self.database.db_url:
            try:
                from database.db_connection import get_database_url  # type: ignore
                self.database.db_url = get_database_url()
            except Exception:
                raise ValueError(
                    "database.enabled=True but database.db_url is not set "
                    "and no .env / environment variables found"
                )

    def prepare(self) -> None:
        self.validate()
        self.paths.ensure_dirs()