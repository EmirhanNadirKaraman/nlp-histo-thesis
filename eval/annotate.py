"""
eval/annotate.py — Terminal annotator for paragraph and media evaluation.

Usage:
    python eval/annotate.py text                  # eval/out/text/     (processed paragraphs)
    python eval/annotate.py text_raw              # eval/out/text_raw/  (raw Docling elements)
    python eval/annotate.py docling_full          # all raw Docling elements before pipeline (recall labeling)
    python eval/annotate.py json_figures          # figures only, from full variant (same across all)
    python eval/annotate.py json_tables_full      # tables from hybrid (Docling + TATR) detector
    python eval/annotate.py json_tables_docling   # tables from Docling-only detector
    python eval/annotate.py json_tables_docling_recon  # tables from Docling recon detector

    # Read crops directly from a sweep dir (no eval/out/ symlinking required) and
    # write per-variant labels.  Pair with `scripts/eval/build_share_map.py` to
    # auto-propagate labels for shared crops to peer-variant annotation files.
    python eval/annotate.py json_tables_full \\
        --sweep out/sweeps/baseline \\
        --variant baseline

Keys:
    y / →    Correct
    n / ←    Incorrect
    s        Skip
    b        Back one item
    r        Show metrics so far
    q        Quit and save

Default annotation file:  eval/annotations/annotations_{mode}.json
With --variant <name>:    eval/annotations/<name>/{mode}.json (and propagation
                          to peer variants that share the same crop, per
                          eval/annotations/share_map.json).
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import termios
import textwrap
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

HERE     = Path(__file__).parent
OUT_DIR  = HERE / "out"
ANN_DIR  = HERE / "annotations"

MAX_PER_DOC = 200   # skip documents with more than this many items (0 = no limit)

TERM_WIDTH = 82
WRAP_WIDTH = TERM_WIDTH - 4

# ── ANSI helpers ───────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN    = "\033[36m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
def CLEAR() -> None:
    os.system("clear")


def c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# ── Item dataclass (shared across all modes) ───────────────────────────────────
@dataclass
class Item:
    doc_id:  str
    label:   str          # section name, "figure", "table", etc.
    index:   int          # global index
    text:    str          # main display text
    meta:    dict = field(default_factory=dict)  # extra fields (page, image_path, …)


# ── Parsers ────────────────────────────────────────────────────────────────────
def parse_text(path: Path) -> list[Item]:
    """Parse eval/out/text/*_text.txt — [Section] headers + plain paragraphs."""
    doc_id  = path.stem.replace("_text", "")
    section = ""
    items   = []

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1]
        elif set(s) <= {"=", "-"} and len(s) > 4:
            pass
        elif s.startswith("Document:"):
            pass
        elif s:
            items.append(Item(doc_id=doc_id, label=section, index=0, text=s))

    return items


def parse_text_raw(path: Path) -> list[Item]:
    """Parse eval/out/text_raw/*_raw.txt — [SECTION_HEADER] / [TEXT] tags."""
    doc_id  = path.stem.replace("_raw", "")
    section = ""
    items   = []

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[SECTION_HEADER]"):
            section = s[len("[SECTION_HEADER]"):].strip()
        elif s.startswith("[TEXT]"):
            text = s[len("[TEXT]"):].strip()
            if text:
                items.append(Item(doc_id=doc_id, label=section, index=0, text=text))

    return items


def parse_docling_full(path: Path) -> list[Item]:
    """Parse eval/out/docling_full/*_layout.json — all raw Docling elements."""
    data   = json.loads(path.read_text(encoding="utf-8"))
    pmcid  = path.stem.replace("_layout", "")
    items: list[Item] = []

    for el in data.get("elements", []):
        el_type = el.get("type", "UNKNOWN")
        text    = (el.get("text") or "").strip()
        if not text:
            continue
        page    = el.get("page", "?")
        bbox    = el.get("bbox", {})
        x1      = bbox.get("x1", 0)
        y1      = bbox.get("y1", 0)
        key     = f"p{page}_{el_type}_{x1:.1f}_{y1:.1f}"
        items.append(
            Item(
                doc_id=pmcid,
                label=el_type,
                index=0,
                text=text,
                meta={
                    "page":           page,
                    "detected_label": key,
                },
            )
        )

    return items


def parse_json(path: Path, only: str | None = None) -> list[Item]:
    """Parse eval/out/json/*_media.json — figure and table detections.

    Args:
        only: ``"figures"`` or ``"tables"`` to restrict output; ``None`` for both.
    """
    data  = json.loads(path.read_text(encoding="utf-8"))
    pmcid = data["pmcid"]
    items: list[Item] = []

    kinds = ("figures", "tables") if only is None else (only,)
    for kind in kinds:
        item_type = kind.rstrip("s")  # "figure" / "table"
        for entry in data.get(kind, []):
            caption = (entry.get("caption") or "").replace("\n", " ")
            items.append(
                Item(
                    doc_id=pmcid,
                    label=item_type,
                    index=0,
                    text=caption or "(no caption)",
                    meta={
                        "detected_label": entry.get("label", ""),
                        "page":           entry.get("page", ""),
                        "image_path":     entry.get("image_path", ""),
                        # bbox in Docling coords (y=0 at bottom).  Used by the
                        # annotator's [p] keystroke to draw a clean overlay.
                        "bbox":           entry.get("bbox", {}),
                    },
                )
            )

    return items


def _loader_for_mode(mode: str, base: Path) -> tuple[Path, str, callable]:
    """Resolve (input_folder, glob, parser) for ``mode`` against a base dir.

    When ``base`` is ``OUT_DIR`` this preserves the historical layout
    (``eval/out/json/full`` etc.).  When ``base`` is a sweep root such as
    ``out/sweeps/baseline`` we flatten the table-variant layout to a single
    ``<base>/json`` directory — each sweep writes one media.json set already.
    """
    from functools import partial
    if base == OUT_DIR:
        return {
            "text":                       (base / "text",                   "*_text.txt",   parse_text),
            "text_raw":                   (base / "text_raw",               "*_raw.txt",    parse_text_raw),
            "docling_full":               (base / "docling_full",           "*_layout.json", parse_docling_full),
            "json_figures":               (base / "json" / "full",          "*_media.json", partial(parse_json, only="figures")),
            "json_tables_full":           (base / "json" / "full",          "*_media.json", partial(parse_json, only="tables")),
            "json_tables_docling":        (base / "json" / "docling",       "*_media.json", partial(parse_json, only="tables")),
            "json_tables_docling_recon":  (base / "json" / "docling_recon", "*_media.json", partial(parse_json, only="tables")),
        }[mode]
    # Sweep mode: each sweep dir produces ONE json/ + one figures/+tables/.
    # All json_tables_* modes resolve to the same dir (the sweep emits a
    # single media.json set per document); the caller picks the appropriate
    # mode based on which detector the sweep ran with.
    return {
        "text":                       (base / "text",          "*_text.txt",    parse_text),
        "text_raw":                   (base / "text_raw",      "*_raw.txt",     parse_text_raw),
        "docling_full":               (base / "docling_full",  "*_layout.json", parse_docling_full),
        "json_figures":               (base / "json",          "*_media.json",  partial(parse_json, only="figures")),
        "json_tables_full":           (base / "json",          "*_media.json",  partial(parse_json, only="tables")),
        "json_tables_docling":        (base / "json",          "*_media.json",  partial(parse_json, only="tables")),
        "json_tables_docling_recon":  (base / "json",          "*_media.json",  partial(parse_json, only="tables")),
    }[mode]


def load_items(mode: str, max_per_doc: int = 0, sweep_dir: Path | None = None) -> list[Item]:
    folder, glob, parser = _loader_for_mode(mode, sweep_dir or OUT_DIR)
    all_items: list[Item] = []
    skipped_docs: list[str] = []
    for f in sorted(folder.glob(glob)):
        doc_items = parser(f)
        if max_per_doc and len(doc_items) > max_per_doc:
            skipped_docs.append(f"{f.stem} ({len(doc_items)} items)")
            continue
        all_items.extend(doc_items)
    if skipped_docs:
        print(f"Skipped {len(skipped_docs)} docs exceeding --max-per-doc {max_per_doc}:")
        for name in skipped_docs:
            print(f"  {name}")
        print()
    for i, item in enumerate(all_items):
        item.index = i
    return all_items


# ── Annotation persistence ─────────────────────────────────────────────────────
SHARE_MAP_PATH = ANN_DIR / "share_map.json"


def ann_path(mode: str, variant: str | None = None) -> Path:
    if variant:
        return ANN_DIR / variant / f"{mode}.json"
    return ANN_DIR / f"annotations_{mode}.json"


def load_annotations(mode: str, variant: str | None = None) -> dict[str, str]:
    p = ann_path(mode, variant)
    return json.loads(p.read_text()) if p.exists() else {}


def _atomic_write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def save_annotations(mode: str, ann: dict[str, str], variant: str | None = None) -> None:
    _atomic_write(ann_path(mode, variant), ann)


def load_share_map() -> dict[str, list[dict] | list[str]]:
    """Load the share map.  Accepts both the new (bbox-aware) and legacy
    (filename-only) formats — callers route on shape at use-time."""
    if SHARE_MAP_PATH.exists():
        try:
            return json.loads(SHARE_MAP_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


# Bbox quantization tolerance used when looking up the matching group.
# Must match the value used by build_share_map.py.
_SHARE_MAP_BBOX_TOL = 1.0


def _quantize_bbox(bbox: dict | None, tol: float = _SHARE_MAP_BBOX_TOL) -> tuple:
    """Quantize the source crop's bbox to the same tolerance used by
    build_share_map.py.  Used to identify which bbox-group the current
    label belongs to during propagation."""
    bbox = bbox or {}
    def q(v):
        try:
            return round(float(v) / tol) * tol
        except (TypeError, ValueError):
            return 0.0
    return (q(bbox.get("x1")), q(bbox.get("y1")),
            q(bbox.get("x2")), q(bbox.get("y2")))


def _find_peers(
    share_map_entry,
    source_variant: str,
    source_bbox: dict | None,
) -> list[str]:
    """Resolve peers for a single share-map entry.

    Handles three input shapes:
      1. List of {"bbox": [...], "variants": [...]} — NEW bbox-aware format.
         Pick the group whose bbox matches ``source_bbox``; return peers.
      2. Flat list of variant names — LEGACY format.  No bbox awareness;
         return all peers minus source.
      3. None / empty — return [].
    """
    if not share_map_entry:
        return []
    # New format: list of {bbox, variants} dicts
    if isinstance(share_map_entry, list) and share_map_entry and isinstance(share_map_entry[0], dict):
        src_key = _quantize_bbox(source_bbox)
        for group in share_map_entry:
            grp_key = tuple(group.get("bbox") or ())
            if grp_key == src_key:
                return [v for v in group.get("variants", []) if v != source_variant]
        return []  # source bbox doesn't match any group → don't propagate
    # Legacy format: flat list of variant names
    if isinstance(share_map_entry, list):
        return [v for v in share_map_entry if v != source_variant]
    return []


def propagate_label(
    key: str,
    label: str,
    mode: str,
    *,
    source_variant: str,
    share_map: dict,
    source_bbox: dict | None = None,
) -> list[str]:
    """Mirror ``key=label`` to peer-variant annotation files whose crop bbox
    matches the source's bbox (per the bbox-aware share map).  Returns the
    list of peer variants written.

    With the legacy filename-only share-map format, propagates to every
    peer that emitted ``key``, regardless of bbox — caller should pass
    ``source_bbox=None`` to opt into that behaviour.

    Best-effort — a single peer-write failure does not raise.
    """
    peers = _find_peers(share_map.get(key), source_variant, source_bbox)
    written: list[str] = []
    for peer in peers:
        peer_path = ann_path(mode, peer)
        try:
            peer_ann = json.loads(peer_path.read_text()) if peer_path.exists() else {}
            peer_ann[key] = label
            _atomic_write(peer_path, peer_ann)
            written.append(peer)
        except Exception as exc:
            print(c(YELLOW, f"  (propagation to {peer} failed: {exc!r})"))
    return written


def ann_key(item: Item) -> str:
    image_path = item.meta.get("image_path", "").strip()
    if image_path:
        return Path(image_path).name
    detected = item.meta.get("detected_label", "").strip()
    if detected:
        return f"{item.doc_id}::{detected}"
    return f"{item.doc_id}::{item.label}::{item.index}"


# ── Input helpers ──────────────────────────────────────────────────────────────

_STANDARD_LABELS = frozenset({"correct", "incorrect", "skipped"})
_RUBRIC_PATH = HERE / "label_rubric.yaml"


def _load_rubric_labels(
    path: Path | None = None,
    item_kind: str | None = None,
) -> list[str]:
    """Return rubric-defined label strings (sorted) for the given item kind.

    Reads the 3-block YAML format (2026-05-19):
      * ``shared_labels`` — always returned.
      * ``figure_labels`` — returned when ``item_kind == "figure"`` or None.
      * ``table_labels``  — returned when ``item_kind == "table"``  or None.

    Falls back to the legacy single ``labels:`` block if no kind-specific
    blocks are found (returns everything regardless of kind).  Returns
    ``[]`` on any failure (missing file, no yaml, malformed) — the
    annotator falls back to the recent-only behaviour transparently.
    """
    if path is None:
        path = _RUBRIC_PATH
    if not path.exists():
        return []
    try:
        import yaml  # type: ignore
    except ImportError:
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    data = data or {}

    has_kind_blocks = any(
        block in data
        for block in ("shared_labels", "figure_labels", "table_labels")
    )
    if not has_kind_blocks:
        # Legacy single-block format — return everything.
        labels = data.get("labels", {}) or {}
        return sorted({
            str(k) for k in labels.keys() if str(k) not in _STANDARD_LABELS
        })

    collected: set[str] = set()
    collected.update(data.get("shared_labels", {}) or {})
    if item_kind in ("figure", None):
        collected.update(data.get("figure_labels", {}) or {})
    if item_kind in ("table", None):
        collected.update(data.get("table_labels", {}) or {})

    return sorted({
        str(k) for k in collected if str(k) not in _STANDARD_LABELS
    })


def _collect_label_menu(ann: dict[str, str],
                         item_kind: str | None = None) -> list[str]:
    """Return the numbered-menu list for the ``[l]`` keystroke.

    Sources:
      1. Rubric labels for the current kind (shared + figure-only or
         shared + table-only, based on ``item_kind``).
      2. Any custom label already in ``ann`` (in case the user typed
         something outside the rubric earlier) — included regardless of
         kind, since the user has already used it once.

    Deduplicated; rubric order preserved, with extras appended.
    """
    rubric_labels = _load_rubric_labels(item_kind=item_kind)
    recent = sorted({v for v in ann.values() if v and v not in _STANDARD_LABELS})
    seen = set(rubric_labels)
    extras = [v for v in recent if v not in seen]
    return rubric_labels + extras


def _collect_recent_labels(ann: dict[str, str]) -> list[str]:
    """Return sorted list of non-standard custom labels seen so far in
    ``ann``.  Kept separate from ``_collect_label_menu`` so callers that
    want only ann-derived labels (tests, downstream tooling) aren't
    forced to filter rubric entries back out.
    """
    return sorted({v for v in ann.values() if v and v not in _STANDARD_LABELS})


def _next_unlabelled_index(items: list, ann: dict[str, str], start: int) -> int:
    """First index ``>= start`` whose ``ann_key(item)`` is NOT in ``ann``.

    Returns ``len(items)`` if no such item exists — the caller treats that
    as "we're done with this pass".
    """
    for i in range(start, len(items)):
        if ann_key(items[i]) not in ann:
            return i
    return len(items)


def read_label(prompt: str = "  Label: ", *,
               recent: list[str] | None = None) -> str | None:
    """Restore normal terminal mode, prompt for a label, return it.

    Returns ``None`` if the user cancels (empty input or Ctrl-C / Ctrl-D).
    If ``recent`` is provided, prints a numbered menu first so the user can
    type a digit to reuse a previously-seen label instead of retyping.
    """
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)  # ensure normal mode
        print()
        if recent:
            print(f"  {c(CYAN, 'Recent labels (type # to reuse):')}")
            for i, label in enumerate(recent, 1):
                print(f"    {c(YELLOW, str(i)):>4}  {label}")
        print(f"  {c(DIM, '(empty to cancel)')}")
        print(prompt, end="", flush=True)
        try:
            raw = input()
        except (EOFError, KeyboardInterrupt):
            print()  # newline so the next render() doesn't collide
            return None
        raw = raw.strip()
        if not raw:
            return None
        if recent and raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(recent):
                return recent[idx - 1]
        return raw
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def getch() -> str:
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == "[":
                if ch3 == "C":
                    return "RIGHT"
                if ch3 == "D":
                    return "LEFT"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── PDF source resolution ──────────────────────────────────────────────────────

def resolve_pdf_dir(sweep_dir: Path | None, explicit: Path | None) -> Path | None:
    """Resolve the source-PDF directory.

    Precedence: explicit ``--pdf-dir`` > the latest manifest's
    ``input.pdf_dir`` under ``<sweep>/run_metadata/`` > ``None``.
    """
    if explicit is not None:
        return explicit
    if sweep_dir is None:
        return None
    manifests = sorted((sweep_dir / "run_metadata").glob("run_*.json"))
    if not manifests:
        return None
    try:
        data = json.loads(manifests[-1].read_text())
    except Exception:
        return None
    pdf_dir = (data.get("input") or {}).get("pdf_dir")
    if not pdf_dir:
        return None
    p = Path(pdf_dir)
    if not p.is_absolute():
        # Manifests usually carry repo-relative paths.  Resolve against repo root.
        p = (HERE.parent / p) if not p.exists() else p
    return p


def resolve_pdf_path(doc_id: str, pdf_dir: Path | None) -> Path | None:
    """Best-effort source PDF for a labelled item (``<pdf_dir>/<doc_id>.pdf``)."""
    if pdf_dir is None:
        return None
    candidate = pdf_dir / f"{doc_id}.pdf"
    return candidate


def resolve_visualization_path(sweep_dir: Path | None, doc_id: str) -> Path | None:
    """Locate the annotated visualization PDF for a doc inside a sweep.

    The visualizer writes ``<sweep>/visualization/{pmcid}_layout_vis.pdf`` with
    coloured bboxes for every detected element (TEXT / FIGURE / TABLE /
    CAPTION / etc.) overlaid on the source pages.  Returns ``None`` if the
    file doesn't exist (e.g. visualization was disabled for that sweep).
    """
    if sweep_dir is None:
        return None
    candidate = sweep_dir / "visualization" / f"{doc_id}_layout_vis.pdf"
    return candidate if candidate.exists() else None


def render_clean_overlay(
    source_pdf: Path,
    bbox_docling: dict,
    page: int,
    out_path: Path,
    *,
    color: tuple[float, float, float] = (0, 0.8, 0),
    width: float = 2.5,
) -> bool:
    """Write a single-bbox overlay PDF: copy of ``source_pdf`` with one
    outline drawn at ``bbox_docling`` on page ``page``.

    Coordinate conversion: media.json bboxes are Docling coords (y=0 at
    bottom of page).  fitz uses y=0 at top.  We flip via ``page.rect.height``.

    Returns ``True`` on success, ``False`` if the input is missing / bbox is
    incomplete / fitz isn't available.
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        return False
    if not source_pdf or not source_pdf.exists():
        return False
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(bbox_docling):
        return False
    try:
        doc = fitz.open(str(source_pdf))
        if not (1 <= page <= len(doc)):
            doc.close()
            return False
        p = doc[page - 1]
        h = p.rect.height
        # Docling y is bottom-origin: convert to fitz top-origin.
        rect = fitz.Rect(
            bbox_docling["x1"], h - bbox_docling["y1"],
            bbox_docling["x2"], h - bbox_docling["y2"],
        )
        rect.normalize()
        p.draw_rect(rect, color=color, width=width)  # outline only — no fill
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
        doc.close()
        return True
    except Exception:
        return False


def open_pdf(pdf_path: Path, page: int | None = None) -> None:
    """Open a PDF in the default viewer.  On macOS optionally navigates to
    ``page`` via Preview's "Go to Page" keyboard shortcut (Cmd-Opt-G).

    Requires Accessibility permission for the shell to control "System
    Events" — macOS will prompt on first use.  If permission is denied or
    the platform isn't macOS, falls back to opening the PDF at page 1.
    """
    if not pdf_path or not pdf_path.exists():
        return
    try:
        os.system(f"open {shlex.quote(str(pdf_path))}")
        if sys.platform == "darwin" and page and page > 1:
            # Give Preview ~0.6s to load before sending keystrokes
            time.sleep(0.6)
            script = (
                'tell application "System Events" to tell process "Preview"\n'
                "set frontmost to true\n"
                'keystroke "g" using {command down, option down}\n'
                "delay 0.2\n"
                f'keystroke "{page}"\n'
                "delay 0.1\n"
                "keystroke return\n"
                "end tell"
            )
            os.system(f"osascript -e {shlex.quote(script)}")
    except Exception:
        pass


# ── Display ────────────────────────────────────────────────────────────────────
def render(item: Item, items: list[Item], ann: dict[str, str], mode: str,
           pdf_dir: Path | None = None,
           sweep_dir: Path | None = None) -> None:
    _known       = {"correct", "incorrect", "skipped"}
    total        = len(items)
    # Count only annotations that belong to items in the current items list.
    # ``ann`` may contain extra keys propagated in from peer variants whose
    # crops this variant doesn't emit — counting those inflates the bar past
    # 100 %.  Filter by the current items' ann_key set.
    in_scope     = {ann_key(it) for it in items}
    scoped       = {k: v for k, v in ann.items() if k in in_scope}
    n_correct    = sum(1 for v in scoped.values() if v == "correct")
    n_incorrect  = sum(1 for v in scoped.values() if v == "incorrect")
    n_skipped    = sum(1 for v in scoped.values() if v == "skipped")
    n_custom     = sum(1 for v in scoped.values() if v not in _known)
    annotated    = n_correct + n_incorrect + n_custom
    pct          = annotated / total * 100 if total else 0

    CLEAR()
    print(c(BOLD, "─" * TERM_WIDTH))

    # Progress bar
    bar_width = TERM_WIDTH - 32
    filled = int(bar_width * annotated / total) if total else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(
        f"  {c(CYAN, f'{annotated}/{total}')}  [{c(GREEN, bar)}]  {pct:.1f}%  "
        f"{c(GREEN, f'✓{n_correct}')}  {c(RED, f'✗{n_incorrect}')}  "
        f"{c(CYAN, f'#{n_custom}')}  {c(DIM, f'~{n_skipped}')}"
    )
    print(c(BOLD, "─" * TERM_WIDTH))

    # Item metadata
    print(f"  {c(DIM, 'mode')}    {c(YELLOW, mode)}")
    print(f"  {c(DIM, 'doc')}     {c(BOLD, item.doc_id)}")
    pdf_path = resolve_pdf_path(item.doc_id, pdf_dir)
    if pdf_path is not None:
        marker = "" if pdf_path.exists() else c(YELLOW, "  (not on disk)")
        print(f"  {c(DIM, 'pdf')}     {c(DIM, str(pdf_path))}{marker}")
    page = item.meta.get("page") if item.meta else None
    if page:
        bbox = item.meta.get("bbox", {}) if item.meta else {}
        bbox_summary = ""
        if bbox.get("x1") is not None:
            bbox_summary = f"  bbox=({bbox.get('x1', 0):.0f},{bbox.get('y1', 0):.0f}-{bbox.get('x2', 0):.0f},{bbox.get('y2', 0):.0f})"
        print(f"  {c(DIM, 'page')}    {c(GREEN, f'{page}')}{c(DIM, bbox_summary)}")
    print(f"  {c(DIM, 'label')}   {c(CYAN, item.label or '(top level)')}")
    if item.meta:
        if item.meta.get("detected_label"):
            print(f"  {c(DIM, 'item')}    {item.meta['detected_label']}  "
                  f"(page {item.meta.get('page', '?')})")
        if item.meta.get("image_path"):
            print(f"  {c(DIM, 'image')}   {c(DIM, item.meta['image_path'])}")
    print(f"  {c(DIM, '#')}       {item.index + 1} / {total}")
    print(c(BOLD, "─" * TERM_WIDTH))
    print()

    # Main text (wrapped)
    clean = item.text.replace("\r", " ")
    for line in textwrap.fill(clean, width=WRAP_WIDTH).splitlines():
        print(f"    {line}")

    print()
    print(c(BOLD, "─" * TERM_WIDTH))

    # Current annotation
    key = ann_key(item)
    if key in ann:
        val    = ann[key]
        colour = GREEN if val == "correct" else (RED if val == "incorrect" else (YELLOW if val == "skipped" else CYAN))
        print(f"  current: {c(colour, val)}")
    else:
        print(f"  {c(DIM, 'not yet annotated')}")

    print()
    print(
        f"  {c(GREEN, '[y/→]')} correct   "
        f"{c(RED, '[n/←]')} incorrect   "
        f"{c(CYAN, '[l]')} label/pick   "
        f"{c(YELLOW, '[s]')} skip   "
        f"{c(BLUE, '[b]')} back   "
        f"{c(DIM, '[space]')} next unlabelled   "
        f"{c(DIM, '[f/tab]')} forward 1   "
        f"{c(DIM, '[p]')} source+bbox@page   "
        f"{c(DIM, '[r]')} metrics   "
        f"{c(DIM, '[q]')} quit"
    )
    print(c(BOLD, "─" * TERM_WIDTH))


def show_metrics(items: list[Item], ann: dict[str, str], mode: str) -> None:
    known     = {"correct", "incorrect", "skipped"}
    # Restrict counting to items currently in scope; peer-variant
    # propagation can leave extra keys in ``ann`` that don't belong here.
    in_scope    = {ann_key(it) for it in items}
    scoped      = {k: v for k, v in ann.items() if k in in_scope}
    n_correct   = sum(1 for v in scoped.values() if v == "correct")
    n_incorrect = sum(1 for v in scoped.values() if v == "incorrect")
    custom_labels = {v: sum(1 for x in scoped.values() if x == v)
                     for v in scoped.values() if v not in known}
    n_custom    = sum(custom_labels.values())
    annotated   = n_correct + n_incorrect + n_custom
    total       = len(items)
    accuracy    = n_correct / annotated if annotated else 0.0

    CLEAR()
    print(c(BOLD, "─" * TERM_WIDTH))
    print(c(BOLD, f"  METRICS — mode: {mode}"))
    print(c(BOLD, "─" * TERM_WIDTH))
    print(f"  Total items   : {total}")
    print(f"  Annotated     : {annotated}")
    print(f"  Correct       : {c(GREEN,   str(n_correct))}")
    print(f"  Incorrect     : {c(RED,     str(n_incorrect))}")
    for lbl, cnt in sorted(custom_labels.items()):
        print(f"  {lbl:<14}: {c(CYAN, str(cnt))}")
    print(f"  Accuracy      : {c(BOLD, f'{accuracy:.1%}')}")
    print()

    # Per-document breakdown
    by_doc: dict[str, dict[str, int]] = {}
    for item in items:
        key = ann_key(item)
        if key not in ann:
            continue
        by_doc.setdefault(item.doc_id, {"correct": 0, "incorrect": 0})
        by_doc[item.doc_id][ann[key]] = by_doc[item.doc_id].get(ann[key], 0) + 1

    if by_doc:
        print(c(BOLD, "  Per-document accuracy:"))
        for doc, counts in sorted(by_doc.items()):
            tot = counts.get("correct", 0) + counts.get("incorrect", 0)
            acc = counts.get("correct", 0) / tot if tot else 0
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {doc:<52} [{bar}] {acc:.0%}")

    print(c(BOLD, "─" * TERM_WIDTH))
    print(f"  {c(DIM, 'Press any key to continue...')}")
    getch()


# ── Main loop ──────────────────────────────────────────────────────────────────
def _parse_cli(argv: list[str] | None = None):
    import argparse
    VALID_MODES = (
        "text", "text_raw", "docling_full",
        "json_figures",
        "json_tables_full", "json_tables_docling", "json_tables_docling_recon",
    )
    p = argparse.ArgumentParser(
        description="Terminal annotator for paragraph / media evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("mode", choices=VALID_MODES,
                   help="What to annotate.")
    p.add_argument("--sweep", type=Path, default=None,
                   help="Read crops directly from this sweep directory "
                        "(e.g. out/sweeps/baseline).  Skips the eval/out/ "
                        "symlink layout.")
    p.add_argument("--variant", default=None,
                   help="Write labels to eval/annotations/<variant>/<mode>.json "
                        "and propagate shared crops via eval/annotations/share_map.json.")
    p.add_argument("--pdf-dir", type=Path, default=None,
                   help="Source PDF directory.  When set, the annotator shows the "
                        "resolved <pdf-dir>/<doc>.pdf path for each item and the "
                        "[p] key opens it in the default viewer.  When omitted, "
                        "auto-resolves from the sweep manifest if --sweep is set.")
    p.add_argument("--no-propagate", action="store_true",
                   help="Disable share-map propagation entirely.  Labels are "
                        "written only to the current variant's annotation file; "
                        "peer-variant files are never touched.  Use for variants "
                        "with extent-sensitive bboxes (footnote-expansion / "
                        "relaxed-threshold sweeps) when you do not trust the "
                        "share map's bbox-grouping for cross-variant equivalence.")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_cli()
    mode = args.mode
    sweep_dir = args.sweep
    variant = args.variant
    pdf_dir = resolve_pdf_dir(sweep_dir, args.pdf_dir)

    if variant and not sweep_dir:
        # Variant without sweep is allowed (legacy eval/out/ layout) — warn so
        # the user knows propagation still works but the input dir is the
        # symlinked eval/out/ tree.
        print(c(YELLOW, f"  (--variant set without --sweep — reading from {OUT_DIR})"))

    items = load_items(mode, max_per_doc=MAX_PER_DOC, sweep_dir=sweep_dir)
    ann   = load_annotations(mode, variant=variant)
    total = len(items)
    no_propagate = bool(args.no_propagate)
    share_map = load_share_map() if (variant and not no_propagate) else {}
    if variant and no_propagate:
        print(c(YELLOW,
                "  (--no-propagate set — labels stay local to this variant)"))
    elif variant and not share_map:
        print(c(YELLOW,
                "  (no share_map.json — peer-variant propagation disabled; "
                "run scripts/eval/build_share_map.py first)"))

    if not items:
        loader_root = sweep_dir or OUT_DIR
        print(f"No items found for mode '{mode}' under {loader_root}")
        sys.exit(1)

    def _set(it: Item, label: str) -> None:
        """Set a label, save to the variant file, propagate to peer variants
        whose bbox matches this crop's bbox (when share_map is bbox-aware)."""
        k = ann_key(it)
        ann[k] = label
        save_annotations(mode, ann, variant=variant)
        if variant and share_map:
            bbox = it.meta.get("bbox") if it.meta else None
            propagate_label(k, label, mode, source_variant=variant,
                            share_map=share_map, source_bbox=bbox)

    # Resume from first un-annotated item
    cursor = next(
        (i for i, item in enumerate(items) if ann_key(item) not in ann),
        0,
    )

    while 0 <= cursor < total:
        item = items[cursor]
        render(item, items, ann, mode, pdf_dir=pdf_dir, sweep_dir=sweep_dir)
        key = getch()

        if key in ("y", "Y", "RIGHT"):
            _set(item, "correct")
            cursor = _next_unlabelled_index(items, ann, cursor + 1)
        elif key in ("n", "N", "LEFT"):
            _set(item, "incorrect")
            cursor = _next_unlabelled_index(items, ann, cursor + 1)
        elif key in ("l", "L"):
            label = read_label(recent=_collect_label_menu(ann, item_kind=item.label))
            if label:
                _set(item, label)
                cursor = _next_unlabelled_index(items, ann, cursor + 1)
            # else: empty / Ctrl-C → stay on current item
        elif key in ("s", "S"):
            _set(item, "skipped")
            cursor = _next_unlabelled_index(items, ann, cursor + 1)
        elif key == " ":
            cursor = _next_unlabelled_index(items, ann, cursor + 1)
        elif key in ("f", "F", "\t"):
            # Forward one item, INCLUDING already-labelled crops.  Distinct
            # from [space] which skips ahead to the next unlabelled item.
            cursor = min(total - 1, cursor + 1)
        elif key in ("b", "B"):
            cursor = max(0, cursor - 1)
        elif key in ("p", "P"):
            page = None
            try:
                page = int(item.meta.get("page")) if item.meta.get("page") else None
            except (TypeError, ValueError):
                page = None
            source = resolve_pdf_path(item.doc_id, pdf_dir)
            bbox = item.meta.get("bbox", {}) if item.meta else {}
            # Clean on-the-fly overlay: source PDF + just THIS crop's bbox
            # outlined.  No fill, no other layout boxes, no masking-look.
            # Falls back to raw source if anything goes wrong.
            target = source
            if source and page and bbox:
                tmp = OUT_DIR / "_overlay" / f"{item.doc_id}_p{page}_{item.index}.pdf"
                if render_clean_overlay(source, bbox, page, tmp):
                    target = tmp
            if target:
                open_pdf(target, page=page)
        elif key in ("r", "R"):
            show_metrics(items, ann, mode)
        elif key in ("q", "Q", "\x03"):
            save_annotations(mode, ann, variant=variant)
            show_metrics(items, ann, mode)
            print(f"\n  Saved to {ann_path(mode, variant)}\n")
            return

    # ── Second pass: revisit skipped items ────────────────────────────────────
    skipped_indices = [i for i, it in enumerate(items)
                       if ann.get(ann_key(it)) == "skipped"]
    if skipped_indices:
        CLEAR()
        print(f"\n  {len(skipped_indices)} skipped item(s) to revisit. Press any key...")
        getch()
        sk = 0
        while 0 <= sk < len(skipped_indices):
            item = items[skipped_indices[sk]]
            render(item, items, ann, mode, pdf_dir=pdf_dir, sweep_dir=sweep_dir)
            key = getch()
            if key in ("y", "Y", "RIGHT"):
                _set(item, "correct"); sk += 1
            elif key in ("n", "N", "LEFT"):
                _set(item, "incorrect"); sk += 1
            elif key in ("s", "S"):
                sk += 1  # keep as skipped, move on
            elif key == " ":
                sk += 1
            elif key in ("b", "B"):
                sk = max(0, sk - 1)
            elif key in ("p", "P"):
                page = None
                try:
                    page = int(item.meta.get("page")) if item.meta.get("page") else None
                except (TypeError, ValueError):
                    page = None
                source = resolve_pdf_path(item.doc_id, pdf_dir)
                bbox = item.meta.get("bbox", {}) if item.meta else {}
                target = source
                if source and page and bbox:
                    tmp = OUT_DIR / "_overlay" / f"{item.doc_id}_p{page}_{item.index}.pdf"
                    if render_clean_overlay(source, bbox, page, tmp):
                        target = tmp
                if target:
                    open_pdf(target, page=page)
            elif key in ("r", "R"):
                show_metrics(items, ann, mode)
            elif key in ("q", "Q", "\x03"):
                save_annotations(mode, ann, variant=variant)
                show_metrics(items, ann, mode)
                print(f"\n  Saved to {ann_path(mode, variant)}\n")
                return

    save_annotations(mode, ann, variant=variant)
    show_metrics(items, ann, mode)
    print(f"\n  All items reviewed! Saved to {ann_path(mode, variant)}\n")


if __name__ == "__main__":
    main()
