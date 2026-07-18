"""
eval/ground_truth.py — Interactively record the number of missed tables,
missed figures, and total tables in each PDF, saving results to
eval/ground_truth.csv.

Runs in sequential passes:
  Pass 1 — missed_figures  (opens detected figure crops in Preview for reference)
  Pass 2 — missed_tables   (opens detected table crops in Preview for reference)
  Pass 3 — total_tables    (opens detected table crops in Preview for reference)

Usage:
    python eval/ground_truth.py [figures|tables|total]   # single pass
    python eval/ground_truth.py                          # all passes in order

Keys:
    <number> + Return   Record count and advance.
    b                   Go back to the previous PDF.
    s                   Skip this PDF.
    q                   Quit and save progress.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

HERE       = Path(__file__).parent
PDF_DIR    = HERE / "pdfs"
CSV_PATH   = HERE / "ground_truth.csv"
FIGURES_DIR = HERE / "out" / "figures"
TABLES_DIR  = HERE / "out" / "tables" / "full"   # canonical detector for ground truth

FIELDNAMES = ["pmcid", "missed_figures", "missed_tables", "total_tables"]

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"


def CLEAR() -> None:
    os.system("clear")


def c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# CSV helpers

def load_csv() -> dict[str, dict]:
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return {row["pmcid"]: row for row in csv.DictReader(f)}


def save_csv(records: dict[str, dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in records.values():
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


# Image helpers

def images_for(pmcid: str, kind: str) -> list[Path]:
    """Return sorted image paths for this pmcid and kind."""
    if kind == "missed_figures":
        return sorted(FIGURES_DIR.glob(f"{pmcid}_Figure_*.png"))
    return sorted(TABLES_DIR.glob(f"{pmcid}_Table_*.png"))


def open_images(paths: list[Path]) -> None:
    """Open images in Preview (macOS). No-op if list is empty."""
    if paths:
        subprocess.Popen(["open"] + [str(p) for p in paths])


# Display

_KIND_LABEL = {
    "missed_figures": "MISSED FIGURES",
    "missed_tables":  "MISSED TABLES",
    "total_tables":   "TOTAL TABLES",
}
_KIND_NOUN = {
    "missed_figures": "missed figures",
    "missed_tables":  "missed tables",
    "total_tables":   "total tables",
}
_KIND_MEDIA_DIR = {
    "missed_figures": FIGURES_DIR,
    "missed_tables":  TABLES_DIR,
    "total_tables":   TABLES_DIR,
}


def render(pmcid: str, idx: int, total: int, records: dict,
           kind: str, images: list[Path]) -> None:
    done = sum(
        1 for r in records.values()
        if r.get(kind, "") != ""
    )
    CLEAR()
    print(c(BOLD, "─" * 64))
    label     = _KIND_LABEL[kind]
    media_dir = _KIND_MEDIA_DIR[kind]
    print(f"  Pass:      {c(CYAN, label)}")
    print(f"  Progress:  {c(GREEN, str(done))}/{total}  "
          f"({c(DIM, f'{idx + 1} of {total}')})")
    print(c(BOLD, "─" * 64))
    print(f"  PDF:       {c(BOLD, pmcid)}")
    print(f"  Detected:  {c(YELLOW, str(len(images)))} crop(s) found in pipeline output")

    if images:
        for p in images:
            print(f"    {c(DIM, p.name)}")
    else:
        print(f"    {c(RED, '(no crops in ' + str(media_dir) + ')')}")

    if pmcid in records and records[pmcid].get(kind, "") != "":
        recorded = records[pmcid][kind]
        print(f"  Current:   {c(YELLOW, recorded)} {_KIND_NOUN[kind]} (recorded)")

    print(c(BOLD, "─" * 64))
    # print(c(DIM, "  Images opened in Preview (if any)."))
    print()


def ask(prompt: str) -> str:
    print(f"  {prompt}: ", end="", flush=True)
    return input().strip()


# Single-pass loop

def run_pass(kind: str, pdfs: list[str], records: dict) -> bool:
    """
    Interactively label one kind ('missed_figures' or 'missed_tables') for all PDFs.
    Returns False if the user quit early.
    """
    total = len(pdfs)

    # Resume from first unlabeled entry for this kind
    cursor = next(
        (i for i, pmcid in enumerate(pdfs)
         if records.get(pmcid, {}).get(kind, "") == ""),
        0,
    )

    while 0 <= cursor < total:
        pmcid  = pdfs[cursor]
        images = images_for(pmcid, kind)

        # Open images whenever we land on this PDF
        # open_images(images)
        render(pmcid, cursor, total, records, kind, images)

        raw = ask(f"{_KIND_NOUN[kind]} in this paper (b=back, q=quit, s=skip)")

        if raw.lower() == "q":
            save_csv(records)
            print(f"\n  Saved to {CSV_PATH}\n")
            return False
        if raw.lower() == "b":
            cursor = max(0, cursor - 1)
            continue
        if raw.lower() == "s":
            cursor += 1
            continue
        if not raw.isdigit():
            print(f"  {c(YELLOW, 'Enter a number (or b / q / s).')}")
            input("  Press Return to continue...")
            continue

        if pmcid not in records:
            records[pmcid] = {"pmcid": pmcid, "missed_figures": "", "missed_tables": "", "total_tables": ""}
        records[pmcid][kind] = raw
        save_csv(records)
        cursor += 1

    print(f"\n  {c(GREEN, _KIND_LABEL[kind] + ' pass complete!')}\n")
    return True


# Entry point

def main() -> None:
    pdfs    = sorted(p.stem for p in PDF_DIR.glob("*.pdf"))
    records = load_csv()

    arg_map = {"figures": "missed_figures", "tables": "missed_tables", "total": "total_tables"}
    if len(sys.argv) >= 2:
        if sys.argv[1] not in arg_map:
            print("Usage: python eval/ground_truth.py [figures|tables|total]")
            sys.exit(1)
        passes = [arg_map[sys.argv[1]]]
    else:
        passes = ["missed_figures", "missed_tables", "total_tables"]

    for kind in passes:
        print(f"\n  Starting {c(CYAN, _KIND_LABEL[kind])} pass — {len(pdfs)} PDFs.\n")
        input("  Press Return to begin...")
        ok = run_pass(kind, pdfs, records)
        if not ok:
            return

    print(f"\n  {c(GREEN, 'All done!')} Saved to {CSV_PATH}\n")


if __name__ == "__main__":
    main()
