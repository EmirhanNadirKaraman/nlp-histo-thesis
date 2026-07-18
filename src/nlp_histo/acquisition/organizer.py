import shutil
from pathlib import Path
from typing import Tuple


# Historical defaults, kept only so callers can reproduce the documented layout.
# Nothing here is resolved against the working directory.
DEFAULT_SOURCE_DIRNAME = "processed_corpus"
DEFAULT_PDF_DIRNAME = "organized_pdfs"
DEFAULT_XML_DIRNAME = "organized_xmls"


def organize_files(source_dir: Path, pdf_dir: Path, xml_dir: Path) -> Tuple[int, int]:
    """
    Organize PDF and XML files from paper directories into separate folders.

    Args:
        source_dir: Directory containing paper subdirectories
        pdf_dir: Destination directory for PDF files
        xml_dir: Destination directory for XML files

    Returns:
        Tuple of (pdf_count, xml_count) - number of files organized
    """
    # Create output directories
    pdf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    pdf_count = 0
    xml_count = 0

    # Find all paper directories (subdirectories in source_dir)
    paper_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

    if not paper_dirs:
        print(f"No paper directories found in {source_dir}")
        return 0, 0

    print(f"Found {len(paper_dirs)} paper directories. Starting organization...")

    for paper_dir in sorted(paper_dirs):
        pmcid = paper_dir.name

        # Find all PDF files in this paper directory
        pdf_files = list(paper_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            # Create a unique name: PMCID_originalname.pdf
            # If the file is already named with PMCID, just use original name
            if pdf_file.stem.startswith(pmcid):
                new_name = pdf_file.name
            else:
                new_name = f"{pmcid}_{pdf_file.name}"

            dest_path = pdf_dir / new_name

            # Handle potential filename conflicts
            counter = 1
            while dest_path.exists():
                stem = dest_path.stem
                new_name = f"{stem}_{counter}.pdf"
                dest_path = pdf_dir / new_name
                counter += 1

            # Copy the file
            shutil.copy2(pdf_file, dest_path)
            pdf_count += 1

        # Find all XML files in this paper directory
        xml_files = list(paper_dir.glob("*.xml")) + list(paper_dir.glob("*.nxml"))
        for xml_file in xml_files:
            # Create a unique name: PMCID.xml or PMCID_2.xml
            if xml_file.stem.startswith(pmcid):
                new_name = xml_file.name
            else:
                # Use .nxml extension if original was .nxml, otherwise .xml
                ext = xml_file.suffix
                new_name = f"{pmcid}{ext}"

            dest_path = xml_dir / new_name

            # Handle potential filename conflicts
            counter = 1
            while dest_path.exists():
                ext = dest_path.suffix
                stem = dest_path.stem.rstrip('_0123456789')  # Remove existing counter
                new_name = f"{stem}_{counter}{ext}"
                dest_path = xml_dir / new_name
                counter += 1

            # Copy the file
            shutil.copy2(xml_file, dest_path)
            xml_count += 1

        if pdf_files or xml_files:
            print(f"Organized {pmcid}: {len(pdf_files)} PDFs, {len(xml_files)} XMLs")
        else:
            print(f"Skipped {pmcid}: No PDF or XML files found")

    return pdf_count, xml_count


def organize_pdfs(
    input_dir: Path,
    pdf_dir: Path,
    xml_dir: Path,
) -> tuple[int, int]:
    """Flatten the extracted corpus in *input_dir* into *pdf_dir* / *xml_dir*.

    Returns ``(pdf_count, xml_count)``. Raises ``FileNotFoundError`` naming the path
    when the input directory is missing, before anything is written. Copy semantics
    (``shutil.copy2``, per-PMCID naming) are unchanged.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Extracted-corpus directory not found: {input_dir} "
            "(the default --source aws route writes this tree directly, so check "
            "the path from `acquire download`; on --source ftp, run "
            "`nlp-histo acquire unpack` first)"
        )

    print(f"Organizing files from: {input_dir}")
    print(f"PDF output: {pdf_dir}")
    print(f"XML output: {xml_dir}")

    pdf_count, xml_count = organize_files(input_dir, Path(pdf_dir), Path(xml_dir))

    print(f"\nOrganization complete — {pdf_count} PDFs, {xml_count} XMLs.")
    return pdf_count, xml_count
