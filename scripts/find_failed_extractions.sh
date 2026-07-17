#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Change to project root (parent of scripts/)
cd "$(dirname "$0")/.."

PDF_DIR="files/organized_pdfs"
JSON_DIR="out/data"
OUTPUT_FILE="out/failed_extractions.txt"

echo "========================================================================"
echo "Finding PDFs with Failed Extractions"
echo "========================================================================"

mkdir -p "$(dirname "$OUTPUT_FILE")"
: > "$OUTPUT_FILE"

failed=0
pdf_files=()  # Initialize empty array to avoid unbound variable error
pdf_files=( "$PDF_DIR"/*.pdf )

for pdf_file in "${pdf_files[@]+"${pdf_files[@]}"}"; do
    base_name="$(basename "$pdf_file" .pdf)"
    json_file="$JSON_DIR/${base_name}.json"

    if [ ! -f "$json_file" ]; then
        echo "$base_name.pdf" >> "$OUTPUT_FILE"
        ((failed++))
    fi
done

echo "Total PDFs checked:   ${#pdf_files[@]:-0}"
echo "Failed extractions:   $failed"
echo ""

if (( failed > 0 )); then
    echo "Failed PDFs saved to: $OUTPUT_FILE"
    echo ""
    echo "First 20 failed files:"
    echo "------------------------------------------------------------------------"
    head -20 "$OUTPUT_FILE"

    if (( failed > 20 )); then
        echo "..."
        echo "(and $((failed - 20)) more)"
    fi
else
    echo "✅ All PDFs have JSON files!"
fi

echo "========================================================================"
