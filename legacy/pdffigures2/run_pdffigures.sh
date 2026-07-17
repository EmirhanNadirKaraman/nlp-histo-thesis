#!/bin/bash
# Run PDFFigures2 only on unprocessed PDFs
# Checks for existing JSON files and skips those PDFs

# Ensure Java 11
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@11/11.0.29/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

echo "Using Java version:"
java -version

PDF_DIR="/Users/emir/Documents/GitHub/nlp-histo/files/organized_pdfs"
JSON_DIR="/Users/emir/Documents/GitHub/nlp-histo/out/data"
IMAGES_DIR="/Users/emir/Documents/GitHub/nlp-histo/out/images"
FIGURES_DIR="/Users/emir/Documents/GitHub/nlp-histo/out/figures"
TABLES_DIR="/Users/emir/Documents/GitHub/nlp-histo/out/tables"
STATS_FILE="/Users/emir/Documents/GitHub/nlp-histo/out/stats.json"
FAILED_LOG="/Users/emir/Documents/GitHub/nlp-histo/out/failed_extractions.txt"

# Create output directories
mkdir -p "$JSON_DIR" "$IMAGES_DIR" "$FIGURES_DIR" "$TABLES_DIR"

# Clear or create failed log
> "$FAILED_LOG"

echo ""
echo "========================================================================"
echo "PDFFigures2 Resume Mode - Process Only Unprocessed PDFs"
echo "========================================================================"

# Count total and processed
total_pdfs=$(ls "$PDF_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')
processed=$(ls "$JSON_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
remaining=$((total_pdfs - processed))

echo "Total PDFs:     $total_pdfs"
echo "Processed:      $processed"
echo "Remaining:      $remaining"
echo "========================================================================"

if [ "$remaining" -eq 0 ]; then
    echo "✅ All PDFs already processed!"
    exit 0
fi

echo ""
echo "Processing $remaining PDFs in TRUE BATCH MODE..."
echo "(Single Java invocation - much faster!)"
echo ""

# Collect all unprocessed PDFs
unprocessed_pdfs=()
for pdf_file in "$PDF_DIR"/*.pdf; do
    if [ ! -f "$pdf_file" ]; then
        continue
    fi

    base_name=$(basename "$pdf_file" .pdf)
    json_file="$JSON_DIR/${base_name}.json"

    # If JSON doesn't exist, add to unprocessed list
    if [ ! -f "$json_file" ]; then
        unprocessed_pdfs+=("$pdf_file")
    fi
done

# Create temporary directory with symlinks to unprocessed PDFs
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

for pdf_file in "${unprocessed_pdfs[@]}"; do
    ln -s "$pdf_file" "$TEMP_DIR/"
done

# Process ALL unprocessed PDFs in ONE batch by passing the temp directory
java -cp legacy/pdffigures2/pdffigures2.jar \
    org.allenai.pdffigures2.FigureExtractorBatchCli \
    -e \
    -q \
    -d "$JSON_DIR/" \
    -m "$IMAGES_DIR/" \
    "$TEMP_DIR"

# Separate figures and tables into different directories
echo ""
echo "========================================================================"
echo "Organizing Figures and Tables..."
echo "========================================================================"

figures_moved=0
tables_moved=0

for img_file in "$IMAGES_DIR"/*-Figure*.png; do
    if [ -f "$img_file" ]; then
        mv "$img_file" "$FIGURES_DIR/"
        figures_moved=$((figures_moved + 1))
    fi
done

for img_file in "$IMAGES_DIR"/*-Table*.png; do
    if [ -f "$img_file" ]; then
        mv "$img_file" "$TABLES_DIR/"
        tables_moved=$((tables_moved + 1))
    fi
done

echo "Figures moved: $figures_moved"
echo "Tables moved:  $tables_moved"

# Check results and log failures
echo ""
echo "========================================================================"
echo "Checking Results..."
echo "========================================================================"

successful=0
failed=0

for pdf_file in "${unprocessed_pdfs[@]}"; do
    base_name=$(basename "$pdf_file" .pdf)
    json_file="$JSON_DIR/${base_name}.json"

    if [ -f "$json_file" ]; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
        echo "$base_name.pdf" >> "$FAILED_LOG"
    fi
done

# Generate statistics
echo ""
echo "========================================================================"
echo "✓ Batch Processing Complete!"
echo "========================================================================"
echo "Successful:       $successful / $remaining"
echo "Failed:           $failed / $remaining"
echo "JSON data:        $JSON_DIR/"
echo "Figures:          $FIGURES_DIR/"
echo "Tables:           $TABLES_DIR/"
echo "Unsorted images:  $IMAGES_DIR/"

if [ "$failed" -gt 0 ]; then
    echo ""
    echo "⚠️  Failed PDFs listed in: $FAILED_LOG"
    echo ""
    echo "First 10 failed files:"
    head -10 "$FAILED_LOG"
fi

echo ""
echo "Next step: python legacy/pdffigures2/process_pdffigures_results.py"
echo "========================================================================"
