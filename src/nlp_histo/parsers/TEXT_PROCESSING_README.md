# Text Processing Utilities

Reusable text processing tools for cleaning and stitching extracted PDF text.

## Location

`parsers/text_processing.py`

## What's Inside

### `ContextAwareStitcher` Class

Intelligently stitches split paragraphs while preserving tables and figures.

**Problem it solves:**
PDF extractors sometimes split paragraphs when they encounter tables or figures:
```
"The results show that"        ← Cut off mid-sentence
"Table 1: Results"             ← Interrupting table
"the treatment was effective." ← Continuation
```

**Solution:**
```python
from nlp_histo.parsers.text_processing import ContextAwareStitcher

stitcher = ContextAwareStitcher()
result = stitcher.reconstruct_paragraphs(paragraphs)
# ["The results show that the treatment was effective.", "Table 1: Results"]
```

**Features:**
- ✅ Detects cut-off sentences (ends with comma, hyphen, connector words like "and"/"the"/"of", or mid-sentence abbreviations like "Fig."/"et al.")
- ✅ Preserves tables and figures
- ✅ Handles hyphenated word breaks
- ✅ Skips over interrupting content to find continuations

---

### `is_reference_entry()` Function

Detects if text is a bibliography/reference entry.

```python
from nlp_histo.parsers.text_processing import is_reference_entry

# Returns True for references
is_reference_entry("1. Smith J, Jones A. Nature. 2020;123:456.")
# True

# Returns False for normal text
is_reference_entry("The results were significant.")
# False
```

**Detection criteria:**
- Starts with number + period (e.g., "14. ")
- Contains author patterns (e.g., "Smith J,")
- Contains year (1900-2099)
- Contains journal identifiers (DOI, PMID, http, etc.)

---

### `remove_citations()` Function

Removes in-text citation numbers while preserving reference list numbers.

```python
from nlp_histo.parsers.text_processing import remove_citations

# Removes in-text citations
remove_citations("The study found. 12 These results")
# "The study found. These results"

# Preserves reference list numbers
remove_citations("14. Smith et al. 2020")
# "14. Smith et al. 2020"
```

**Removes:**
- ". 1 " → ". "
- ". 19,20 " → ". "
- ", 5 " → ", "
- " 1-3 " → " "

**Preserves:**
- Start-of-line numbers (reference list)

---

### `clean_text()` Function

Convenience function combining multiple cleaning operations.

```python
from nlp_histo.parsers.text_processing import clean_text

# Remove citations and skip references
clean_text("The study found. 12 These results")
# "The study found. These results"

# Reference entries return empty string
clean_text("1. Smith J. Nature. 2020", remove_refs=True)
# ""
```

---

## Usage Examples

### Example 1: Clean Extracted Text

```python
from nlp_histo.parsers.text_processing import ContextAwareStitcher, clean_text

# Extract text from PDF (using any parser)
text_elements = extract_from_pdf("paper.pdf")

# Stitch split paragraphs
stitcher = ContextAwareStitcher()
stitched = stitcher.reconstruct_paragraphs([elem['text'] for elem in text_elements])

# Clean each paragraph
cleaned = [clean_text(text) for text in stitched]
```

### Example 2: Filter References

```python
from nlp_histo.parsers.text_processing import is_reference_entry

# Separate narrative from references
narrative = [elem for elem in text_elements if not is_reference_entry(elem['text'])]
references = [elem for elem in text_elements if is_reference_entry(elem['text'])]
```

### Example 3: Pipeline Processing

```python
from nlp_histo.parsers.text_processing import ContextAwareStitcher, remove_citations, is_reference_entry

def process_extracted_text(text_elements):
    """Complete text processing pipeline."""

    # Step 1: Stitch split paragraphs
    stitcher = ContextAwareStitcher()
    stitched = stitcher.reconstruct_paragraphs(text_elements)

    # Step 2: Remove citations and filter references
    cleaned = []
    for text in stitched:
        # Skip reference entries
        if is_reference_entry(text):
            continue

        # Remove citation numbers
        text = remove_citations(text)
        cleaned.append(text)

    return cleaned
```

---

## Why This Module Exists

**Before:** Text processing utilities were scattered in individual scripts
- ❌ `parse_single_pdf.py` had its own copy (~130 lines)
- ❌ Duplicate code if other scripts needed same functionality
- ❌ Hard to maintain and test

**After:** Centralized, reusable module
- ✅ Single source of truth
- ✅ Can be imported by any script
- ✅ Well-documented with docstrings
- ✅ Easier to test and maintain

---

## Files Using This Module

- `parsers/layout_utils.py` — `extract_text` calls `ContextAwareStitcher`, `remove_citations`, and `is_reference_entry`.
- `pipeline/stages/pdf_text_extraction/components/text_assembler.py` — delegates to `extract_text` and re-applies `is_reference_entry`.

---

## Testing

```python
# Test stitcher
from nlp_histo.parsers.text_processing import ContextAwareStitcher

stitcher = ContextAwareStitcher()
test_paras = [
    "The results show that",
    "Table 1: Data",
    "the treatment was effective."
]
result = stitcher.reconstruct_paragraphs(test_paras)
assert result == ["The results show that the treatment was effective.", "Table 1: Data"]

# Test reference detection
from nlp_histo.parsers.text_processing import is_reference_entry

assert is_reference_entry("1. Smith J. Nature. 2020") == True
assert is_reference_entry("Normal paragraph") == False

# Test citation removal
from nlp_histo.parsers.text_processing import remove_citations

assert remove_citations("Text. 12 More") == "Text. More"
```

---

## Summary

| Utility | Purpose | Input | Output |
|---------|---------|-------|--------|
| `ContextAwareStitcher` | Stitch split paragraphs | List of strings | List of merged strings |
| `is_reference_entry()` | Detect references | String | Boolean |
| `remove_citations()` | Remove citations | String | Cleaned string |
| `clean_text()` | Combined cleaning | String | Cleaned string |

**Lines of code:** ~370 lines (`parsers/text_processing.py`). Originally
extracted from the legacy `parse_single_pdf.py`, which has since been removed
from the repo; the current callers are `parsers/layout_utils.py` and
`pipeline/stages/pdf_text_extraction/components/text_assembler.py`.

**Reusability:** ⭐⭐⭐⭐⭐ High - can be used by any text processing script
