# EIDAT Clean Extraction Pipeline v2.0

A streamlined, modular extraction pipeline for EIDAT project. Replaces legacy `scanner.core.py` with focused, maintainable components.

## Architecture

```
extraction/
├── __init__.py              # Module exports
├── ocr_engine.py            # PDF → Tesseract TSV tokens
├── table_detection.py       # Bordered cell detection
├── token_projector.py       # Project tokens into cells
├── page_analyzer.py         # Headers, footers, paragraph flow
├── debug_exporter.py        # JSON debug format export
├── batch_processor.py       # Main pipeline orchestrator
└── README.md               # This file
```

## Features

✅ **Pure geometric table detection** - Detects bordered cells using contour + corner intersection
✅ **Side-by-side table separation** - Correctly splits adjacent tables
✅ **Chart filtering** - Filters out chart frames and axis grids
✅ **Token projection** - Maps OCR tokens to table cells with spatial overlap
✅ **Compatible output** - Same JSON format as legacy scanner
✅ **Clean architecture** - ~2,000 lines vs 20,000+ in legacy

## Usage

### Command Line

```bash
# Process single PDF
python obs/cli_tools/run_extraction.py document.pdf

# Process specific pages
python obs/cli_tools/run_extraction.py document.pdf --pages 1,2,3

# Process directory
python obs/cli_tools/run_extraction.py ./pdfs --dpi 300

# Custom output location
python obs/cli_tools/run_extraction.py document.pdf --output ./my_output --verbose
```

### Python API

```python
from pathlib import Path
from extraction.batch_processor import ExtractionPipeline

# Create pipeline
pipeline = ExtractionPipeline(dpi=900, lang="eng", psm=6)

# Process single page
result = pipeline.process_page(
    pdf_path=Path("document.pdf"),
    page_num=0,  # 0-indexed
    debug_dir=Path("output"),
    verbose=True
)

# Process entire PDF
results = pipeline.process_pdf(
    pdf_path=Path("document.pdf"),
    output_dir=Path("output"),
    verbose=True
)

# Access results
for page_result in results:
    tokens = page_result['tokens']
    tables = page_result['tables']

    for table in tables:
        cells = table['cells']
        for cell in cells:
            print(f"Cell [{cell['row']}, {cell['col']}]: {cell['text']}")
```

## Module Details

### ocr_engine.py
- **Renders** PDF pages at specified DPI using PyMuPDF
- **Runs** Tesseract OCR in TSV mode
- **Parses** TSV output into token dicts with bboxes and confidence

### table_detection.py
- **Contour method**: Finds closed rectangular regions (cells)
- **Corner method**: Detects line intersections and builds cells from corners
- **Clusters** cells into tables based on direct adjacency
- **Filters** out chart frames and single-cell noise

### token_projector.py
- **Projects** OCR tokens into cells via spatial overlap
- **Organizes** cells into row/column grid structure
- **Extracts** table data as 2D matrices

### page_analyzer.py
- **Detects** headers and footers based on page position
- **Filters** table tokens from flow text
- **Groups** tokens into lines and paragraphs

### debug_exporter.py
- **Exports** page JSON in compatible format
- **Generates** combined text output
- **Creates** summary statistics

### batch_processor.py
- **Orchestrates** the full pipeline
- **Processes** single pages or entire PDFs
- **Handles** errors and exports debug output

## Output Format

### Page JSON
```json
{
  "page": 2,
  "pdf_file": "path/to/document.pdf",
  "img_w": 7650,
  "img_h": 9900,
  "dpi": 900,
  "artifacts": {
    "tokens": [...],
    "tables": [
      {
        "bbox_px": [x0, y0, x1, y1],
        "num_cells": 60,
        "cells": [
          {
            "bbox_px": [x0, y0, x1, y1],
            "text": "Cell text",
            "row": 0,
            "col": 0,
            "tokens": [...]
          }
        ]
      }
    ]
  }
}
```

## Configuration

Environment variables:
- `EIDAT_TESS_LANG` - Tesseract language (default: `eng`)
- `EIDAT_TESS_PSM` - Tesseract PSM mode (default: `6`)
- `EIDAT_TABLE_CELL_PREFIX_CLEAN` - Strip leading table-border OCR artifacts in cell text (default: `1`)

## Performance

| Metric | Legacy Scanner | Clean Pipeline |
|--------|---------------|----------------|
| Lines of code | ~20,000 | ~2,000 |
| Modules | 1 monolithic file | 7 focused modules |
| Side-by-side tables | ❌ Merged | ✅ Separated |
| Chart false positives | ⚠️ Some | ✅ Filtered |
| Maintainability | Low | High |

## Migration from Legacy

The legacy `EIDAT_App_Files/Application/eidp_term_scanner.core.py` is now **obsolete**. Use this clean pipeline instead.

**Key differences:**
- ✅ No interactive mode - batch processing only
- ✅ No term matching - pure extraction
- ✅ No Excel extraction - PDF only
- ✅ Bordered cell detection instead of line-based
- ✅ Token projection instead of per-cell OCR

## Testing

Tested against existing debug output:
```bash
python obs/dev_tests/test_cell_projection.py global_run_mirror/debug/ocr/*/page_*_page.json
```

Results: ✅ 6/6 pages processed correctly with improved table detection

## License

Internal EIDAT project
