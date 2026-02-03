#!/usr/bin/env python3
"""
EIDAT Clean Extraction Pipeline - CLI Entry Point

Uses token projection as the PRIMARY method for cell text extraction:
1. Full-page OCR at 450 DPI (optimal for text recognition)
2. Table detection at 900 DPI (optimal for border detection)
3. Project tokens into cells based on spatial overlap
4. Selective re-OCR only for low-confidence cells

Usage:
    python run_extraction.py <pdf_path> [options]
    python run_extraction.py <directory> [options]  # Process all PDFs in directory

Options:
    --output DIR        Output directory (default: global_run_mirror)
    --ocr-dpi INT      OCR render DPI (default: 450, optimal for text)
    --detection-dpi INT Table detection DPI (default: 900, optimal for borders)
    --pages N,M,K      Specific pages to process (1-indexed)
    --lang LANG        Tesseract language (default: eng)
    --psm INT          Tesseract PSM mode (default: 6)
    --verbose          Verbose output

Examples:
    # Process single PDF
    python run_extraction.py document.pdf

    # Process specific pages
    python run_extraction.py document.pdf --pages 1,2,3

    # Process directory
    python run_extraction.py ./pdfs

    # Custom output location
    python run_extraction.py document.pdf --output ./my_output
"""

import sys
import argparse
from pathlib import Path
from typing import List

# NOTE: extraction/ moved to EIDAT_App_Files/extraction/ (Jan 2026)
# To use, add EIDAT_App_Files to sys.path first:
#   sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "EIDAT_App_Files"))
from extraction.batch_processor import ExtractionPipeline, process_pdf_batch


def parse_pages(pages_str: str) -> List[int]:
    """
    Parse page string into list of 0-indexed page numbers.

    Examples:
        "1,2,3" -> [0, 1, 2]
        "1-3" -> [0, 1, 2]
        "1,3-5,7" -> [0, 2, 3, 4, 6]
    """
    pages = []

    for part in pages_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            pages.extend(range(int(start) - 1, int(end)))
        else:
            pages.append(int(part) - 1)

    return pages


def find_pdfs(directory: Path) -> List[Path]:
    """Find all PDF files in directory."""
    return sorted(directory.glob('*.pdf'))


def main():
    parser = argparse.ArgumentParser(
        description='EIDAT Clean Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input', type=str,
                        help='PDF file or directory containing PDFs')
    parser.add_argument('--output', type=str, default='global_run_mirror',
                        help='Output directory (default: global_run_mirror)')
    parser.add_argument('--ocr-dpi', type=int, default=450,
                        help='OCR render DPI (default: 450, optimal for text)')
    parser.add_argument('--detection-dpi', type=int, default=900,
                        help='Table detection DPI (default: 900, optimal for borders)')
    parser.add_argument('--pages', type=str,
                        help='Comma-separated page numbers (1-indexed), e.g., "1,2,3" or "1-5"')
    parser.add_argument('--lang', type=str, default='eng',
                        help='Tesseract language (default: eng)')
    parser.add_argument('--psm', type=int, default=3,
                        help='Tesseract PSM mode (default: 3, auto page segmentation)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Parse input
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Determine PDFs to process
    if input_path.is_file():
        pdf_paths = [input_path]
    elif input_path.is_dir():
        pdf_paths = find_pdfs(input_path)
        if not pdf_paths:
            print(f"Error: No PDF files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: Input must be PDF file or directory")
        sys.exit(1)

    # Parse page selection
    pages = None
    if args.pages:
        pages = parse_pages(args.pages)

    # Create pipeline
    pipeline = ExtractionPipeline(
        ocr_dpi=args.ocr_dpi,
        detection_dpi=args.detection_dpi,
        lang=args.lang,
        psm=args.psm
    )

    # Process PDFs
    if args.verbose:
        print(f"EIDAT Clean Extraction Pipeline v2.1.0 (Token Projection)")
        print(f"OCR DPI: {args.ocr_dpi}, Detection DPI: {args.detection_dpi}")
        print(f"Lang: {args.lang}, PSM: {args.psm}")
        print(f"Output: {output_dir}")
        print(f"Processing {len(pdf_paths)} PDF(s)...\n")

    for pdf_path in pdf_paths:
        try:
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"PDF: {pdf_path.name}")
                print('='*60)

            results = pipeline.process_pdf(
                pdf_path,
                pages=pages,
                output_dir=output_dir,
                verbose=args.verbose
            )

            if args.verbose:
                total_pages = len(results)
                total_tokens = sum(len(r.get('tokens', [])) for r in results)
                total_tables = sum(len(r.get('tables', [])) for r in results)
                print(f"\nCompleted: {total_pages} pages, {total_tokens} tokens, {total_tables} tables")

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    if args.verbose:
        print(f"\n{'='*60}")
        print("Extraction complete!")
        print(f"Debug files saved to: {output_dir}")
        print('='*60)


if __name__ == "__main__":
    main()
