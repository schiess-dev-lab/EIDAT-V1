"""
EIDAT Clean Extraction Pipeline

A streamlined, modular extraction pipeline for EIDAT project.
Replaces the legacy scanner.core.py with focused, maintainable components.

Modules:
- ocr_engine: PDF rendering and Tesseract OCR
- table_detection: Bordered cell detection and clustering
- borderless_table_detection: Token-grid table detection
- page_analyzer: Headers, footers, paragraph flow
- token_projector: Project OCR tokens into table cells
- chart_detection: Chart region detection
- debug_exporter: Export to JSON debug format
- batch_processor: Main extraction pipeline
- term_value_extractor: Extract structured test data from combined.txt
- project_manager: Manage projects, term registries, and trending
"""

__version__ = "2.0.0"
__all__ = [
    "ocr_engine",
    "table_detection",
    "borderless_table_detection",
    "page_analyzer",
    "token_projector",
    "chart_detection",
    "debug_exporter",
    "batch_processor",
    "term_value_extractor",
    "project_manager"
]
