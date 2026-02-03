"""
Term Value Extractor - Extract structured test data from combined.txt OCR output

Parses acceptance criteria tables and other structured data, extracting:
- Terms/Tags with their measured values
- Requirements (ranges, min/max, comparisons)
- Units and pass/fail validation
- Document metadata

Outputs to JSON and SQLite formats.
"""

import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Requirement:
    """Parsed requirement specification."""
    type: str  # 'range', 'max', 'min', 'exact', 'unknown'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    operator: Optional[str] = None  # '<=', '>=', '=', etc.
    raw: str = ""

    def check_pass(self, measured: float) -> bool:
        """Check if measured value passes this requirement."""
        if self.type == 'range':
            if self.min_value is not None and self.max_value is not None:
                return self.min_value <= measured <= self.max_value
        elif self.type == 'max':
            if self.max_value is not None:
                return measured <= self.max_value
        elif self.type == 'min':
            if self.min_value is not None:
                return measured >= self.min_value
        elif self.type == 'exact':
            if self.max_value is not None:
                return abs(measured - self.max_value) < 0.001
        return True  # Unknown requirements pass by default


@dataclass
class MeasuredValue:
    """Parsed measured value with optional notes."""
    value: Optional[float] = None
    raw: str = ""
    note: Optional[str] = None


@dataclass
class AcceptanceTest:
    """Single acceptance test result."""
    term: str
    description: str
    requirement: Requirement
    measured: MeasuredValue
    units: Optional[str] = None
    result: Optional[str] = None  # Original PASS/FAIL from document
    computed_pass: Optional[bool] = None  # Our computed result
    page: int = 0
    table_index: int = 0


@dataclass
class DocumentMetadata:
    """Document-level metadata."""
    program: Optional[str] = None
    serial: Optional[str] = None
    title: Optional[str] = None
    revision: Optional[str] = None
    test_date: Optional[str] = None
    report_date: Optional[str] = None
    operator: Optional[str] = None
    facility: Optional[str] = None
    valve_model: Optional[str] = None
    valve_size: Optional[str] = None
    valve_rating: Optional[str] = None


@dataclass
class ExtractionResult:
    """Complete extraction result."""
    document: DocumentMetadata
    acceptance_tests: List[AcceptanceTest] = field(default_factory=list)
    test_data: Dict[str, List[Dict]] = field(default_factory=dict)
    calibration: List[Dict] = field(default_factory=list)
    source_file: str = ""
    extracted_at: str = ""


# =============================================================================
# Requirement Parsing
# =============================================================================

class RequirementParser:
    """Parse requirement strings into structured Requirement objects."""

    # Regex patterns for different requirement formats
    PATTERNS = [
        # Range: "12.0-14.0" or "12-14" or "-20-120"
        (r'^(-?\d+\.?\d*)\s*[-–—]\s*(-?\d+\.?\d*)$', 'range'),
        # Less than or equal: "0.45<=" or "0.45=<" or "<=0.45" or "0,45<="
        (r'^(\d+[,.]?\d*)\s*[<≤]=?\s*$', 'max'),
        (r'^[<≤]=?\s*(\d+\.?\d*)$', 'max'),
        # Greater than or equal: "600>=" or ">=600"
        (r'^(\d+\.?\d*)\s*[>≥]=?\s*$', 'min'),
        (r'^[>≥]=?\s*(\d+\.?\d*)$', 'min'),
        # Single value (treated as max threshold): "5.0"
        (r'^(\d+\.?\d*)$', 'max'),
    ]

    @classmethod
    def parse(cls, raw: str) -> Requirement:
        """Parse a requirement string."""
        if not raw or not raw.strip():
            return Requirement(type='unknown', raw=raw)

        cleaned = raw.strip()
        # Normalize comma to decimal point
        cleaned_norm = cleaned.replace(',', '.')

        for pattern, req_type in cls.PATTERNS:
            match = re.match(pattern, cleaned_norm)
            if match:
                groups = match.groups()

                if req_type == 'range':
                    min_val = float(groups[0])
                    max_val = float(groups[1])
                    return Requirement(
                        type='range',
                        min_value=min_val,
                        max_value=max_val,
                        raw=raw
                    )
                elif req_type == 'max':
                    max_val = float(groups[0])
                    return Requirement(
                        type='max',
                        max_value=max_val,
                        operator='<=',
                        raw=raw
                    )
                elif req_type == 'min':
                    min_val = float(groups[0])
                    return Requirement(
                        type='min',
                        min_value=min_val,
                        operator='>=',
                        raw=raw
                    )

        return Requirement(type='unknown', raw=raw)


# =============================================================================
# Measured Value Parsing
# =============================================================================

class MeasuredValueParser:
    """Parse measured value strings."""

    # Pattern for value with optional note: "600 (hold)" or "13.48"
    VALUE_PATTERN = re.compile(r'^(-?\d+\.?\d*)\s*(?:\(([^)]+)\))?$')

    @classmethod
    def parse(cls, raw: str) -> MeasuredValue:
        """Parse a measured value string."""
        if not raw or not raw.strip():
            return MeasuredValue(raw=raw)

        cleaned = raw.strip()
        # Handle brackets from OCR: "[17.71" -> "17.71"
        cleaned = re.sub(r'[\[\]]', '', cleaned)

        match = cls.VALUE_PATTERN.match(cleaned)
        if match:
            value = float(match.group(1))
            note = match.group(2) if match.group(2) else None
            return MeasuredValue(value=value, raw=raw, note=note)

        # Try to extract just a number
        num_match = re.search(r'(-?\d+\.?\d*)', cleaned)
        if num_match:
            return MeasuredValue(
                value=float(num_match.group(1)),
                raw=raw,
                note=cleaned.replace(num_match.group(1), '').strip() or None
            )

        return MeasuredValue(raw=raw)


# =============================================================================
# Combined.txt Parser
# =============================================================================

class CombinedTxtParser:
    """Parse combined.txt OCR output format."""

    TABLE_START = re.compile(r'^\[Table(?:\s+\d+)?\]$')
    TABLE_HEADER = re.compile(r'^\[Table\s+(\d+)\]$')
    PAGE_MARKER = re.compile(r'^=== Page (\d+) ===$')
    SECTION_MARKER = re.compile(r'^\[([^\]]+)\]$')
    TABLE_ROW = re.compile(r'^\|(.+)\|$')
    TABLE_SEPARATOR = re.compile(r'^\+[-+]+\+$')

    def __init__(self, content: str):
        self.content = content
        self.lines = content.split('\n')
        self.current_page = 0

    def parse(self) -> ExtractionResult:
        """Parse the entire combined.txt file."""
        result = ExtractionResult(
            document=DocumentMetadata(),
            extracted_at=datetime.now().isoformat()
        )

        tables = self._extract_all_tables()

        # Extract document metadata from first tables
        result.document = self._extract_metadata(tables)

        # Extract acceptance tests
        result.acceptance_tests = self._extract_acceptance_tests(tables)

        # Extract test data tables
        result.test_data = self._extract_test_data(tables)

        # Extract calibration data
        result.calibration = self._extract_calibration(tables)

        return result

    def _extract_all_tables(self) -> List[Dict]:
        """Extract all tables with their context."""
        tables = []
        current_table = None
        current_page = 0
        preceding_title = None
        table_index = 0

        for i, line in enumerate(self.lines):
            line = line.strip()

            # Track page numbers
            page_match = self.PAGE_MARKER.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            # Track table/chart titles
            if line == '[Table/Chart Title]' and i + 1 < len(self.lines):
                preceding_title = self.lines[i + 1].strip()
                continue

            # Start of table
            if self.TABLE_START.match(line):
                if current_table:
                    tables.append(current_table)
                header_match = self.TABLE_HEADER.match(line)
                table_index = int(header_match.group(1)) if header_match else 0
                current_table = {
                    'page': current_page,
                    'title': preceding_title,
                    'table_index': table_index,
                    'rows': [],
                    'headers': []
                }
                preceding_title = None
                continue

            # Table row
            if current_table is not None:
                if self.TABLE_SEPARATOR.match(line):
                    continue
                row_match = self.TABLE_ROW.match(line)
                if row_match:
                    cells = [c.strip() for c in row_match.group(1).split('|')]
                    if not current_table['headers']:
                        current_table['headers'] = cells
                    else:
                        current_table['rows'].append(cells)
                elif line and not line.startswith('['):
                    # End of table
                    if current_table['rows']:
                        tables.append(current_table)
                    current_table = None

        if current_table and current_table['rows']:
            tables.append(current_table)

        return tables

    def _extract_metadata(self, tables: List[Dict]) -> DocumentMetadata:
        """Extract document metadata from tables."""
        meta = DocumentMetadata()

        # Look for Document Profile table or Field/Value tables on page 1-2
        field_mapping = {
            'PROGRAM': 'program',
            'SERIAL': 'serial',
            'TITLE': 'title',
            'REV': 'revision',
            'TEST DATE': 'test_date',
            'REPORT DATE': 'report_date',
            'OPERATOR': 'operator',
            'FACILITY': 'facility',
            'VALVE MODEL': 'valve_model',
            'VALVE SIZE': 'valve_size',
            'VALVE RATING': 'valve_rating',
            'Test Operator': 'operator',
        }

        for table in tables:
            if table['page'] > 2:
                continue
            headers = [h.lower() for h in table['headers']]
            if 'field' in headers and 'value' in headers:
                field_idx = headers.index('field')
                value_idx = headers.index('value')
                for row in table['rows']:
                    if len(row) > max(field_idx, value_idx):
                        field_name = row[field_idx].strip()
                        field_value = row[value_idx].strip()
                        for key, attr in field_mapping.items():
                            if key.lower() == field_name.lower():
                                setattr(meta, attr, field_value)
                                break

        return meta

    def _extract_acceptance_tests(self, tables: List[Dict]) -> List[AcceptanceTest]:
        """Extract acceptance test results."""
        tests = []

        # Look for Acceptance Criteria tables
        for table in tables:
            title = (table.get('title') or '').lower()
            headers = [h.lower() for h in table['headers']]

            # Check if this looks like an acceptance criteria table
            has_tag = any(h in ['tag', 'term', 'id', 'parameter'] for h in headers)
            has_requirement = any(h in ['requirement', 'criteria', 'spec', 'acceptance'] for h in headers)
            has_measured = any(h in ['measured', 'actual', 'value', 'result'] for h in headers)

            if not (has_tag and (has_requirement or has_measured)):
                continue

            # Find column indices
            tag_idx = next((i for i, h in enumerate(headers)
                           if h in ['tag', 'term', 'id', 'parameter']), 0)
            desc_idx = next((i for i, h in enumerate(headers)
                            if h in ['description', 'desc', 'name']), 1)
            req_idx = next((i for i, h in enumerate(headers)
                           if h in ['requirement', 'criteria', 'spec', 'acceptance']), 2)
            meas_idx = next((i for i, h in enumerate(headers)
                            if h in ['measured', 'actual', 'value']), 3)
            units_idx = next((i for i, h in enumerate(headers)
                             if h in ['units', 'unit']), None)
            result_idx = next((i for i, h in enumerate(headers)
                              if headers[i] == 'result' or h in ['pass?', 'status']), None)

            for row in table['rows']:
                if len(row) <= tag_idx:
                    continue

                term = row[tag_idx].strip()
                if not term:
                    continue

                desc = row[desc_idx].strip() if desc_idx < len(row) else ''
                req_raw = row[req_idx].strip() if req_idx < len(row) else ''
                meas_raw = row[meas_idx].strip() if meas_idx < len(row) else ''
                units = row[units_idx].strip() if units_idx and units_idx < len(row) else None
                result = row[result_idx].strip() if result_idx and result_idx < len(row) else None

                requirement = RequirementParser.parse(req_raw)
                measured = MeasuredValueParser.parse(meas_raw)

                # Compute pass/fail
                computed_pass = None
                if measured.value is not None:
                    computed_pass = requirement.check_pass(measured.value)

                test = AcceptanceTest(
                    term=term,
                    description=desc,
                    requirement=requirement,
                    measured=measured,
                    units=units,
                    result=result,
                    computed_pass=computed_pass,
                    page=table['page'],
                    table_index=table['table_index']
                )
                tests.append(test)

        return tests

    def _extract_test_data(self, tables: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract test data tables (step tests, leak tests, etc.)."""
        test_data = {}

        for table in tables:
            title = table.get('title') or ''
            headers = table['headers']

            # Skip metadata and acceptance tables
            if any(h.lower() in ['field', 'tag', 'role'] for h in headers):
                continue

            # Categorize by title
            category = 'other'
            title_lower = title.lower()
            if 'step' in title_lower or 'pressure' in title_lower or 'flow' in title_lower:
                category = 'step_test'
            elif 'leak' in title_lower:
                category = 'leak_test'
            elif 'cycle' in title_lower or 'stroke' in title_lower:
                category = 'stroke_test'
            elif 'excerpt' in title_lower or 'raw' in title_lower:
                category = 'raw_data'
            elif 'seal' in title_lower or 'visual' in title_lower or 'inspection' in title_lower:
                category = 'inspection'

            if category not in test_data:
                test_data[category] = []

            for row in table['rows']:
                row_dict = {}
                for i, header in enumerate(headers):
                    if i < len(row):
                        row_dict[header] = row[i].strip()
                row_dict['_page'] = table['page']
                row_dict['_title'] = title
                test_data[category].append(row_dict)

        return test_data

    def _extract_calibration(self, tables: List[Dict]) -> List[Dict]:
        """Extract calibration data."""
        calibration = []

        for table in tables:
            title = (table.get('title') or '').lower()
            headers = [h.lower() for h in table['headers']]

            # Look for calibration tables
            if 'calibration' in title or 'as-found' in headers or 'as-left' in headers:
                for row in table['rows']:
                    row_dict = {}
                    for i, header in enumerate(table['headers']):
                        if i < len(row):
                            row_dict[header] = row[i].strip()
                    row_dict['_page'] = table['page']
                    calibration.append(row_dict)

        return calibration


# =============================================================================
# JSON Export
# =============================================================================

def extraction_to_dict(result: ExtractionResult) -> Dict:
    """Convert ExtractionResult to JSON-serializable dict."""
    return {
        'document': asdict(result.document),
        'acceptance_tests': [
            {
                'term': t.term,
                'description': t.description,
                'requirement': asdict(t.requirement),
                'measured': asdict(t.measured),
                'units': t.units,
                'result': t.result,
                'computed_pass': t.computed_pass,
                'page': t.page
            }
            for t in result.acceptance_tests
        ],
        'test_data': result.test_data,
        'calibration': result.calibration,
        'source_file': result.source_file,
        'extracted_at': result.extracted_at
    }


def export_json(result: ExtractionResult, output_path: Path) -> None:
    """Export extraction result to JSON file."""
    data = extraction_to_dict(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# SQLite Export
# =============================================================================

SQLITE_SCHEMA = """
-- Document metadata
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program TEXT,
    serial TEXT,
    title TEXT,
    revision TEXT,
    test_date TEXT,
    report_date TEXT,
    operator TEXT,
    facility TEXT,
    valve_model TEXT,
    valve_size TEXT,
    valve_rating TEXT,
    source_file TEXT,
    extracted_at TEXT,
    certification_status TEXT,
    certification_analyzed_at TEXT,
    certification_passed_count INTEGER,
    certification_failed_count INTEGER,
    certification_pending_count INTEGER,
    UNIQUE(program, serial)
);

-- Acceptance test terms with parsed requirements
CREATE TABLE IF NOT EXISTS acceptance_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    description TEXT,
    requirement_type TEXT,
    requirement_min REAL,
    requirement_max REAL,
    requirement_raw TEXT,
    measured_value REAL,
    measured_raw TEXT,
    measured_note TEXT,
    units TEXT,
    result TEXT,
    computed_pass INTEGER,
    page INTEGER,
    table_index INTEGER
);

-- Test data points (step tests, leak tests, etc.)
CREATE TABLE IF NOT EXISTS test_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    category TEXT,
    data_json TEXT,
    page INTEGER,
    title TEXT
);

-- Calibration records
CREATE TABLE IF NOT EXISTS calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    channel TEXT,
    as_found REAL,
    as_left REAL,
    status TEXT,
    page INTEGER
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_acceptance_term ON acceptance_tests(term);
CREATE INDEX IF NOT EXISTS idx_acceptance_doc ON acceptance_tests(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_serial ON documents(serial);
CREATE INDEX IF NOT EXISTS idx_documents_program ON documents(program);
"""


class SQLiteExporter:
    """Export extraction results to SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript(SQLITE_SCHEMA)
        self.conn.commit()
        # Migration: add certification columns for existing databases
        self._migrate_certification_columns()

    def _migrate_certification_columns(self):
        """Add certification columns to existing databases that don't have them."""
        cert_columns = [
            ("certification_status", "TEXT"),
            ("certification_analyzed_at", "TEXT"),
            ("certification_passed_count", "INTEGER"),
            ("certification_failed_count", "INTEGER"),
            ("certification_pending_count", "INTEGER"),
        ]
        cursor = self.conn.cursor()
        for col_name, col_type in cert_columns:
            try:
                cursor.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        self.conn.commit()

    def export(self, result: ExtractionResult) -> int:
        """Export extraction result to database. Returns document ID."""
        cursor = self.conn.cursor()

        # Insert document
        doc = result.document
        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (program, serial, title, revision, test_date, report_date,
             operator, facility, valve_model, valve_size, valve_rating,
             source_file, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.program, doc.serial, doc.title, doc.revision,
            doc.test_date, doc.report_date, doc.operator, doc.facility,
            doc.valve_model, doc.valve_size, doc.valve_rating,
            result.source_file, result.extracted_at
        ))
        doc_id = cursor.lastrowid

        # Clear existing acceptance tests for this document
        cursor.execute("DELETE FROM acceptance_tests WHERE document_id = ?", (doc_id,))

        # Insert acceptance tests
        for test in result.acceptance_tests:
            cursor.execute("""
                INSERT INTO acceptance_tests
                (document_id, term, description, requirement_type,
                 requirement_min, requirement_max, requirement_raw,
                 measured_value, measured_raw, measured_note,
                 units, result, computed_pass, page, table_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, test.term, test.description,
                test.requirement.type, test.requirement.min_value,
                test.requirement.max_value, test.requirement.raw,
                test.measured.value, test.measured.raw, test.measured.note,
                test.units, test.result,
                1 if test.computed_pass else (0 if test.computed_pass is False else None),
                test.page, test.table_index
            ))

        # Clear and insert test data
        cursor.execute("DELETE FROM test_data WHERE document_id = ?", (doc_id,))
        for category, rows in result.test_data.items():
            for row in rows:
                page = row.pop('_page', None)
                title = row.pop('_title', None)
                cursor.execute("""
                    INSERT INTO test_data (document_id, category, data_json, page, title)
                    VALUES (?, ?, ?, ?, ?)
                """, (doc_id, category, json.dumps(row), page, title))

        # Clear and insert calibration
        cursor.execute("DELETE FROM calibration WHERE document_id = ?", (doc_id,))
        for cal in result.calibration:
            cursor.execute("""
                INSERT INTO calibration
                (document_id, channel, as_found, as_left, status, page)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                cal.get('Channel', cal.get('Tag', '')),
                _try_float(cal.get('As-Found', cal.get('as-found'))),
                _try_float(cal.get('As-Left', cal.get('as-left'))),
                cal.get('Status', cal.get('status')),
                cal.get('_page')
            ))

        self.conn.commit()
        return doc_id


def _try_float(val) -> Optional[float]:
    """Try to convert value to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Main API
# =============================================================================

def extract_from_combined_txt(
    combined_txt_path: Path,
    output_json: Optional[Path] = None,
    output_db: Optional[Path] = None,
    auto_project: bool = False
) -> Tuple['ExtractionResult', Optional[int], Optional[int]]:
    """
    Extract structured data from combined.txt file.

    Args:
        combined_txt_path: Path to combined.txt file
        output_json: Optional path for JSON output
        output_db: Optional path for SQLite database
        auto_project: If True, auto-create/assign project and register terms

    Returns:
        Tuple of (ExtractionResult, document_id, project_id)
        document_id and project_id are None if output_db is not provided
    """
    content = Path(combined_txt_path).read_text(encoding='utf-8')
    parser = CombinedTxtParser(content)
    result = parser.parse()
    result.source_file = str(combined_txt_path)

    if output_json:
        export_json(result, output_json)

    doc_id = None
    project_id = None

    if output_db:
        with SQLiteExporter(output_db) as exporter:
            doc_id = exporter.export(result)

        if auto_project and doc_id:
            from extraction.project_manager import setup_project_from_extraction
            project_id, terms_enabled = setup_project_from_extraction(output_db, doc_id)

    return result, doc_id, project_id


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python term_value_extractor.py <combined.txt> [output.json] [output.db] [--auto-project]")
        sys.exit(1)

    args = sys.argv[1:]
    auto_project = '--auto-project' in args
    args = [a for a in args if not a.startswith('--')]

    input_file = Path(args[0])
    json_out = Path(args[1]) if len(args) > 1 else None
    db_out = Path(args[2]) if len(args) > 2 else None

    result, doc_id, project_id = extract_from_combined_txt(
        input_file, json_out, db_out, auto_project=auto_project
    )

    print(f"Extracted from: {result.source_file}")
    print(f"Document: {result.document.program} / {result.document.serial}")
    if doc_id:
        print(f"Document ID: {doc_id}")
    if project_id:
        print(f"Project ID: {project_id}")
    print(f"Acceptance tests: {len(result.acceptance_tests)}")
    for test in result.acceptance_tests:
        status = "PASS" if test.computed_pass else "FAIL" if test.computed_pass is False else "?"
        print(f"  [{status}] {test.term}: {test.measured.value} {test.units or ''} "
              f"(req: {test.requirement.raw})")
