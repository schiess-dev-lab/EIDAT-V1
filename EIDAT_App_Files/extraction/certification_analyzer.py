"""
Certification Analyzer for EIDAT

Analyzes measured terms in extracted_terms.db, determines pass/fail status
based on requirements, and auto-certifies products.

Certification Logic:
- CERTIFIED: All acceptance tests have computed_pass = 1
- FAILED: Any acceptance test has computed_pass = 0
- PENDING: Any acceptance test has computed_pass = NULL (no measurement)
- NO_DATA: No acceptance tests found for the document
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class CertificationStatus(Enum):
    """Certification status for a document."""
    CERTIFIED = "CERTIFIED"   # All tests pass (computed_pass = 1)
    FAILED = "FAILED"         # Any test fails (computed_pass = 0)
    PENDING = "PENDING"       # Any test has NULL computed_pass
    NO_DATA = "NO_DATA"       # No acceptance_tests found
    ERROR = "ERROR"           # Unable to analyze


@dataclass
class CertificationResult:
    """Result of certification analysis for a single document."""
    document_id: int
    status: CertificationStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    pending_tests: int
    failed_terms: list[str] = field(default_factory=list)
    pending_terms: list[str] = field(default_factory=list)
    analyzed_at: str = ""
    source_db: str = ""

    def __post_init__(self):
        if not self.analyzed_at:
            self.analyzed_at = datetime.now().isoformat()

    @property
    def pass_rate(self) -> str:
        """Return pass rate string like '5/7'."""
        return f"{self.passed_tests}/{self.total_tests}"

    @property
    def display_text(self) -> str:
        """Return display text like 'CERTIFIED (5/5)'."""
        if self.status == CertificationStatus.NO_DATA:
            return "NO DATA"
        return f"{self.status.value} ({self.pass_rate})"


class CertificationAnalyzer:
    """Analyze acceptance tests and determine certification status."""

    def __init__(self, db_path: Path):
        """
        Initialize analyzer with path to extracted_terms.db.

        Args:
            db_path: Path to the extracted_terms.db SQLite database
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_certification_columns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _ensure_certification_columns(self):
        """Add certification columns to documents table if they don't exist."""
        if not self.conn:
            return
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

    def analyze_document(self, document_id: int) -> CertificationResult:
        """
        Analyze a single document's acceptance tests.

        Args:
            document_id: ID of the document in the documents table

        Returns:
            CertificationResult with status and counts
        """
        if not self.conn:
            raise RuntimeError("Analyzer not connected. Use with context manager.")

        cursor = self.conn.cursor()

        # Get all acceptance tests for this document
        cursor.execute("""
            SELECT term, computed_pass
            FROM acceptance_tests
            WHERE document_id = ?
        """, (document_id,))
        tests = cursor.fetchall()

        if not tests:
            return CertificationResult(
                document_id=document_id,
                status=CertificationStatus.NO_DATA,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                pending_tests=0,
                failed_terms=[],
                pending_terms=[],
                source_db=str(self.db_path)
            )

        passed_terms = []
        failed_terms = []
        pending_terms = []

        for row in tests:
            term = row["term"] or "unknown"
            computed_pass = row["computed_pass"]

            if computed_pass == 1:
                passed_terms.append(term)
            elif computed_pass == 0:
                failed_terms.append(term)
            else:  # NULL
                pending_terms.append(term)

        # Determine status (priority: FAILED > PENDING > CERTIFIED)
        if failed_terms:
            status = CertificationStatus.FAILED
        elif pending_terms:
            status = CertificationStatus.PENDING
        else:
            status = CertificationStatus.CERTIFIED

        return CertificationResult(
            document_id=document_id,
            status=status,
            total_tests=len(tests),
            passed_tests=len(passed_terms),
            failed_tests=len(failed_terms),
            pending_tests=len(pending_terms),
            failed_terms=failed_terms,
            pending_terms=pending_terms,
            source_db=str(self.db_path)
        )

    def analyze_all_documents(self) -> list[CertificationResult]:
        """
        Analyze all documents in the database.

        Returns:
            List of CertificationResult for each document
        """
        if not self.conn:
            raise RuntimeError("Analyzer not connected. Use with context manager.")

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents")
        doc_ids = [row["id"] for row in cursor.fetchall()]

        results = []
        for doc_id in doc_ids:
            result = self.analyze_document(doc_id)
            results.append(result)

        return results

    def update_certification_status(self, document_id: int, result: CertificationResult) -> None:
        """
        Update the certification status in the documents table.

        Args:
            document_id: ID of the document to update
            result: CertificationResult with the analysis results
        """
        if not self.conn:
            raise RuntimeError("Analyzer not connected. Use with context manager.")

        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE documents
            SET certification_status = ?,
                certification_analyzed_at = ?,
                certification_passed_count = ?,
                certification_failed_count = ?,
                certification_pending_count = ?
            WHERE id = ?
        """, (
            result.status.value,
            result.analyzed_at,
            result.passed_tests,
            result.failed_tests,
            result.pending_tests,
            document_id
        ))
        self.conn.commit()

    def analyze_and_update_all(self) -> list[CertificationResult]:
        """
        Analyze all documents and update their certification status.

        Returns:
            List of CertificationResult for each document
        """
        results = self.analyze_all_documents()
        for result in results:
            self.update_certification_status(result.document_id, result)
        return results


def analyze_artifacts_folder(artifacts_dir: Path) -> Optional[CertificationResult]:
    """
    Analyze extracted_terms.db in an artifacts folder.

    Args:
        artifacts_dir: Path to the artifacts folder (e.g., debug/ocr/DEBUGPROG_V1_VALVE01_SN4001)

    Returns:
        CertificationResult or None if no database found
    """
    db_path = artifacts_dir / "extracted_terms.db"
    if not db_path.exists():
        return None

    try:
        with CertificationAnalyzer(db_path) as analyzer:
            results = analyzer.analyze_and_update_all()
            # Return first result (typically one document per artifacts folder)
            return results[0] if results else None
    except Exception as e:
        return CertificationResult(
            document_id=0,
            status=CertificationStatus.ERROR,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            pending_tests=0,
            failed_terms=[],
            pending_terms=[],
            source_db=str(db_path)
        )


def analyze_all_in_support_dir(support_dir: Path) -> dict[str, CertificationResult]:
    """
    Scan EIDAT Support/debug/ocr/* for all artifacts folders,
    analyze each extracted_terms.db, and return results keyed by artifacts_rel.

    Args:
        support_dir: Path to the EIDAT Support directory

    Returns:
        Dict mapping artifacts_rel path to CertificationResult
    """
    results: dict[str, CertificationResult] = {}
    ocr_root = support_dir / "debug" / "ocr"

    if not ocr_root.exists():
        return results

    for artifacts_dir in ocr_root.iterdir():
        if not artifacts_dir.is_dir():
            continue

        db_path = artifacts_dir / "extracted_terms.db"
        if not db_path.exists():
            continue

        # Get relative path for key
        artifacts_rel = f"debug/ocr/{artifacts_dir.name}"

        result = analyze_artifacts_folder(artifacts_dir)
        if result:
            results[artifacts_rel] = result

    return results


def get_certification_from_db(db_path: Path) -> Optional[dict]:
    """
    Read certification status directly from extracted_terms.db.

    Args:
        db_path: Path to extracted_terms.db

    Returns:
        Dict with certification fields or None if not found
    """
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT certification_status, certification_analyzed_at,
                   certification_passed_count, certification_failed_count,
                   certification_pending_count
            FROM documents
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        status = row["certification_status"]
        passed = row["certification_passed_count"] or 0
        failed = row["certification_failed_count"] or 0
        pending = row["certification_pending_count"] or 0
        total = passed + failed + pending

        return {
            "certification_status": status,
            "certification_analyzed_at": row["certification_analyzed_at"],
            "certification_passed_count": passed,
            "certification_failed_count": failed,
            "certification_pending_count": pending,
            "certification_pass_rate": f"{passed}/{total}" if total > 0 else ""
        }
    except Exception:
        return None


# CLI for standalone testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python certification_analyzer.py <path_to_extracted_terms.db>")
        print("   or: python certification_analyzer.py <path_to_support_dir> --all")
        sys.exit(1)

    path = Path(sys.argv[1])

    if len(sys.argv) > 2 and sys.argv[2] == "--all":
        # Analyze all in support directory
        results = analyze_all_in_support_dir(path)
        print(f"\nAnalyzed {len(results)} documents:\n")
        for artifacts_rel, result in results.items():
            print(f"  {artifacts_rel}")
            print(f"    Status: {result.display_text}")
            if result.failed_terms:
                print(f"    Failed: {', '.join(result.failed_terms)}")
            if result.pending_terms:
                print(f"    Pending: {', '.join(result.pending_terms)}")
            print()
    else:
        # Analyze single database
        if path.is_dir():
            path = path / "extracted_terms.db"

        if not path.exists():
            print(f"Error: Database not found: {path}")
            sys.exit(1)

        with CertificationAnalyzer(path) as analyzer:
            results = analyzer.analyze_and_update_all()

            print(f"\nAnalyzed {len(results)} documents:\n")
            for result in results:
                print(f"  Document {result.document_id}: {result.display_text}")
                if result.failed_terms:
                    print(f"    Failed tests: {', '.join(result.failed_terms)}")
                if result.pending_terms:
                    print(f"    Pending tests: {', '.join(result.pending_terms)}")
                print()
