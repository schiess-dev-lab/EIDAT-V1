"""
Project Manager - Manage projects, term registries, and trending for EIDPs

Features:
- Create/manage projects to group related EIDPs
- Auto-group documents by program, asset, or similarity
- Term registry for consistent tracking across documents
- Trending data aggregation and spreadsheet export
- Auto-populate term values from extracted data
"""

import re
import json
import sqlite3
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# Enums and Data Classes
# =============================================================================

class GroupingStrategy(Enum):
    """How to group documents into projects."""
    PROGRAM = "program"      # Group by program name (e.g., DEBUGPROG_V1)
    ASSET = "asset"          # Group by valve model/asset (e.g., VLV-42A)
    SERIAL_PREFIX = "serial_prefix"  # Group by serial prefix (e.g., SN40xx)
    CUSTOM = "custom"        # Manual grouping


@dataclass
class TermDefinition:
    """Definition of a term to track across documents."""
    term_id: str                    # Unique identifier (e.g., "CV50")
    display_name: str               # Human-readable name
    description: Optional[str] = None
    expected_units: Optional[str] = None
    category: Optional[str] = None  # e.g., "flow", "pressure", "timing"
    requirement_type: Optional[str] = None  # "range", "max", "min"
    default_min: Optional[float] = None
    default_max: Optional[float] = None
    aliases: List[str] = field(default_factory=list)  # Alternative names


@dataclass
class Project:
    """A project grouping related EIDPs."""
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    grouping_strategy: GroupingStrategy = GroupingStrategy.PROGRAM
    grouping_value: Optional[str] = None  # The value to match (e.g., "DEBUGPROG_V1")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class TrendingRow:
    """A row in the trending spreadsheet."""
    serial: str
    test_date: Optional[str] = None
    operator: Optional[str] = None
    term_values: Dict[str, Any] = field(default_factory=dict)  # term_id -> value
    term_results: Dict[str, bool] = field(default_factory=dict)  # term_id -> pass/fail


# =============================================================================
# Extended SQLite Schema
# =============================================================================

PROJECT_SCHEMA = """
-- Projects for grouping EIDPs
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    grouping_strategy TEXT DEFAULT 'program',
    grouping_value TEXT,
    created_at TEXT,
    updated_at TEXT
);

-- Link documents to projects (many-to-many)
CREATE TABLE IF NOT EXISTS project_documents (
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    added_at TEXT,
    PRIMARY KEY (project_id, document_id)
);

-- Term registry - known terms to track
CREATE TABLE IF NOT EXISTS term_registry (
    term_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    expected_units TEXT,
    category TEXT,
    requirement_type TEXT,
    default_min REAL,
    default_max REAL,
    aliases TEXT,  -- JSON array
    created_at TEXT,
    updated_at TEXT
);

-- Project-specific term settings (override defaults)
CREATE TABLE IF NOT EXISTS project_terms (
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    term_id TEXT REFERENCES term_registry(term_id) ON DELETE CASCADE,
    enabled INTEGER DEFAULT 1,
    display_order INTEGER,
    custom_min REAL,
    custom_max REAL,
    PRIMARY KEY (project_id, term_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_project_docs_project ON project_documents(project_id);
CREATE INDEX IF NOT EXISTS idx_project_docs_doc ON project_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_term_category ON term_registry(category);
"""

# Common valve/test terms with their typical requirements
DEFAULT_TERMS = [
    TermDefinition("LEAK150", "Seat Leak @ 150 psig", "Seat leak rate at 150 psig", "sccm", "leak", "max"),
    TermDefinition("CV50", "Flow Coefficient @ 50%", "Cv at 50% open position", None, "flow", "range"),
    TermDefinition("CV100", "Flow Coefficient @ 100%", "Cv at fully open", None, "flow", "range"),
    TermDefinition("OPEN_T", "Open Stroke Time", "Time to fully open", "s", "timing", "max"),
    TermDefinition("CLOSE_T", "Close Stroke Time", "Time to fully close", "s", "timing", "max"),
    TermDefinition("HYST", "Hysteresis", "Position hysteresis", "%FS", "performance", "max"),
    TermDefinition("ACT_P", "Actuation Pressure", "Required actuation pressure", "psig", "pressure", "range"),
    TermDefinition("PROOF", "Proof Pressure", "Proof pressure hold", "psig", "pressure", "min"),
    TermDefinition("DEADBAND", "Deadband", "Control deadband", "%", "performance", "max"),
    TermDefinition("LINEARITY", "Linearity", "Position linearity", "%FS", "performance", "max"),
]


# =============================================================================
# Project Manager
# =============================================================================

class ProjectManager:
    """Manage projects, terms, and trending data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _init_schema(self):
        """Initialize project management schema."""
        self.conn.executescript(PROJECT_SCHEMA)
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Project Management
    # -------------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        grouping_strategy: GroupingStrategy = GroupingStrategy.PROGRAM,
        grouping_value: Optional[str] = None
    ) -> int:
        """Create a new project. Returns project ID."""
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO projects (name, description, grouping_strategy, grouping_value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, description, grouping_strategy.value, grouping_value, now, now))
        self.conn.commit()
        return cursor.lastrowid

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get project by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if row:
            return Project(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                grouping_strategy=GroupingStrategy(row['grouping_strategy']),
                grouping_value=row['grouping_value'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        return None

    def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get project by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return Project(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                grouping_strategy=GroupingStrategy(row['grouping_strategy']),
                grouping_value=row['grouping_value'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        return None

    def list_projects(self) -> List[Project]:
        """List all projects."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects ORDER BY name")
        return [
            Project(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                grouping_strategy=GroupingStrategy(row['grouping_strategy']),
                grouping_value=row['grouping_value'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            for row in cursor.fetchall()
        ]

    def delete_project(self, project_id: int) -> bool:
        """Delete a project."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Document-Project Association
    # -------------------------------------------------------------------------

    def add_document_to_project(self, project_id: int, document_id: int) -> None:
        """Add a document to a project."""
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO project_documents (project_id, document_id, added_at)
            VALUES (?, ?, ?)
        """, (project_id, document_id, now))
        self.conn.commit()

    def remove_document_from_project(self, project_id: int, document_id: int) -> None:
        """Remove a document from a project."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM project_documents WHERE project_id = ? AND document_id = ?
        """, (project_id, document_id))
        self.conn.commit()

    def get_project_documents(self, project_id: int) -> List[Dict]:
        """Get all documents in a project."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.*, pd.added_at
            FROM documents d
            JOIN project_documents pd ON d.id = pd.document_id
            WHERE pd.project_id = ?
            ORDER BY d.test_date, d.serial
        """, (project_id,))
        return [dict(row) for row in cursor.fetchall()]

    def auto_assign_documents(self, project_id: int) -> int:
        """
        Auto-assign documents to project based on grouping strategy.
        Returns count of documents added.
        """
        project = self.get_project(project_id)
        if not project or not project.grouping_value:
            return 0

        cursor = self.conn.cursor()
        count = 0

        if project.grouping_strategy == GroupingStrategy.PROGRAM:
            cursor.execute("""
                SELECT id FROM documents WHERE program = ?
            """, (project.grouping_value,))
        elif project.grouping_strategy == GroupingStrategy.ASSET:
            cursor.execute("""
                SELECT id FROM documents WHERE valve_model = ?
            """, (project.grouping_value,))
        elif project.grouping_strategy == GroupingStrategy.SERIAL_PREFIX:
            cursor.execute("""
                SELECT id FROM documents WHERE serial LIKE ?
            """, (project.grouping_value + '%',))
        else:
            return 0

        for row in cursor.fetchall():
            self.add_document_to_project(project_id, row[0])
            count += 1

        return count

    def find_or_create_project_for_document(self, document_id: int) -> Optional[int]:
        """
        Find an existing project for a document or create one based on program.
        Returns project ID.
        """
        cursor = self.conn.cursor()

        # Get document info
        cursor.execute("SELECT program, valve_model, serial FROM documents WHERE id = ?", (document_id,))
        doc = cursor.fetchone()
        if not doc:
            return None

        program = doc['program']
        if not program:
            return None

        # Look for existing project with this program
        cursor.execute("""
            SELECT id FROM projects
            WHERE grouping_strategy = 'program' AND grouping_value = ?
        """, (program,))
        row = cursor.fetchone()

        if row:
            project_id = row[0]
        else:
            # Create new project
            project_id = self.create_project(
                name=f"{program} Project",
                description=f"Auto-created project for program {program}",
                grouping_strategy=GroupingStrategy.PROGRAM,
                grouping_value=program
            )

        # Add document to project
        self.add_document_to_project(project_id, document_id)
        return project_id

    # -------------------------------------------------------------------------
    # Term Registry
    # -------------------------------------------------------------------------

    def register_term(self, term: TermDefinition) -> None:
        """Register a term in the registry."""
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO term_registry
            (term_id, display_name, description, expected_units, category,
             requirement_type, default_min, default_max, aliases, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            term.term_id, term.display_name, term.description, term.expected_units,
            term.category, term.requirement_type, term.default_min, term.default_max,
            json.dumps(term.aliases), now, now
        ))
        self.conn.commit()

    def register_default_terms(self) -> int:
        """Register all default terms. Returns count."""
        for term in DEFAULT_TERMS:
            self.register_term(term)
        return len(DEFAULT_TERMS)

    def get_term(self, term_id: str) -> Optional[TermDefinition]:
        """Get a term definition."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM term_registry WHERE term_id = ?", (term_id,))
        row = cursor.fetchone()
        if row:
            return TermDefinition(
                term_id=row['term_id'],
                display_name=row['display_name'],
                description=row['description'],
                expected_units=row['expected_units'],
                category=row['category'],
                requirement_type=row['requirement_type'],
                default_min=row['default_min'],
                default_max=row['default_max'],
                aliases=json.loads(row['aliases']) if row['aliases'] else []
            )
        return None

    def list_terms(self, category: Optional[str] = None) -> List[TermDefinition]:
        """List all registered terms, optionally filtered by category."""
        cursor = self.conn.cursor()
        if category:
            cursor.execute("SELECT * FROM term_registry WHERE category = ? ORDER BY term_id", (category,))
        else:
            cursor.execute("SELECT * FROM term_registry ORDER BY category, term_id")
        return [
            TermDefinition(
                term_id=row['term_id'],
                display_name=row['display_name'],
                description=row['description'],
                expected_units=row['expected_units'],
                category=row['category'],
                requirement_type=row['requirement_type'],
                default_min=row['default_min'],
                default_max=row['default_max'],
                aliases=json.loads(row['aliases']) if row['aliases'] else []
            )
            for row in cursor.fetchall()
        ]

    def auto_register_terms_from_documents(self) -> int:
        """
        Scan all acceptance_tests and register any new terms found.
        Returns count of new terms registered.
        """
        cursor = self.conn.cursor()

        # Get unique terms from acceptance tests
        cursor.execute("""
            SELECT DISTINCT term, description, units, requirement_type,
                   requirement_min, requirement_max
            FROM acceptance_tests
            WHERE term IS NOT NULL AND term != ''
        """)

        count = 0
        for row in cursor.fetchall():
            term_id = row['term']
            # Check if already registered
            if not self.get_term(term_id):
                term = TermDefinition(
                    term_id=term_id,
                    display_name=term_id,
                    description=row['description'],
                    expected_units=row['units'],
                    requirement_type=row['requirement_type'],
                    default_min=row['requirement_min'],
                    default_max=row['requirement_max']
                )
                self.register_term(term)
                count += 1

        return count

    # -------------------------------------------------------------------------
    # Project Term Configuration
    # -------------------------------------------------------------------------

    def enable_term_for_project(
        self,
        project_id: int,
        term_id: str,
        display_order: Optional[int] = None,
        custom_min: Optional[float] = None,
        custom_max: Optional[float] = None
    ) -> None:
        """Enable and configure a term for a specific project."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO project_terms
            (project_id, term_id, enabled, display_order, custom_min, custom_max)
            VALUES (?, ?, 1, ?, ?, ?)
        """, (project_id, term_id, display_order, custom_min, custom_max))
        self.conn.commit()

    def disable_term_for_project(self, project_id: int, term_id: str) -> None:
        """Disable a term for a specific project."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE project_terms SET enabled = 0
            WHERE project_id = ? AND term_id = ?
        """, (project_id, term_id))
        self.conn.commit()

    def get_project_terms(self, project_id: int) -> List[Dict]:
        """Get all terms configured for a project with their settings."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT tr.*, pt.enabled, pt.display_order, pt.custom_min, pt.custom_max
            FROM term_registry tr
            LEFT JOIN project_terms pt ON tr.term_id = pt.term_id AND pt.project_id = ?
            ORDER BY COALESCE(pt.display_order, 999), tr.category, tr.term_id
        """, (project_id,))
        return [dict(row) for row in cursor.fetchall()]

    def auto_enable_terms_for_project(self, project_id: int) -> int:
        """
        Auto-enable terms that appear in project documents.
        Returns count of terms enabled.
        """
        cursor = self.conn.cursor()

        # Find terms used in project documents
        cursor.execute("""
            SELECT DISTINCT at.term
            FROM acceptance_tests at
            JOIN project_documents pd ON at.document_id = pd.document_id
            WHERE pd.project_id = ?
              AND at.term IS NOT NULL AND at.term != ''
        """, (project_id,))

        count = 0
        order = 1
        for row in cursor.fetchall():
            term_id = row[0]
            # Make sure term is registered
            if not self.get_term(term_id):
                self.auto_register_terms_from_documents()
            self.enable_term_for_project(project_id, term_id, display_order=order)
            order += 1
            count += 1

        return count

    # -------------------------------------------------------------------------
    # Trending Data
    # -------------------------------------------------------------------------

    def get_trending_data(self, project_id: int) -> List[TrendingRow]:
        """
        Get trending data for all documents in a project.
        Returns list of TrendingRow with term values for each document.
        """
        cursor = self.conn.cursor()

        # Get enabled terms for this project (ordered)
        cursor.execute("""
            SELECT tr.term_id
            FROM term_registry tr
            JOIN project_terms pt ON tr.term_id = pt.term_id
            WHERE pt.project_id = ? AND pt.enabled = 1
            ORDER BY pt.display_order, tr.term_id
        """, (project_id,))
        term_ids = [row[0] for row in cursor.fetchall()]

        # If no terms configured, get all terms from documents
        if not term_ids:
            cursor.execute("""
                SELECT DISTINCT at.term
                FROM acceptance_tests at
                JOIN project_documents pd ON at.document_id = pd.document_id
                WHERE pd.project_id = ? AND at.term IS NOT NULL
                ORDER BY at.term
            """, (project_id,))
            term_ids = [row[0] for row in cursor.fetchall()]

        # Get documents in project
        docs = self.get_project_documents(project_id)

        rows = []
        for doc in docs:
            row = TrendingRow(
                serial=doc['serial'],
                test_date=doc['test_date'],
                operator=doc['operator']
            )

            # Get term values for this document
            cursor.execute("""
                SELECT term, measured_value, computed_pass, units
                FROM acceptance_tests
                WHERE document_id = ?
            """, (doc['id'],))

            for at_row in cursor.fetchall():
                term = at_row['term']
                if term in term_ids or not term_ids:
                    row.term_values[term] = at_row['measured_value']
                    row.term_results[term] = bool(at_row['computed_pass'])

            rows.append(row)

        return rows

    def get_trending_summary(self, project_id: int) -> Dict[str, Dict]:
        """
        Get statistical summary of trending data.
        Returns dict of term_id -> {min, max, avg, count, pass_rate}
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT at.term,
                   MIN(at.measured_value) as min_val,
                   MAX(at.measured_value) as max_val,
                   AVG(at.measured_value) as avg_val,
                   COUNT(*) as count,
                   SUM(CASE WHEN at.computed_pass = 1 THEN 1 ELSE 0 END) as pass_count
            FROM acceptance_tests at
            JOIN project_documents pd ON at.document_id = pd.document_id
            WHERE pd.project_id = ?
              AND at.term IS NOT NULL
              AND at.measured_value IS NOT NULL
            GROUP BY at.term
        """, (project_id,))

        summary = {}
        for row in cursor.fetchall():
            summary[row['term']] = {
                'min': row['min_val'],
                'max': row['max_val'],
                'avg': row['avg_val'],
                'count': row['count'],
                'pass_rate': row['pass_count'] / row['count'] if row['count'] > 0 else None
            }

        return summary

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_trending_csv(self, project_id: int, output_path: Path) -> None:
        """Export trending data to CSV spreadsheet."""
        rows = self.get_trending_data(project_id)
        if not rows:
            return

        # Collect all term IDs
        all_terms = set()
        for row in rows:
            all_terms.update(row.term_values.keys())
        term_list = sorted(all_terms)

        # Build CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header row
            header = ['Serial', 'Test Date', 'Operator'] + term_list
            writer.writerow(header)

            # Data rows
            for row in rows:
                csv_row = [row.serial, row.test_date or '', row.operator or '']
                for term in term_list:
                    val = row.term_values.get(term)
                    csv_row.append(val if val is not None else '')
                writer.writerow(csv_row)

        # Also write a summary section
        summary = self.get_trending_summary(project_id)
        with open(output_path.with_suffix('.summary.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Term', 'Min', 'Max', 'Avg', 'Count', 'Pass Rate'])
            for term in term_list:
                if term in summary:
                    s = summary[term]
                    writer.writerow([
                        term,
                        f"{s['min']:.4f}" if s['min'] is not None else '',
                        f"{s['max']:.4f}" if s['max'] is not None else '',
                        f"{s['avg']:.4f}" if s['avg'] is not None else '',
                        s['count'],
                        f"{s['pass_rate']*100:.1f}%" if s['pass_rate'] is not None else ''
                    ])

    def export_trending_json(self, project_id: int, output_path: Path) -> None:
        """Export trending data to JSON."""
        project = self.get_project(project_id)
        rows = self.get_trending_data(project_id)
        summary = self.get_trending_summary(project_id)

        # Convert project to dict with enum as string
        project_dict = None
        if project:
            project_dict = {
                'id': project.id,
                'name': project.name,
                'description': project.description,
                'grouping_strategy': project.grouping_strategy.value,
                'grouping_value': project.grouping_value,
                'created_at': project.created_at,
                'updated_at': project.updated_at
            }

        data = {
            'project': project_dict,
            'exported_at': datetime.now().isoformat(),
            'documents': [
                {
                    'serial': row.serial,
                    'test_date': row.test_date,
                    'operator': row.operator,
                    'terms': row.term_values,
                    'results': row.term_results
                }
                for row in rows
            ],
            'summary': summary
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_project_from_extraction(
    db_path: Path,
    document_id: int,
    auto_register_terms: bool = True
) -> Tuple[int, int]:
    """
    Set up project and terms from a newly extracted document.
    Returns (project_id, terms_enabled).
    """
    with ProjectManager(db_path) as pm:
        # Register default terms if needed
        if auto_register_terms:
            pm.register_default_terms()

        # Auto-register any new terms from this document
        pm.auto_register_terms_from_documents()

        # Find or create project
        project_id = pm.find_or_create_project_for_document(document_id)

        # Auto-enable terms used in this project
        terms_enabled = 0
        if project_id:
            terms_enabled = pm.auto_enable_terms_for_project(project_id)

        return project_id, terms_enabled


def get_or_create_project(
    db_path: Path,
    name: str,
    grouping_strategy: GroupingStrategy = GroupingStrategy.PROGRAM,
    grouping_value: Optional[str] = None,
    auto_assign: bool = True
) -> int:
    """Get existing project or create new one."""
    with ProjectManager(db_path) as pm:
        project = pm.get_project_by_name(name)
        if project:
            project_id = project.id
        else:
            project_id = pm.create_project(name, None, grouping_strategy, grouping_value)

        if auto_assign and grouping_value:
            pm.auto_assign_documents(project_id)

        return project_id


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python project_manager.py <database.db> <command> [args...]")
        print("\nCommands:")
        print("  list-projects                    List all projects")
        print("  create-project <name> [strategy] Create project (strategy: program/asset/serial_prefix)")
        print("  list-terms                       List registered terms")
        print("  register-defaults                Register default terms")
        print("  auto-register                    Auto-register terms from documents")
        print("  trending <project_id>            Show trending data")
        print("  export-csv <project_id> <file>   Export to CSV")
        print("  setup <document_id>              Setup project from document")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    command = sys.argv[2]

    with ProjectManager(db_path) as pm:
        if command == 'list-projects':
            for p in pm.list_projects():
                docs = pm.get_project_documents(p.id)
                print(f"[{p.id}] {p.name} ({p.grouping_strategy.value}: {p.grouping_value}) - {len(docs)} docs")

        elif command == 'create-project':
            name = sys.argv[3]
            strategy = GroupingStrategy(sys.argv[4]) if len(sys.argv) > 4 else GroupingStrategy.PROGRAM
            value = sys.argv[5] if len(sys.argv) > 5 else None
            pid = pm.create_project(name, None, strategy, value)
            print(f"Created project {pid}: {name}")

        elif command == 'list-terms':
            for t in pm.list_terms():
                print(f"  {t.term_id}: {t.display_name} [{t.category}] ({t.requirement_type})")

        elif command == 'register-defaults':
            count = pm.register_default_terms()
            print(f"Registered {count} default terms")

        elif command == 'auto-register':
            count = pm.auto_register_terms_from_documents()
            print(f"Registered {count} new terms from documents")

        elif command == 'trending':
            project_id = int(sys.argv[3])
            rows = pm.get_trending_data(project_id)
            if rows:
                # Get all terms
                terms = set()
                for r in rows:
                    terms.update(r.term_values.keys())
                terms = sorted(terms)

                # Print header
                print(f"{'Serial':<12} {'Date':<12} " + " ".join(f"{t:<10}" for t in terms))
                print("-" * (24 + 11 * len(terms)))

                # Print rows
                for r in rows:
                    vals = " ".join(f"{r.term_values.get(t, ''):<10}" for t in terms)
                    print(f"{r.serial:<12} {r.test_date or '':<12} {vals}")

        elif command == 'export-csv':
            project_id = int(sys.argv[3])
            output = Path(sys.argv[4])
            pm.export_trending_csv(project_id, output)
            print(f"Exported to {output}")

        elif command == 'setup':
            document_id = int(sys.argv[3])
            project_id, terms = setup_project_from_extraction(db_path, document_id)
            print(f"Document added to project {project_id}, {terms} terms enabled")
