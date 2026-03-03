# EIDAT (EIDAT-V1)

This repo contains the EIDAT prototype UI + extraction pipeline, plus an MVP “Production Nodes” layer that lets you:

- Deposit **nodes** into shared repositories (per-team/per-project roots)
- Keep **code centralized** (admin-controlled “Central Runtime”)
- Let end users run **Files Explorer** and **Projects Manager** from simple `.bat` files
- Support **multiple project writers** safely via a SQLite registry (WAL) instead of a shared JSON file

## Glossary

### Central Runtime
The single folder that contains the EIDAT code (`EIDAT_App_Files/`) that everyone runs **from**.

Example (recommended on a share):
- `\\share\EIDAT\CentralRuntime\EIDAT-V1\`
  - `EIDAT_App_Files\...`

You update EIDAT by updating this Central Runtime folder (git pull / copy new version).

### Node Root
The “bite-size” repository root you want EIDAT to manage for a team (where their docs live).
EIDAT deposits an `EIDAT\` folder alongside their content and writes artifacts to `EIDAT\EIDAT Support\` (legacy nodes may still use `EIDAT Support\`).

Example:
- `\\share\TeamA\Repo1\` *(node root)*
  - `EIDAT\...` *(deposited)*
  - `EIDAT\EIDAT Support\...` *(generated/managed)*

## Node layout (deposited into each Node Root)
After deploying a node, the node root contains:

- `EIDAT\`
  - `EIDAT.bat` *(launches the end-user Node GUI; optional arg: `files` / `projects`)*
- `EIDAT\Runtime\`
  - `requirements.lock.txt`
  - `sys_python.txt` *(optional; only needed if `py -3` and `python` are unavailable)*
- `EIDAT\ExtractionNode\`
  - `eidat_node.sqlite3`
- `EIDAT\EIDAT Support\`
  - `projects\projects_registry.sqlite3` *(multi-writer projects registry; WAL)*
  - other pipeline artifacts (debug/ocr, logs, staging, eidat_index.sqlite3, …)

## Permissions (important)
End users must have **Modify** rights on:

- `<node_root>\EIDAT\EIDAT Support\projects\` *(to create/edit projects and update registry)*
- `<node_root>\EIDAT\UserData\` *(to persist user_inputs + GUI settings under the node)*

If these are read-only, the Projects UI will show a clear error telling you what path needs rights.

## Strict metadata allowlists (Program / Vendor / Asset / Doc Type)
EIDAT can enforce strict allowlists for metadata fields via `user_inputs/metadata_candidates.json`:

- Node-local (production nodes): `<node_root>\EIDAT\UserData\user_inputs\metadata_candidates.json`
- Repo-local (dev/runtime fallback): `<runtime_root>\user_inputs\metadata_candidates.json`

Keys:
- `program_titles`: allowed `program_title` values (non-empty list enables enforcement)
- `vendors`: allowed `vendor` values
- `asset_types`: allowed `asset_type` values
- `asset_specific_types`: allowed `asset_specific_type` values
- `part_numbers`: allowed `part_number` values
- `acceptance_test_plan_numbers`: allowed `acceptance_test_plan_number` values
- `document_types`: allowed document types (uses `name`/`acronym`/`aliases`)

Each allowlist can be either:
- `list[str]` (canonical values only), or
- `list[{"name": "...", "aliases": ["...", "..."]}]` (canonical + alias variants)

If an allowlist is empty/missing, the extracted value is forced to `Unknown`.

To backfill old metadata (e.g., remove non-allowlisted program titles), run **Files Explorer → Refresh Metadata Only** on the affected files (it rewrites metadata JSONs and triggers an index refresh).

## Table labeling (combined.txt)
EIDAT can insert stable table labels into OCR `combined.txt` output using rules in `user_inputs/table_label_rules.json`.

- Each matching table gets a label block inserted immediately before it:
  - `[TABLE_LABEL]`
  - `Your Label` (or `Your Label (2)` for repeats)
- Multiple table types are supported by adding multiple entries under `rules`.
- Rule selection is per-table: the highest `priority` match wins (ties default to earlier rule order).
- Optional: set `"tie_breaker": "priority_then_specificity"` to prefer the most-specific rule when priorities tie.

## Table merging (multi-page ASCII tables)
EIDAT can merge bordered ASCII tables that continue across page boundaries in `combined.txt` using `user_inputs/table_merge_heuristics.json`.

- Default behavior is heuristic (layout/headers/titles) and only considers adjacent pages.
- Optional: enable label-driven merging by setting `label_merge_rules` (allowlist of base table labels) and `protect_labeled_tables` to avoid accidental merges of labeled tables.

## Quickstart (admin, shared-drive testing)

### 1) Prepare Central Runtime
Place this repo on a shared drive and treat that folder as the Central Runtime. Confirm:

- `<runtime_root>\EIDAT_App_Files\` exists

Optional (recommended for the admin dashboard): create a runtime venv:

- Run `install.bat` once from the runtime root (creates `EIDAT_App_Files\.venv\...`).

### 2) Launch the Admin Dashboard (GUI)
From the Central Runtime folder, run:

- `run_admin_dashboard.bat`

This opens **EIDAT Admin Dashboard (Nodes)**.

When launched via `run_admin_dashboard.bat`, the registry defaults to:

- `<runtime_root>\admin_registry.sqlite3`

Override registry location (optional):

- Set `EIDAT_ADMIN_REGISTRY_PATH=\\share\...\admin_registry.sqlite3`

### 3) Add/Deploy a node (deposit scaffolding)
In the Admin Dashboard:

1. Click **Add/Deploy Node…**
2. Pick the **Node Root** folder (the repo you want to manage)
3. The dashboard will deposit the node run files + metadata DB and bootstrap the node UI venv

You can also deploy from CLI (run from anywhere that can import `Production`):

```powershell
$env:PYTHONPATH="\\share\\EIDAT\\CentralRuntime\\EIDAT-V1\\EIDAT_App_Files"
py -3 -m Production.deploy --node-root "\\share\\TeamA\\Repo1" --runtime-root "\\share\\EIDAT\\CentralRuntime\\EIDAT-V1"
```

Optional: grant Modify rights during deploy (Windows):

```powershell
py -3 -m Production.deploy --node-root "\\share\\TeamA\\Repo1" --runtime-root "\\share\\EIDAT\\CentralRuntime\\EIDAT-V1" --grant-writers "DOMAIN\\EIDAT_Users"
```

Debugging helper (repo-local node mirror):

- Each deploy also creates/updates a mirror entry under `<runtime_root>\node_mirror\nodes\...` pointing at the node root (junction/symlink/copy fallback).
- To (re)create mirrors for all nodes in the admin registry, run: `tools\mirror_nodes_from_registry.bat`

### 4) End-user usage (no Python packages required globally)
Users run this from the node root:

- `EIDAT\EIDAT.bat`

On launch, the node UI auto-runs a scan so PDFs appear in the Files tab immediately (even before processing).

Venv location overrides (optional):

- Set `EIDAT_VENV_DIR=C:\path\to\.venv` to pin a specific venv folder, or
- Set `EIDAT_VENV_DIR=<node_root>\EIDAT\ExtractionNode\node-ui\.venv` to use the default node-local location

By default, the Admin Dashboard bootstraps the node UI venv during **Add/Deploy Node…** so end users usually won’t see any installation prompts.

If deploy was run with `--no-bootstrap-node-ui` (or requirements changed), first run may prompt to:

1. Create an EIDAT UI environment under `<node_root>\EIDAT\ExtractionNode\node-ui\.venv`
2. Install/upgrade required packages into that environment

The UI is then launched bound to that node root.

Notes:

- The Node GUI is **Files + Projects only** (no extraction controls).
- File status (processed / not processed) is driven by admin-generated `EIDAT Support` SQLite DBs.
  If a file does not appear in the Files list, it has not been scanned/tracked yet.

### 5) Admin node processing
Process nodes from the admin dashboard (per-node **Process** button).

Admin processing always uses the node’s `EIDAT/UserData/.env` (open from the dashboard).
This file is refreshed from the Central Runtime `user_inputs/scanner.env` during Deploy/Repair.

- `EIDAT_PROCESS_FORCE=1` to reprocess already-processed PDFs
- `EIDAT_PROCESS_LIMIT=100` to cap PDFs per run
- `EIDAT_PROCESS_DPI=900` to override DPI for that run (0/blank = use normal config)

Per-node shortcut:

- **Scan+Force Candidates** runs a scan and then force-processes only the newly detected candidates (overwrites pointer tokens without reprocessing unchanged files).

## Projects: multi-writer registry
Projects are registered in:

- `<node_root>\EIDAT\EIDAT Support\projects\projects_registry.sqlite3` *(legacy: `<node_root>\EIDAT Support\projects\...`)*

This replaces the old shared JSON registry and is safe for concurrent writers using SQLite WAL.
If `projects.json` exists, EIDAT will migrate it into SQLite automatically the first time.

## Project-level overrides (per project)
For **EIDP Trending** / **EIDP Raw File Trending** projects, you can override selected `scanner.env` knobs for a single project by creating:

- `<project_dir>\scanner.project.env`

Precedence (last wins):

- Central Runtime `user_inputs/scanner.env`
- Node-local `user_inputs/scanner.local.env`
- Project override `<project_dir>\scanner.project.env`

Node GUI access:

- **Projects** tab → select a project → **Project Env**

## Node-local writable config/data (avoids writing into Central Runtime)
When launched from a node run file, EIDAT sets:

- `EIDAT_NODE_ROOT=<node_root>`
- `EIDAT_DATA_ROOT=<node_root>\EIDAT\UserData`

This makes GUI settings and `user_inputs/*` persist under the node (not under the Central Runtime folder).

## Troubleshooting

### “Python 3 not found” when running a node `.bat`
Install Python 3 (Windows) and ensure either:

- `py -3` works, or
- `python` is on PATH

### “Projects folder is not writable”
Grant Modify rights to:

- `<node_root>\EIDAT\EIDAT Support\projects\` *(legacy: `<node_root>\EIDAT Support\projects\`)*

### Package installation prompts fail (pip/network)
If your company requires an internal index, set env vars before running the `.bat`:

- `PIP_INDEX_URL`
- `PIP_EXTRA_INDEX_URL`
- `PIP_TRUSTED_HOST`

If you use **EIDP Trending** (Trend / Analyze Data), the node UI environment must be able to install `pandas` + `matplotlib`.
If the UI shows “Plotting unavailable. Install matplotlib…”, re-run `EIDAT\\EIDAT.bat` after fixing pip access, or delete the node UI venv and launch again to force a clean bootstrap.

### OCR DPI settings
OCR DPI knobs are documented in `user_inputs/scanner.env` and overridden locally via `user_inputs/scanner.local.env` (for both GUI and pipelines).
Legacy `user_inputs/ocr_force.env` is kept for backward compatibility, but `scanner.env`/`scanner.local.env` take precedence.

### Excel workbook update errors
If EIDAT tells you the workbook isn’t writable, close it in Excel and retry.
