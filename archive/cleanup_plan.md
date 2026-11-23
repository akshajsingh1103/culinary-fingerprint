# Repository Cleanup & Audit Plan (`culinary-fingerprint`)

**Pre-Cleanup Backup Status (Step 1 Completed):**
- **External Zip Backup**: `C:\Users\Akshaj\Documents\culinary-fingerprint_backup_20261123.zip` (`48,937,331` bytes, `267` files; excludes `.git/` and `.venv/`).
- **Pre-Cleanup Git Snapshot**: Committed dirty working tree locally (`Pre-cleanup snapshot: D4 experiments and Streamlit app`; nothing pushed, history intact).
- **Deliverable Naming Check (`D5` / `d5`)**: Searched the entire repo (excluding `.venv/` and `.git/`). Zero `D5` / `Deliverable 5` references exist anywhere (the 29 substring matches are hex characters inside SHA-256 hashes). All deliverables remain strictly **D1–D4**, with the Streamlit app treated as **D4.3** under Deliverable 4.

---

## A. KEEP (Required by D1–D4, Tests, `run_all.py`, App, or Research Record)

| Path / Pattern | Count / Size | One-line Reason |
| --- | --- | --- |
| `.gitignore`, `config.yaml`, `Makefile`, `requirements.txt`, `run_all.py`, `evaluate.py` | 6 root files | Core repo configuration, build/validation orchestration, and entry points (`Makefile`, `run_all.py`, `evaluate.py`). |
| `data/raw/train.json` | 1 file (12.42 MB) | Raw Kaggle *What's Cooking* source dataset (already gitignored). |
| `data/processed/split.json` | 1 file (203.9 KB) | Frozen stratified 80/20 train/test recipe ID split (`seed=42`) read by `src/`, `tests/`, and `outputs/app/export_artifacts.py`. |
| `data/processed/recipes_normalized.json` | 1 file (6.29 MB) | Preprocessed normalized corpus loaded by `src/preprocess.py` (already gitignored). |
| `src/*.py`, `src/d4_experiments/*.py` | 20 Python files | Complete D1–D4 pipeline and Phase 1–3 research codebase imported by `run_all.py`, `evaluate.py`, and `tests/`. |
| `tests/conftest.py`, `tests/test_*.py` | 10 Python files | Full automated test suite (52 passing tests across D1–D4, leakage guards, and app parity/UI). |
| `outputs/d1_dataset/*` | 8 files | D1 node/edge tables, merge log, and dataset/normalization summaries validated by `tests/test_d1.py` and `run_all.py`. |
| `outputs/d2_backbone/*` | 14 files | D2 Curveball-validated backbones, null stats/diagnostics, and offline PyVis graphs validated by `tests/test_d2.py` and `run_all.py`. |
| `outputs/d3_report/*` | 15 files (incl. `figures/`) | D3 hubs, Louvain communities, bridges, `report.md`, `report.pdf`, and 5 figures validated by `tests/test_d3.py` and `run_all.py`. |
| `outputs/d4_scorer/*` | 15 files (incl. `training_artifacts/`) | D4 v1 baseline & NB results, confusion matrices, `garam_masala_diagnostic.json`, and training-only backbones validated by `tests/test_evaluation.py`, `tests/test_fingerprint.py`, `tests/test_no_leakage.py`, `run_all.py`, and `outputs/app/export_artifacts.py`. |
| `outputs/d4_experiments/` non-cache artifacts | 15 files (`results.csv`, `cv_folds.json`, `phase1_summary.json`, `phase2_summary.json`, `phase2_report.md`, `diagnostics.md`, `coverage_accuracy_table.csv`, `accuracy_vs_matched_pairs.png`, `final_test_results.csv`, `final_test_summary.json`, `final_test_report.md`, `phase3/phase3_design.md`, `phase3/phase3_summary.json`, `phase3/phase3_report.md`, `phase3/results.csv`) | Every one of these files is read directly by `tests/test_d4_phase2.py`, `tests/test_d4_phase3.py`, `tests/test_app.py`, or `src/d4_experiments/` and forms the permanent D4 research record. |
| `outputs/poster_outline.md`, `outputs/validation_summary.json` | 2 files | Final summary deliverables checked by `run_all.py --validate-only` (to be updated in Step 5). |
| `outputs/app/` core files (`app.py`, `scorer_runtime.py`, `export_artifacts.py`, `.streamlit/config.toml`, `assets/ingredient_pattern.svg`, `artifacts/*` [17 files], `README.md`, `requirements_app.txt`) | 24 files | Self-contained Streamlit app (D4.3) verified by `tests/test_app.py`. |

---

## B. DELETE (Safe, Regenerable Caches, Junk, or Temporary Backups)

All candidates below were searched across `src/`, `tests/`, `run_all.py`, `Makefile`, `config.yaml`, `evaluate.py`, and `outputs/app/` — **none** are required by `pytest`, `run_all.py --validate-only`, or the running Streamlit app.

| Path / Pattern | Count / Size | One-line Reason |
| --- | --- | --- |
| Stray root file `e Culinary Fingerprint research pipelinegit commit -m Initial release Culinary Fingerprint research pipeline` | 1 file (6,406 B) | Mangled ANSI-colored `git commit --stat` stdout artifact from an accidental shell redirection; zero code/test references. |
| `outputs/app/app_backup_before_cleanup.py`, `outputs/app/app_backup_before_bugfix.py` | 2 files (81.8 KB) | Temporary pre-refactor backups of `app.py`; superseded by git commit + external backup zip; zero code/test references. |
| `.pytest_cache/` | 6 files (4.6 KB) | Regenerable pytest cache; already listed in `.gitignore`. |
| `__pycache__/` in `src/`, `src/d4_experiments/`, `tests/`, `outputs/app/` | 41 files (1.59 MB) | Regenerable Python `.pyc` and Numba `.nbc`/`.nbi` caches; already listed in `.gitignore`. |
| `data/raw/__MACOSX/` (`._train.json`) | 1 file (539 B) | macOS AppleDouble resource-fork metadata junk from zip extraction; already listed in `.gitignore`; zero code/test references. |
| `data/processed/cache/` | 6 `.npz` files (6.65 MB) | Regenerable Curveball null-model `.npz` cache; never read by `pytest`, `--validate-only`, or the app; already listed in `.gitignore`. |
| `outputs/d4_experiments/cache/` & `outputs/d4_experiments/phase3/cache/` | 69 `.npz` files (24.89 MB total: 8.32 MB + 16.57 MB) | Regenerable Curveball null-model `.npz` caches from Phase 1–3 CV builds; never read by `pytest` (`test_d4_phase3.py` uses `tmp_path`), `--validate-only`, or the app; will also add both cache folders to `.gitignore` and untrack them in git (`git rm --cached` + delete). |

*(Note: After Step 4 completes, this temporary `cleanup_plan.md` can either be deleted or moved to `archive/cleanup_plan.md` — please indicate your preference if you'd like it archived rather than removed.)*

---

## C. ARCHIVE (Moved to Top-Level `archive/` and Documented in `archive/README.md`)

| Current Path | Destination in `archive/` | Size | One-line Reason |
| --- | --- | --- | --- |
| `README.md` | `archive/README_old.md` | 41.3 KB | Superseded by the new GitHub-style `README.md` in Step 6; explicitly preserved per instructions. |
| `audit_ingredients.py` | `archive/audit_ingredients.py` | 3.3 KB | Early exploratory raw-ingredient audit script; zero imports or references in `src/`, `tests/`, `run_all.py`, `Makefile`, `config.yaml`, or `outputs/app/`. |
| `outputs/app/screenshots/` (`predict_indian.png`, `predict_italian.png`, `predict_mexican.png`) | `archive/app_screenshots/` | 3 files (2.11 MB) | Visual verification snapshots from the UI bug-fix pass; zero references in code/tests/app; archived to preserve visual record without cluttering `outputs/app/`. |

---

## D. `.gitignore` Audit

Current `.gitignore` covers `.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.numba_cache/`, `data/raw/__MACOSX/`, `data/raw/`, `data/processed/recipes_normalized.json`, and `data/processed/cache/` (with duplicate `data/raw/` and `data/processed/cache/` entries).

In Step 4 (after approval), `.gitignore` will be cleaned up to deduplicate existing entries and also ignore:
- `outputs/d4_experiments/cache/`
- `outputs/d4_experiments/phase3/cache/`
- `*.nbc`, `*.nbi` (Numba compiled cache files)
