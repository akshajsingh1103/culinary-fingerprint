# Archive Directory

This folder holds historical, superseded, or standalone exploratory files retained for auditability and provenance while keeping the active repository clean.

| Item | Original Location | Description |
| --- | --- | --- |
| `README_old.md` | `README.md` | Original project README written before the D4 Phase 1–3 research extensions and Streamlit app (`v2` scorer). Replaced by the root `README.md`. |
| `audit_ingredients.py` | `audit_ingredients.py` | Early standalone script used to inspect raw Kaggle *What's Cooking* ingredient formatting collisions and recipe-level duplicates prior to `src/preprocess.py` and `src/normalize.py`. Not imported by any pipeline stage, test, or app module. |
| `app_screenshots/` | `outputs/app/screenshots/` | Visual verification screenshots (`predict_indian.png`, `predict_italian.png`, `predict_mexican.png`) captured after the Streamlit Predict tab UI bug-fix pass. |
| `cleanup_plan.md` | `cleanup_plan.md` | The read-only repository audit and cleanup plan approved prior to executing the final-submission cleanup. |
