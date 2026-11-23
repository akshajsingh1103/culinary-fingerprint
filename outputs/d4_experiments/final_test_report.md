# Deliverable 4 — Final Held-Out Test Set Evaluation Report (`N = 3,456` Test Recipes)

> [!IMPORTANT]
> **One-Time Pre-Registered Final Evaluation**:
> - Evaluated **once** on the 3,456 held-out test recipes (`test_ids` in `data/processed/split.json`), with zero retuning or post-hoc changes.
> - All four pre-registered models (`M0`, `M1`, `M4`, `M5`) were trained strictly on the **13,823 training recipes** using the existing training-only D2 Curveball backbones (`outputs/d4_scorer/training_artifacts`, 1,000 randomizations) and D3 definitions (`src/analysis.py`, `src/graph_utils.py`).
> - Training meta-features for `M4` (25 features) and `M5` (22 features) were constructed via 3-fold inner cross-fitting on the 13,823 training recipes (`seed = 42`), and meta-model L2 regularization $C$ was chosen by inner CV on the training set using **Macro-F1** (primary metric).
> - Original D1–D3 and D4 artifacts (`outputs/d4_scorer/results.csv`, etc.) remain completely untouched.

---

## 1. Primary Held-Out Test Results (`N = 3,456` Test Recipes; Macro-F1 Primary, Accuracy Secondary)

Saved to `outputs/d4_experiments/final_test_results.csv`:

| Pre-Registered Model | Test Macro-F1 [95% Bootstrap CI] (PRIMARY) | 5-Fold Train CV Macro-F1 (Mean ± SD) | Test Accuracy [95% Bootstrap CI] (SECONDARY) | 5-Fold Train CV Accuracy (Mean ± SD) | Test F1 `indian` | Test F1 `italian` | Test F1 `mexican` | Selected Meta $C$ | Zero-Score Fallbacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Original Fingerprint Scorer** | **87.12%** [85.83%, 88.34%] | 85.82% ± 0.92% | **87.67%** [86.54%, 88.80%] | 86.62% ± 0.77% | 85.49% | 89.37% | 86.50% | — | 139 |
| **M1: Naive Bayes** | **95.99%** [95.30%, 96.61%] | 95.53% ± 0.62% | **95.98%** [95.31%, 96.59%] | 95.66% ± 0.58% | 96.25% | 96.33% | 95.40% | — | 0 |
| **M4: Full Stack (NB + network features)** | **95.92%** [95.25%, 96.57%] | 95.82% ± 0.44% | **95.89%** [95.25%, 96.53%] | 95.88% ± 0.46% | 96.24% | 96.28% | 95.24% | `1.0` | 0 |
| **M5: Network-Only Fingerprint Scorer v2** | **93.99%** [93.10%, 94.77%] | 93.84% ± 0.46% | **94.13%** [93.29%, 94.88%] | 94.04% ± 0.47% | 93.68% | 94.80% | 93.49% | `1.0` | 0 |

### 1.1 Per-Cuisine Precision, Recall, and F1 on the Held-Out Test Set (`N = 3,456`)

| Model | `indian` ($N=600$) P / R / F1 (%) | `italian` ($N=1,568$) P / R / F1 (%) | `mexican` ($N=1,288$) P / R / F1 (%) | Test Macro-F1 (%) | Test Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M0: Original Fingerprint Scorer** | 91.78 / 80.00 / **85.49** | 87.15 / 91.71 / **89.37** | 86.67 / 86.34 / **86.50** | **87.12%** | **87.67%** |
| **M1: Naive Bayes** | 96.33 / 96.17 / **96.25** | 94.70 / 98.02 / **96.33** | 97.49 / 93.40 / **95.40** | **95.99%** | **95.98%** |
| **M4: Full Stack (NB + network features)** | 96.48 / 96.00 / **96.24** | 95.20 / 97.39 / **96.28** | 96.49 / 94.02 / **95.24** | **95.92%** | **95.89%** |
| **M5: Network-Only Fingerprint Scorer v2** | 96.14 / 91.33 / **93.68** | 93.11 / 96.56 / **94.80** | 94.52 / 92.47 / **93.49** | **93.99%** | **94.13%** |

---

## 2. Paired Statistical Comparisons & Recall Regression Guard on the Held-Out Test Set

| Paired Comparison (`Model A vs. Model B`) | Test $\Delta$ Macro-F1 (`A - B`) [95% Bootstrap CI] | CI Excludes Zero? | Test $\Delta$ Accuracy (`A - B`) [95% Bootstrap CI] | A Right, B Wrong ($b$) | A Wrong, B Right ($c$) | Discordant ($b+c$) | Net Test Recipes ($b-c$) | McNemar $\chi^2$ (CC) | Exact Binomial $p$-value | $\Delta$ Recall (`IN / IT / MX`, pp) | Recall Regression Guard ($>1\text{ pp}$ drop)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M5 (Network-Only Fingerprint Scorer v2) vs. M0 (Original Fingerprint Scorer)** | **+6.87 pp** [+5.66, +8.04] | **YES** | **+6.45 pp** [+5.35, +7.52] | 287 | 64 | 351 | **+223** | 140.41 | `7.658e-35` | `+11.33 / +4.85 / +6.13` | Clear (No drop > 1 pp) |
| **M4 (Full Stack: NB + network features) vs. M1 (Naive Bayes)** | **-0.07 pp** [-0.56, +0.40] | No (includes 0) | **-0.09 pp** [-0.52, +0.35] | 27 | 30 | 57 | **-3** | 0.07 | `0.7914` | `-0.17 / -0.64 / +0.62` | Clear (No drop > 1 pp) |

---

## 3. Held-Out Test Confusion Matrices (`N = 3,456`; Rows = True Cuisine, Columns = Predicted Cuisine)

| Model | True Cuisine | Pred `indian` | Pred `italian` | Pred `mexican` | Row Total | Per-Cuisine Precision (%) | Per-Cuisine Recall (%) | Per-Cuisine F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Original Fingerprint Scorer** | `indian` | 480 | 57 | 63 | 600 | 91.78% | 80.00% | **85.49%** |
| **M0: Original Fingerprint Scorer** | `italian` | 22 | 1,438 | 108 | 1,568 | 87.15% | 91.71% | **89.37%** |
| **M0: Original Fingerprint Scorer** | `mexican` | 21 | 155 | 1,112 | 1,288 | 86.67% | 86.34% | **86.50%** |
| **M1: Naive Bayes** | `indian` | 577 | 15 | 8 | 600 | 96.33% | 96.17% | **96.25%** |
| **M1: Naive Bayes** | `italian` | 8 | 1,537 | 23 | 1,568 | 94.70% | 98.02% | **96.33%** |
| **M1: Naive Bayes** | `mexican` | 14 | 71 | 1,203 | 1,288 | 97.49% | 93.40% | **95.40%** |
| **M4: Full Stack (NB + network features)** | `indian` | 576 | 11 | 13 | 600 | 96.48% | 96.00% | **96.24%** |
| **M4: Full Stack (NB + network features)** | `italian` | 10 | 1,527 | 31 | 1,568 | 95.20% | 97.39% | **96.28%** |
| **M4: Full Stack (NB + network features)** | `mexican` | 11 | 66 | 1,211 | 1,288 | 96.49% | 94.02% | **95.24%** |
| **M5: Network-Only Fingerprint Scorer v2** | `indian` | 548 | 24 | 28 | 600 | 96.14% | 91.33% | **93.68%** |
| **M5: Network-Only Fingerprint Scorer v2** | `italian` | 13 | 1,514 | 41 | 1,568 | 93.11% | 96.56% | **94.80%** |
| **M5: Network-Only Fingerprint Scorer v2** | `mexican` | 9 | 88 | 1,191 | 1,288 | 94.52% | 92.47% | **93.49%** |

---

## 4. Full-Training D2/D3 Network Summary & Standardized Meta-Model Coefficients (`13,823` Training Recipes)

- **Full-Training D2 Backbone (`1,000` Curveball randomizations from `outputs/d4_scorer/training_artifacts`)**:
  - Backbone nodes: `indian`: `237`, `italian`: `413`, `mexican`: `350`.
  - Backbone edges: `indian`: `1,466`, `italian`: `2,973`, `mexican`: `2,223` (union validated backbone pairs = `6,149`).
  - Louvain communities (`seed=42, resolution=1.0`): `indian`: `7`, `italian`: `7`, `mexican`: `7`.
  - Top-10 Weighted-Degree Hubs: `indian`: `bay leaf, cinnamon sticks, clove, garlic paste, green cardamom, green chilies, mace, oil, red chili powder, star anise` | `italian`: `all-purpose flour, baking powder, carrot, cooking spray, dried basil, dry white wine, onion, sugar, tomato paste, vanilla extract` | `mexican`: `all-purpose flour, ancho chile pepper, baking powder, cumin, dried oregano, garlic cloves, ground cinnamon, ground cumin, oregano, white onion`.
  - D3 Top-5 Pairwise Bridges per Cuisine: `indian`: `bay leaf, bay leaves, black peppercorn, cinnamon, cinnamon sticks, clove, garlic, heavy cream, onion, paprika` | `italian`: `all-purpose flour, baking powder, bay leaf, bay leaves, dried oregano, garlic, garlic cloves, heavy cream, onion` | `mexican`: `all-purpose flour, baking powder, black peppercorn, cinnamon, cinnamon sticks, clove, dried oregano, garlic, garlic cloves, paprika`.
- **Inner 3-Fold CV Selected Regularization $C$ (on 13,823 Cross-Fitted Training Rows)**:
  - `M4` selected $C = 1.0$ (inner-CV Macro-F1 = `95.75%`).
  - `M5` selected $C = 1.0$ (inner-CV Macro-F1 = `93.65%`).

---

## Appendix: Supplementary Test Weighted-F1 (Metric Hierarchy Level 5)

| Model | Held-Out Test Weighted-F1 (%) |
| :--- | :---: |
| **M0: Original Fingerprint Scorer** | 87.63% |
| **M1: Naive Bayes** | 95.97% |
| **M4: Full Stack (NB + network features)** | 95.89% |
| **M5: Network-Only Fingerprint Scorer v2** | 94.12% |
