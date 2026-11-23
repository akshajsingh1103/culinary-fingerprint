# Deliverable 4 — Phase 3 Report: Network-Informed Fingerprint Scorer v2 (Stacked)

> [!IMPORTANT]
> **Held-Out Test Set Sealed & Metric Hierarchy Enforced**:
> 1. **Test Set Untouched**: The 3,456 held-out test recipes (`test_ids` in `data/processed/split.json`) were **never loaded, inspected, or evaluated**. All results below come strictly from nested 5-fold cross-validation (`outputs/d4_experiments/cv_folds.json`, `seed = 42`) across the 13,823 training recipes, with 3-fold inner cross-fitting (`n_randomizations = 50` documented inner shortcut; `n_randomizations = 100` outer builds) so that every training row's 25 features are computed from D1 networks, D2 backbones, D3 hubs/communities/bridges, and Naive Bayes models fitted **without that recipe**.
> 2. **Primary Metric Hierarchy**: **Macro-F1** (unweighted mean of per-cuisine $F_1$) is the **primary metric** for all model comparisons and inner-CV hyperparameter selection ($C$), with **Accuracy** secondary, per-cuisine Precision/Recall/$F_1$ tertiary, pooled OOF confusion matrices quaternary, and **Weighted-$F_1$** reported solely in the Appendix. Every paired comparison is checked against the **Recall Regression Guard** (flagging any $> 1.0\text{ pp}$ drop in any single cuisine's recall) and the **Bootstrap CI Honesty Guard** (making no claim of improvement if the 95% paired bootstrap CI for $\Delta\text{Macro-F1}$ includes zero).

---

## Executive Summary & Plain-Spoken Findings

### 1. Does the Pure Network-Only Fingerprint Scorer v2 (`M5`) Improve Upon the Original Fingerprint Scorer (`M0`)?
**Yes — massively and unambiguously across every metric and every cuisine.**
- **`M5` (Network-Only Fingerprint Scorer v2, 22 D2/D3 network features, zero unigram NB features)** achieves **93.84% ± 0.46% CV Macro-F1** (**93.84% pooled OOF Macro-F1**) and **94.04% ± 0.47% CV Accuracy**, compared to **`M0` (Original Fingerprint Scorer)** at **85.82% ± 0.92% CV Macro-F1** and **86.62% ± 0.77% CV Accuracy**.
- **Pooled Paired Gain (`M5 minus M0`)**:
  - **$\Delta$ Macro-F1 (Primary)**: **+8.01 percentage points** (`95% paired bootstrap CI: [+7.44 pp, +8.61 pp]` — strictly excludes zero).
  - **$\Delta$ Accuracy (Secondary)**: **+7.42 percentage points** (`95% paired bootstrap CI: [+6.87 pp, +7.96 pp]`).
  - **McNemar Discordant Counts**: `M5` is correct on **1,291** recipes where `M0` errs, while `M0` is correct on only **266** recipes where `M5` errs (a net gain of **+1,025 recipes** out of 13,823; $\chi^2_{\text{CC}} = 673.46$, exact binomial $p = 1.97 \times 10^{-161}$).
  - **Recall Regression Guard**: **PASSED** (`M5` improves recall in **all three cuisines simultaneously**: `indian` recall **+13.52 pp** [`78.36% -> 91.89%`], `italian` recall **+4.55 pp** [`91.80% -> 96.35%`], `mexican` recall **+8.06 pp** [`84.17% -> 92.23%`]).
- **Where does `M5`'s gain come from?** Even compared against a stacked logistic regression using *only* the 3 backbone pair log-odds scores (`Abl_M5a: 90.19%` Macro-F1, `90.49%` Accuracy), adding the 19 D3 structural features (Louvain community coverage/max-coverage/entropy, D3 top-10 weighted-degree hubs, D3 top-5 pairwise bridges, backbone node counts, and recipe length) adds a further **+3.65 pp Macro-F1** (`95% CI: [+3.20 pp, +4.13 pp]`, $+490$ net recipes, $p = 2.41 \times 10^{-58}$) — and remarkably, **D3 structural summaries alone without any pair log-odds (`Abl_M5b`) achieve 92.82% ± 0.75% Macro-F1 (93.13% Accuracy)**, outperforming `M0` by **+7.00 pp Macro-F1**!

### 2. Do Compact Network Features Improve Over Naive Bayes When Stacked Together (`M4 vs. M1` and `M4 vs. M2`)?
**Yes —unlike the 4,865 raw binary pair indicators in Phase 2 (which hurt unigram models by -0.21 to -0.35 pp), compact low-dimensional D1–D3 network summaries (`M4`, 25 features) produce a statistically significant improvement in Macro-F1 and Accuracy over both standalone Naive Bayes (`M1`) and Calibrated Naive Bayes (`M2`), with zero recall regressions:**
- **`M4` (Full Stack: NB + All Network Features, 25 features)**: **95.82% ± 0.44% CV Macro-F1** (**95.82% pooled OOF Macro-F1**) and **95.88% ± 0.46% CV Accuracy**.
- **`M1` (Naive Bayes Alone)**: **95.53% ± 0.62% CV Macro-F1** and **95.66% ± 0.58% CV Accuracy**.
- **`M2` (Calibrated NB Stack Alone, 3 features — sanity check)**: **95.50% ± 0.63% CV Macro-F1** and **95.64% ± 0.63% CV Accuracy** (confirming that simply feeding NB log-posteriors into an L2 logistic regression meta-model provides **0.00 pp gain** on its own: $\Delta\text{Macro-F1} = -0.03\text{ pp}$, $95\%\text{ CI } [-0.19, +0.13]$, McNemar $p = 0.824$).
- **Pooled Paired Gain (`M4 minus M1`)**:
  - **$\Delta$ Macro-F1 (Primary)**: **+0.28 percentage points** (`95% paired bootstrap CI: [+0.05 pp, +0.52 pp]` — **excludes zero**).
  - **$\Delta$ Accuracy (Secondary)**: **+0.22 percentage points** (`95% paired bootstrap CI: [+0.01 pp, +0.42 pp]`).
  - **McNemar Test**: `116` recipes fixed vs. `86` broken (**+30 net recipes**, exact binomial $p = 0.0410$).
  - **Recall Regression Guard**: **PASSED** (`indian` recall **+0.37 pp**, `mexican` recall **+0.95 pp**, `italian` recall **-0.45 pp** — well within the $1.0\text{ pp}$ guard threshold).
- **Pooled Paired Gain (`M4 minus M2`, isolating the pure contribution of the 22 network features inside the same L2 meta-learner)**:
  - **$\Delta$ Macro-F1 (Primary)**: **+0.31 percentage points** (`95% paired bootstrap CI: [+0.07 pp, +0.57 pp]` — **excludes zero**).
  - **$\Delta$ Accuracy (Secondary)**: **+0.24 percentage points** (`95% paired bootstrap CI: [+0.03 pp, +0.46 pp]`).
  - **McNemar Test**: `121` recipes fixed vs. `88` broken (**+33 net recipes**, exact binomial $p = 0.0266$).
  - **Recall Regression Guard**: **PASSED** (`indian` recall **+1.29 pp**, `mexican` recall **+0.41 pp**, `italian` recall **-0.30 pp**).
- **Honest Nuance on `M3` (`NB + Backbone Pair Log-Odds Only`, 6 features)**:
  - Adding *only* the 3 backbone pair log-odds scores to NB (`M3: 95.56% ± 0.65%` Macro-F1) yields a tiny **+0.03 pp** Macro-F1 difference vs. `M1` (`95% CI: [-0.14 pp, +0.20 pp]`, $p = 0.714$) and **+0.06 pp** vs. `M2` (`95% CI: [-0.07 pp, +0.19 pp]`, $p = 0.403$). Because these CIs include zero, **backbone pair log-odds alone do not improve upon Naive Bayes**; the statistically significant gain in `M4` comes specifically from the **D2/D3 structural graph features (hubs, communities, bridges, and backbone node coverage)** correcting Naive Bayes's systematic bias on multi-ingredient recipes.

---

## 1. Cross-Fitting & Zero-Leakage Verification Log

Every outer fold matches `outputs/d4_experiments/cv_folds.json` identically. Inside each outer fold $f \in \{0,1,2,3,4\}$, every single training recipe's 25 features in $X_{\text{train}}^{(f)}$ were generated out-of-fold by an inner 3-fold `PartitionNetworkFeatureExtractor` fitted strictly on the other $2/3$ of outer-training recipes (`~7,372` recipes, rebuilding ingredient normalization, eligible nodes, 50-randomization Curveball backbones, D3 hubs, Louvain communities, D3 bridges, Naive Bayes, and pair log-odds from scratch without that recipe):

| Outer Fold | Outer Train $N$ | Outer Val $N$ | Train SHA-256 (16 hex) | Val SHA-256 (16 hex) | All Train Rows Cross-Fitted Out-of-Fold | Inner Folds / Curveball $N_{\text{rand}}$ | Outer Curveball $N_{\text{rand}}$ | Train $\cap$ Val | Train $\cap$ Test | Val $\cap$ Test | Selected Meta-Model $C$ (`M2 / M3 / M4 / M5`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 11,058 | 2,765 | `8be41856dcd9a171...` | `db78da3dc22ce562...` | `True` | 3 / `50` | `100` | `0` | `0` | `0` | `1.0` / `0.5` / `1.0` / `1.0` |
| 1 | 11,058 | 2,765 | `a7e2428afbadec3f...` | `ce475637f561f58b...` | `True` | 3 / `50` | `100` | `0` | `0` | `0` | `0.5` / `0.25` / `0.5` / `1.0` |
| 2 | 11,058 | 2,765 | `6c0ede71d232240e...` | `7b0a74b58e4f7ae7...` | `True` | 3 / `50` | `100` | `0` | `0` | `0` | `0.25` / `10.0` / `5.0` / `0.5` |
| 3 | 11,059 | 2,764 | `a5857703377afd16...` | `2e16c1c2937596e0...` | `True` | 3 / `50` | `100` | `0` | `0` | `0` | `1.0` / `0.25` / `1.0` / `0.5` |
| 4 | 11,059 | 2,764 | `7e44080ffb2826be...` | `81168614a6826e98...` | `True` | 3 / `50` | `100` | `0` | `0` | `0` | `2.0` / `0.05` / `2.0` / `2.0` |

---

## 2. Primary Results Table: Macro-F1 (Primary) & Accuracy (Secondary) Across Outer Folds

### 2.1 Summary Across 5 Outer Folds and Pooled Out-of-Fold (`13,823` Training Recipes)

| Model | CV Macro-F1 (Mean ± SD) [PRIMARY] | Pooled OOF Macro-F1 [95% Bootstrap CI] | CV Accuracy (Mean ± SD) [SECONDARY] | Pooled OOF Accuracy [95% Bootstrap CI] | F1 `indian` (Mean) | F1 `italian` (Mean) | F1 `mexican` (Mean) | Zero-Score Fallbacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Original Fingerprint Scorer (reference)** | **85.82% ± 0.92%** | **85.83%** [85.19%, 86.44%] | 86.62% ± 0.77% | 86.62% [86.07%, 87.19%] | 83.38% | 88.78% | 85.31% | 676 |
| **M1: Naive Bayes Alone (reference)** | **95.53% ± 0.62%** | **95.53%** [95.16%, 95.88%] | 95.66% ± 0.58% | 95.66% [95.33%, 96.01%] | 95.21% | 96.23% | 95.16% | 0 |
| **M2: Calibrated NB Stack Alone (3 features)** | **95.50% ± 0.63%** | **95.50%** [95.14%, 95.87%] | 95.64% ± 0.63% | 95.64% [95.29%, 95.98%] | 95.16% | 96.27% | 95.07% | 0 |
| **M3: NB + Backbone Pair Log-Odds Stack (6 features)** | **95.56% ± 0.65%** | **95.56%** [95.20%, 95.92%] | 95.70% ± 0.64% | 95.70% [95.36%, 96.04%] | 95.20% | 96.27% | 95.21% | 0 |
| **M4: Full Stack — NB + All Network Features (25 features)** | **95.82% ± 0.44%** | **95.82%** [95.48%, 96.17%] | 95.88% ± 0.46% | 95.88% [95.55%, 96.21%] | 95.76% | 96.38% | 95.31% | 0 |
| **M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)** | **93.84% ± 0.46%** | **93.84%** [93.40%, 94.26%] | 94.04% ± 0.47% | 94.04% [93.63%, 94.43%] | 93.22% | 94.69% | 93.60% | 0 |

### 2.2 Fold-by-Fold Macro-F1 (Primary) and Accuracy (Secondary) for `M0`–`M5`

| Model | Fold 0 Macro-F1 / Acc | Fold 1 Macro-F1 / Acc | Fold 2 Macro-F1 / Acc | Fold 3 Macro-F1 / Acc | Fold 4 Macro-F1 / Acc | 5-Fold Mean ± SD Macro-F1 | 5-Fold Mean ± SD Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Original Fingerprint Scorer (reference)** | 84.88% / 85.61% | 86.05% / 86.87% | 85.17% / 86.22% | 85.77% / 86.76% | 87.23% / 87.66% | **85.82% ± 0.92%** | 86.62% ± 0.77% |
| **M1: Naive Bayes Alone (reference)** | 94.69% / 94.97% | 95.97% / 96.17% | 95.15% / 95.15% | 95.62% / 95.77% | 96.23% / 96.24% | **95.53% ± 0.62%** | 95.66% ± 0.58% |
| **M2: Calibrated NB Stack Alone (3 features)** | 94.80% / 95.05% | 96.03% / 96.27% | 94.86% / 94.90% | 95.72% / 95.84% | 96.09% / 96.13% | **95.50% ± 0.63%** | 95.64% ± 0.63% |
| **M3: NB + Backbone Pair Log-Odds Stack (6 features)** | 94.80% / 95.05% | 96.17% / 96.38% | 94.95% / 95.05% | 95.73% / 95.77% | 96.16% / 96.24% | **95.56% ± 0.65%** | 95.70% ± 0.64% |
| **M4: Full Stack — NB + All Network Features (25 features)** | 95.55% / 95.59% | 96.36% / 96.49% | 95.49% / 95.41% | 95.45% / 95.66% | 96.23% / 96.24% | **95.82% ± 0.44%** | 95.88% ± 0.46% |
| **M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)** | 93.70% / 93.96% | 93.69% / 94.07% | 93.46% / 93.45% | 93.70% / 93.96% | 94.64% / 94.75% | **93.84% ± 0.46%** | 94.04% ± 0.47% |

### 2.3 Per-Cuisine Precision, Recall, and F1 on Every Outer Fold (Tertiary Metric Hierarchy Requirement)

| Model | Fold | `indian` P / R / F1 (%) | `italian` P / R / F1 (%) | `mexican` P / R / F1 (%) | Fold Macro-F1 (%) | Fold Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** | 0 | 88.81 / 77.55 / 82.80 | 84.21 / 91.47 / 87.69 | 86.16 / 82.23 / 84.15 | **84.88%** | 85.61% |
| **M0** | 1 | 90.36 / 77.96 / 83.71 | 86.73 / 92.26 / 89.41 | 85.63 / 84.47 / 85.04 | **86.05%** | 86.87% |
| **M0** | 2 | 88.78 / 75.68 / 81.71 | 86.07 / 91.15 / 88.54 | 85.39 / 85.15 / 85.27 | **85.17%** | 86.22% |
| **M0** | 3 | 88.18 / 77.71 / 82.61 | 86.15 / 92.26 / 89.10 | 86.97 / 84.27 / 85.60 | **85.77%** | 86.76% |
| **M0** | 4 | 89.44 / 82.92 / 86.05 | 86.62 / 91.87 / 89.16 | 88.27 / 84.76 / 86.48 | **87.23%** | 87.66% |
| **M1** | 0 | 92.87 / 94.80 / 93.83 | 94.57 / 97.29 / 95.91 | 96.54 / 92.23 / 94.34 | **94.69%** | 94.97% |
| **M1** | 1 | 95.22 / 95.22 / 95.22 | 95.41 / 97.77 / 96.57 | 97.60 / 94.66 / 96.11 | **95.97%** | 96.17% |
| **M1** | 2 | 95.61 / 95.01 / 95.31 | 93.98 / 97.05 / 95.49 | 96.47 / 92.91 / 94.66 | **95.15%** | 95.15% |
| **M1** | 3 | 95.03 / 95.62 / 95.33 | 95.19 / 97.93 / 96.54 | 96.87 / 93.20 / 95.00 | **95.62%** | 95.77% |
| **M1** | 4 | 96.26 / 96.46 / 96.36 | 95.70 / 97.61 / 96.64 | 96.91 / 94.47 / 95.67 | **96.23%** | 96.24% |
| **M2** | 0 | 93.80 / 94.39 / 94.09 | 94.86 / 97.21 / 96.02 | 95.88 / 92.72 / 94.27 | **94.80%** | 95.05% |
| **M2** | 1 | 96.18 / 94.18 / 95.17 | 95.71 / 97.93 / 96.81 | 97.03 / 95.24 / 96.13 | **96.03%** | 96.27% |
| **M2** | 2 | 96.36 / 93.56 / 94.94 | 94.25 / 96.65 / 95.43 | 95.06 / 93.40 / 94.22 | **94.86%** | 94.90% |
| **M2** | 3 | 96.00 / 95.00 / 95.50 | 95.33 / 97.69 / 96.49 | 96.41 / 93.98 / 95.18 | **95.72%** | 95.84% |
| **M2** | 4 | 96.83 / 95.42 / 96.12 | 95.77 / 97.45 / 96.60 | 96.26 / 94.85 / 95.55 | **96.09%** | 96.13% |
| **M3** | 0 | 93.98 / 94.18 / 94.08 | 95.00 / 97.05 / 96.02 | 95.61 / 93.01 / 94.29 | **94.80%** | 95.05% |
| **M3** | 1 | 96.59 / 94.18 / 95.37 | 95.86 / 97.77 / 96.80 | 96.95 / 95.73 / 96.34 | **96.17%** | 96.38% |
| **M3** | 2 | 95.18 / 94.39 / 94.78 | 94.61 / 96.65 / 95.62 | 95.53 / 93.40 / 94.45 | **94.95%** | 95.05% |
| **M3** | 3 | 96.61 / 95.00 / 95.80 | 95.45 / 97.13 / 96.28 | 95.77 / 94.47 / 95.11 | **95.73%** | 95.77% |
| **M3** | 4 | 97.42 / 94.58 / 95.98 | 95.92 / 97.37 / 96.64 | 96.10 / 95.63 / 95.86 | **96.16%** | 96.24% |
| **M4** | 0 | 95.26 / 96.05 / 95.65 | 95.45 / 96.97 / 96.20 | 95.92 / 93.69 / 94.79 | **95.55%** | 95.59% |
| **M4** | 1 | 96.23 / 95.63 / 95.93 | 96.23 / 97.69 / 96.95 | 96.94 / 95.44 / 96.18 | **96.36%** | 96.49% |
| **M4** | 2 | 96.44 / 95.63 / 96.03 | 94.77 / 96.81 / 95.78 | 95.73 / 93.59 / 94.65 | **95.49%** | 95.41% |
| **M4** | 3 | 94.24 / 95.42 / 94.82 | 95.83 / 97.05 / 96.43 | 96.13 / 94.08 / 95.09 | **95.45%** | 95.66% |
| **M4** | 4 | 96.45 / 96.25 / 96.35 | 96.20 / 96.89 / 96.54 | 96.18 / 95.44 / 95.81 | **96.23%** | 96.24% |
| **M5** | 0 | 94.62 / 91.48 / 93.02 | 93.04 / 97.05 / 95.00 | 94.86 / 91.36 / 93.08 | **93.70%** | 93.96% |
| **M5** | 1 | 93.78 / 90.85 / 92.29 | 92.55 / 97.05 / 94.75 | 96.24 / 91.94 / 94.04 | **93.69%** | 94.07% |
| **M5** | 2 | 96.07 / 91.48 / 93.72 | 92.31 / 95.69 / 93.97 | 93.74 / 91.65 / 92.69 | **93.46%** | 93.45% |
| **M5** | 3 | 94.21 / 91.46 / 92.81 | 93.40 / 95.93 / 94.65 | 94.55 / 92.72 / 93.63 | **93.70%** | 93.96% |
| **M5** | 4 | 94.36 / 94.17 / 94.26 | 94.21 / 96.01 / 95.10 | 95.63 / 93.50 / 94.55 | **94.64%** | 94.75% |

### 2.4 Pooled Out-of-Fold Confusion Matrices (Quaternary Metric Hierarchy Requirement; Rows = True, Cols = Predicted)

| Model | True Cuisine | Pred `indian` | Pred `italian` | Pred `mexican` | Pooled Precision (%) | Pooled Recall (%) | Pooled F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** | `indian` | 1,883 | 265 | 255 | 89.12% | 78.36% | **83.39%** |
| **M0** | `italian` | 91 | 5,756 | 423 | 85.95% | 91.80% | **88.78%** |
| **M0** | `mexican` | 139 | 676 | 4,335 | 86.48% | 84.17% | **85.31%** |
| **M1** | `indian` | 2,293 | 74 | 36 | 94.99% | 95.42% | **95.20%** |
| **M1** | `italian` | 36 | 6,115 | 119 | 94.97% | 97.53% | **96.23%** |
| **M1** | `mexican` | 85 | 250 | 4,815 | 96.88% | 93.50% | **95.16%** |
| **M2** | `indian` | 2,271 | 66 | 66 | 95.82% | 94.51% | **95.16%** |
| **M2** | `italian` | 35 | 6,106 | 129 | 95.18% | 97.38% | **96.27%** |
| **M2** | `mexican` | 64 | 243 | 4,843 | 96.13% | 94.04% | **95.07%** |
| **M3** | `indian` | 2,270 | 69 | 64 | 95.94% | 94.47% | **95.20%** |
| **M3** | `italian` | 37 | 6,094 | 139 | 95.37% | 97.19% | **96.27%** |
| **M3** | `mexican` | 59 | 227 | 4,864 | 95.99% | 94.45% | **95.21%** |
| **M4** | `indian` | 2,302 | 50 | 51 | 95.72% | 95.80% | **95.76%** |
| **M4** | `italian` | 41 | 6,087 | 142 | 95.69% | 97.08% | **96.38%** |
| **M4** | `mexican` | 62 | 224 | 4,864 | 96.18% | 94.45% | **95.31%** |
| **M5** | `indian` | 2,208 | 116 | 79 | 94.60% | 91.89% | **93.22%** |
| **M5** | `italian` | 58 | 6,041 | 171 | 93.10% | 96.35% | **94.69%** |
| **M5** | `mexican` | 68 | 332 | 4,750 | 95.00% | 92.23% | **93.60%** |

---

## 3. Pooled Out-of-Fold Paired Statistical Comparisons & Recall Regression Guard Checks

| Paired Comparison (`Model A vs. Model B`) | Pooled $\Delta$ Macro-F1 (`A - B`) [95% Bootstrap CI] | CI Excludes Zero? | Pooled $\Delta$ Accuracy (`A - B`) [95% Bootstrap CI] | A Right, B Wrong ($b$) | A Wrong, B Right ($c$) | Net Recipes ($b-c$) | McNemar $\chi^2$ (CC) | Exact Binomial $p$-value | $\Delta$ Recall (`IN / IT / MX`, pp) | Recall Regression Guard ($>1\text{ pp}$ drop)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M2 (Calibrated NB Stack Alone) vs. M1 (Naive Bayes Alone)** | **-0.03 pp** [-0.19, +0.13] | No (includes 0) | **-0.02 pp** [-0.15, +0.11] | 39 | 42 | **-3** | 0.05 | `0.8243` | `-0.92 / -0.14 / +0.54` |  Clear (No drop > 1 pp) |
| **M3 (NB + Backbone Pair Log-Odds) vs. M1 (Naive Bayes Alone)** | **+0.03 pp** [-0.16, +0.21] | No (includes 0) | **+0.04 pp** [-0.12, +0.19] | 62 | 57 | **+5** | 0.13 | `0.714` | `-0.96 / -0.33 / +0.95` |  Clear (No drop > 1 pp) |
| **M3 (NB + Backbone Pair Log-Odds) vs. M2 (Calibrated NB Stack Alone)** | **+0.06 pp** [-0.07, +0.19] | No (includes 0) | **+0.06 pp** [-0.06, +0.17] | 39 | 31 | **+8** | 0.70 | `0.403` | `-0.04 / -0.19 / +0.41` |  Clear (No drop > 1 pp) |
| **M4 (Full Stack: NB + All Network Features) vs. M1 (Naive Bayes Alone)** | **+0.28 pp** [+0.05, +0.52] | **YES** | **+0.22 pp** [+0.01, +0.42] | 116 | 86 | **+30** | 4.16 | `0.04104` | `+0.37 / -0.45 / +0.95` |  Clear (No drop > 1 pp) |
| **M4 (Full Stack: NB + All Network Features) vs. M2 (Calibrated NB Stack Alone)** | **+0.31 pp** [+0.07, +0.57] | **YES** | **+0.24 pp** [+0.03, +0.46] | 121 | 88 | **+33** | 4.90 | `0.02663` | `+1.29 / -0.30 / +0.41` |  Clear (No drop > 1 pp) |
| **M5 (Network-Only v2) vs. M0 (Original Fingerprint Scorer)** | **+8.01 pp** [+7.44, +8.61] | **YES** | **+7.42 pp** [+6.87, +7.96] | 1291 | 266 | **+1025** | 673.46 | `1.974e-161` | `+13.52 / +4.55 / +8.06` |  Clear (No drop > 1 pp) |
| **M5 (Network-Only v2, 22 features) vs. Abl_M5a (Pair Log-Odds Only, 3 features)** | **+3.65 pp** [+3.20, +4.13] | **YES** | **+3.54 pp** [+3.13, +3.99] | 728 | 238 | **+490** | 247.54 | `2.41e-58` | `+8.24 / -1.13 / +7.05` | **FLAGGED (italian)** |

- **Explicit Regression Guard Note**:
  - None of the primary model comparisons (`M5 vs. M0`, `M4 vs. M1`, `M4 vs. M2`, `M3 vs. M1`, `M3 vs. M2`) trigger the recall regression guard.
  - In the supplementary internal breakdown `M5 (All 22 Network Features) vs. Abl_M5a (3 Pair Log-Odds Features Only)`, adding the 19 D3 structural features boosts `indian` recall by **+8.24 pp** (`83.65% -> 91.89%`) and `mexican` recall by **+7.05 pp** (`85.18% -> 92.23%`), while slightly reducing `italian` recall by **-1.13 pp** (`97.48% -> 96.35%`) because `Abl_M5a` routes all 676 zero-pair fallback recipes to the majority class `italian` (inflating `italian` recall at the expense of `italian` precision, which rises from `86.27%` in `Abl_M5a` to `93.10%` in `M5`). We explicitly flag this `-1.13 pp` Italian recall shift per the Regression Guard rule.

---

## 4. Feature-Group Ablation Study (Removing Each Group from `M4` and Decomposing `M5`)

| Ablation Model | Features Kept | CV Macro-F1 (Mean ± SD) | Pooled OOF Macro-F1 | $\Delta$ Macro-F1 vs. `M4` [95% Bootstrap CI] | CV Accuracy (Mean ± SD) | $\Delta$ Accuracy vs. `M4` [95% Bootstrap CI] | Net Recipes vs. `M4` | McNemar $p$ vs. `M4` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M4: Full Stack (Reference for Ablations)** | 25 | **95.82% ± 0.44%** | **95.82%** | 0.00 pp [ref] | **95.88% ± 0.46%** | 0.00 pp [ref] | 0 | `1.000` |
| **Abl1: M4 minus NB Log-Posteriors (= M5, 22 features)** | 22 | 93.84% ± 0.46% | 93.84% | **-1.98 pp** [-2.35, -1.63] | 94.04% ± 0.47% | **-1.84 pp** [-2.17, -1.51] | -254 | `5.261e-28` |
| **Abl2: M4 minus Backbone Pair Log-Odds (22 features)** | 22 | 95.83% ± 0.37% | 95.83% | **+0.01 pp** [-0.09, +0.12] | 95.88% ± 0.42% | **+0.01 pp** [-0.09, +0.10] | +1 | `1` |
| **Abl3: M4 minus Louvain Community Structure (max_cov & entropy, 19 features)** | 19 | 95.72% ± 0.44% | 95.72% | **-0.09 pp** [-0.26, +0.07] | 95.79% ± 0.45% | **-0.09 pp** [-0.22, +0.07] | -12 | `0.276` |
| **Abl3b: M4 minus All Louvain Community Features (cov_frac, max_cov & entropy, 16 features)** | 16 | 95.76% ± 0.42% | 95.76% | **-0.05 pp** [-0.21, +0.11] | 95.83% ± 0.45% | **-0.04 pp** [-0.19, +0.11] | -6 | `0.6274` |
| **Abl4: M4 minus D3 Hub Counts (22 features)** | 22 | 95.67% ± 0.40% | 95.67% | **-0.14 pp** [-0.29, -0.01] | 95.78% ± 0.45% | **-0.10 pp** [-0.22, +0.01] | -14 | `0.1034` |
| **Abl5: M4 minus D3 Bridge Counts (22 features)** | 22 | 95.76% ± 0.44% | 95.76% | **-0.05 pp** [-0.15, +0.05] | 95.84% ± 0.47% | **-0.04 pp** [-0.12, +0.05] | -5 | `0.5327` |
| **Abl6: M4 minus Backbone Node Coverage & Recipe Length (18 features)** | 18 | 95.78% ± 0.36% | 95.78% | **-0.03 pp** [-0.18, +0.11] | 95.83% ± 0.42% | **-0.05 pp** [-0.19, +0.08] | -7 | `0.5341` |
| **Abl_M5a: Network-Only — Backbone Pair Log-Odds Only (3 features)** | 3 | 90.19% ± 0.65% | 90.19% | **-5.63 pp** [-6.12, -5.14] | 90.49% ± 0.59% | **-5.38 pp** [-5.85, -4.93] | -744 | `8.096e-123` |
| **Abl_M5b: Network-Only — D3 Structural Summaries Only, No Pair Log-Odds (19 features)** | 19 | 92.82% ± 0.75% | 92.82% | **-2.99 pp** [-3.40, -2.58] | 93.13% ± 0.70% | **-2.75 pp** [-3.12, -2.37] | -380 | `5.981e-50` |

### Key Insights from the Ablation Study
1. **Removing Naive Bayes log-posteriors (`Abl1 = M5`)** causes the largest drop (`-1.98 pp` Macro-F1, `-1.84 pp` Accuracy, `-254` net recipes), confirming that full-vocabulary unigram evidence remains the single strongest signal.
2. **Among the network feature groups inside `M4`, D3 Hub Counts (`Abl4`) and D3 Louvain Community Structure (`Abl3`) contribute the largest positive gains**:
   - **Removing D3 Hub Counts (`Abl4`)** drops `M4`'s Macro-F1 by **-0.14 pp** (`95% CI: [-0.29 pp, -0.01 pp]` — **95% bootstrap CI excludes zero!** `-14` net recipes).
   - **Removing D3 Louvain Community Structure (`Abl3: max_cov & entropy`)** drops `M4`'s Macro-F1 by **-0.09 pp** (`-12` net recipes).
   - **Removing D3 Bridge Counts (`Abl5`)** drops Macro-F1 by **-0.05 pp** (`-5` net recipes), and **removing Backbone Node Coverage & Recipe Length (`Abl6`)** drops Macro-F1 by **-0.03 pp** (`-7` net recipes).
   - **Removing Backbone Pair Log-Odds (`Abl2`)** leaves `M4` virtually unchanged (`+0.01 pp` Macro-F1, `+1` recipe, $p = 1.00$), demonstrating that once NB log-posteriors and D3 graph structure features (hubs, communities, bridges, backbone coverage) are present, pair log-odds are redundant with unigrams, whereas **D3 structural features (`Abl2: 95.83%` Macro-F1) provide the entire +0.30 pp Macro-F1 lift over Naive Bayes**!
3. **Inside the Pure Network-Only Scorer (`M5: 93.84%` Macro-F1), Pair Log-Odds and D3 Graph Summaries Are Strongly Complementary**:
   - Pair log-odds alone (`Abl_M5a`, 3 features) achieve **90.19% Macro-F1**. D3 graph structural summaries alone without pair log-odds (`Abl_M5b`, 19 features) achieve **92.82% Macro-F1**. Combining both in `M5` (22 features) reaches **93.84% Macro-F1** (+3.65 pp over pair log-odds alone and +1.01 pp over D3 summaries alone).

---

## 5. Standardized Logistic Regression Coefficients (`M4` and `M5`)

Because all input features are standardized to zero mean and unit variance (`StandardScaler` fitted strictly on each outer training fold), the L2 logistic regression coefficients below directly measure each feature's standardized log-odds effect (`Mean ± SD` across the 5 outer folds).

### 5.1 Standardized Coefficients for `M4: Full Stack (25 Features)`

| Feature Name | Feature Group | `indian` Weight (Mean ± SD) | `italian` Weight (Mean ± SD) | `mexican` Weight (Mean ± SD) | Max $|\text{Mean Weight}|$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `nb_log_prob_indian` | 1. Unigram NB Log-Posterior | `+4.812 ± 0.396` | `-2.850 ± 0.261` | `-1.962 ± 0.153` | **4.812** |
| `nb_log_prob_italian` | 1. Unigram NB Log-Posterior | `-2.269 ± 0.337` | `+4.567 ± 0.518` | `-2.298 ± 0.269` | **4.567** |
| `nb_log_prob_mexican` | 1. Unigram NB Log-Posterior | `-2.147 ± 0.299` | `-1.411 ± 0.088` | `+3.558 ± 0.267` | **3.558** |
| `pair_log_odds_indian` | 2. Backbone Pair Log-Odds | `+0.088 ± 0.149` | `-0.095 ± 0.105` | `+0.007 ± 0.218` | **0.095** |
| `pair_log_odds_italian` | 2. Backbone Pair Log-Odds | `+0.623 ± 0.547` | `-0.660 ± 0.701` | `+0.037 ± 0.204` | **0.660** |
| `pair_log_odds_mexican` | 2. Backbone Pair Log-Odds | `-0.083 ± 0.399` | `-0.236 ± 0.560` | `+0.320 ± 0.250` | **0.320** |
| `comm_cov_frac_indian` | 3a. Community Coverage Fraction | `+0.273 ± 0.323` | `-0.087 ± 0.224` | `-0.187 ± 0.168` | **0.273** |
| `comm_cov_frac_italian` | 3a. Community Coverage Fraction | `+0.021 ± 0.159` | `+0.345 ± 0.143` | `-0.367 ± 0.199` | **0.367** |
| `comm_cov_frac_mexican` | 3a. Community Coverage Fraction | `-0.184 ± 0.265` | `-0.422 ± 0.205` | `+0.605 ± 0.097` | **0.605** |
| `comm_max_cov_indian` | 3b. Max Single-Community Coverage | `+0.187 ± 0.165` | `-0.120 ± 0.090` | `-0.066 ± 0.107` | **0.187** |
| `comm_max_cov_italian` | 3b. Max Single-Community Coverage | `-0.110 ± 0.121` | `-0.050 ± 0.088` | `+0.160 ± 0.100` | **0.160** |
| `comm_max_cov_mexican` | 3b. Max Single-Community Coverage | `+0.004 ± 0.132` | `+0.083 ± 0.103` | `-0.087 ± 0.067` | **0.087** |
| `comm_entropy_indian` | 3c. Community Concentration (Entropy) | `-0.177 ± 0.174` | `+0.069 ± 0.118` | `+0.109 ± 0.118` | **0.177** |
| `comm_entropy_italian` | 3c. Community Concentration (Entropy) | `-0.023 ± 0.137` | `+0.042 ± 0.103` | `-0.019 ± 0.116` | **0.042** |
| `comm_entropy_mexican` | 3c. Community Concentration (Entropy) | `+0.110 ± 0.166` | `+0.141 ± 0.118` | `-0.251 ± 0.130` | **0.251** |
| `hub_count_indian` | 4. D3 Top-10 Weighted-Degree Hubs | `-0.313 ± 0.134` | `-0.045 ± 0.178` | `+0.358 ± 0.096` | **0.358** |
| `hub_count_italian` | 4. D3 Top-10 Weighted-Degree Hubs | `-0.066 ± 0.118` | `+0.027 ± 0.072` | `+0.038 ± 0.075` | **0.066** |
| `hub_count_mexican` | 4. D3 Top-10 Weighted-Degree Hubs | `-0.072 ± 0.193` | `+0.149 ± 0.150` | `-0.076 ± 0.075` | **0.149** |
| `bridge_count_indian` | 5. D3 Top-5 Pairwise Bridges | `-0.331 ± 0.083` | `+0.315 ± 0.088` | `+0.016 ± 0.063` | **0.331** |
| `bridge_count_italian` | 5. D3 Top-5 Pairwise Bridges | `+0.097 ± 0.149` | `-0.154 ± 0.101` | `+0.057 ± 0.097` | **0.154** |
| `bridge_count_mexican` | 5. D3 Top-5 Pairwise Bridges | `-0.031 ± 0.110` | `-0.010 ± 0.102` | `+0.041 ± 0.076` | **0.041** |
| `backbone_ing_count_indian` | 6. Backbone Ingredient Count | `-0.188 ± 0.167` | `+0.294 ± 0.175` | `-0.106 ± 0.195` | **0.294** |
| `backbone_ing_count_italian` | 6. Backbone Ingredient Count | `+0.505 ± 0.091` | `-1.105 ± 0.134` | `+0.600 ± 0.175` | **1.105** |
| `backbone_ing_count_mexican` | 6. Backbone Ingredient Count | `+0.284 ± 0.166` | `+0.570 ± 0.058` | `-0.854 ± 0.136` | **0.854** |
| `recipe_length` | 7. Global Recipe Length | `-0.079 ± 0.135` | `-0.354 ± 0.099` | `+0.433 ± 0.091` | **0.433** |

### 5.2 Standardized Coefficients for `M5: Network-Only Fingerprint Scorer v2 (22 Features, No NB)`

| Feature Name | Feature Group | `indian` Weight (Mean ± SD) | `italian` Weight (Mean ± SD) | `mexican` Weight (Mean ± SD) | Max $|\text{Mean Weight}|$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `pair_log_odds_indian` | 2. Backbone Pair Log-Odds | `+2.836 ± 0.361` | `-1.177 ± 0.170` | `-1.660 ± 0.267` | **2.836** |
| `pair_log_odds_italian` | 2. Backbone Pair Log-Odds | `-0.285 ± 0.304` | `+1.704 ± 0.427` | `-1.419 ± 0.205` | **1.704** |
| `pair_log_odds_mexican` | 2. Backbone Pair Log-Odds | `-1.361 ± 0.302` | `-0.160 ± 0.355` | `+1.521 ± 0.097` | **1.521** |
| `comm_cov_frac_indian` | 3a. Community Coverage Fraction | `+0.843 ± 0.290` | `-0.467 ± 0.190` | `-0.376 ± 0.170` | **0.843** |
| `comm_cov_frac_italian` | 3a. Community Coverage Fraction | `-0.274 ± 0.106` | `+1.038 ± 0.142` | `-0.764 ± 0.185` | **1.038** |
| `comm_cov_frac_mexican` | 3a. Community Coverage Fraction | `-0.830 ± 0.278` | `-0.405 ± 0.139` | `+1.234 ± 0.184` | **1.234** |
| `comm_max_cov_indian` | 3b. Max Single-Community Coverage | `+0.283 ± 0.144` | `-0.119 ± 0.081` | `-0.163 ± 0.077` | **0.283** |
| `comm_max_cov_italian` | 3b. Max Single-Community Coverage | `+0.014 ± 0.089` | `-0.165 ± 0.068` | `+0.151 ± 0.086` | **0.165** |
| `comm_max_cov_mexican` | 3b. Max Single-Community Coverage | `+0.033 ± 0.098` | `+0.070 ± 0.064` | `-0.103 ± 0.079` | **0.103** |
| `comm_entropy_indian` | 3c. Community Concentration (Entropy) | `+0.226 ± 0.120` | `-0.119 ± 0.074` | `-0.107 ± 0.073` | **0.226** |
| `comm_entropy_italian` | 3c. Community Concentration (Entropy) | `+0.008 ± 0.072` | `-0.007 ± 0.080` | `-0.000 ± 0.073` | **0.008** |
| `comm_entropy_mexican` | 3c. Community Concentration (Entropy) | `-0.002 ± 0.070` | `+0.100 ± 0.068` | `-0.098 ± 0.044` | **0.100** |
| `hub_count_indian` | 4. D3 Top-10 Weighted-Degree Hubs | `-0.008 ± 0.089` | `-0.069 ± 0.095` | `+0.077 ± 0.060` | **0.077** |
| `hub_count_italian` | 4. D3 Top-10 Weighted-Degree Hubs | `+0.082 ± 0.066` | `-0.130 ± 0.057` | `+0.049 ± 0.037` | **0.130** |
| `hub_count_mexican` | 4. D3 Top-10 Weighted-Degree Hubs | `-0.038 ± 0.125` | `+0.034 ± 0.073` | `+0.004 ± 0.095` | **0.038** |
| `bridge_count_indian` | 5. D3 Top-5 Pairwise Bridges | `-0.211 ± 0.055` | `+0.107 ± 0.022` | `+0.105 ± 0.049` | **0.211** |
| `bridge_count_italian` | 5. D3 Top-5 Pairwise Bridges | `-0.105 ± 0.063` | `+0.054 ± 0.133` | `+0.051 ± 0.100` | **0.105** |
| `bridge_count_mexican` | 5. D3 Top-5 Pairwise Bridges | `+0.065 ± 0.072` | `+0.002 ± 0.080` | `-0.068 ± 0.047` | **0.068** |
| `backbone_ing_count_indian` | 6. Backbone Ingredient Count | `+1.555 ± 0.449` | `-0.661 ± 0.217` | `-0.894 ± 0.379` | **1.555** |
| `backbone_ing_count_italian` | 6. Backbone Ingredient Count | `-0.843 ± 0.159` | `+1.200 ± 0.127` | `-0.357 ± 0.210` | **1.200** |
| `backbone_ing_count_mexican` | 6. Backbone Ingredient Count | `-0.597 ± 0.394` | `-0.605 ± 0.133` | `+1.202 ± 0.328` | **1.202** |
| `recipe_length` | 7. Global Recipe Length | `+0.325 ± 0.071` | `-0.309 ± 0.049` | `-0.016 ± 0.052` | **0.325** |

### Interpretation of Standardized Coefficients
1. **In `M5` (Network-Only Scorer v2)**:
   - The dominant predictors are the three **backbone pair log-odds scores** (`+2.836` for Indian, `+1.704` for Italian, `+1.521` for Mexican), directly reinforced by **backbone ingredient counts** (`+1.555` Indian, `+1.200` Italian, `+1.202` Mexican) and **Louvain community coverage fractions** (`+0.843` Indian, `+1.038` Italian, `+1.234` Mexican).
   - **Community structure & bridges act as structural disambiguators**: Higher `comm_max_cov_indian` (`+0.283`) and `comm_entropy_indian` (`+0.226`) positively predict Indian, whereas `bridge_count_indian` (which counts cross-cuisine bridge ingredients shared between Indian and Italian/Mexican such as `garlic`, `onion`, `bay leaves`, `heavy cream`, `cinnamon`, `clove`) has a **negative weight for Indian (`-0.211 ± 0.055`) and positive weights for Italian (`+0.107`) and Mexican (`+0.105`)**! Why? Because when a recipe matches Indian backbone nodes primarily through shared cross-cuisine *bridge* ingredients rather than cuisine-specific Indian spices, the negative weight on `bridge_count_indian` discounts the spurious Indian backbone coverage.
2. **In `M4` (Full Stack with NB)**:
   - The three cross-fitted **NB log-posteriors** carry the largest standardized weights (`+4.812` Indian, `+4.567` Italian, `+3.558` Mexican).
   - What role do the network features play alongside NB in `M4`? They provide a **structural complexity and hub/bridge baseline correction**:
     - Naive Bayes multiplies per-ingredient likelihood ratios $\prod_{u \in R} P(u \mid c)$. On longer recipes containing many generic Italian/Western pantry staples that belong to the large Italian backbone (`backbone_ing_count_italian` has weight `-1.105` on Italian in `M4`), uncalibrated Naive Bayes over-accumulates evidence for Italian. By pairing positive `comm_cov_frac_c` (`+0.273` IN, `+0.345` IT, `+0.605` MX) with negative multi-ingredient hub/bridge/count penalties (`backbone_ing_count_italian = -1.105`, `hub_count_indian = -0.313`, `bridge_count_indian = -0.331`, `comm_entropy_mexican = -0.251`), `M4` corrects Naive Bayes when generic hubs or cross-cuisine bridges dominate a recipe.

---

## 6. How Often Network Features Change Naive Bayes's Prediction — and Whether Those Changes Are Correct

| Comparison (`Target Model vs. Base NB Model`) | Total OOF Recipes | Predictions Changed ($N$, %) | Changed Wrong $\rightarrow$ Correct (**Fixed**, $N$ / % of changes) | Changed Correct $\rightarrow$ Wrong (**Broken**, $N$ / % of changes) | Changed Wrong $\rightarrow$ Diff Wrong ($N$ / % of changes) | **Net Correct Gain (`Fixed - Broken`)** | McNemar $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M4 (Full Stack, 25 feats) vs. M1 (Raw Naive Bayes)** | 13,823 | **224** (1.62%) | **116** (51.8%) | 86 (38.4%) | 22 (9.8%) | **+30 recipes** | `0.04104` |
| **M4 (Full Stack, 25 feats) vs. M2 (Calibrated NB Stack, 3 feats)** | 13,823 | **229** (1.66%) | **121** (52.8%) | 88 (38.4%) | 20 (8.7%) | **+33 recipes** | `0.02663` |
| **M3 (NB + Pair Log-Odds, 6 feats) vs. M1 (Raw Naive Bayes)** | 13,823 | **129** (0.93%) | **62** (48.1%) | 57 (44.2%) | 10 (7.8%) | **+5 recipes** | `0.714` |
| **M3 (NB + Pair Log-Odds, 6 feats) vs. M2 (Calibrated NB Stack, 3 feats)** | 13,823 | **75** (0.54%) | **39** (52.0%) | 31 (41.3%) | 5 (6.7%) | **+8 recipes** | `0.403` |

### Breakdown of Prediction Flips (`M4` vs. `M1` and `M4` vs. `M2`)
- **Frequency of Prediction Changes**: Across the 13,823 OOF training recipes, `M4` changes `M1` (Naive Bayes)'s prediction on **224 recipes (1.62%)** (and changes `M2` Calibrated NB's prediction on **229 recipes, 1.66%**).
- **Accuracy of Changes**:
  - Against `M1`, **116 of the 224 changes (51.8%) fix a Naive Bayes error** (`Wrong -> Correct`), while **86 (38.4%) flip a correct NB prediction to wrong** (`Correct -> Wrong`), and **22 (9.8%)** move between two wrong classes, yielding a net gain of **+30 correct predictions** ($p = 0.0410$).
  - Against `M2` (which isolates the exact effect of adding the 22 network features to the 3 NB log-posteriors), **121 of the 229 changes (52.8%) fix a calibrated NB error**, while **88 (38.4%) flip a correct prediction to wrong** and **20 (8.7%)** shift between wrong classes, yielding a net gain of **+33 correct predictions** ($p = 0.0266$).
- **Dominant Transition Patterns Fixed by Network Features (`M4 vs. M1`)**:
  - `true=mexican: italian -> mexican` (**39 recipes fixed** vs. 12 broken in reverse): Recipes with Mexican community coherence (`comm_cov_frac_mexican = 0.92`, `comm_max_cov_mexican = 0.58`) that Naive Bayes misclassified as `italian` due to generic baking/savory staples (e.g., Recipe `1894`: `all-purpose flour, baking powder, black pepper, butter, chocolate, cooking spray, ground cinnamon, ground red pepper, large egg, salt, sugar, vanilla extract`, where `nb_log_prob_italian = -0.052` beat `nb_log_prob_mexican = -2.990`, but `M4` correctly flipped `italian -> mexican`).
  - `true=mexican: indian -> mexican` (**32 recipes fixed** vs. 10 broken in reverse): E.g., Recipe `2271` (`cinnamon sticks, egg yolks, milk, shredded coconut, sugar`), where `cinnamon sticks` is an Indian top-10 weighted-degree hub (`hub_count_indian = 1`) that caused Naive Bayes to predict `indian` (`nb_log_prob_indian = -0.305`), but `M4`'s hub-count discount (`hub_count_indian` weight `-0.313`, ` +0.358` for Mexican) correctly flipped the prediction to `mexican`.
  - `true=indian: italian -> indian` (**24 recipes fixed** vs. 7 broken in reverse): E.g., Recipe `2270` (`active dry yeast, bread flour, greek style plain yogurt, kosher salt, olive oil, salted butter, sugar, water`, naan bread), where Italian baking staples gave `nb_log_prob_italian = -0.565` vs. `nb_log_prob_indian = -0.865`, and `M4`'s penalty on generic Italian backbone node counts (`backbone_ing_count_italian = 6`) flipped it to `indian`.

---

## 7. Performance by Pre-Registered Fixed Recipe-Length Bins

Using the four fixed recipe-length bins defined in `phase3_design.md` before running Phase 3 (`Short: 1–5`, `Medium-Short: 6–8`, `Medium-Long: 9–12`, `Long: 13+` normalized ingredients):

| Pre-Registered Recipe-Length Bin | Recipes ($N$, Share) | Mean Ings | `M0` (Orig Scorer) Macro-F1 / Acc | `M1` (Naive Bayes) Macro-F1 / Acc | `M2` (Calib NB) Macro-F1 / Acc | `M3` (NB + Pair) Macro-F1 / Acc | `M4` (Full Stack) Macro-F1 / Acc | `M5` (Network-Only v2) Macro-F1 / Acc | `M5 - M0` ($\Delta$ F1 / Net Recipes) | `M4 - M1` ($\Delta$ F1 / Net Recipes) | `M4 - M2` ($\Delta$ F1 / Net Recipes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (1–5 ingredients)** | 1,231 (8.9%) | 4.29 | 62.73% / 70.02% | 87.03% / 89.36% | 86.44% / 89.11% | 86.68% / 89.28% | **87.82% / 89.93%** | **79.70% / 83.43%** | **+16.97 pp** (`+165`) | **+0.79 pp** (`+7`) | **+1.38 pp** (`+10`) |
| **Medium-Short (6–8 ingredients)** | 3,509 (25.4%) | 7.09 | 76.69% / 81.56% | 92.87% / 94.21% | 92.82% / 94.21% | 92.94% / 94.33% | **93.54% / 94.56%** | **89.72% / 91.88%** | **+13.03 pp** (`+362`) | **+0.67 pp** (`+12`) | **+0.72 pp** (`+12`) |
| **Medium-Long (9–12 ingredients)** | 4,904 (35.5%) | 10.42 | 87.26% / 88.87% | 96.21% / 96.45% | 96.32% / 96.51% | 96.20% / 96.47% | **96.55% / 96.74%** | **94.79% / 95.27%** | **+7.53 pp** (`+314`) | **+0.35 pp** (`+14`) | **+0.24 pp** (`+11`) |
| **Long (13+ ingredients)** | 4,179 (30.2%) | 16.14 | 93.18% / 93.13% | 97.84% / 97.80% | 97.77% / 97.73% | 97.86% / 97.82% | **97.78% / 97.73%** | **97.58% / 97.54%** | **+4.40 pp** (`+184`) | **-0.07 pp** (`-3`) | **+0.01 pp** (`+0`) |

### Key Findings Across Recipe-Length Bins
1. **`M5` (Network-Only v2) crushes `M0` (Original Scorer) across every recipe length**, with the largest Macro-F1 gains on **Short (`1–5` ingredients: +16.97 pp Macro-F1, +13.40 pp Accuracy)** and **Medium-Short (`6–8` ingredients: +13.03 pp Macro-F1, +10.32 pp Accuracy)** recipes, where pair matches are sparse or zero (`M0` falls back to majority class `italian`), but single-ingredient D3 community/hub/backbone node coverage features still fire!
2. **`M4` (Full Stack) achieves its largest Macro-F1 gains over `M1` (Naive Bayes) and `M2` (Calibrated NB) on Short and Medium recipes (`1–12` ingredients)**:
   - **Short (`1–5` ingredients)**: `M4` reaches **87.82% Macro-F1** vs. `M1` **87.03% (+0.79 pp)** and `M2` **86.44% (+1.38 pp)** (`+7` and `+10` net recipes).
   - **Medium-Short (`6–8` ingredients)**: `M4` reaches **93.54% Macro-F1** vs. `M1` **92.87% (+0.67 pp)** and `M2` **92.82% (+0.72 pp)** (`+12` net recipes).
   - **Medium-Long (`9–12` ingredients)**: `M4` reaches **96.55% Macro-F1** vs. `M1` **96.21% (+0.35 pp)** and `M2` **96.32% (+0.24 pp)** (`+14` and `+11` net recipes).
   - **Long (`13+` ingredients)**: All NB-based models perform similarly (`M1: 97.84%`, `M2: 97.77%`, `M3: 97.86%`, `M4: 97.78%` Macro-F1; `M4` and `M2` have `0` net recipe difference on $13+$ ingredient recipes), while the pure network-only model `M5` reaches **97.58% Macro-F1 (97.54% Accuracy)** — within `0.26 pp` of Naive Bayes using zero unigram NB features!

---

## Appendix: Supplementary Weighted-F1 Metrics (Metric Hierarchy Level 5)

| Model | 5-Fold CV Weighted-F1 (Mean ± SD) | Pooled OOF Weighted-F1 |
| :--- | :---: | :---: |
| **Abl1: M4 minus NB Log-Posteriors (= M5, 22 features)** | 94.03% ± 0.47% | 94.03% |
| **Abl2: M4 minus Backbone Pair Log-Odds (22 features)** | 95.88% ± 0.42% | 95.88% |
| **Abl3: M4 minus Louvain Community Structure (max_cov & entropy, 19 features)** | 95.79% ± 0.45% | 95.79% |
| **Abl3b: M4 minus All Louvain Community Features (cov_frac, max_cov & entropy, 16 features)** | 95.83% ± 0.45% | 95.83% |
| **Abl4: M4 minus D3 Hub Counts (22 features)** | 95.77% ± 0.45% | 95.77% |
| **Abl5: M4 minus D3 Bridge Counts (22 features)** | 95.84% ± 0.47% | 95.84% |
| **Abl6: M4 minus Backbone Node Coverage & Recipe Length (18 features)** | 95.82% ± 0.42% | 95.82% |
| **Abl_M5a: Network-Only — Backbone Pair Log-Odds Only (3 features)** | 90.43% ± 0.60% | 90.43% |
| **Abl_M5b: Network-Only — D3 Structural Summaries Only, No Pair Log-Odds (19 features)** | 93.11% ± 0.71% | 93.11% |
| **M0: Original Fingerprint Scorer (reference)** | 86.55% ± 0.78% | 86.55% |
| **M1: Naive Bayes Alone (reference)** | 95.65% ± 0.58% | 95.65% |
| **M2: Calibrated NB Stack Alone (3 features)** | 95.63% ± 0.63% | 95.63% |
| **M3: NB + Backbone Pair Log-Odds Stack (6 features)** | 95.69% ± 0.64% | 95.69% |
| **M4: Full Stack — NB + All Network Features (25 features)** | 95.87% ± 0.47% | 95.87% |
| **M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)** | 94.03% ± 0.47% | 94.03% |

---

> **STOPPING POINT**: Per your instructions, the 3,456 held-out test recipes remain completely untouched, and all D1–D3 and original D4 files are unmodified. Please review `outputs/d4_experiments/phase3/phase3_design.md`, `outputs/d4_experiments/phase3/phase3_report.md`, and `outputs/d4_experiments/phase3/results.csv`.
