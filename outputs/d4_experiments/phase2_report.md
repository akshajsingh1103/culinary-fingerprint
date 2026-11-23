# Deliverable 4 — Phase 2 Controlled Experiments Report (Nested 5-Fold CV on Training Split Only)

> [!IMPORTANT]
> **Held-Out Test Set Sealed & Untouched**: In strict compliance with Phase 2 rules and Addendum #6, the 3,456 held-out test recipes (`test_ids` in `data/processed/split.json`) were **never loaded, inspected, or evaluated** at any point. Every model, hyperparameter selection, vocabulary, Curveball backbone, pair set, calibration statistic, statistical significance test, and error analysis below was computed exclusively within **nested 5-fold stratified cross-validation on the 13,823 training recipes** (`seed = 42`), using the exact same outer splits (`outputs/d4_experiments/cv_folds.json`) as Phase 1 so that all results are strictly paired.

---

## Executive Summary & Honest Answer to the Central Question

### Do Ingredient Pairs Add Information Beyond Unigrams?
**Plain answer: No.** Across all controlled comparisons in 5-fold nested cross-validation on the 13,823 training recipes, **adding ingredient pair-presence features to an L2-regularized logistic regression over unigram features does not improve accuracy or macro-F1 over unigrams alone**—in fact, it slightly degrades performance due to overfitting in the expanded quadratic feature space:

1. **On the Backbone Vocabulary (~610 unigrams vs. ~610 unigrams + ~4,865 validated backbone pairs)**:
   - **Unigram-only L2 LogReg (`ExpA2`)**: **95.13% ± 0.74%** CV accuracy (**94.98% ± 0.77%** macro-F1).
   - **Hybrid Unigram + Pair L2 LogReg (`ExpC_bb`)**: **94.92% ± 0.66%** CV accuracy (**94.76% ± 0.73%** macro-F1).
   - **Paired difference (`Hybrid - Unigram`)**: **-0.21 pp** (`-29` net recipes out of 13,823; Hybrid right / Unigram wrong = `182`, Hybrid wrong / Unigram right = `211`; 95% paired bootstrap CI `[-0.49 pp, +0.07 pp]`, McNemar exact binomial $p = 0.1577$ — statistically indistinguishable from zero, slightly favoring unigrams).
2. **On the Full Training Vocabulary (~3,834 unigrams vs. ~3,834 unigrams + ~4,865 backbone pairs)**:
   - **Unigram-only L2 LogReg (`ExpA3`)**: **95.82% ± 0.86%** CV accuracy (**95.71% ± 0.90%** macro-F1) — **the highest-performing model across all Phase 2 experiments**.
   - **Hybrid Full Unigram + Backbone Pair L2 LogReg (`ExpC_full_bb`)**: **95.47% ± 0.68%** CV accuracy (**95.31% ± 0.76%** macro-F1).
   - **Paired difference (`Hybrid - Unigram`)**: **-0.35 pp** (`-48` net recipes; Hybrid right / Unigram wrong = `112`, Hybrid wrong / Unigram right = `160`; 95% paired bootstrap CI `[-0.59 pp, -0.12 pp]`, McNemar exact binomial $p = 0.0043$ in favor of unigrams alone).
3. **On the Full Training Vocabulary + All Frequent Pairs (`min_support >= 5`, ~25,050 pairs)**:
   - **Hybrid Full Unigram + All Pairs L2 LogReg (`ExpC_full_all`)**: **95.46% ± 0.54%** CV accuracy (**95.29% ± 0.61%** macro-F1), trailing unigram-only `ExpA3` by **-0.35 pp** (`-49` net recipes; $p = 0.0061$).
4. **Does pair coverage change this conclusion?**
   - **No.** In our pre-registered coverage-stratified analysis (Section 7), unigram-only logistic regression (`ExpA2` and `ExpA3`) matches or slightly outperforms the corresponding hybrid model (`ExpC_bb` and `ExpC_full_bb`) in **every single coverage bin** (Low `0–4` matched pairs, Mid `5–15` matched pairs, and High `16+` matched pairs, as well as all three training-quantile tertiles).
5. **Why does the original Fingerprint Scorer trail unigram baselines by ~9 pp (86.62% vs. 95.66%–95.82%), and what actually improves pair-only scoring?**
   - Replacing raw intra-cuisine NPMI sums (`86.62%`) with **discriminative log-odds pair weights** (`ExpB1_bb`, which contrast $P(p\mid c)$ against $P(p\mid \neg c)$ and naturally supply negative evidence against competing cuisines) jumps pair-only accuracy on the exact same backbone pairs by **+4.01 pp** to **90.63% ± 0.78%** (and to **94.62% ± 0.61%** when expanded to all pairs with support $\ge 5$ in `ExpB1_all`).
   - Replacing the majority-class (`italian`) zero-score fallback on the 676 zero-match recipes with a **unigram Naive Bayes fallback** (`ExpD_nb_full`) lifts accuracy on the zero-score fallback subset from **50.30% to 89.50% (+39.20 pp)**, improving overall Fingerprint Scorer accuracy from **86.62% to 88.54% (+1.92 pp)** without altering a single non-zero backbone score.
   - In **Experiment E** (matched feature count $K \approx 4,865$ pairs), **Curveball statistical validation (`z > 3.0`, 89.39%) decisively beats raw NPMI thresholding (`87.44%`, +1.95 pp) and random eligible pairs (`88.01% ± 0.20%`, +1.37 pp)** for pair-only classification because raw NPMI prioritizes rare, low-coverage pairs with inflated conditional PMI (`1,082` zero-match fallbacks vs. `676`), whereas **top-$K$ most frequent pairs (`91.07%`)** outperform Curveball pairs in pair-only classification due to higher recipe coverage (`608` zero-match fallbacks).

---

## 1. Nested Validation Protocol & Zero-Leakage Verification (Addendum #1, #6, #7)

### 1.1 Nested Cross-Validation Design
- **Outer Evaluation Splits (5 Folds)**: Identical to Phase 1 (`outputs/d4_experiments/cv_folds.json`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 13,823 training recipes). Each outer fold partitions the 13,823 training recipes into ~11,058 outer-training recipes and ~2,765 outer-validation recipes.
- **Inner Hyperparameter Tuning Splits (3 Folds inside each Outer Training Partition)**: Inside each outer fold $f \in \{0,1,2,3,4\}$, a 3-fold `StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + f)` splits only that outer fold's ~11,058 training recipes into ~7,372 inner-training and ~3,686 inner-validation recipes.
- **Parameters Tuned Strictly by Inner CV**:
  - L2 logistic regression inverse regularization strength $C \in \{0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0\}$ for `ExpA2`, `ExpA3`, `ExpB2_bb`, `ExpB2_all`, `ExpC_bb`, `ExpC_full_bb`, `ExpC_full_all`, and all `ExpE` variants.
  - Log-odds smoothing constant $\varepsilon \in \{10^{-6}, 10^{-5}, 10^{-4}, 5\times 10^{-4}, 10^{-3}, 5\times 10^{-3}, 10^{-2}, 5\times 10^{-2}\}$ for `ExpB1_bb`, `ExpB1_elig5`, and `ExpB1_all`.
  - Minimum pair co-occurrence support $\text{min\_support} \in \{5, 10, 20\}$ jointly with $\varepsilon$ (for `ExpB1_all`) and jointly with $C$ (for `ExpB2_all` and `ExpC_full_all`).
  - Negative evidence penalty weight $\lambda \in \{0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0\}$ and competing-pair scope (`exclusive` vs. `all` competing backbone edges) for `ExpF2_neg_raw` and `ExpF2_neg_zscore`.
- **Documented Compute Approximation (Addendum #1)**: Each outer fold's Curveball null-model backbone (`n_randomizations = 100` per cuisine, `burn_in = 10 * N_c`, `thinning = 2 * N_c`) is constructed once on the outer training partition and reused during inner-fold hyperparameter tuning of parameters ($C, \varepsilon, \lambda$) that do not change the backbone.

### 1.2 Explicit Partition & Zero-Leakage Verification Log (Addendum #7)

| Outer Fold | Outer Train $N$ | Outer Val $N$ | Outer Train SHA-256 (first 16 hex) | Outer Val SHA-256 (first 16 hex) | Matches `cv_folds.json` | Train $\cap$ Val | Train $\cap$ Test | Val $\cap$ Test | Inner 3-Fold Splits Disjoint from Val & Test |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 11,058 | 2,765 | `8be41856dcd9a171...` | `db78da3dc22ce562...` | `True` | `0` | `0` | `0` | `True` |
| 1 | 11,058 | 2,765 | `a7e2428afbadec3f...` | `ce475637f561f58b...` | `True` | `0` | `0` | `0` | `True` |
| 2 | 11,058 | 2,765 | `6c0ede71d232240e...` | `7b0a74b58e4f7ae7...` | `True` | `0` | `0` | `0` | `True` |
| 3 | 11,059 | 2,764 | `a5857703377afd16...` | `2e16c1c2937596e0...` | `True` | `0` | `0` | `0` | `True` |
| 4 | 11,059 | 2,764 | `7e44080ffb2826be...` | `81168614a6826e98...` | `True` | `0` | `0` | `0` | `True` |

- **Dataset SHA-256**: `f18e7e568d1c606774bfb134ece51ef0a143cd5069e6f9241bb6036b45b7d2b0`
- **Normalizer SHA-256**: `760d350b8906154df72208039266707d9ae2590f2208fae9923aed581220d11d`
- **Verification Summary**: All 30 evaluated models share the exact same 5 outer folds (`13,823` total OOF predictions; `0` overlap with the `3,456` sealed test IDs), and every fitted component (ingredient normalizer vocabulary, eligible networks, Curveball backbones, NPMI/frequency/random pair sets, log-odds weights, L2 logistic regression coefficients, score standardization $\mu_c, \sigma_c$, and training-quantile coverage cutoffs) was estimated strictly from its own outer training partition.

---

## 2. Fold-Wise Feature Counts & Selected Hyperparameters (Inner CV)

| Outer Fold | Full Unigram Vocab | Backbone Unigram Vocab | Validated Backbone Pairs ($K_f$) | Eligible Universe ($\text{co} \ge 5$) | All Pairs ($\text{sup} \ge 5$) | `ExpA2` $C$ (BB Uni) | `ExpA3` $C$ (Full Uni) | `ExpB1_bb` $\varepsilon$ | `ExpB1_all` ($\text{sup}, \varepsilon$) | `ExpB2_bb` $C$ | `ExpB2_all` ($\text{sup}, C$) | `ExpC_bb` $C$ | `ExpC_full_bb` $C$ | `ExpC_full_all` $C$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 3,834 | 606 | **4,865** | 19,890 | 25,051 | `2.0` | `5.0` | `1e-06` | `(5, 1e-06)` | `0.5` | `(5, 0.25)` | `0.25` | `0.25` | `0.5` |
| 1 | 3,861 | 615 | **4,859** | 19,950 | 25,130 | `5.0` | `2.0` | `1e-06` | `(5, 1e-05)` | `0.5` | `(5, 0.25)` | `0.25` | `0.5` | `0.5` |
| 2 | 3,818 | 608 | **4,909** | 19,971 | 25,124 | `5.0` | `0.25` | `1e-06` | `(5, 1e-06)` | `0.5` | `(5, 0.25)` | `0.25` | `0.25` | `1.0` |
| 3 | 3,809 | 605 | **4,823** | 19,833 | 24,983 | `5.0` | `2.0` | `1e-06` | `(5, 1e-06)` | `0.5` | `(5, 0.25)` | `2.0` | `0.5` | `0.5` |
| 4 | 3,849 | 615 | **4,867** | 19,966 | 25,074 | `5.0` | `1.0` | `1e-06` | `(5, 1e-06)` | `0.5` | `(5, 0.25)` | `1.0` | `1.0` | `1.0` |

---

## 3. Master Results Table Across All Phase 2 Experiments (5-Fold Nested CV)

Complete machine-readable metrics are saved in `outputs/d4_experiments/results.csv`. Below is the consolidated comparison across all reference baselines and Experiments A–F (`Mean ± SD` across the 5 outer folds, plus pooled OOF accuracy/macro-F1 across all 13,823 training recipes):

| Experiment / Model | CV Accuracy (Mean ± SD) | CV Macro-F1 (Mean ± SD) | Pooled OOF Acc | Pooled OOF Macro-F1 | F1 `indian` | F1 `italian` | F1 `mexican` | Zero-Score Fallbacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ExpA1: Unigram Naive Bayes (backbone vocab)** | 94.83% ± 0.78% | 94.63% ± 0.84% | 94.83% | 94.63% | 94.08% | 95.64% | 94.19% | 23 |
| **ExpA2: Unigram L2 LogReg (backbone vocab)** | 95.13% ± 0.74% | 94.98% ± 0.77% | 95.13% | 94.98% | 94.55% | 95.79% | 94.58% | 0 |
| **ExpA3: Unigram L2 LogReg (full vocab)** | 95.82% ± 0.86% | 95.71% ± 0.90% | 95.82% | 95.71% | 95.44% | 96.32% | 95.37% | 0 |
| **ExpB1_all: Pair Log-Odds (all pairs >= min_support)** | 94.62% ± 0.61% | 94.39% ± 0.59% | 94.62% | 94.40% | 93.72% | 95.43% | 94.03% | 201 |
| **ExpB1_bb: Pair Log-Odds (backbone pairs)** | 90.63% ± 0.78% | 90.07% ± 0.90% | 90.63% | 90.07% | 88.34% | 92.22% | 89.65% | 676 |
| **ExpB1_elig5: Pair Log-Odds (all eligible pairs co>=5)** | 94.26% ± 0.69% | 93.99% ± 0.73% | 94.26% | 93.99% | 93.15% | 95.19% | 93.62% | 238 |
| **ExpB2_all: Pair L2 LogReg (all pairs >= min_support)** | 92.85% ± 0.70% | 92.41% ± 0.74% | 92.85% | 92.42% | 90.96% | 93.65% | 92.63% | 201 |
| **ExpB2_bb: Pair L2 LogReg (backbone pairs)** | 89.39% ± 0.78% | 88.75% ± 0.78% | 89.39% | 88.76% | 86.86% | 90.80% | 88.59% | 676 |
| **ExpC_bb: Hybrid L2 LogReg (backbone unigrams + backbone pairs)** | 94.92% ± 0.66% | 94.76% ± 0.73% | 94.92% | 94.76% | 94.37% | 95.65% | 94.27% | 0 |
| **ExpC_full_all: Hybrid L2 LogReg (full unigrams + all pairs >= min_support)** | 95.46% ± 0.54% | 95.29% ± 0.61% | 95.46% | 95.29% | 94.82% | 96.18% | 94.87% | 0 |
| **ExpC_full_bb: Hybrid L2 LogReg (full unigrams + backbone pairs)** | 95.47% ± 0.68% | 95.31% ± 0.76% | 95.47% | 95.32% | 94.88% | 96.10% | 94.96% | 0 |
| **ExpD_nb_bb: Fingerprint Scorer + Unigram NB Fallback (backbone vocab)** | 88.19% ± 0.85% | 87.35% ± 1.02% | 88.19% | 87.35% | 84.50% | 90.40% | 87.14% | 0 |
| **ExpD_nb_full: Fingerprint Scorer + Unigram NB Fallback (full vocab)** | 88.54% ± 0.85% | 87.74% ± 0.97% | 88.54% | 87.74% | 85.03% | 90.67% | 87.51% | 0 |
| **ExpE_B2_i_curveball: Pair L2 LogReg (D2 Curveball z>3 pairs, matched K)** | 89.39% ± 0.78% | 88.75% ± 0.78% | 89.39% | 88.76% | 86.86% | 90.80% | 88.59% | 676 |
| **ExpE_B2_ii_npmi: Pair L2 LogReg (NPMI-thresholded pairs, matched K)** | 87.44% ± 0.36% | 87.01% ± 0.40% | 87.44% | 87.01% | 86.03% | 88.80% | 86.19% | 1082 |
| **ExpE_B2_iii_freq: Pair L2 LogReg (Top-K frequent pairs, matched K)** | 91.07% ± 0.48% | 90.77% ± 0.51% | 91.07% | 90.77% | 90.00% | 92.01% | 90.30% | 608 |
| **ExpE_B2_iv_random: Pair L2 LogReg (Random eligible pairs, 10-seed mean)** | 88.01% ± 0.57% | 87.41% ± 0.68% | 88.01% | 87.41% | 85.45% | 89.61% | 87.05% | 0 |
| **ExpE_C_i_curveball: Hybrid L2 LogReg (BB unigrams + Curveball z>3 pairs, matched K)** | 94.92% ± 0.66% | 94.76% ± 0.73% | 94.92% | 94.76% | 94.37% | 95.65% | 94.27% | 0 |
| **ExpE_C_ii_npmi: Hybrid L2 LogReg (BB unigrams + NPMI-thresholded pairs, matched K)** | 95.04% ± 0.54% | 94.86% ± 0.63% | 95.04% | 94.87% | 94.36% | 95.75% | 94.48% | 0 |
| **ExpE_C_iii_freq: Hybrid L2 LogReg (BB unigrams + Top-K frequent pairs, matched K)** | 94.88% ± 0.59% | 94.74% ± 0.64% | 94.88% | 94.74% | 94.44% | 95.64% | 94.14% | 0 |
| **ExpE_C_iv_random: Hybrid L2 LogReg (BB unigrams + Random eligible pairs, 10-seed mean)** | 94.95% ± 0.64% | 94.81% ± 0.67% | 94.95% | 94.81% | 94.50% | 95.65% | 94.40% | 0 |
| **ExpF1_zscore_all: Fingerprint Scorer (per-cuisine z-score on train fold)** | 73.69% ± 1.13% | 72.13% ± 1.08% | 73.69% | 72.13% | 59.49% | 80.48% | 76.43% | 676 |
| **ExpF1_zscore_pos: Fingerprint Scorer (per-cuisine z-score on positive train scores)** | 82.11% ± 0.63% | 80.32% ± 0.63% | 82.11% | 80.33% | 72.23% | 86.54% | 82.20% | 676 |
| **ExpF2_neg_raw: Fingerprint Scorer + Negative Evidence (raw score - lambda * competing NPMI)** | 86.96% ± 0.68% | 86.05% ± 0.90% | 86.96% | 86.06% | 83.23% | 89.14% | 85.79% | 676 |
| **ExpF2_neg_zscore: Fingerprint Scorer (z-score + Negative Evidence)** | 77.75% ± 0.82% | 76.22% ± 0.81% | 77.75% | 76.22% | 65.98% | 81.57% | 81.11% | 676 |
| **ExpF3_edge_div: Fingerprint Scorer (divide by backbone edge count |E_c| ablation)** | 82.82% ± 0.48% | 82.19% ± 0.48% | 82.82% | 82.19% | 79.62% | 84.81% | 82.15% | 676 |
| **Ref1: Fingerprint Scorer (original)** | 86.62% ± 0.77% | 85.82% ± 0.92% | 86.62% | 85.83% | 83.38% | 88.78% | 85.31% | 676 |
| **Ref2: Fingerprint (normalized ablation)** | 47.43% ± 1.05% | 46.87% ± 1.14% | 47.43% | 46.89% | 43.63% | 52.36% | 44.62% | 676 |
| **Ref3: Naive Bayes (full vocab)** | 95.66% ± 0.58% | 95.53% ± 0.62% | 95.66% | 95.53% | 95.21% | 96.23% | 95.16% | 0 |
| **Ref4: Naive Bayes (backbone vocab)** | 94.83% ± 0.78% | 94.63% ± 0.84% | 94.83% | 94.63% | 94.08% | 95.64% | 94.19% | 23 |

---

## 4. Detailed Findings by Experiment (A, B, C, D, E, F)

### 4.1 Reference Models & Experiment A: Unigram-Only Models
- **Backbone Vocabulary (~605–615 unigrams per fold) vs. Full Training Vocabulary (~3,809–3,861 unigrams per fold)**:
  - Restricting Multinomial Naive Bayes from the full training vocabulary (`Ref3: 95.66% ± 0.58%`) to the ~610-ingredient union backbone vocabulary (`Ref4 / ExpA1: 94.83% ± 0.78%`) costs **-0.82 pp** accuracy (`95% CI [-1.05 pp, -0.61 pp]`), showing that ~84% of the unigram vocabulary (rare/long-tail ingredients occurring fewer than 20 times per cuisine) accounts for less than 1 percentage point of accuracy.
  - On the **backbone unigram vocabulary**, L2 logistic regression (`ExpA2: 95.13% ± 0.74%`, macro-F1 `94.98% ± 0.77%`, inner-CV tuned $C \in \{2.0, 5.0\}$) outperforms Naive Bayes (`ExpA1: 94.83% ± 0.78%`, macro-F1 `94.63% ± 0.84%`) by **+0.30 pp** accuracy (+0.34 pp macro-F1).
  - On the **full training vocabulary**, L2 logistic regression (`ExpA3: 95.82% ± 0.86%`, macro-F1 `95.71% ± 0.90%`, inner-CV tuned $C \in \{0.25, 1.0, 2.0, 5.0\}$) outperforms full-vocabulary Naive Bayes (`Ref3: 95.66% ± 0.58%`, macro-F1 `95.53% ± 0.62%`) by **+0.16 pp** accuracy (+0.18 pp macro-F1) and achieves the **highest accuracy and macro-F1 of all 30 models tested in Phase 2**.

### 4.2 Experiment B: Pair-Only Scoring & Classification (B1 Log-Odds, B2 L2 LogReg, B3 Backbone vs. All Pairs)
- **B1 Log-Odds Pair Scoring Formula**:
  $$w(c, p) = \log\left(\frac{P(p \mid c) + \varepsilon}{P(p \mid \neg c) + \varepsilon}\right), \quad \text{Score}_{\text{B1}}(c, R) = \sum_{p \in \binom{R}{2} \cap P_{\text{cand}}} w(c, p)$$
  where $P(p \mid c) = \text{count}_c(p) / N_c$ and $P(p \mid \neg c) = \text{count}_{\neg c}(p) / N_{\neg c}$ are estimated strictly from the outer/inner training partition, and $\varepsilon$ is chosen by 3-fold inner CV from $\{10^{-6}, 10^{-5}, 10^{-4}, 5\times 10^{-4}, 10^{-3}, 5\times 10^{-3}, 10^{-2}, 5\times 10^{-2}\}$.
- **Why B1 Log-Odds (`90.63%`) dramatically outperforms raw NPMI summation (`86.62%`, +4.01 pp) on the exact same backbone pairs**:
  - Inner CV consistently selected small smoothing $\varepsilon = 10^{-6}$ (or $10^{-5}$ in Fold 1). Unlike raw within-cuisine NPMI (which is positive for every backbone edge and therefore rewards Italian whenever a recipe contains generic culinary/baking pairs like `active dry yeast + warm water` or `garlic powder + onion powder`), the log-odds weight $w(c, p)$ is **positive only when pair $p$ is more prevalent in cuisine $c$ than in the complement $\neg c$, and negative when $p$ is more prevalent in competing cuisines**.
- **B2 Pair-Presence L2 Logistic Regression**:
  - On the ~4,865 backbone pairs (`ExpB2_bb`, tuned $C = 0.5$ in all 5 folds), L2 logistic regression achieves **89.39% ± 0.78%** accuracy (**88.75% ± 0.78%** macro-F1), beating the original Fingerprint Scorer by **+2.76 pp** (`95% CI [+2.24, +3.30]`), though trailing `ExpB1_bb` log-odds (`90.63%`) by 1.24 pp because L2 logistic regression's intercept routes low-activation and zero-pair recipes heavily toward the majority class (`italian` recall `96.70%`, `indian` recall `79.03%`).
- **B3 Restricted to Backbone Pairs vs. All Pairs with Minimum Support**:
  - When pair selection is expanded from the ~4,865 Curveball backbone pairs to **all eligible-node pairs with co-occurrence $\ge 5$** (~19,920 pairs, `ExpB1_elig5`), log-odds accuracy jumps from **90.63% to 94.26% ± 0.69%** (and zero-score fallbacks drop from `676` to `238`).
  - When expanded to **all pairs across the full vocabulary with inner-CV tuned $\text{min\_support} = 5$** (~25,050 pairs), log-odds accuracy reaches **94.62% ± 0.61%** (`ExpB1_all`, `94.39%` macro-F1, `201` zero-score fallbacks), and pair-only L2 logistic regression rises from **89.39% to 92.85% ± 0.70%** (`ExpB2_all`, tuned $\text{min\_support} = 5, C = 0.25$).
  - **Takeaway from Experiment B**: Most of the gap between pair-only models and unigram models was caused by (1) non-contrastive NPMI edge weights and (2) aggressively pruning the pair space down to ~4,865 Curveball edges. However, even the best 25,050-pair model (`ExpB1_all: 94.62%`) still trails the 610-unigram backbone models (`ExpA1: 94.83%`, `ExpA2: 95.13%`) and full-vocab unigram models (`Ref3: 95.66%`, `ExpA3: 95.82%`).

---

## 5. Experiment C (Key Experiment): Hybrid Unigram + Pair-Presence vs. Unigram-Only L2 Logistic Regression (Addendum #5)

Per Addendum #5, we report the 5-fold CV mean/SD accuracy and macro-F1, the exact fold-wise differences, the pooled out-of-fold paired difference across all 13,823 recipes, the 95% paired bootstrap confidence interval (2,000 resamples, `seed = 42`), and the paired McNemar test with discordant counts (without running a t-test across the 5 non-independent fold scores).

### 5.1 Paired Statistical Comparison Summary Table

| Paired Comparison (Hybrid vs. Unigram-Only L2 LogReg) | Hybrid CV Acc (Mean ± SD) | Unigram CV Acc (Mean ± SD) | Pooled $\Delta$ Acc (`Hyb - Uni`) [95% Bootstrap CI] | Pooled $\Delta$ Macro-F1 (`Hyb - Uni`) [95% Bootstrap CI] | Hyb Right, Uni Wrong ($b$) | Hyb Wrong, Uni Right ($c$) | Discordant ($b+c$) | Net Recipes ($b-c$) | McNemar $\chi^2$ (CC) | Exact Binomial $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Backbone Vocab: Hybrid (ExpC_bb) vs. Unigram-Only (ExpA2)** | 94.92% ± 0.66% | 95.13% ± 0.74% | **-0.21 pp** [-0.49, +0.07] | **-0.21 pp** [-0.53, +0.10] | 182 | 211 | 393 | **-29** | 1.99 | `0.1577` |
| **Full Vocab + All Pairs >= min_support: Hybrid (ExpC_full_all) vs. Unigram-Only (ExpA3)** | 95.46% ± 0.54% | 95.82% ± 0.86% | **-0.35 pp** [-0.61, -0.10] | **-0.42 pp** [-0.71, -0.15] | 129 | 178 | 307 | **-49** | 7.50 | `0.006062` |
| **Full Vocab + Backbone Pairs: Hybrid (ExpC_full_bb) vs. Unigram-Only (ExpA3)** | 95.47% ± 0.68% | 95.82% ± 0.86% | **-0.35 pp** [-0.59, -0.12] | **-0.40 pp** [-0.66, -0.16] | 112 | 160 | 272 | **-48** | 8.12 | `0.004289` |

### 5.2 Fold-Wise Paired Differences (`Hybrid minus Unigram-Only`)

| Outer Fold | `ExpA2` (BB Uni) Acc | `ExpC_bb` (Hyb BB) Acc | $\Delta$ Acc (`ExpC_bb - ExpA2`) | $\Delta$ Macro-F1 (`ExpC_bb - ExpA2`) | `ExpA3` (Full Uni) Acc | `ExpC_full_bb` (Hyb Full+BB) Acc | $\Delta$ Acc (`ExpC_full_bb - ExpA3`) | `ExpC_full_all` (Hyb Full+All) Acc | $\Delta$ Acc (`ExpC_full_all - ExpA3`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 94.97% | 94.61% | **-0.36 pp** | -0.42 pp | 95.52% | 95.08% | **-0.43 pp** | 95.08% | **-0.43 pp** |
| 1 | 95.66% | 95.33% | **-0.33 pp** | -0.23 pp | 96.56% | 95.84% | **-0.72 pp** | 95.62% | **-0.94 pp** |
| 2 | 93.92% | 93.96% | **+0.04 pp** | -0.02 pp | 94.47% | 94.47% | **+0.00 pp** | 94.76% | **+0.29 pp** |
| 3 | 95.37% | 95.04% | **-0.33 pp** | -0.28 pp | 96.09% | 95.95% | **-0.14 pp** | 96.13% | **+0.04 pp** |
| 4 | 95.73% | 95.66% | **-0.07 pp** | -0.12 pp | 96.45% | 96.02% | **-0.43 pp** | 95.73% | **-0.72 pp** |

- **Why adding pairs to unigrams does not help (and slightly hurts)**:
  1. **Collinearity & feature dilution under L2 regularization**: A pair indicator $I(a \in R \land b \in R)$ can only fire when both unigram indicators $I(a \in R)$ and $I(b \in R)$ are already active. Appending ~4,865 (or ~25,050) binary pair columns to ~610 (or ~3,834) unigram columns forces inner CV to choose stronger L2 regularization ($C \approx 0.25 - 1.0$ in Hybrid vs. $C \approx 2.0 - 5.0$ in Unigram-only) to prevent variance explosion on the thousands of correlated pair features. This stronger L2 penalty shrinks informative unigram weights (especially for rarer signature ingredients), while dense recipes with 15–45 active staple pairs accumulate noisy pair-weight sums that occasionally override a decisive single ingredient.
  2. **Additive single-ingredient sufficiency**: For distinguishing Indian, Italian, and Mexican cuisines, culinary identity is overwhelmingly encoded by additive single-ingredient presence (e.g., `garam masala`, `turmeric`, `curry leaves`, `parmesan`, `gnocchi`, `prosciutto`, `tortilla`, `jalapeno`, `salsa`, `enchilada sauce`). Genuinely non-linear XOR-like interactions where ingredients $a$ and $b$ individually point to Cuisine X but their combination $(a, b)$ points to Cuisine Y are exceedingly rare in 3-cuisine classification.

---

## 6. Experiment D, Experiment E, and Experiment F Detailed Analysis

### 6.1 Experiment D: Original Scorer with Unigram Naive Bayes Fallback on Zero-Score Recipes

| Evaluation Subset | Recipe Count ($N$) | Original Scorer (`italian` Fallback) Acc / Macro-F1 | `ExpD_nb_bb` (Backbone-Vocab NB Fallback) Acc / Macro-F1 | `ExpD_nb_full` (Full-Vocab NB Fallback) Acc / Macro-F1 | Gain (`ExpD_nb_full - Original`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Score Fallback Subset (`score = 0` in all 3 cuisines)** | **676** (4.89%) | 50.30% / 22.31% | 82.40% / 79.19% | **89.50% / 88.03%** | **+39.20 pp Acc** |
| **Non-Fallback Subset (`max(score) > 0`)** | **13,147** (95.11%) | 88.49% | 88.49% | 88.49% | 0.00 pp (unchanged) |
| **All Training Recipes (OOF Pooled)** | **13,823** (100.0%) | 86.62% / 85.83% | 88.19% / 87.35% | **88.54% / 87.74%** | **+1.92 pp Acc / +1.91 pp F1** |

- **Interpretation**: On the 676 recipes where no validated backbone pair matches, replacing the majority-class default (`italian`, 50.30% accuracy) with fold-trained unigram Naive Bayes recovers **265 additional correct recipes** (`89.50%` subset accuracy with full-vocab NB, or `82.40%` with backbone-vocab NB), lifting overall Fingerprint Scorer accuracy by **+1.92 pp** (`86.62% -> 88.54%`).

### 6.2 Experiment E: Matched-Size Pair-Set Comparison Inside Pair-Only (`B2`) and Hybrid (`C`) L2 Logistic Regression (Addendum #2)

In every outer fold $f$, all four pair sets have the **exact same number of pair features $K_f$** (`4,865`, `4,859`, `4,909`, `4,823`, `4,867` in Folds 0–4), the **exact same binary recipe-level pair-presence representation**, the **exact same eligible node (`count >= 20`) and pair co-occurrence support constraint (`co_count >= 5`)**, and the **exact same inner-CV L2 regularization search over $C$**:

| Pair Set (Matched Size $K_f \in [4823, 4909]$ per Fold) | Overlap with Curveball Backbone Pairs (Mean across Folds) | Pair-Only (`ExpE_B2`) CV Acc (Mean ± SD) | Pair-Only (`ExpE_B2`) CV Macro-F1 (Mean ± SD) | Zero-Pair Recipes (OOF) | Hybrid BB Unigrams + Pairs (`ExpE_C`) CV Acc (Mean ± SD) | Hybrid BB Unigrams + Pairs (`ExpE_C`) CV Macro-F1 (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **(i) D2 Curveball Validated Pairs ($z > 3.0, \text{co} \ge 5$)** | 100.0% (~4,865) | **89.39% ± 0.78%** | **88.75% ± 0.78%** | 676 (4.89%) | 94.92% ± 0.66% | 94.76% ± 0.73% |
| **(ii) NPMI-Thresholded Pairs (Top $K_f$ by max NPMI, $\text{co} \ge 5$, cutoff $\approx 0.153$)** | 84.0% (~4,090) | 87.44% ± 0.36% | 87.01% ± 0.40% | 1,082 (7.83%) | 95.04% ± 0.54% | 94.86% ± 0.63% |
| **(iii) Top-$K_f$ Most Frequent Pairs ($\text{co} \ge 5$, cutoff $\ge 21$ recipes)** | 37.7% (~1,833) | **91.07% ± 0.48%** | **90.77% ± 0.51%** | **608 (4.40%)** | 94.88% ± 0.59% | 94.74% ± 0.64% |
| **(iv) Random Eligible Pairs ($\text{co} \ge 5$, 10 seeds, across-seed SD = 0.20% / 0.10%)** | ~24.4% (~1,187) | 88.01% ± 0.57% (seed SD 0.20%) | 87.41% ± 0.68% (seed SD 0.21%) | ~780 (5.64%) | 94.95% ± 0.64% (seed SD 0.10%) | 94.81% ± 0.67% (seed SD 0.11%) |

- **10-Seed Random Control Distribution (Addendum #2)**:
  - Across the 10 random seeds sampling $K_f$ pairs from the ~19,922-pair eligible universe (`co_count >= 5`), pair-only `ExpE_B2_iv_random` 5-fold CV accuracy ranged from `87.75%` to `88.39%` with **10-seed mean = 88.01% and across-seed SD = 0.20%** (macro-F1 mean = **87.41%**, across-seed SD = **0.21%**).
  - For hybrid `ExpE_C_iv_random`, 5-fold CV accuracy across the 10 seeds ranged from `94.83%` to `95.09%` with **10-seed mean = 94.95% and across-seed SD = 0.10%** (macro-F1 mean = **94.81%**, across-seed SD = **0.11%**).
- **Does Curveball validation beat NPMI-thresholded, Top-K frequent, and Random pairs?**
  1. **In Pair-Only Classification (`ExpE_B2`)**:
     - **Curveball (`89.39%`) decisively beats NPMI-thresholded pairs (`87.44%`, +1.95 pp) and Random eligible pairs (`88.01%`, +1.37 pp, >6.8 SDs above the random control)**. Why does raw NPMI thresholding perform worst (even 0.57 pp *below* random pairs)? Because ranking solely by NPMI selects pairs of rare ingredients that co-occur 5–10 times with low marginal frequencies (inflating conditional PMI/NPMI), leaving **1,082 validation recipes (7.83%) with zero matched pairs**! Curveball null-model validation ($z > 3.0$ holding row and column margins fixed) filters out noisy small-sample pairs while retaining statistically significant medium- and high-frequency associations (`676` zero-pair recipes).
     - **However, Top-$K$ Most Frequent Pairs (`91.07%`) beat Curveball pairs (`89.39%`) by +1.68 pp** in supervised pair-only L2 logistic regression. Because logistic regression learns discriminative weights directly, giving it the $K_f$ highest-frequency pairs maximizes feature density and coverage across recipes (reducing zero-pair recipes to `608`), whereas Curveball prunes high-prevalence staple pairs whose co-occurrence is well-explained by their high marginal frequencies under the null model.
  2. **In Hybrid Classification (`ExpE_C`)**:
     - Once the ~610 backbone unigrams are present (`ExpA2: 95.13%`), **all four matched-size pair sets yield virtually identical hybrid accuracy between `94.88%` and `95.04%`** (within 0.16 pp of each other and slightly below the `95.13%` unigram-only baseline). Even adding 4,865 random pairs (`94.95%`) performs identically to adding 4,865 Curveball pairs (`94.92%`), confirming that pairs provide no incremental predictive signal beyond unigrams.

### 6.3 Experiment F: Calibration & Negative Evidence on the Original Fingerprint Scorer (Addendum #3)

Per Addendum #3, all calibration parameters ($\mu_c, \sigma_c, \lambda$) were estimated **strictly from the outer training partition** (never from validation data), prioritizing per-cuisine z-score standardization and negative evidence, and reporting division by backbone edge count $|E_c|$ as a separate simple ablation:

| Experiment F Variant | CV Accuracy (Mean ± SD) | CV Macro-F1 (Mean ± SD) | F1 `indian` | F1 `italian` | F1 `mexican` | Precision / Recall `indian` | Precision / Recall `italian` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref1: Original Scorer (Raw NPMI Sum)** | 86.62% ± 0.77% | 85.82% ± 0.92% | 83.38% | 88.78% | 85.31% | 89.11% / 78.36% | 85.96% / 91.80% |
| **ExpF2_neg_raw: Raw Score - $\lambda \times$ Competing Exclusive NPMI** | **86.96% ± 0.68%** | **86.05% ± 0.90%** | 83.23% | **89.14%** | **85.79%** | **90.31%** / 77.20% | 85.93% / **92.60%** |
| **ExpF3_edge_div: Divide by Backbone Edge Count $|E_c|$ (Ablation)** | 82.82% ± 0.48% | 82.19% ± 0.48% | 79.62% | 84.81% | 82.15% | 73.20% / 87.31% | 88.97% / 81.02% |
| **ExpF1_zscore_pos: Per-Cuisine Z-Score (Positive Train Scores)** | 82.11% ± 0.63% | 80.32% ± 0.63% | 72.23% | 86.54% | 82.20% | 62.43% / 85.69% | 86.83% / 86.27% |
| **ExpF2_neg_zscore: Per-Cuisine Z-Score + Negative Evidence** | 77.75% ± 0.82% | 76.22% ± 0.81% | 65.98% | 81.57% | 81.11% | 52.32% / 89.31% | 87.39% / 76.49% |
| **ExpF1_zscore_all: Per-Cuisine Z-Score (All Train Scores)** | 73.69% ± 1.13% | 72.13% ± 1.08% | 59.49% | 80.48% | 76.43% | 44.48% / 89.81% | 87.33% / 74.64% |

- **Why Negative Evidence on Raw Scores (`ExpF2_neg_raw: 86.96%`) Helps (+0.33 pp)**:
  - Subtracting $\lambda \times \sum_{c' \ne c} \text{NPMI}_{c'}(p)$ for pairs $p \in E_{c'} \setminus E_c$ (with inner-CV selected $\lambda = 1.0$ in 4 folds and $0.5$ in Fold 3) penalizes a cuisine when a recipe contains backbone pairs exclusive to competing cuisines, improving accuracy across all 5 outer folds (`86.62% -> 86.96%`, macro-F1 `85.82% -> 86.05%`).
- **Why Unconditional Per-Cuisine Z-Score Standardization (`ExpF1_zscore_all: 73.69%`) and Dividing by $|E_c|$ (`ExpF3_edge_div: 82.82%`) Hurt Accuracy**:
  - In the training folds, `indian` makes up only 17.4% of recipes (so 82.6% of training recipes are non-Indian and have near-zero Indian scores). Consequently, the unconditional training-fold mean score for Indian is much lower ($\mu_{\text{indian}} \approx 0.91$) than for Italian ($\mu_{\text{italian}} \approx 1.38$) or Mexican ($\mu_{\text{mexican}} \approx 1.37$).
  - When a non-Indian recipe has a small raw score in all cuisines (say, $S_{\text{indian}} = 0.20, S_{\text{italian}} = 0.50, S_{\text{mexican}} = 0.45$), subtracting $\mu_c$ gives $(0.20 - 0.91)/2.08 = -0.34$ for Indian vs. $(0.50 - 1.38)/1.95 = -0.45$ for Italian and $(0.45 - 1.37)/1.75 = -0.53$ for Mexican—causing z-score standardization to predict **`indian` whenever all raw scores are below their respective training means**! This inflates `indian` recall to `89.81%` while collapsing `indian` precision from `89.11%` down to `44.48%`.
  - Similarly, dividing by backbone edge count $|E_c|$ (`~1,151` Indian vs. `~2,339` Italian and `~1,739` Mexican) multiplies every Indian score by $\sim 2.03\times$ relative to Italian, over-correcting and reducing `indian` precision to `73.20%` (`82.82%` overall accuracy). True discriminative calibration requires per-pair contrastive weights (`ExpB1_bb: 90.63%`), not post-hoc rescaling of non-contrastive sums.

---

## 7. Coverage-Stratified Accuracy & Macro-F1 Table (Addendum #4)

Per Addendum #4, coverage bins were defined **prior to inspecting Phase 2 results** and computed purely from each recipe's total matched backbone pairs $M(R) = \sum_{c} |\binom{R}{2} \cap E_c|$, **completely independent of any model's predictions**. The exact same bins are applied to all models below under both pre-registered rules:
1. **Rule 1 (Fixed Boundaries)**: **Low (`0–4` matched pairs, 20.79% of recipes)**, **Mid (`5–15` matched pairs, 34.36% of recipes)**, **High (`16+` matched pairs, 44.85% of recipes)**.
2. **Rule 2 (Outer-Training-Partition Quantile Tertiles)**: Cutoffs $(q_{33.3\%}, q_{66.7\%})$ computed strictly on each outer fold's training partition (`[8–9, 22–23]` matched pairs).

### 7.1 Rule 1: Fixed Matched-Pair Coverage Bins (`Low: 0–4`, `Mid: 5–15`, `High: 16+`)

| Coverage Bin | Recipes ($N$, Share) | Mean Recipe Ings | Mean Matched Pairs | Original Scorer Acc | Pair-Only `ExpB2_bb` Acc | Backbone Unigram `ExpA2` Acc / F1 | Backbone Hybrid `ExpC_bb` Acc / F1 | $\Delta$ Acc (`ExpC_bb - ExpA2`) [Hyb>Uni / Uni>Hyb, $p$] | Full Unigram `ExpA3` Acc / F1 | Full Hybrid `ExpC_full_bb` Acc / F1 | $\Delta$ Acc (`ExpC_full_bb - ExpA3`) [Hyb>Uni / Uni>Hyb, $p$] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low (0–4 matched pairs)** | 2,874 (20.8%) | 6.32 | 1.89 | 72.72% | 73.59% | 90.61% / 89.17% | 90.33% / 88.81% | **-0.28 pp** [`62`/`70`, $p=0.54$] | 92.52% / 91.38% | 91.72% / 90.30% | **-0.80 pp** [`31`/`54`, $p=0.017$] |
| **Mid (5–15 matched pairs)** | 4,750 (34.4%) | 9.20 | 9.60 | 87.39% | 90.84% | 95.37% / 94.94% | 95.22% / 94.68% | **-0.15 pp** [`62`/`69`, $p=0.60$] | 95.56% / 95.21% | 95.52% / 95.05% | **-0.04 pp** [`42`/`44`, $p=0.914$] |
| **High (16+ matched pairs)** | 6,199 (44.8%) | 14.01 | 33.16 | 92.48% | 95.60% | 97.05% / 97.07% | 96.82% / 96.85% | **-0.23 pp** [`58`/`72`, $p=0.25$] | 97.55% / 97.57% | 97.18% / 97.20% | **-0.37 pp** [`39`/`62`, $p=0.028$] |

### 7.2 Rule 2: Outer-Training-Partition Quantile Tertiles (`Tertile 1: Low`, `Tertile 2: Mid`, `Tertile 3: High`)

| Training-Quantile Coverage Bin | Recipes ($N$, Share) | Mean Recipe Ings | Mean Matched Pairs | Original Scorer Acc | Pair-Only `ExpB2_bb` Acc | Backbone Unigram `ExpA2` Acc / F1 | Backbone Hybrid `ExpC_bb` Acc / F1 | $\Delta$ Acc (`ExpC_bb - ExpA2`) [Hyb>Uni / Uni>Hyb, $p$] | Full Unigram `ExpA3` Acc / F1 | Full Hybrid `ExpC_full_bb` Acc / F1 | $\Delta$ Acc (`ExpC_full_bb - ExpA3`) [Hyb>Uni / Uni>Hyb, $p$] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tertile 1 (Low)** | 5,212 (37.7%) | 7.26 | 4.12 | 78.84% | 80.35% | 92.59% / 91.55% | 92.34% / 91.22% | **-0.25 pp** [`98`/`111`, $p=0.41$] | 93.73% / 92.91% | 93.21% / 92.16% | **-0.52 pp** [`53`/`80`, $p=0.024$] |
| **Tertile 2 (Mid)** | 4,494 (32.5%) | 10.72 | 15.31 | 89.90% | 93.72% | 96.19% / 95.99% | 95.93% / 95.64% | **-0.27 pp** [`40`/`52`, $p=0.25$] | 96.51% / 96.31% | 96.26% / 95.98% | **-0.24 pp** [`31`/`42`, $p=0.242$] |
| **Tertile 3 (High)** | 4,117 (29.8%) | 15.23 | 40.39 | 92.91% | 96.09% | 97.18% / 97.25% | 97.09% / 97.18% | **-0.10 pp** [`44`/`48`, $p=0.75$] | 97.72% / 97.78% | 97.47% / 97.56% | **-0.24 pp** [`28`/`38`, $p=0.268$] |

- **Key Finding from Coverage Stratification**:
  - Contrary to the hypothesis that pairs might improve classification on short or low-coverage recipes, **Hybrid trails Unigram-only in every single coverage bin** under both binning rules (`-0.28 pp`, `-0.15 pp`, `-0.23 pp` for Backbone Hybrid vs. Backbone Unigrams in Low/Mid/High; and `-0.80 pp`, `-0.04 pp`, `-0.37 pp` for Full Hybrid vs. Full Unigrams in Low/Mid/High).
  - In the **Low coverage bin (`0–4` matched pairs)**, recipes have very few pair features firing (mean 1.89 matched pairs), so the stronger L2 regularization chosen to control the 4,865 pair features slightly dampens the unigram weights on the 6.32 ingredients present (`31` Hybrid wins vs. `54` Unigram wins on full vocab, $p = 0.0165$).
  - In the **High coverage bin (`16+` matched pairs)**, recipes average 14.01 ingredients and 33.16 matched pairs; adding dozens of collinear pair weights introduces extra variance that flips `62` correct unigram predictions to wrong while fixing only `39` (`-0.37 pp`, $p = 0.0281$).

---

## 8. Error Analysis: Discordant Recipes Between Hybrid (`ExpC`) and Unigram-Only (`ExpA`) and the Pairs Responsible

Across all 13,823 OOF training recipes, Backbone Hybrid (`ExpC_bb`) and Backbone Unigram-Only (`ExpA2`) disagree on **393 recipes (2.84%)**: Hybrid is right on **182** and Unigram-only is right on **211** (`net = -29`). On the full vocabulary (`ExpC_full_bb` vs. `ExpA3`), they disagree on **272 recipes (1.97%)**: Hybrid is right on **112** and Unigram-only is right on **160** (`net = -48`). Below are concrete examples of both regimes showing the exact learned pair weights responsible.

### 8.1 Cases Where Hybrid Beats Unigram-Only (`Hybrid Correct, Unigram-Only Wrong`)

When does a pair feature genuinely rescue a misclassified unigram recipe? Primarily in **ambiguous dessert, baking, or crossover recipes composed entirely of generic staples** (`flour`, `butter`, `sugar`, `milk`, `salt`, `vanilla`, `chocolate`, `corn starch`) where no single cuisine-exclusive spice or herb is present, so specific pair co-occurrence patterns disambiguate the culinary tradition:

#### Hybrid-Wins Example 1: Recipe ID `11000` (Fold 3, Mid (5–15 matched pairs)) — True: `italian` | Hybrid (`ExpC_bb`): `italian` ($P=0.973$) | Unigram (`ExpA2`): `mexican` ($P=0.482$)
- **Normalized Ingredients (5 total, 5 active backbone pairs)**: `bittersweet chocolate, corn starch, salt, sugar, whole milk`
- **Total Pair Logit Shift in Hybrid Model**: `indian`: `-0.89` | `italian`: `+2.49` | `mexican`: `-1.61`
- **Top Responsible Pairs (Hybrid L2 LogReg Weights)**: `(corn starch + sugar)` [IN:-0.22, IT:+1.02, MX:-0.80], `(bittersweet chocolate + sugar)` [IN:-0.10, IT:+0.72, MX:-0.62], `(sugar + whole milk)` [IN:-0.33, IT:+0.37, MX:-0.04], `(corn starch + whole milk)` [IN:-0.12, IT:+0.23, MX:-0.11]
- **Top Unigram Weights (`ExpA2`)**: `whole milk` [IN:+0.86, IT:-0.70, MX:-0.16], `corn starch` [IN:-0.86, IT:+0.29, MX:+0.56], `bittersweet chocolate` [IN:-0.76, IT:+0.13, MX:+0.63], `salt` [IN:+0.60, IT:-0.58, MX:-0.01]

#### Hybrid-Wins Example 2: Recipe ID `42941` (Fold 4, Mid (5–15 matched pairs)) — True: `mexican` | Hybrid (`ExpC_bb`): `mexican` ($P=0.803$) | Unigram (`ExpA2`): `italian` ($P=0.737$)
- **Normalized Ingredients (6 total, 8 active backbone pairs)**: `all-purpose flour, butter, chopped pecan, confectioners sugar, salt, vanilla extract`
- **Total Pair Logit Shift in Hybrid Model**: `indian`: `-1.84` | `italian`: `-1.02` | `mexican`: `+2.86`
- **Top Responsible Pairs (Hybrid L2 LogReg Weights)**: `(butter + confectioners sugar)` [IN:-0.17, IT:-0.84, MX:+1.01], `(confectioners sugar + vanilla extract)` [IN:-0.22, IT:+0.43, MX:-0.21], `(all-purpose flour + vanilla extract)` [IN:-0.30, IT:-0.11, MX:+0.42], `(all-purpose flour + salt)` [IN:-0.24, IT:-0.17, MX:+0.41]
- **Top Unigram Weights (`ExpA2`)**: `salt` [IN:+0.64, IT:-0.55, MX:-0.10], `all-purpose flour` [IN:-0.63, IT:+0.41, MX:+0.22], `vanilla extract` [IN:-0.54, IT:+0.36, MX:+0.18], `confectioners sugar` [IN:+0.16, IT:+0.10, MX:-0.26]

#### Hybrid-Wins Example 3: Recipe ID `43930` (Fold 4, High (16+ matched pairs)) — True: `mexican` | Hybrid (`ExpC_bb`): `mexican` ($P=0.807$) | Unigram (`ExpA2`): `indian` ($P=0.534$)
- **Normalized Ingredients (11 total, 16 active backbone pairs)**: `baking powder, brown sugar, cinnamon, egg, flour, lime, melted butter, milk, salt, shredded coconut, whole wheat flour`
- **Total Pair Logit Shift in Hybrid Model**: `indian`: `-2.76` | `italian`: `+2.15` | `mexican`: `+0.60`
- **Top Responsible Pairs (Hybrid L2 LogReg Weights)**: `(baking powder + milk)` [IN:+0.11, IT:+0.63, MX:-0.75], `(baking powder + egg)` [IN:-0.49, IT:+0.58, MX:-0.09], `(lime + salt)` [IN:-0.46, IT:+0.35, MX:+0.11], `(flour + milk)` [IN:-0.40, IT:+0.30, MX:+0.10]
- **Top Unigram Weights (`ExpA2`)**: `whole wheat flour` [IN:+2.65, IT:-0.84, MX:-1.81], `lime` [IN:-0.01, IT:-1.53, MX:+1.54], `milk` [IN:+1.35, IT:-0.80, MX:-0.55], `egg` [IN:-1.20, IT:+0.95, MX:+0.25]

#### Hybrid-Wins Example 4: Recipe ID `42379` (Fold 1, High (16+ matched pairs)) — True: `indian` | Hybrid (`ExpC_bb`): `indian` ($P=0.867$) | Unigram (`ExpA2`): `italian` ($P=0.523$)
- **Normalized Ingredients (13 total, 21 active backbone pairs)**: `almond, black pepper, boneless skinless chicken breast, cooked white rice, curry powder, fresh basil leaves, ground cinnamon, heavy cream, kosher salt, low sodium chicken broth, olive oil, yellow onion, zucchini`
- **Total Pair Logit Shift in Hybrid Model**: `indian`: `+1.59` | `italian`: `-1.66` | `mexican`: `+0.08`
- **Top Responsible Pairs (Hybrid L2 LogReg Weights)**: `(curry powder + olive oil)` [IN:+0.72, IT:-0.56, MX:-0.16], `(boneless skinless chicken breast + olive oil)` [IN:+0.33, IT:-0.37, MX:+0.04], `(olive oil + zucchini)` [IN:-0.05, IT:-0.20, MX:+0.25], `(olive oil + yellow onion)` [IN:+0.19, IT:-0.25, MX:+0.06]
- **Top Unigram Weights (`ExpA2`)**: `curry powder` [IN:+4.84, IT:-3.29, MX:-1.55], `fresh basil leaves` [IN:-1.57, IT:+1.88, MX:-0.31], `ground cinnamon` [IN:-0.47, IT:-0.86, MX:+1.33], `olive oil` [IN:-0.51, IT:+0.88, MX:-0.36]

### 8.2 Cases Where Unigram-Only Beats Hybrid (`Unigram-Only Correct, Hybrid Wrong`)

Why does Unigram-only beat Hybrid more often (`211 vs. 182` on backbone vocab; `160 vs. 112` on full vocab)? In longer recipes (10–20 ingredients, 14–45 active backbone pairs), **a recipe may contain 1 or 2 clear cuisine-identifying unigrams** (e.g., `chopped cilantro fresh` + `white rice` + `cayenne` for Mexican, or `ground cinnamon` + `anise seed` for Mexican pan dulce), **but also contain a dense cluster of generic savory or baking ingredients** (`ground beef`, `tomato sauce`, `onion`, `olive oil`, `baking powder`, `egg`, `white sugar`, `flour`). Because the 15–45 active pairs among those generic ingredients all fire simultaneously, their cumulative pair logit shift (+2.0 to +3.7 logits) overwhelms the single discriminative unigram!

#### Unigram-Wins Example 1: Recipe ID `21021` (Fold 3, High (16+ matched pairs)) — True: `italian` | Unigram (`ExpA2`): `italian` ($P=0.849$) | Hybrid (`ExpC_bb`): `mexican` ($P=0.974$)
- **Normalized Ingredients (17 total, 52 active backbone pairs)**: `active dry yeast, all-purpose flour, chili powder, chopped onion, crushed red pepper flakes, dried basil, dried oregano, garlic powder, ground black pepper, ground cumin, marjoram, salt, shortening, tomato paste, vegetable oil, water, white sugar`
- **Total Pair Logit Shift Driving the Hybrid Error**: `indian`: `-0.86` | `italian`: `-2.53` | `mexican`: `+3.39`
- **Top Responsible Pairs Causing the Flip**: `(vegetable oil + white sugar)` [IN:+1.17, IT:-0.91, MX:-0.26], `(vegetable oil + water)` [IN:+0.49, IT:-0.86, MX:+0.38], `(chili powder + garlic powder)` [IN:+0.14, IT:-0.79, MX:+0.65], `(chopped onion + ground cumin)` [IN:-0.26, IT:-0.46, MX:+0.71]
- **Top Unigram Weights That Got It Right (`ExpA2`)**: `dried oregano` [IN:-2.48, IT:+1.42, MX:+1.06], `ground cumin` [IN:+1.39, IT:-2.29, MX:+0.90], `garlic powder` [IN:-1.55, IT:+0.60, MX:+0.95], `dried basil` [IN:-0.22, IT:+1.38, MX:-1.16]

#### Unigram-Wins Example 2: Recipe ID `34568` (Fold 4, High (16+ matched pairs)) — True: `mexican` | Unigram (`ExpA2`): `mexican` ($P=0.887$) | Hybrid (`ExpC_bb`): `italian` ($P=0.843$)
- **Normalized Ingredients (19 total, 36 active backbone pairs)**: `black pepper, carrot, cayenne, chicken stock, chopped cilantro fresh, chopped parsley, dried oregano, egg, fresh mint, fresh peas, ground beef, large garlic cloves, olive oil, onion, pepper, salt, string beans, tomato sauce, white rice`
- **Total Pair Logit Shift Driving the Hybrid Error**: `indian`: `-1.20` | `italian`: `+2.89` | `mexican`: `-1.69`
- **Top Responsible Pairs Causing the Flip**: `(chicken stock + onion)` [IN:+0.36, IT:+0.20, MX:-0.56], `(carrot + chopped cilantro fresh)` [IN:+0.55, IT:-0.24, MX:-0.31], `(olive oil + salt)` [IN:-0.51, IT:+0.05, MX:+0.46], `(ground beef + pepper)` [IN:-0.09, IT:+0.45, MX:-0.36]
- **Top Unigram Weights That Got It Right (`ExpA2`)**: `dried oregano` [IN:-2.24, IT:+1.12, MX:+1.12], `fresh mint` [IN:+1.82, IT:-0.88, MX:-0.94], `white rice` [IN:-0.04, IT:-1.48, MX:+1.52], `chopped cilantro fresh` [IN:+0.59, IT:-1.52, MX:+0.93]

#### Unigram-Wins Example 3: Recipe ID `37142` (Fold 3, High (16+ matched pairs)) — True: `mexican` | Unigram (`ExpA2`): `mexican` ($P=0.537$) | Hybrid (`ExpC_bb`): `italian` ($P=0.838$)
- **Normalized Ingredients (10 total, 14 active backbone pairs)**: `all-purpose flour, anise seed, baking powder, cream of tartar, egg, ground cinnamon, orange juice, salt, shortening, white sugar`
- **Total Pair Logit Shift Driving the Hybrid Error**: `indian`: `-2.98` | `italian`: `+3.75` | `mexican`: `-0.77`
- **Top Responsible Pairs Causing the Flip**: `(baking powder + egg)` [IN:-0.35, IT:+0.87, MX:-0.52], `(baking powder + white sugar)` [IN:-0.51, IT:+0.85, MX:-0.34], `(all-purpose flour + white sugar)` [IN:-0.27, IT:+0.53, MX:-0.26], `(baking powder + salt)` [IN:-0.33, IT:-0.19, MX:+0.52]
- **Top Unigram Weights That Got It Right (`ExpA2`)**: `ground cinnamon` [IN:-0.53, IT:-0.84, MX:+1.37], `egg` [IN:-0.94, IT:+0.78, MX:+0.16], `orange juice` [IN:-0.89, IT:+0.19, MX:+0.71], `all-purpose flour` [IN:-0.79, IT:+0.47, MX:+0.33]

#### Unigram-Wins Example 4: Recipe ID `32867` (Fold 0, High (16+ matched pairs)) — True: `italian` | Unigram (`ExpA2`): `italian` ($P=0.816$) | Hybrid (`ExpC_bb`): `mexican` ($P=0.877$)
- **Normalized Ingredients (16 total, 45 active backbone pairs)**: `all-purpose flour, baking soda, brown sugar, butter, coffee granules, cooking spray, granulated sugar, ground cinnamon, ground nutmeg, large egg, large egg whites, roasted cashew, salt, unsweetened cocoa powder, vanilla extract, white chocolate`
- **Total Pair Logit Shift Driving the Hybrid Error**: `indian`: `-2.04` | `italian`: `-0.78` | `mexican`: `+2.82`
- **Top Responsible Pairs Causing the Flip**: `(butter + vanilla extract)` [IN:-0.17, IT:-0.33, MX:+0.50], `(granulated sugar + salt)` [IN:-0.18, IT:-0.25, MX:+0.42], `(ground cinnamon + large egg)` [IN:-0.11, IT:-0.30, MX:+0.40], `(all-purpose flour + unsweetened cocoa powder)` [IN:-0.08, IT:+0.29, MX:-0.21]
- **Top Unigram Weights That Got It Right (`ExpA2`)**: `large egg whites` [IN:-1.48, IT:+1.02, MX:+0.46], `ground cinnamon` [IN:-0.33, IT:-0.92, MX:+1.25], `ground nutmeg` [IN:+0.28, IT:+0.77, MX:-1.05], `large egg` [IN:-1.03, IT:+0.86, MX:+0.16]

---

## 9. Final Configuration Selection (Based Strictly on 5-Fold Nested CV) — Awaiting Approval Before Touching Held-Out Test Set

Based strictly on the 5-fold nested cross-validation results above (with the 3,456-recipe held-out test set still completely untouched), we recommend the following final configurations for the held-out test set evaluation once approved:

1. **Best Overall Classifier (`ExpA3: Unigram L2 LogReg on Full Training Vocab`)**:
   - **CV Performance**: **95.82% ± 0.86% accuracy**, **95.71% ± 0.90% macro-F1** (beats Naive Bayes `95.66%` and all hybrid models).
   - **Configuration**: Binary recipe-level ingredient presence over full training vocabulary, `LogisticRegression(penalty='l2', solver='lbfgs', C=2.0, random_state=42)` (modal/median inner-CV selected $C = 2.0$, or tuned by 5-fold CV on the full 13,823 training split).
2. **Best Backbone-Vocab Unigram Classifier (`ExpA2: Unigram L2 LogReg on Union Backbone Vocab`)**:
   - **CV Performance**: **95.13% ± 0.74% accuracy**, **94.98% ± 0.77% macro-F1** (`C = 5.0` selected in 4/5 folds).
3. **Key Hybrid Comparison Models (`ExpC_bb` and `ExpC_full_bb`)**:
   - **CV Performance**: `ExpC_bb` (**94.92% ± 0.66% accuracy**, **94.76% ± 0.73% macro-F1**, `C = 0.25`) and `ExpC_full_bb` (**95.47% ± 0.68% accuracy**, **95.31% ± 0.76% macro-F1**, `C = 0.5`), included to test whether the null/negative finding for pairs beyond unigrams replicates on the held-out test split.
4. **Best Interpretable Pair-Only Scorers (`ExpB1_bb` and `ExpB1_all`)**:
   - **CV Performance**: Log-odds pair scoring on Curveball backbone pairs `ExpB1_bb` (**90.63% ± 0.78% accuracy**, **90.07% ± 0.90% macro-F1**, $\varepsilon = 10^{-6}$) and on all pairs with support $\ge 5$ `ExpB1_all` (**94.62% ± 0.61% accuracy**, **94.39% ± 0.59% macro-F1**, $\text{min\_support} = 5, \varepsilon = 10^{-6}$).
5. **Best Improved Original Fingerprint Scorer (`ExpD_nb_full` and `ExpF2_neg_raw`)**:
   - **CV Performance**: Original NPMI backbone scorer with unigram Naive Bayes fallback on zero-score recipes `ExpD_nb_full` (**88.54% ± 0.85% accuracy**, **87.74% ± 0.97% macro-F1**, zero-score subset accuracy **89.50%** vs. **50.30%** majority fallback) and negative-evidence scorer `ExpF2_neg_raw` (**86.96% ± 0.68% accuracy**, $\lambda = 1.0$ on exclusive competing backbone edges).

> **STOPPING POINT**: Per your instructions, the held-out test set has **not** been touched. Please review `outputs/d4_experiments/phase2_report.md` and `outputs/d4_experiments/results.csv` and confirm which final configurations you approve for the final held-out test evaluation.
