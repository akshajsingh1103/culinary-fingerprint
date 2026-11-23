# Deliverable 4 — Phase 1 Diagnostics Report (5-Fold CV on Training Split Only)

> **Test Set Isolation Statement**: The 3,456 held-out test recipes (`test_ids` in `data/processed/split.json`) were **never loaded, inspected, or evaluated** in any calculation below. Every metric, fallback count, coverage distribution, statistical test, and misclassified recipe example is computed strictly from **5-fold stratified out-of-fold (OOF) predictions across the 13,823 training recipes** (`seed = 42`). Within each fold, only that fold's training partition ($4/5$ of $13,823 \approx 11,058$ recipes) was used to normalize ingredients, build eligible networks (`min_ingredient_count = 20`), run Curveball null chains (`n_randomizations = 100`, documented approximation of the 1,000-sample chain), extract statistically validated backbones ($z > 3.0$, $\text{co\_count} \ge 5$), and derive fold-specific vocabularies.

---

## 0. Reproducibility & Fold-Specific Feature Summary

- **Global Seed**: `42` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
- **Total Training Recipes Evaluated (OOF)**: `13,823` (`indian`: 2,403 | `italian`: 6,270 | `mexican`: 5,150)
- **Held-Out Test Recipes (Untouched)**: `3,456`
- **Curveball Randomizations per Fold per Cuisine**: `100` (`burn_in = 10 * N_c`, `thinning = 2 * N_c`; approximation to the 1,000-randomization full training run)
- **Backbone Thresholds**: `min_ingredient_count = 20`, `z_threshold = 3.0` (strict `z > 3.0`), `min_co_count = 5`, `zero_variance_policy = exclude`
- **Naive Bayes Hyperparameters**: `MultinomialNB(alpha=1.0, fit_prior=True)` over binary ingredient incidence
- **Total 5-Fold CV Runtime**: `8.23 s` (fold assignments saved to `outputs/d4_experiments/cv_folds.json`)

| Fold | Train / Val Recipes | Full Unigram Vocab | Union Eligible Nodes | Union Backbone Vocab (~692-style) | Union Backbone Pairs | Indian Nodes / Edges (Elig -> BB) | Italian Nodes / Edges (Elig -> BB) | Mexican Nodes / Edges (Elig -> BB) | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 11,058 / 2,765 | 3,834 | 623 | **606** | **4,865** | 207/212 (15110->1146) | 355/368 (31265->2329) | 304/316 (27479->1749) | 1.58 |
| 1 | 11,058 / 2,765 | 3,861 | 634 | **615** | **4,859** | 201/207 (14568->1116) | 369/379 (32265->2391) | 304/323 (27902->1732) | 1.50 |
| 2 | 11,058 / 2,765 | 3,818 | 632 | **608** | **4,909** | 209/213 (15154->1168) | 361/376 (31747->2348) | 306/325 (28346->1752) | 1.62 |
| 3 | 11,059 / 2,764 | 3,809 | 630 | **605** | **4,823** | 206/209 (14735->1147) | 354/373 (31612->2309) | 307/323 (28106->1722) | 1.66 |
| 4 | 11,059 / 2,764 | 3,849 | 626 | **615** | **4,867** | 210/213 (15097->1178) | 364/371 (31755->2316) | 308/322 (27905->1741) | 1.62 |

> **Note on Fold vs. Full-Training Backbone Sizes**: On the full 13,823 training set (1,000 randomizations), the per-cuisine backbone node counts are 237 (Indian), 413 (Italian), and 350 (Mexican) with a union backbone vocabulary of 642 ingredients (whereas the descriptive full-dataset D2 backbones on all 17,279 recipes had 261 / 473 / 408 nodes and 692 union backbone ingredients). In 5-fold CV, each training fold has 80% of the training split ($\approx 11,058$ recipes), so thresholding at `min_ingredient_count = 20` yields 605–615 union backbone nodes and 4,823–4,909 unique validated backbone pairs per fold.

---

## 1. Exact Scoring Code, Zero-Score Fallback, and Normalization Formula

### 1.1 Exact Source Code (`src/fingerprint.py`, lines 47–94)
```python
class FingerprintScorer:
    """Score recipes against per-cuisine backbones. ``normalized=True`` gives the ablation."""

    def __init__(self, backbones: dict[str, pd.DataFrame], class_counts: dict[str, int], normalized: bool = False):
        self.cuisines = list(backbones)
        missing = [c for c in self.cuisines if c not in class_counts]
        if missing:
            raise ValueError(f"class_counts missing for {missing}")
        self.normalized = normalized
        self.edges: dict[str, dict[tuple[str, str], float]] = {}
        for c, df in backbones.items():
            d = {}
            for a, b, w in zip(df["ingredient_a"], df["ingredient_b"], df["NPMI"]):
                a, b = (a, b) if a < b else (b, a)
                d[(a, b)] = float(w)
            self.edges[c] = d
        self.class_counts = {c: int(class_counts[c]) for c in self.cuisines}
        self.priority = sorted(self.cuisines, key=lambda c: (-self.class_counts[c], c))
        self.vocab = {c: {x for p in self.edges[c] for x in p} for c in self.cuisines}

    def explain(self, ingredients: list[str]) -> dict:
        norm = normalize_recipe(list(ingredients))
        per = {}
        for c in self.cuisines:
            matched = []
            for a, b in itertools.combinations(norm, 2):  # norm is sorted -> a < b
                w = self.edges[c].get((a, b))
                if w is not None:
                    matched.append({"pair": [a, b], "NPMI": w})
            total = float(sum(m["NPMI"] for m in matched))
            score = (total / len(matched) if matched else 0.0) if self.normalized else total
            per[c] = {"matched_pairs": matched, "n_matched": len(matched), "npmi_sum": total, "score": score,
                      "inputs_in_backbone_vocab": [x for x in norm if x in self.vocab[c]],
                      "training_class_count": self.class_counts[c]}
        scores = {c: per[c]["score"] for c in self.cuisines}
        label, fb = self._decide(scores)
        return {"normalized_input": norm, "per_cuisine": per, "scores": scores, "prediction": label,
                "fallback": fb, "priority_order": self.priority, "variant": "normalized" if self.normalized else "primary"}

    def _decide(self, scores: dict) -> tuple[str, str | None]:
        m = max(scores.values())
        tied = [c for c in self.cuisines if abs(scores[c] - m) <= TIE_TOL]
        if len(tied) == 1:
            return tied[0], None
        choice = next(c for c in self.priority if c in tied)
        if all(abs(s) <= TIE_TOL for s in scores.values()):
            return choice, "zero_score"
        return choice, "tied_score"
```

### 1.2 Normalization Formula in Plain Words
- **Original (Primary) Fingerprint Scorer**:
  $$\text{Score}(c) = \sum_{\substack{a < b \\ a, b \in R \\ (a,b) \in E_c}} \text{NPMI}_c(a, b)$$
  In plain words: For a recipe $R$ (after normalizing, deduplicating, and sorting its ingredient names), we generate all $\binom{|R|}{2}$ unordered ingredient pairs and **sum the NPMI weights of all pairs that appear in cuisine $c$'s training backbone $E_c$** (pairs absent from $E_c$ contribute $0$).
- **Normalized Ablation (`normalized=True`)**:
  $$\text{Score}_{\text{norm}}(c) = \begin{cases} \dfrac{\text{Score}(c)}{M_c(R)} & \text{if } M_c(R) > 0 \\[6pt] 0 & \text{if } M_c(R) = 0 \end{cases} \quad \text{where } M_c(R) = \left|\{(a,b) \in E_c : a, b \in R\}\right|$$
  In plain words: For each cuisine $c$, the normalized ablation divides the total NPMI sum for cuisine $c$ by **the number of matched backbone pairs $M_c(R)$ in that same cuisine $c$ for that recipe** (or returns $0$ if no pairs match in cuisine $c$). That is, it scores each cuisine by the **arithmetic mean NPMI of the pairs that matched in that cuisine's backbone**, rather than dividing by the total possible pairs in the recipe $\binom{|R|}{2}$ or by the size of cuisine $c$'s backbone $|E_c|$.
- **Zero-Score & Tied-Score Fallback**:
  When two or more cuisines tie for the maximum score within `TIE_TOL = 1e-12`, `_decide` picks the tied cuisine with the highest training-split recipe count (`self.priority = ['italian', 'mexican', 'indian']`). If all three cuisines have score $0$ (`all(abs(s) <= 1e-12)`), it is logged as a `zero_score` fallback and deterministically predicts **`italian`**.

---

## 2. Per-Class Metrics, Prediction Counts, and Normalized Ablation Bug/Pathology Analysis

### 2.1 5-Fold Cross-Validation Summary (13,823 Training Recipes)

| Model | CV Accuracy (Mean ± SD) | CV Macro-F1 (Mean ± SD) | Pooled OOF Accuracy [95% Bootstrap CI] | Pooled OOF Macro-F1 [95% Bootstrap CI] | Zero-Score Fallbacks | Tied-Score Fallbacks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fingerprint Scorer** | 86.62% ± 0.77% | 85.82% ± 0.92% | 86.62% [86.07%, 87.19%] | 85.83% [85.19%, 86.44%] | 676 | 0 |
| **Fingerprint (normalized ablation)** | 47.43% ± 1.05% | 46.87% ± 1.14% | 47.43% [46.57%, 48.30%] | 46.89% [46.01%, 47.78%] | 676 | 0 |
| **Naive Bayes** | 95.66% ± 0.58% | 95.53% ± 0.62% | 95.66% [95.33%, 96.01%] | 95.53% [95.16%, 95.88%] | 0 | 0 |

### 2.2 Per-Class Precision, Recall, F1, and Prediction Counts (Pooled 5-Fold OOF)

| Model | Cuisine | True Count | Predicted Count | Pred Share (%) | Precision (%) | Recall (%) | F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fingerprint Scorer** | `indian` | 2,403 (17.4%) | **2,113** | **15.29%** | 89.12% | 78.36% | 83.39% |
| **Fingerprint Scorer** | `italian` | 6,270 (45.4%) | **6,697** | **48.45%** | 85.95% | 91.80% | 88.78% |
| **Fingerprint Scorer** | `mexican` | 5,150 (37.3%) | **5,013** | **36.27%** | 86.48% | 84.17% | 85.31% |
| **Fingerprint (normalized ablation)** | `indian` | 2,403 (17.4%) | **5,037** | **36.44%** | 32.26% | 67.62% | 43.68% |
| **Fingerprint (normalized ablation)** | `italian` | 6,270 (45.4%) | **4,658** | **33.70%** | 61.40% | 45.61% | 52.34% |
| **Fingerprint (normalized ablation)** | `mexican` | 5,150 (37.3%) | **4,128** | **29.86%** | 50.17% | 40.21% | 44.64% |
| **Naive Bayes** | `indian` | 2,403 (17.4%) | **2,414** | **17.46%** | 94.99% | 95.42% | 95.20% |
| **Naive Bayes** | `italian` | 6,270 (45.4%) | **6,439** | **46.58%** | 94.97% | 97.53% | 96.23% |
| **Naive Bayes** | `mexican` | 5,150 (37.3%) | **4,970** | **35.95%** | 96.88% | 93.50% | 95.16% |

### 2.3 Out-of-Fold Confusion Matrices (Rows = True Cuisine, Columns = Predicted Cuisine)

| Model | True \ Pred | `indian` | `italian` | `mexican` |
| :--- | :--- | :---: | :---: | :---: |
| **Fingerprint Scorer** | `indian` | 1,883 | 265 | 255 |
| **Fingerprint Scorer** | `italian` | 91 | 5,756 | 423 |
| **Fingerprint Scorer** | `mexican` | 139 | 676 | 4,335 |
| **Fingerprint (normalized ablation)** | `indian` | 1,625 | 358 | 420 |
| **Fingerprint (normalized ablation)** | `italian` | 1,773 | 2,860 | 1,637 |
| **Fingerprint (normalized ablation)** | `mexican` | 1,639 | 1,440 | 2,071 |
| **Naive Bayes** | `indian` | 2,293 | 74 | 36 |
| **Naive Bayes** | `italian` | 36 | 6,115 | 119 |
| **Naive Bayes** | `mexican` | 85 | 250 | 4,815 |

### 2.4 Diagnosis of the Normalized Ablation: Is It Predicting One Class Mostly? Is There a Bug?

1. **Yes — the normalized ablation massively over-predicts `indian` (the smallest class)**:
   - Although `indian` accounts for only **17.38% (2,403 / 13,823)** of the training recipes, the normalized ablation makes **`indian` its most frequently predicted class (5,037 predictions, or 36.44% of all recipes — more than double the true Indian share)**!
   - Consequently, `indian` precision collapses from **89.12%** (original scorer) down to **32.26%**, while `italian` recall collapses from **91.80%** down to **45.61%** (predicting only 4,658 Italian recipes out of 6,270 true Italian recipes) and `mexican` recall collapses from **84.17%** down to **40.21%**.
2. **Why this happens (Mathematical Pathology of Dividing by `len(matched)`)**:
   - The code in `src/fingerprint.py` line 76 (`score = (total / len(matched) if matched else 0.0)`) implements the literal specification in its docstring (`Score_norm(c) = Score(c) / (#matched pairs in E_c)`). Thus, it is **not a coding typo for backbone size $|E_c|$**, but rather a **fundamental statistical flaw** in normalizing by the per-cuisine matched-pair count $M_c(R)$:
     - **Loss of volume evidence**: Dividing by `len(matched)` discards how *many* backbone edges matched. A recipe matching 30 Italian backbone edges with average NPMI $0.17$ (`npmi_sum = 5.10`) loses to a cuisine where just **1 single incidental pair** matches with NPMI $0.22$ (`npmi_sum = 0.22`, `score_norm = 0.22 > 0.17`).
     - **Small-denominator + inverse-prevalence amplification**: For a true Italian or Mexican recipe, many common staples match in the true cuisine's large backbone (pulling the true cuisine's *average* NPMI down toward the mean backbone edge weight of ~0.16–0.19 because high-frequency staples like `salt`, `garlic`, `olive oil`, `onion` have moderate NPMI). Meanwhile, if that recipe contains just 1 or 2 pairs that happen to exist in another cuisine's backbone (especially `indian`, which has the smallest backbone of ~1,150 edges), those ingredients often have lower marginal prevalence $P(a), P(b)$ in that competing cuisine, which inflates their conditional PMI/NPMI!
     - **Empirical proof from CV**: Across the 13,823 OOF recipes, **3,000 non-Indian recipes** that the primary Fingerprint Scorer classified *correctly* were flipped to **`indian`** by the normalized ablation. Among those flipped recipes, the average number of matched pairs in the true cuisine was **15.25 pairs**, whereas the average number of matched pairs in `indian` was only **3.46 pairs** (and **1,517** of those flipped recipes had only **1 or 2 matched pairs** in Indian!).
     - By contrast, if one wanted to normalize for recipe length without discarding match volume relative to other cuisines, dividing by total recipe pairs $\binom{|R|}{2}$ is a positive constant across all three cuisines for a given recipe and therefore produces **identical argmax predictions** to the unnormalized sum, whereas calibrating for unequal cuisine backbone sizes ($|E_c|$ or $|V_c|$) requires dividing or shifting by a **cuisine-level** constant rather than the recipe-specific matched count $M_c(R)$ (explored in Phase 2 Experiment F).

---

## 3. Fallback Analysis (5-Fold CV Out-of-Fold Predictions)

Across the 13,823 OOF training predictions, the primary Fingerprint Scorer triggered **676 zero-score fallbacks (4.89%)** and **0 tied-score fallbacks**, leaving **13,147 recipes (95.11%)** with a unique positive winning score.

### 3.1 Accuracy and True-Class Distribution: Zero-Score Fallback vs. Non-Fallback Recipes

| Subset | Recipe Count | True `indian` | True `italian` | True `mexican` | Fingerprint Scorer Accuracy | Naive Bayes Accuracy on Same Subset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Score Fallback (`score = 0` in all 3 cuisines)** | **676** (4.89%) | 91 (13.46%) | 340 (50.30%) | 245 (36.24%) | **50.30%** | **89.50%** |
| **Non-Fallback (`max(score) > 0`)** | **13,147** (95.11%) | 2,312 (17.59%) | 5,930 (45.11%) | 4,905 (37.31%) | **88.49%** | **95.98%** |
| **All Training Recipes (OOF Pooled)** | **13,823** (100.0%) | 2,403 (17.38%) | 6,270 (45.36%) | 5,150 (37.26%) | **86.62%** | **95.66%** |

- **Why Zero-Score Fallback Accuracy Equals the True Italian Share**: Because `italian` is the largest class in every training fold, `self.priority[0]` is always `'italian'`. Thus all 676 zero-score recipes are predicted as `'italian'`, yielding 340/676 = **50.30%** accuracy (0% recall on the 91 Indian and 245 Mexican fallback recipes).
- **Unigram Evidence Exists on Zero-Score Fallbacks**: On the exact same 676 zero-score fallback recipes where no backbone *pair* matched, **Naive Bayes achieves 89.50% accuracy** (+39.20 percentage points over the majority-class fallback!). Replacing the majority-class fallback with a unigram fallback on just these 676 recipes would recover 265 additional correct predictions (+1.92% overall accuracy). This directly motivates **Phase 2 Experiment D**.

### 3.2 Ingredient Count Distribution of Zero-Score Fallback Recipes

- **Normalized Ingredient Count (`n_ingredients`)**:
  - Zero-score fallback recipes ($n=676$): **mean = 4.99 ± 1.85**, **median = 5.0**, **min = 1**, **max = 13**.
  - Non-fallback recipes ($n=13,147$): **mean = 11.05 ± 4.38**, **median = 10.0**, **min = 2**, **max = 65**.
- **Ingredients Present in Any Backbone Vocabulary (`n_in_any_backbone`)**:
  - Zero-score fallback recipes ($n=676$): **mean = 3.03 ± 1.57**, **median = 3.0**, **min = 0**, **max = 8**.
  - Notice that even when a recipe has 2–8 ingredients inside the union backbone vocabulary, none of its ingredient pairs may cross the $z > 3.0, \text{co\_count} \ge 5$ threshold in any cuisine's backbone.

| Ingredient Count ($k$) | Zero-Score Recipes with $k$ Total Normalized Ingredients | Zero-Score Recipes with $k$ Ingredients in Union Backbone Vocab |
| :---: | :---: | :---: |
| 0 | 0 (0.0%) | 23 (3.4%) |
| 1 | 8 (1.2%) | 98 (14.5%) |
| 2 | 44 (6.5%) | 140 (20.7%) |
| 3 | 94 (13.9%) | 169 (25.0%) |
| 4 | 142 (21.0%) | 125 (18.5%) |
| 5 | 136 (20.1%) | 77 (11.4%) |
| 6 | 116 (17.2%) | 33 (4.9%) |
| 7 | 75 (11.1%) | 8 (1.2%) |
| 8 | 41 (6.1%) | 3 (0.4%) |
| 9 | 12 (1.8%) | 0 (0.0%) |
| 10 | 4 (0.6%) | 0 (0.0%) |
| 11 | 2 (0.3%) | 0 (0.0%) |
| 12 | 1 (0.1%) | 0 (0.0%) |
| 13 | 1 (0.1%) | 0 (0.0%) |

---

## 4. Coverage Analysis & Accuracy vs. Matched-Pair Count

### 4.1 Ingredient & Pair Coverage Across All 13,823 Training Recipes (5-Fold OOF)

- **Total Normalized Ingredients per Recipe**: mean = **10.76 ± 4.49**, median = **10.0**, range = **[1, 65]**.
- **Ingredients in Any Cuisine's Backbone Vocab per Recipe**: mean = **9.33 ± 4.28** (**85.6%** of a recipe's ingredients on average), median = **9.0**, range = **[0, 53]**.

| Cuisine Backbone | Mean Matched Pairs (All 13,823 Recipes) | Median Matched Pairs (All) | Mean Matched Pairs When True Cuisine = $c$ | Mean Matched Pairs When True Cuisine $\ne c$ | Recipes with 0 Matched Pairs in $c$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `indian` | 4.34 | 2.0 | **14.96** | 2.10 | 4,263 (30.8%) |
| `italian` | 7.07 | 4.0 | **10.85** | 3.93 | 2,216 (16.0%) |
| `mexican` | 7.15 | 4.0 | **11.46** | 4.59 | 1,864 (13.5%) |

### 4.2 Accuracy vs. Total Matched Backbone Pair Count

Plot saved to: `outputs/d4_experiments/accuracy_vs_matched_pairs.png` (and data in `outputs/d4_experiments/coverage_accuracy_table.csv`).

| Total Matched Pairs Bin (Sum Across Cuisines) | Recipes ($N$) | Share of Training Set (%) | Mean Recipe Ingredients | Mean Backbone Ingredients | Fingerprint Scorer Acc (%) | Normalized Ablation Acc (%) | Naive Bayes Acc (%) | Gap (NB - Scorer) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (zero-score fallback)** | 676 | 4.89% | 4.99 | 3.03 | **50.30%** | 50.30% | **89.50%** | +39.20 pp |
| **1** | 579 | 4.19% | 6.03 | 4.36 | **83.94%** | 83.94% | **94.30%** | +10.36 pp |
| **2** | 548 | 3.96% | 6.50 | 4.89 | **71.35%** | 71.35% | **92.70%** | +21.35 pp |
| **3–4** | 1,071 | 7.75% | 7.22 | 5.74 | **81.51%** | 63.68% | **94.58%** | +13.07 pp |
| **5–7** | 1,500 | 10.85% | 8.15 | 6.80 | **85.47%** | 53.40% | **94.93%** | +9.47 pp |
| **8–12** | 2,169 | 15.69% | 9.28 | 7.98 | **88.11%** | 43.85% | **95.85%** | +7.75 pp |
| **13–20** | 2,613 | 18.90% | 11.01 | 9.67 | **90.32%** | 36.59% | **95.98%** | +5.66 pp |
| **21–35** | 2,626 | 19.00% | 13.02 | 11.65 | **91.89%** | 39.22% | **97.14%** | +5.26 pp |
| **36+** | 2,041 | 14.77% | 17.26 | 15.75 | **93.97%** | 45.03% | **97.45%** | +3.48 pp |

- **Key Finding**: Fingerprint Scorer accuracy rises with matched-pair coverage, from **50.30%** at 0 matched pairs (fallback) and **71.35%–81.51%** at 2–4 matched pairs up to **93.97%** when $\ge 36$ backbone pairs match. However, **Naive Bayes outperforms the Fingerprint Scorer in every single coverage bin**, with the largest gaps occurring in low-to-moderate pair coverage recipes (0 to 7 matched pairs, representing 31.64% of the training set), where single ingredients carry strong discriminative signal even when few validated pairs fire.

---

## 5. Error Overlap & Paired Statistical Significance Tests (McNemar & Bootstrap CIs)

### 5.1 Error Overlap: Fingerprint Scorer vs. Naive Bayes (13,823 Training Recipes OOF)

| Contingency Cell | Recipe Count | Percentage of Training Set (%) |
| :--- | :---: | :---: |
| **Both Correct** (Fingerprint Scorer right & Naive Bayes right) | **11,736** | 84.90% |
| **Scorer Wrong, Naive Bayes Right** ($b$) | **1,487** | **10.76%** |
| **Scorer Right, Naive Bayes Wrong** ($c$) | **238** | **1.72%** |
| **Both Wrong** (Fingerprint Scorer wrong & Naive Bayes wrong) | **362** | 2.62% |
| **Total Discordant Recipes** ($b + c$) | **1,725** | 12.48% |

### 5.2 Paired McNemar Tests & 95% Bootstrap Confidence Intervals for Differences

| Comparison (Model A vs. Model B) | A Right, B Wrong | A Wrong, B Right | Discordant ($b+c$) | McNemar $\chi^2$ (CC) | Exact Binomial $p$-value | $\Delta$ Accuracy (A - B) [95% Bootstrap CI] | $\Delta$ Macro-F1 (A - B) [95% Bootstrap CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fingerprint Scorer vs. Naive Bayes** | 238 | 1,487 | 1,725 | 902.90 | `1.426e-220` | -9.03 pp [-9.63, -8.46] | -9.70 pp [-10.37, -9.10] |
| **Fingerprint Scorer vs. Normalized Ablation** | 5,772 | 354 | 6,126 | 4790.06 | `< 1e-300` | +39.19 pp [+38.34, +40.08] | +38.94 pp [+38.04, +39.85] |
| **Naive Bayes vs. Normalized Ablation** | 6,957 | 290 | 7,247 | 6131.58 | `< 1e-300` | +48.23 pp [+47.29, +49.14] | +48.64 pp [+47.71, +49.57] |

- **Interpretation**: Naive Bayes gets **1,487** recipes right that the Fingerprint Scorer gets wrong, whereas the Fingerprint Scorer gets only **238** recipes right that Naive Bayes gets wrong (a net deficit of **1,249** recipes, or **9.04 percentage points**, exact binomial $p = 1.426e-220$). However, the fact that the Fingerprint Scorer correctly classifies **238 recipes** where Naive Bayes errs motivates testing whether combining unigram and pair features (Phase 2 Experiment C) or using discriminative log-odds pair weights with negative evidence (Phase 2 Experiment B) can improve upon unigrams alone.

---

## 6. Fifteen Concrete Misclassified Recipes (Out-of-Fold Training Examples)

Below are 15 misclassified training recipes (with distinct ingredient lists) from the 5-fold CV out-of-fold predictions, showing their normalized ingredients, per-cuisine scores, Naive Bayes prediction, and the top matched backbone pairs (`pair: NPMI`) in each cuisine that drove the incorrect score.

### Example 1: Recipe ID `32349` (Fold 1) — True: `indian` | Predicted: `italian` | NB: `indian`
- **Failure Mechanism**: Zero-score fallback (0 backbone pairs matched -> defaults to Italian; NB correct)
- **Normalized Ingredients (13 total, 7 in union backbone vocab)**: `boneless skinless chicken breast, bread, cashew chop unsalt, curry powder, dried apricot, fresh parsley, grated orange, orange juice, peeled fresh ginger, reduced fat mayonnaise, seedless green grape, sliced green onion, watercress`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **0.0000** (0 pairs) | `italian`: **0.0000** (0 pairs) | `mexican`: **0.0000** (0 pairs) | Fallback: `zero_score`
  - `indian` matched pairs (0): *none*
  - `italian` matched pairs (0): *none*
  - `mexican` matched pairs (0): *none*

### Example 2: Recipe ID `36893` (Fold 3) — True: `mexican` | Predicted: `italian` | NB: `mexican`
- **Failure Mechanism**: Zero-score fallback (0 backbone pairs matched -> defaults to Italian; NB correct)
- **Normalized Ingredients (12 total, 5 in union backbone vocab)**: `asadero, buns, canola oil, carnitas, chiles, hass avocado, iceberg lettuce, mayonaise, onion rings, papalo, refried beans, sliced tomato`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **0.0000** (0 pairs) | `italian`: **0.0000** (0 pairs) | `mexican`: **0.0000** (0 pairs) | Fallback: `zero_score`
  - `indian` matched pairs (0): *none*
  - `italian` matched pairs (0): *none*
  - `mexican` matched pairs (0): *none*

### Example 3: Recipe ID `36370` (Fold 0) — True: `indian` | Predicted: `italian` | NB: `indian`
- **Failure Mechanism**: True Indian -> Predicted Italian (Italian backbone density/shared staples outweigh Indian pairs; NB correct)
- **Normalized Ingredients (10 total, 10 in union backbone vocab)**: `active dry yeast, bread flour, butter, egg, garam masala, milk, minced garlic, salt, warm water, white sugar`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **0.7674** (3 pairs) | `italian`: **4.6560** (17 pairs) | `mexican`: **1.0910** (4 pairs) | Fallback: `None`
  - `indian` matched pairs (3, sum=0.7674): (egg + milk: 0.307), (egg + warm water: 0.299), (butter + milk: 0.161)
  - `italian` matched pairs (17, sum=4.6560): (active dry yeast + warm water: 0.710), (bread flour + warm water: 0.651), (active dry yeast + bread flour: 0.561), (egg + white sugar: 0.408), (active dry yeast + white sugar: 0.313), (milk + white sugar: 0.310) ... (+11 more)
  - `mexican` matched pairs (4, sum=1.0910): (egg + milk: 0.466), (butter + milk: 0.316), (butter + white sugar: 0.171), (butter + egg: 0.139)

### Example 4: Recipe ID `31330` (Fold 4) — True: `indian` | Predicted: `italian` | NB: `indian`
- **Failure Mechanism**: True Indian -> Predicted Italian (Italian backbone density/shared staples outweigh Indian pairs; NB correct)
- **Normalized Ingredients (11 total, 11 in union backbone vocab)**: `active dry yeast, all-purpose flour, baking powder, butter, chopped cilantro fresh, egg, garlic, honey, plain yogurt, salt, warm water`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **1.4590** (6 pairs) | `italian`: **4.3125** (19 pairs) | `mexican`: **3.1215** (12 pairs) | Fallback: `None`
  - `indian` matched pairs (6, sum=1.4590): (all-purpose flour + baking powder: 0.428), (egg + warm water: 0.306), (butter + honey: 0.226), (all-purpose flour + warm water: 0.205), (butter + garlic: 0.165), (garlic + plain yogurt: 0.129)
  - `italian` matched pairs (19, sum=4.3125): (active dry yeast + warm water: 0.725), (all-purpose flour + baking powder: 0.524), (baking powder + egg: 0.416), (all-purpose flour + warm water: 0.311), (active dry yeast + all-purpose flour: 0.274), (active dry yeast + honey: 0.248) ... (+13 more)
  - `mexican` matched pairs (12, sum=3.1215): (all-purpose flour + baking powder: 0.588), (baking powder + warm water: 0.435), (baking powder + egg: 0.339), (all-purpose flour + egg: 0.317), (all-purpose flour + warm water: 0.313), (all-purpose flour + butter: 0.279) ... (+6 more)

### Example 5: Recipe ID `13654` (Fold 1) — True: `indian` | Predicted: `italian` | NB: `indian`
- **Failure Mechanism**: True Indian -> Predicted Italian (Italian backbone density/shared staples outweigh Indian pairs; NB correct)
- **Normalized Ingredients (18 total, 15 in union backbone vocab)**: `active dry yeast, bread flour, butter, cilantro leaves, cumin, egg, greek style plain yogurt, lemon juice, mint leaves, plain yogurt, purple onion, red bell pepper, salt, sauce, sliced cucumber, warm water, wheat flour, white sugar`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **1.2380** (6 pairs) | `italian`: **4.0345** (15 pairs) | `mexican`: **0.9791** (7 pairs) | Fallback: `None`
  - `indian` matched pairs (6, sum=1.2380): (warm water + white sugar: 0.362), (egg + warm water: 0.338), (cilantro leaves + mint leaves: 0.305), (lemon juice + salt: 0.084), (mint leaves + salt: 0.082), (cilantro leaves + salt: 0.067)
  - `italian` matched pairs (15, sum=4.0345): (active dry yeast + warm water: 0.713), (bread flour + warm water: 0.671), (active dry yeast + bread flour: 0.509), (egg + white sugar: 0.443), (active dry yeast + white sugar: 0.295), (warm water + white sugar: 0.245) ... (+9 more)
  - `mexican` matched pairs (7, sum=0.9791): (purple onion + red bell pepper: 0.200), (egg + white sugar: 0.183), (cilantro leaves + purple onion: 0.166), (cumin + salt: 0.131), (lemon juice + purple onion: 0.129), (salt + warm water: 0.100) ... (+1 more)

### Example 6: Recipe ID `21157` (Fold 2) — True: `indian` | Predicted: `mexican` | NB: `indian`
- **Failure Mechanism**: True Indian -> Predicted Mexican (shared chili/cilantro/cumin/lime/onion pairs; NB correct)
- **Normalized Ingredients (14 total, 12 in union backbone vocab)**: `cayenne pepper, chicken drumstick, cinnamon, coriander, cumin, dried basil, garlic powder, ginger, lemon, low fat plain yoghurt, onion powder, paprika, salt, spices`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **2.0429** (10 pairs) | `italian`: **1.4044** (5 pairs) | `mexican`: **4.1795** (14 pairs) | Fallback: `None`
  - `indian` matched pairs (10, sum=2.0429): (coriander + cumin: 0.368), (cayenne pepper + paprika: 0.235), (cumin + paprika: 0.220), (ginger + lemon: 0.197), (coriander + ginger: 0.191), (coriander + lemon: 0.188) ... (+4 more)
  - `italian` matched pairs (5, sum=1.4044): (garlic powder + onion powder: 0.575), (garlic powder + paprika: 0.346), (dried basil + garlic powder: 0.226), (dried basil + onion powder: 0.196), (dried basil + salt: 0.061)
  - `mexican` matched pairs (14, sum=4.1795): (garlic powder + onion powder: 0.612), (garlic powder + paprika: 0.393), (cayenne pepper + paprika: 0.375), (onion powder + paprika: 0.370), (cayenne pepper + onion powder: 0.361), (cayenne pepper + garlic powder: 0.337) ... (+8 more)

### Example 7: Recipe ID `29623` (Fold 4) — True: `indian` | Predicted: `mexican` | NB: `indian`
- **Failure Mechanism**: True Indian -> Predicted Mexican (shared chili/cilantro/cumin/lime/onion pairs; NB correct)
- **Normalized Ingredients (12 total, 10 in union backbone vocab)**: `all-purpose flour, baking powder, butter, egg, milk, oil, powdered milk, salt, sesame seed, sugar, water, yeast`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **2.3252** (9 pairs) | `italian`: **3.7961** (18 pairs) | `mexican`: **4.2880** (16 pairs) | Fallback: `None`
  - `indian` matched pairs (9, sum=2.3252): (all-purpose flour + baking powder: 0.428), (baking powder + milk: 0.365), (all-purpose flour + milk: 0.298), (egg + milk: 0.290), (milk + sugar: 0.282), (baking powder + sugar: 0.207) ... (+3 more)
  - `italian` matched pairs (18, sum=3.7961): (all-purpose flour + baking powder: 0.524), (baking powder + egg: 0.416), (baking powder + sugar: 0.322), (baking powder + milk: 0.308), (egg + milk: 0.286), (all-purpose flour + milk: 0.238) ... (+12 more)
  - `mexican` matched pairs (16, sum=4.2880): (all-purpose flour + baking powder: 0.588), (egg + milk: 0.417), (all-purpose flour + milk: 0.343), (baking powder + egg: 0.339), (butter + milk: 0.326), (baking powder + milk: 0.318) ... (+10 more)

### Example 8: Recipe ID `39969` (Fold 4) — True: `mexican` | Predicted: `italian` | NB: `mexican`
- **Failure Mechanism**: True Mexican -> Predicted Italian (larger Italian backbone accumulates more generic pair score; NB correct)
- **Normalized Ingredients (24 total, 24 in union backbone vocab)**: `avocado, bay leaves, carrot, celery, chicken, corn tortilla, dried basil, dried oregano, epazote, garlic, green bell pepper, ground black pepper, ground cumin, olive oil, onion, potato, red bell pepper, salt, shredded monterey jack cheese, spinach, tomato paste, water, whole peeled tomato, zucchini`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **2.7270** (14 pairs) | `italian`: **13.6928** (71 pairs) | `mexican`: **9.2221** (55 pairs) | Fallback: `None`
  - `indian` matched pairs (14, sum=2.7270): (green bell pepper + red bell pepper: 0.505), (carrot + zucchini: 0.278), (carrot + red bell pepper: 0.264), (carrot + green bell pepper: 0.251), (carrot + potato: 0.224), (garlic + tomato paste: 0.190) ... (+8 more)
  - `italian` matched pairs (71, sum=13.6928): (carrot + celery: 0.576), (dried basil + dried oregano: 0.561), (bay leaves + tomato paste: 0.414), (green bell pepper + red bell pepper: 0.398), (bay leaves + celery: 0.371), (bay leaves + carrot: 0.360) ... (+65 more)
  - `mexican` matched pairs (55, sum=9.2221): (carrot + celery: 0.464), (green bell pepper + red bell pepper: 0.429), (carrot + potato: 0.338), (dried oregano + ground cumin: 0.303), (bay leaves + dried oregano: 0.292), (bay leaves + carrot: 0.265) ... (+49 more)

### Example 9: Recipe ID `29917` (Fold 1) — True: `mexican` | Predicted: `italian` | NB: `mexican`
- **Failure Mechanism**: True Mexican -> Predicted Italian (larger Italian backbone accumulates more generic pair score; NB correct)
- **Normalized Ingredients (11 total, 10 in union backbone vocab)**: `all-purpose flour, baking soda, brown sugar, butter, egg, ground cinnamon, mexican chocolate, rolled oats, salt, vanilla extract, white sugar`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **0.8226** (3 pairs) | `italian`: **6.8788** (25 pairs) | `mexican`: **3.6615** (13 pairs) | Fallback: `None`
  - `indian` matched pairs (3, sum=0.8226): (all-purpose flour + baking soda: 0.416), (all-purpose flour + egg: 0.214), (butter + ground cinnamon: 0.193)
  - `italian` matched pairs (25, sum=6.8788): (baking soda + vanilla extract: 0.555), (vanilla extract + white sugar: 0.543), (ground cinnamon + vanilla extract: 0.523), (egg + white sugar: 0.443), (ground cinnamon + white sugar: 0.438), (all-purpose flour + vanilla extract: 0.386) ... (+19 more)
  - `mexican` matched pairs (13, sum=3.6615): (ground cinnamon + vanilla extract: 0.518), (all-purpose flour + vanilla extract: 0.406), (vanilla extract + white sugar: 0.397), (butter + vanilla extract: 0.326), (all-purpose flour + egg: 0.325), (ground cinnamon + white sugar: 0.283) ... (+7 more)

### Example 10: Recipe ID `7508` (Fold 0) — True: `mexican` | Predicted: `italian` | NB: `mexican`
- **Failure Mechanism**: True Mexican -> Predicted Italian (larger Italian backbone accumulates more generic pair score; NB correct)
- **Normalized Ingredients (17 total, 15 in union backbone vocab)**: `barbecue sauce, bay leaves, beer, chili beans, chili powder, crushed tomato, dried basil, dried oregano, dried thyme, ground beef, ground cumin, jalapeno chilies, large garlic cloves, low sodium chicken broth, olive oil, sweet onion, tomato paste`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **0.6608** (3 pairs) | `italian`: **6.8642** (28 pairs) | `mexican`: **3.8579** (18 pairs) | Fallback: `None`
  - `indian` matched pairs (3, sum=0.6608): (crushed tomato + jalapeno chilies: 0.264), (crushed tomato + tomato paste: 0.224), (ground cumin + large garlic cloves: 0.172)
  - `italian` matched pairs (28, sum=6.8642): (chili powder + ground cumin: 0.666), (dried basil + dried oregano: 0.549), (bay leaves + tomato paste: 0.374), (low sodium chicken broth + sweet onion: 0.357), (ground beef + tomato paste: 0.337), (dried basil + tomato paste: 0.314) ... (+22 more)
  - `mexican` matched pairs (18, sum=3.8579): (bay leaves + dried thyme: 0.400), (dried oregano + dried thyme: 0.311), (dried oregano + ground cumin: 0.306), (chili powder + ground cumin: 0.284), (bay leaves + dried oregano: 0.269), (low sodium chicken broth + tomato paste: 0.234) ... (+12 more)

### Example 11: Recipe ID `49687` (Fold 0) — True: `mexican` | Predicted: `indian` | NB: `mexican`
- **Failure Mechanism**: True Mexican -> Predicted Indian (spice/aromatic pairs score higher NPMI in Indian; NB correct)
- **Normalized Ingredients (16 total, 14 in union backbone vocab)**: `carrot, cilantro leaves, flour tortilla, garlic cloves, garlic paste, garlic sauce, lime juice, marinade, mayonaise, oil, pepper, red chili powder, salt, shredded cabbage, shrimp, taco shells`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **2.3496** (10 pairs) | `italian`: **0.4019** (2 pairs) | `mexican`: **1.0670** (7 pairs) | Fallback: `None`
  - `indian` matched pairs (10, sum=2.3496): (garlic paste + red chili powder: 0.375), (oil + red chili powder: 0.358), (garlic paste + oil: 0.339), (cilantro leaves + oil: 0.312), (cilantro leaves + garlic paste: 0.232), (oil + salt: 0.200) ... (+4 more)
  - `italian` matched pairs (2, sum=0.4019): (pepper + salt: 0.304), (carrot + garlic cloves: 0.098)
  - `mexican` matched pairs (7, sum=1.0670): (pepper + salt: 0.340), (pepper + shredded cabbage: 0.200), (lime juice + pepper: 0.131), (salt + shrimp: 0.105), (lime juice + salt: 0.102), (garlic cloves + salt: 0.099) ... (+1 more)

### Example 12: Recipe ID `14171` (Fold 1) — True: `mexican` | Predicted: `indian` | NB: `mexican`
- **Failure Mechanism**: True Mexican -> Predicted Indian (spice/aromatic pairs score higher NPMI in Indian; NB correct)
- **Normalized Ingredients (8 total, 7 in union backbone vocab)**: `chili, fresh coriander, fresh lime juice, garlic cloves, green onion, lime, plum tomato, vegetable oil`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **1.2473** (6 pairs) | `italian`: **0.1227** (1 pairs) | `mexican`: **0.4990** (3 pairs) | Fallback: `None`
  - `indian` matched pairs (6, sum=1.2473): (fresh lime juice + green onion: 0.293), (green onion + lime: 0.217), (garlic cloves + plum tomato: 0.204), (fresh coriander + lime: 0.200), (garlic cloves + vegetable oil: 0.178), (fresh lime juice + vegetable oil: 0.156)
  - `italian` matched pairs (1, sum=0.1227): (garlic cloves + plum tomato: 0.123)
  - `mexican` matched pairs (3, sum=0.4990): (fresh lime juice + plum tomato: 0.201), (fresh lime juice + garlic cloves: 0.185), (garlic cloves + vegetable oil: 0.113)

### Example 13: Recipe ID `37777` (Fold 1) — True: `italian` | Predicted: `mexican` | NB: `italian`
- **Failure Mechanism**: True Italian -> Predicted Mexican (shared baking/spice/vegetable pairs have higher NPMI in Mexican; NB correct)
- **Normalized Ingredients (10 total, 9 in union backbone vocab)**: `baking powder, cinnamon, egg, flour, salt, strawberry, sugar, unsalted butter, whipped cream, whole milk`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **1.1053** (4 pairs) | `italian`: **2.6206** (12 pairs) | `mexican`: **4.6236** (17 pairs) | Fallback: `None`
  - `indian` matched pairs (4, sum=1.1053): (baking powder + egg: 0.403), (baking powder + sugar: 0.285), (egg + flour: 0.229), (sugar + whole milk: 0.189)
  - `italian` matched pairs (12, sum=2.6206): (baking powder + egg: 0.405), (strawberry + sugar: 0.352), (baking powder + sugar: 0.335), (unsalted butter + whole milk: 0.278), (sugar + whole milk: 0.260), (egg + flour: 0.236) ... (+6 more)
  - `mexican` matched pairs (17, sum=4.6236): (unsalted butter + whole milk: 0.393), (sugar + whole milk: 0.352), (cinnamon + unsalted butter: 0.346), (cinnamon + flour: 0.329), (cinnamon + sugar: 0.326), (baking powder + egg: 0.320) ... (+11 more)

### Example 14: Recipe ID `19217` (Fold 2) — True: `italian` | Predicted: `mexican` | NB: `italian`
- **Failure Mechanism**: True Italian -> Predicted Mexican (shared baking/spice/vegetable pairs have higher NPMI in Mexican; NB correct)
- **Normalized Ingredients (20 total, 20 in union backbone vocab)**: `angel hair, black pepper, bread crumbs, broccoli, butter, canola oil, cayenne pepper, chicken breast, diced tomato, dried oregano, egg yolks, freshly grated parmesan, garlic, garlic powder, ground black pepper, heavy cream, kosher salt, milk, paprika, sugar`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **4.5574** (21 pairs) | `italian`: **3.3174** (18 pairs) | `mexican`: **4.7616** (26 pairs) | Fallback: `None`
  - `indian` matched pairs (21, sum=4.5574): (cayenne pepper + heavy cream: 0.305), (chicken breast + heavy cream: 0.296), (milk + sugar: 0.279), (butter + heavy cream: 0.276), (ground black pepper + kosher salt: 0.263), (heavy cream + kosher salt: 0.260) ... (+15 more)
  - `italian` matched pairs (18, sum=3.3174): (garlic powder + paprika: 0.346), (diced tomato + dried oregano: 0.260), (dried oregano + garlic powder: 0.259), (ground black pepper + kosher salt: 0.249), (freshly grated parmesan + heavy cream: 0.227), (egg yolks + heavy cream: 0.226) ... (+12 more)
  - `mexican` matched pairs (26, sum=4.7616): (garlic powder + paprika: 0.393), (cayenne pepper + paprika: 0.375), (cayenne pepper + garlic powder: 0.337), (butter + milk: 0.331), (ground black pepper + kosher salt: 0.263), (black pepper + paprika: 0.245) ... (+20 more)

### Example 15: Recipe ID `47616` (Fold 2) — True: `italian` | Predicted: `indian` | NB: `indian`
- **Failure Mechanism**: True Italian -> Predicted Indian (spice/aromatic crossover)
- **Normalized Ingredients (14 total, 13 in union backbone vocab)**: `chili powder, cilantro leaves, clove, coriander powder, garam masala, ground turmeric, jeera, olive oil, onion, pasta, salt, sugar, tomato, water`
- **Fingerprint Scores (Matched Pairs)**: `indian`: **5.8407** (30 pairs) | `italian`: **0.8005** (8 pairs) | `mexican`: **1.1822** (11 pairs) | Fallback: `None`
  - `indian` matched pairs (30, sum=5.8407): (chili powder + coriander powder: 0.349), (coriander powder + ground turmeric: 0.312), (coriander powder + tomato: 0.304), (coriander powder + garam masala: 0.301), (cilantro leaves + coriander powder: 0.296), (cilantro leaves + tomato: 0.287) ... (+24 more)
  - `italian` matched pairs (8, sum=0.8005): (olive oil + onion: 0.126), (onion + pasta: 0.120), (sugar + water: 0.113), (onion + tomato: 0.109), (olive oil + salt: 0.096), (salt + water: 0.095) ... (+2 more)
  - `mexican` matched pairs (11, sum=1.1822): (clove + sugar: 0.164), (olive oil + salt: 0.135), (sugar + water: 0.127), (onion + tomato: 0.117), (cilantro leaves + olive oil: 0.115), (chili powder + salt: 0.108) ... (+5 more)

---

## 7. Summary of Phase 1 Diagnostic Findings (Implications for Phase 2)

1. **Why the Original Fingerprint Scorer Trails Naive Bayes by 9.04 pp (86.62% vs. 95.66% in 5-Fold CV)**:
   - **Positive-only NPMI summation lacks negative evidence and discriminative contrast**: A pair like `(garlic, onion)`, `(active dry yeast, warm water)`, or `(garlic powder, onion powder)` has positive NPMI in every backbone where it is retained, adding positive score to a cuisine even when the pair is shared across cuisines or when other ingredients in the recipe strongly rule that cuisine out. Unlike log-odds or logistic regression, summing raw intra-cuisine NPMI never penalizes a cuisine for pairs characteristic of a competing cuisine (addressed in **Phase 2 Experiments B & C**).
   - **Severe backbone size imbalance**: In every fold, the Italian backbone has ~2.0× as many validated edges (~2,339 edges) as the Indian backbone (~1,151 edges) and ~1.34× as many as the Mexican backbone (~1,739 edges). Recipes with generic culinary or baking staples accumulate larger sums in the denser Italian backbone, causing Italian to be over-predicted (6,697 predicted vs. 6,270 true) and Indian/Mexican to be under-predicted (addressed in **Phase 2 Experiment F**).
   - **Majority-class zero-score fallback wastes unigram signal**: 676 OOF recipes (4.89%) match zero backbone pairs and default blindly to `italian` (50.30% accuracy), even though unigram Naive Bayes achieves **89.50% accuracy** on those exact same 676 recipes (addressed in **Phase 2 Experiment D**).
   - **Quadratic sparsity of pairs vs. unigrams**: Low-coverage recipes with 1–4 matched backbone pairs (15.90% of recipes) achieve only 71.35%–83.94% accuracy under pair-only NPMI scoring vs. 92.70%–94.58% under unigram Naive Bayes (addressed in **Phase 2 Experiments A, C, E**).
2. **Why the Normalized Ablation Collapses to 47.43% Accuracy**:
   - Dividing `Score(c)` by the per-cuisine matched-pair count `len(matched)` turns the score into the *mean NPMI of matched edges in cuisine $c$*, completely discarding the number of matched edges. When a non-Indian recipe matches many pairs in its true cuisine (averaging 15.25 matched pairs across staples and signature pairs) but happens to match a few incidental pairs in `indian` (averaging 3.46 matched pairs, where lower marginal ingredient frequencies yield a higher per-pair NPMI), the mean-NPMI ablation flips the prediction to `indian` (causing `indian` to become the most predicted class at 36.44% of all recipes, with precision dropping to 32.26%).
