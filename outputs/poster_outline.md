# The Culinary Fingerprint: Statistically Validated Ingredient Co-Occurrence Networks for Identifying World Cuisines

## 1. Research Problem and Motivation
- **Problem Statement**: Raw ingredient co-occurrence graphs in recipe corpora are dominated by high-prevalence culinary staples (`salt`, `water`, `onion`, `garlic`), creating dense hairballs where trivial frequency masks genuine pairing structure.
- **Research Objective**: Construct Pointwise Mutual Information (PMI) and Normalized Pointwise Mutual Information (NPMI) networks for **Indian**, **Italian**, and **Mexican** cuisines, validate edges against a strict degree-preserving bipartite null model (Curveball Markov chain), analyze structural hubs, communities, and bridges, and evaluate an interpretable backbone-only **Fingerprint Scorer** against a **Multinomial Naive Bayes** baseline under strict train/test separation.

## 2. Dataset and Preprocessing
- **Source Corpus**: Kaggle *What's Cooking* (`data/raw/train.json`, SHA-256 `f18e7e568d1c6067...`), containing **39,774** total recipes and **428,275** ingredient occurrences across 20 cuisines (**6,714** raw distinct strings; **6,703** after basic lowercase/whitespace cleanup).
- **Target Cuisines**: **17,279** recipes retained — **Indian** (3,003), **Italian** (7,838), and **Mexican** (6,438).
- **Conservative Deterministic Normalization**: 5 auditable stages (`unicode` -> `lowercase` -> `whitespace` -> `singularize` via a 71-noun count-plural whitelist -> `synonym` via a 5-entry documented dictionary) reduce **4,416** raw strings in the 3 cuisines to **4,402** canonical names (`outputs/d1_dataset/merge_log.csv` logs **720** transformations across **14** multi-string merge groups; **4** recipes had duplicate ingredients deduplicated).
- **Leakage-Free Split**: Descriptive analyses (Deliverables 1-3) characterize all **17,279** selected recipes; predictive evaluation (Deliverable 4) uses a stratified 80/20 split (seed `42`: **13,823** train / **3,456** test recipes) performed on recipe IDs *prior* to all frequency filtering, null-model randomization, backbone extraction, and classifier fitting.

## 3. NPMI-Based Network Construction
- **Node Eligibility**: Ingredients appearing in $\ge 20$ recipes within a cuisine are retained as nodes (`outputs/d1_dataset/nodes.csv`):
  - **Indian**: 268 eligible nodes, 22,284 raw co-occurring pairs.
  - **Italian**: 487 eligible nodes, 48,488 raw co-occurring pairs.
  - **Mexican**: 423 eligible nodes, 42,703 raw co-occurring pairs.
- **Association Weighting**: Binary recipe presence defines $P(a) = n(a)/N$, $P(a,b) = n(a,b)/N$, $\text{PMI}(a,b) = \log_2(P(a,b)/[P(a)P(b)])$, and $\text{NPMI}(a,b) = \text{PMI}(a,b)/(-\log_2 P(a,b)) \in [-1, 1]$. Negative PMI/NPMI values are preserved in all edge tables.

## 4. Null-Model and Backbone Methodology
- **Degree-Preserving Bipartite Randomization**: One Curveball Markov chain (Strona et al., 2014) per cuisine on the full binary recipe-by-ingredient incidence matrix ($10 \times N$ burn-in trades, **1,000** samples, $2 \times N$ thinning trades/sample), preserving every recipe's exact basket size and every ingredient's exact frequency.
- **Mixing Adequacy**: Effective sample size $\text{ESS}(S)$ of $S = \sum_{a<b} n(a,b)^2$ is **655** (Indian), **667** (Italian), and **691** (Mexican).
- **Backbone Filtering**: Retain edges with strict $z = (n(a,b) - \mu_{\text{null}})/\sigma_{\text{null}} > 3$ and $\text{co\_count} \ge 5$ (excluding zero-variance pairs where $\sigma_{\text{null}} = 0$):
  - **Indian**: 22,284 $\to$ **1,829** edges (**8.21%** retained, 261 backbone nodes).
  - **Italian**: 48,488 $\to$ **3,758** edges (**7.75%** retained, 473 backbone nodes).
  - **Mexican**: 42,703 $\to$ **2,847** edges (**6.67%** retained, 408 backbone nodes).

## 5. Hub, Community, and Bridge Analysis
- **Top Weighted-Degree Hubs** ($\text{Strength}(a) = \sum \max(\text{NPMI}, 0)$):
  - **Indian**: clove (12.77), garlic paste (12.28), green cardamom (12.18); top weighted-betweenness broker: `peeled fresh ginger` (0.0540).
  - **Italian**: vanilla extract (18.24), carrot (18.11), sugar (15.50); top weighted-betweenness broker: `carrot` (0.0527).
  - **Mexican**: ancho chile pepper (13.80), ground cinnamon (13.04), dried oregano (11.51); top weighted-betweenness broker: `cooking spray` (0.0460).
- **Universal vs Cuisine-Specific Hubs**: Across 26 distinct top-10 strength hubs, **0** are universal across all 3 cuisines, **4** are two-cuisine hubs (`clove, all-purpose flour, baking powder, sugar`), and **22** are cuisine-specific.
- **Louvain Communities**: 7 communities in Indian ($Q = 0.488$), 6 in Italian ($Q = 0.473$), and 7 in Mexican ($Q = 0.497$).
- **Strongest Structural Bridges** ($\text{Bridge}(a; A,B) = \min(\text{Strength}_A(a), \text{Strength}_B(a))$):
  - **Indian-Italian**: `bay leaf` (score 8.21).
  - **Indian-Mexican**: `clove` (score 9.60).
  - **Italian-Mexican**: `dried oregano` (score 11.51).

## 6. Fingerprint Scorer and Naive Bayes Comparison
- **Models Evaluated on Held-Out Test Split ($N_{\text{test}} = 3,456$)**:
  1. **Fingerprint Scorer** ($\text{Score}(c) = \sum_{(a,b) \in E_c^{\text{train}}} \text{NPMI}_c(a,b)$): Accuracy = **87.67%** (`0.8767`), Macro-F1 = **0.8712** (F1: Indian `0.855`, Italian `0.894`, Mexican `0.865`; zero-score fallbacks = `139`, tied-score fallbacks = `0`).
  2. **Fingerprint (normalized ablation)** ($\text{Score}_{\text{norm}}(c) = \text{Score}(c) / |M_c(R)|$ for $|M_c(R)| > 0$): Accuracy = **43.69%** (`0.4369`), Macro-F1 = **0.4314** (empirically 43.98 percentage points below the sum scorer; structurally, dividing by the matched-edge count $|M_c(R)|$ discards cumulative pair-count evidence so a single matched edge in one backbone can exceed the multi-edge average in another).
  3. **Multinomial Naive Bayes Baseline** (binary presence over all 4,105 training ingredients): Accuracy = **95.98%** (`0.9598`), Macro-F1 = **0.9599** (F1: Indian `0.962`, Italian `0.963`, Mexican `0.954`).
- **Diagnostic Probe**: `predict(['garam masala', 'cumin', 'turmeric'])` $\to$ **`indian`** via edge `('cumin', 'garam masala')` ($\text{NPMI} = 0.1918$ in Indian vs $0.0$ in Italian/Mexican); exact dataset strings `['garam masala', 'ground cumin', 'ground turmeric']` match all 3 pairs in Indian ($\text{Score} = 0.4447$) $\to$ **`indian`**.

## 7. Three Recommended Poster Figures (Generated Files)
1. `outputs/d3_report/figures/fig1_backbone_edges.png` — Raw vs statistically validated backbone edge counts and retention percentages across Indian, Italian, and Mexican cuisines.
2. `outputs/d3_report/figures/fig2_top_hubs.png` — Side-by-side comparison of top-10 weighted-degree hubs per cuisine.
3. `outputs/d4_scorer/confusion_matrix_normalized.png` (paired with `outputs/d4_scorer/confusion_matrix_normalized_nb.png`) — Row-normalized confusion matrices with test recipe counts comparing the Fingerprint Scorer and Multinomial Naive Bayes.

## 8. Key Numerical Results Summary
| Metric | Indian | Italian | Mexican |
| --- | ---: | ---: | ---: |
| Recipes (Full / Train / Test) | 3,003 / 2,403 / 600 | 7,838 / 6,270 / 1,568 | 6,438 / 5,150 / 1,288 |
| Eligible Nodes / Backbone Nodes | 268 / 261 | 487 / 473 | 423 / 408 |
| Raw Edges $\to$ Backbone Edges (% Retained) | 22,284 $\to$ 1,829 (8.21%) | 48,488 $\to$ 3,758 (7.75%) | 42,703 $\to$ 2,847 (6.67%) |
| Louvain Communities (Modularity $Q$) | 7 (0.488) | 6 (0.473) | 7 (0.497) |
| Fingerprint Scorer Test F1 | 0.8549 | 0.8937 | 0.8650 |
| Naive Bayes Test F1 | 0.9625 | 0.9633 | 0.9540 |

## 9. Conclusion
- Degree-preserving bipartite null models prune over **91-93%** of raw co-occurrence edges while uncovering modular sub-cuisine communities ($Q = 0.47\text{-}0.50$) and specialized spice/aromatics hubs.
- Summing validated backbone NPMI weights alone achieves **87.67%** three-way accuracy (**0.8712** macro-F1) with completely transparent pair-level explanations, while full-vocabulary Multinomial Naive Bayes achieves **95.98%** accuracy by leveraging singleton ingredient marginals and rare (<20 recipe) ingredients.

## 10. Methodological Limitations
- **Multiple Testing**: The threshold $z > 3$ with $\text{co\_count} \ge 5$ is a standard heuristic filter rather than a family-wise or FDR-corrected significance level across $18\text{k}-48\text{k}$ observed pairs.
- **Eligibility & Sparsity Cutoff**: Filtering nodes at $\ge 20$ recipes and requiring $\ge 2$ matched backbone nodes per test recipe causes **139 / 3,456 (4.02%)** short or rare-ingredient test recipes to receive zero backbone score and fall back to the majority training class (`italian`).
- **Binary Presence**: Recipe-level incidence ignores ingredient quantities and preparation techniques.
