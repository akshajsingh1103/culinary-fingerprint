# The Culinary Fingerprint: Statistically Validated Ingredient Co-Occurrence Networks for Identifying World Cuisines

## 1. Research Problem and Motivation
- **Problem Statement**: Raw ingredient co-occurrence graphs in recipe corpora are dominated by high-prevalence culinary staples (`salt`, `water`, `onion`, `garlic`), creating dense hairballs where trivial frequency masks genuine pairing structure.
- **Research Objective**: Construct Pointwise Mutual Information (PMI) and Normalized Pointwise Mutual Information (NPMI) networks for **Indian**, **Italian**, and **Mexican** cuisines (Deliverable 1), validate edges against a strict degree-preserving bipartite null model (Curveball Markov chain, Deliverable 2), analyze structural hubs, communities, and bridges (Deliverable 3), and build and evaluate an interpretable network-based **Fingerprint Scorer** — evolving from **Scorer v1** (raw NPMI sum) to **Scorer v2** (22-feature network-informed model) alongside a live **Streamlit application** (`outputs/app/app.py`) — under strict train/test separation (Deliverable 4).

## 2. Dataset and Preprocessing
- **Source Corpus**: Kaggle *What's Cooking* (`data/raw/train.json`, SHA-256 `f18e7e568d1c6067...`), containing **39,774** total recipes and **428,275** ingredient occurrences across 20 cuisines (**6,714** raw distinct strings; **6,703** after basic lowercase/whitespace cleanup).
- **Target Cuisines**: **17,279** recipes retained — **Indian** (3,003), **Italian** (7,838), and **Mexican** (6,438) — totaling **185,852** raw ingredient occurrences.
- **Conservative Deterministic Normalization**: 5 auditable stages (`unicode` -> `lowercase` -> `whitespace` -> `singularize` via a 71-noun count-plural whitelist -> `synonym` via a 5-entry documented dictionary) reduce **4,416** raw strings in the 3 cuisines to **4,402** canonical names (`outputs/d1_dataset/merge_log.csv` logs **720** transformations across **14** multi-string merge groups; **4** recipes had duplicate ingredients deduplicated).
- **Leakage-Free Split**: Descriptive analyses (Deliverables 1–3) characterize all **17,279** selected recipes; predictive modeling and evaluation (Deliverable 4) use a stratified 80/20 split (seed `42`: **13,823** train / **3,456** test recipes) performed on recipe IDs *prior* to all frequency filtering, null-model randomization, backbone extraction, and scorer fitting.

## 3. NPMI-Based Network Construction (D1)
- **Node Eligibility**: Ingredients appearing in $\ge 20$ recipes within a cuisine are retained as nodes (`outputs/d1_dataset/nodes.csv`):
  - **Indian**: 268 eligible nodes (1,659 distinct ingredients), 22,284 raw co-occurring pairs in `edges_indian.csv` (8,109 with `NPMI < 0`).
  - **Italian**: 487 eligible nodes (2,924 distinct ingredients), 48,488 raw co-occurring pairs in `edges_italian.csv` (14,514 with `NPMI < 0`).
  - **Mexican**: 423 eligible nodes (2,676 distinct ingredients), 42,703 raw co-occurring pairs in `edges_mexican.csv` (13,162 with `NPMI < 0`).
- **Association Weighting**: Binary recipe presence defines $P(a) = n(a)/N$, $P(a,b) = n(a,b)/N$, $\text{PMI}(a,b) = \log_2(P(a,b)/[P(a)P(b)])$, and $\text{NPMI}(a,b) = \text{PMI}(a,b)/(-\log_2 P(a,b)) \in [-1, 1]$. Negative PMI/NPMI values are preserved in all `edges_{cuisine}.csv` tables.

## 4. Null-Model and Backbone Methodology (D2)
- **Degree-Preserving Bipartite Randomization**: One Curveball Markov chain (Strona et al., 2014) per cuisine on the full binary recipe-by-ingredient incidence matrix ($10 \times N$ burn-in trades, **1,000** samples, $2 \times N$ thinning trades/sample), preserving every recipe's exact basket size and every ingredient's exact frequency.
- **Mixing Adequacy**: Effective sample size $\text{ESS}(S)$ of $S = \sum_{a<b} n(a,b)^2$ is **655** (Indian), **667** (Italian), and **691** (Mexican).
- **Backbone Filtering**: Retain edges with strict $z = (n(a,b) - \mu_{\text{null}})/\sigma_{\text{null}} > 3$ and $\text{co\_count} \ge 5$ (excluding zero-variance pairs where $\sigma_{\text{null}} = 0$):
  - **Indian**: 22,284 $\to$ **1,829** edges (**8.21%** retained, 261 backbone nodes).
  - **Italian**: 48,488 $\to$ **3,758** edges (**7.75%** retained, 473 backbone nodes).
  - **Mexican**: 42,703 $\to$ **2,847** edges (**6.67%** retained, 408 backbone nodes).
  - **Union**: **789** backbone nodes and **7,759** unique backbone pairs across the 3 descriptive backbones.

## 5. Hub, Community, and Bridge Analysis (D3)
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

## 6. Deliverable 4: Fingerprint Scorer Evolution (v1 $\to$ v2) & Live App (D4.3)
- **Training-Only Network Artifacts (`N_train = 13,823`, 1,000 Curveball randomizations)**:
  - Training-only backbones: Indian (`237` nodes / `1,466` edges), Italian (`413` nodes / `2,973` edges), Mexican (`350` nodes / `2,223` edges) — union of **692** backbone ingredients and **6,149** validated pairs.
- **Scorer v1 (Original Baseline, `outputs/d4_scorer/`)**:
  - Sums positive within-cuisine backbone NPMI over matched pairs ($\text{Score}(c) = \sum_{(a,b) \in M_c(R)} \text{NPMI}_c(a,b)$), falling back to the majority training class (`italian`) when zero pairs match.
  - Held-out test ($N = 3,456$): **Macro-F1 = 87.12%** (`0.8712`), **Accuracy = 87.67%** (`0.8767`), with **139** zero-score fallbacks. (Dividing by $|M_c(R)|$ in the normalized ablation drops to **43.14% Macro-F1 / 43.69% Accuracy**.)
- **Scorer v2 (Network-Informed 22-Feature Model, `outputs/d4_experiments/`)**:
  - Combines **22 purely network-derived features** (3 contrastive backbone pair log-odds sums, 9 Louvain community coverage/max-share/entropy features, 3 top-10 hub counts, 3 top-5 bridge counts, 3 backbone node counts, and recipe length) via L2-regularized logistic regression ($C = 1.0$, trained with 3-fold inner cross-fitting on the 13,823 training recipes).
  - Held-out test ($N = 3,456$, from `outputs/d4_experiments/final_test_results.csv` & `final_test_summary.json`): **Macro-F1 = 93.99%** (`0.9399`, 95% CI `[93.10%, 94.77%]`), **Accuracy = 94.13%** (`0.9413`, 95% CI `[93.29%, 94.88%]`), **0** fallbacks.
  - **Paired Gain (v2 minus v1)**: **+6.87 pp Macro-F1** (95% bootstrap CI `[+5.66 pp, +8.04 pp]`), **+6.45 pp Accuracy** (95% CI `[+5.35 pp, +7.52 pp]`), McNemar exact $p = 7.66 \times 10^{-35}$ (**287** fixed vs. **64** broken, **+223 net recipes**), and recall gains across every cuisine (Indian **+11.33 pp** to 91.33%, Italian **+4.85 pp** to 96.56%, Mexican **+6.13 pp** to 92.47%).
- **Held-Out Test Comparison (`N = 3,456`, Primary Metric: Macro-F1)**:

| Model | Test Macro-F1 | Test Accuracy | Indian Recall / F1 | Italian Recall / F1 | Mexican Recall / F1 | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Scorer v1 (Original NPMI Sum)** | **87.12%** | 87.67% | 80.00% / 85.49% | 91.71% / 89.37% | 86.34% / 86.50% | 139 |
| **Scorer v2 (Network-Informed, 22 features)** | **93.99%** | 94.13% | 91.33% / 93.68% | 96.56% / 94.80% | 92.47% / 93.49% | 0 |

- **D4.3 Live Interactive Demo App (`outputs/app/app.py`)**:
  - Self-contained Streamlit web application powered by Scorer v2 (`scorer_runtime.py` + `outputs/app/artifacts/`, parity-verified against the research pipeline on 250 training recipes to $< 10^{-6}$ probability difference), featuring search-as-you-type vocabulary selection, 5-stage text normalization, a "Why This Prediction" structural breakdown (matched pairs, hubs, Louvain communities, cross-cuisine bridges), an offline interactive PyVis backbone subgraph viewer, and a held-out v1 vs. v2 benchmark tab.

## 7. Three Recommended Poster Figures (Generated Files)
1. `outputs/d3_report/figures/fig1_backbone_edges.png` — Raw vs. statistically validated backbone edge counts and retention percentages across Indian, Italian, and Mexican cuisines.
2. `outputs/d3_report/figures/fig2_top_hubs.png` — Side-by-side comparison of top-10 weighted-degree hubs per cuisine.
3. `outputs/d4_scorer/confusion_matrix_normalized.png` (paired with `outputs/d4_experiments/accuracy_vs_matched_pairs.png`) — Confusion matrix and pair-coverage diagnostics showing Scorer v1's zero-pair bottleneck and Scorer v2's improvement.

## 8. Key Numerical Results Summary
| Metric | Indian | Italian | Mexican |
| --- | ---: | ---: | ---: |
| Recipes (Full / Train / Test) | 3,003 / 2,403 / 600 | 7,838 / 6,270 / 1,568 | 6,438 / 5,150 / 1,288 |
| Eligible Nodes / Backbone Nodes (Full D2) | 268 / 261 | 487 / 473 | 423 / 408 |
| Raw Edges $\to$ Backbone Edges (% Retained, Full D2) | 22,284 $\to$ 1,829 (8.21%) | 48,488 $\to$ 3,758 (7.75%) | 42,703 $\to$ 2,847 (6.67%) |
| Louvain Communities (Modularity $Q$, Full D2) | 7 (0.488) | 6 (0.473) | 7 (0.497) |
| Scorer v1 Test F1 (Historical Baseline) | 0.8549 | 0.8937 | 0.8650 |
| Scorer v2 Test F1 (Network-Informed, 22 features) | **0.9368** | **0.9480** | **0.9349** |

## 9. Conclusion
- Degree-preserving bipartite null models prune **91.8%–93.3%** of raw co-occurrence edges while uncovering modular sub-cuisine communities ($Q = 0.47\text{-}0.50$) and specialized spice/aromatics hubs.
- Summing validated backbone NPMI weights alone (**Scorer v1**) achieves **87.12% Macro-F1** (**87.67%** accuracy) with 139 zero-score fallbacks, whereas combining contrastive backbone pair log-odds with D3 community, hub, bridge, and node-coverage summaries (**Scorer v2**) improves held-out performance by **+6.87 pp Macro-F1** to **93.99% Macro-F1** (**94.13%** accuracy) with **0 fallbacks**.

## 10. Methodological Limitations
- **Three-Way Classification Scope**: Only Indian, Italian, and Mexican are modeled; any input is forced into one of those three cuisines.
- **Multiple Testing**: The threshold $z > 3$ with $\text{co\_count} \ge 5$ is a standard heuristic filter rather than a family-wise or FDR-corrected significance level across `22,284–48,488` observed pairs (`18k–40k` in training).
- **Eligibility & Coverage**: Restricting backbones to ingredients appearing in $\ge 20$ training recipes leaves rare (<20 recipe) ingredients outside Scorer v2's scope; part of v2's gain over v1 comes from resolving the 139 zero-pair recipes using single-node D3 community, hub, and node-coverage signals plus recipe length.
- **CV Randomization Budget**: Inner/outer Phase 1–3 cross-validation builds used 100 Curveball randomizations (and 50 in inner cross-fitting) as documented compute approximations to the 1,000-randomization full-training backbone.
