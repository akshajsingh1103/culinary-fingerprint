# The Culinary Fingerprint: Statistically Validated Ingredient Networks for Identifying World Cuisines

## 1. Project Overview

**The Culinary Fingerprint** constructs statistically validated ingredient co-occurrence networks from the Kaggle *What's Cooking* recipe dataset for three world cuisines: **Indian**, **Italian**, and **Mexican**. Raw co-occurrence graphs are dominated by high-frequency culinary staples (`salt`, `water`, `onion`, `garlic`), obscuring sub-cuisine structure. We weight edges using Pointwise Mutual Information (PMI) and Normalized Pointwise Mutual Information (NPMI), validate every co-occurring ingredient pair against a strict degree-preserving bipartite **Curveball Markov chain null model** (1,000 samples per cuisine after burn-in and thinning), extract statistically significant backbones ($z > 3$, $\text{co\_count} \ge 5$), render offline-capable interactive PyVis HTML graphs, analyze structural hubs, Louvain communities, and cross-cuisine bridges, and evaluate an interpretable **Fingerprint Scorer** (plus a normalized-score ablation) against a **Multinomial Naive Bayes** baseline under strict 80/20 stratified train/test separation.

> **Separation of Descriptive vs. Predictive Products**:
> - **Descriptive networks (Deliverables 1–3)** use **all 17,279 selected recipes** across Indian, Italian, and Mexican cuisines to characterize the full empirical structure of each cuisine.
> - **Predictive model artifacts (Deliverable 4)** rebuild every ingredient frequency count, eligible node set, PMI/NPMI weight, 1,000-sample Curveball null distribution, and validated backbone exclusively from the **80% stratified training split (13,823 recipes)**. Held-out test recipes (**3,456 recipes**) never touch network construction or vocabulary fitting.

---

## 2. Research Question

1. **Structural**: When recipe basket sizes and ingredient frequencies are held strictly fixed via a degree-preserving bipartite null model, which ingredient pairings survive in the statistically validated backbones of Indian, Italian, and Mexican cuisines, and how do those backbones organize into hubs, modular communities, and cross-cuisine structural bridges?
2. **Predictive**: How well can a purely graph-based **Fingerprint Scorer**—which sums training-backbone NPMI weights over matched ingredient pairs—classify unseen recipes compared with a normalized-score ablation and a full-vocabulary binary **Multinomial Naive Bayes** baseline?

---

## 3. Dataset Provenance and Exact Placement Instructions

- **Dataset**: Official Kaggle *"What's Cooking?"* competition training dataset (sourced from Yummly).
- **Required Location**:
  ```text
  culinary-fingerprint/data/raw/train.json
  ```
- **Verified File Provenance**:
  - **File size**: `12,415,067` bytes
  - **SHA-256**: `f18e7e568d1c606774bfb134ece51ef0a143cd5069e6f9241bb6036b45b7d2b0`
  - **Total records**: `39,774` recipes across 20 cuisines (`0` malformed records, `0` empty ingredient lists, `0` duplicate recipe IDs)
  - **Target cuisine counts**:
    - `indian`: **3,003** recipes
    - `italian`: **7,838** recipes
    - `mexican`: **6,438** recipes
    - **Selected total**: **17,279** recipes (**22,495** recipes belong to the remaining 17 cuisines)

---

## 4. Installation Instructions

Requires **Python 3.10+** (verified on Python `3.13.15` on Windows). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Pinned dependencies in `requirements.txt`: `numpy==2.2.6`, `scipy==1.15.3`, `pandas==2.2.3`, `networkx==3.4.2`, `pyvis==0.3.2`, `scikit-learn==1.6.1`, `matplotlib==3.10.3`, `seaborn==0.13.2`, `reportlab==4.4.1`, `pypdf==5.6.0`, `pytest==8.3.5`, `PyYAML==6.0.2`, `numba==0.61.2`.

---

## 5. Folder Map

```text
culinary-fingerprint/
├── data/
│   ├── raw/
│   │   └── train.json                         # Official Kaggle dataset (read-only)
│   └── processed/
│       ├── recipes_normalized.json            # Normalized 17,279 target-cuisine recipes
│       └── split.json                         # Stratified 80/20 train/test recipe IDs (seed 42)
├── outputs/
│   ├── d1_dataset/
│   │   ├── nodes.csv                          # 1,178 (ingredient, cuisine) rows with recipe_count >= 20
│   │   ├── edges_indian.csv                   # 22,284 raw co-occurrence edges for Indian
│   │   ├── edges_italian.csv                  # 48,488 raw co-occurrence edges for Italian
│   │   ├── edges_mexican.csv                  # 42,703 raw co-occurrence edges for Mexican
│   │   ├── merge_log.csv                      # 720 auditable normalization transformations
│   │   ├── ingredient_prevalence_all.csv      # Prevalence of all 4,402 normalized ingredients per cuisine
│   │   ├── normalization_report.json          # Full-dataset & 3-cuisine normalization audit
│   │   └── d1_summary.json                    # Per-cuisine node/edge summary
│   ├── d2_backbone/
│   │   ├── backbone_indian.csv                # 1,829 validated Indian backbone edges (z > 3, co >= 5)
│   │   ├── backbone_italian.csv               # 3,758 validated Italian backbone edges (z > 3, co >= 5)
│   │   ├── backbone_mexican.csv               # 2,847 validated Mexican backbone edges (z > 3, co >= 5)
│   │   ├── backbone_summary.csv               # Node, edge, retention %, community & modularity table
│   │   ├── graph_indian.html                  # Interactive PyVis HTML graph for Indian backbone
│   │   ├── graph_italian.html                 # Interactive PyVis HTML graph for Italian backbone
│   │   ├── graph_mexican.html                 # Interactive PyVis HTML graph for Mexican backbone
│   │   ├── null_model_checks.json             # Degree-preservation & seed reproducibility verification
│   │   ├── null_benchmark.json                # Pre-run Curveball timing benchmark & extrapolation
│   │   └── null_model_metadata.json           # Chain trades, ESS, autocorrelation & Geweke diagnostics
│   ├── d3_report/
│   │   ├── report.pdf                         # 4-page formatted academic analysis report
│   │   ├── report.md                          # Consistent Markdown analysis report
│   │   ├── hubs_indian.csv                    # Top-10 weighted degree & top-10 betweenness hubs (Indian)
│   │   ├── hubs_italian.csv                   # Top-10 weighted degree & top-10 betweenness hubs (Italian)
│   │   ├── hubs_mexican.csv                   # Top-10 weighted degree & top-10 betweenness hubs (Mexican)
│   │   ├── universal_vs_specific_hubs.csv     # Classification across all 26 distinct top-10 strength hubs
│   │   ├── communities_indian.csv             # Louvain communities & top-5 members (Indian)
│   │   ├── communities_italian.csv            # Louvain communities & top-5 members (Italian)
│   │   ├── communities_mexican.csv            # Louvain communities & top-5 members (Mexican)
│   │   ├── bridges.csv                        # Top-5 structural bridges for each of the 3 cuisine pairs
│   │   └── figures/                           # Programmatically generated PNG figures (fig1..fig5)
│   ├── d4_scorer/
│   │   ├── results.csv                        # Held-out test metrics for all 3 models
│   │   ├── confusion_matrix.png               # Fingerprint Scorer confusion matrix (counts)
│   │   ├── confusion_matrix_nb.png            # Naive Bayes confusion matrix (counts)
│   │   ├── confusion_matrix_normalized.png    # Fingerprint Scorer row-normalized confusion matrix
│   │   ├── confusion_matrix_normalized_nb.png # Naive Bayes row-normalized confusion matrix
│   │   ├── garam_masala_diagnostic.json       # Vocabulary & pair-matching diagnostic for spice test
│   │   ├── training_artifacts/                # Training-only backbones & metadata (13,823 train recipes)
│   │   └── evaluation_metadata.json           # Provenance, hashes, confusion matrices & phase runtimes
│   ├── poster_outline.md                      # Academic poster outline populated from real outputs
│   └── validation_summary.json                # Machine-readable acceptance check summary
├── src/
│   ├── common.py                              # Config, logging, SHA-256 hashing & dataset validation
│   ├── normalize.py                           # Shared deterministic, idempotent normalizer
│   ├── preprocess.py                          # Dataset audit, normalization & merge_log.csv builder
│   ├── split_data.py                          # Stratified 80/20 recipe-ID split (seed 42)
│   ├── build_networks.py                      # Node eligibility, sparse B^T B co-occurrence & PMI/NPMI
│   ├── null_model.py                          # Numba Curveball Markov chain, mixing diagnostics & benchmark
│   ├── graph_utils.py                         # Strength, weighted betweenness & deterministic Louvain
│   ├── backbone.py                            # Null statistics, zero-variance exclusion & z > 3 filtering
│   ├── visualize.py                           # Self-contained PyVis HTML generator
│   ├── analysis.py                            # Hubs, hub classification, communities, bridges & figures
│   ├── report.py                              # Programmatic Markdown & 3-4 page PDF report generator
│   ├── fingerprint.py                         # FingerprintScorer, predict() & predict_scores() API
│   ├── evaluate.py                            # End-to-end training-only D4 evaluation & Naive Bayes
│   └── poster.py                              # Programmatic poster_outline.md generator
├── tests/
│   ├── conftest.py                            # Shared pytest fixtures
│   ├── test_d1.py                             # D1 & normalization tests (8 tests)
│   ├── test_d2.py                             # D2 & Curveball null-model tests (6 tests)
│   ├── test_d3.py                             # D3 hubs, communities, bridges & PDF report tests (4 tests)
│   ├── test_fingerprint.py                    # FingerprintScorer & garam masala tests (4 tests)
│   ├── test_evaluation.py                     # Evaluation & confusion matrix tests (3 tests)
│   └── test_no_leakage.py                     # Strict train/test leakage boundary tests (3 tests)
├── config.yaml                                # Centralized configuration
├── requirements.txt                           # Pinned dependencies
├── Makefile                                   # Convenience targets
├── run_all.py                                 # Single entry point for full pipeline & validation
├── evaluate.py                                # Root entry point reproducing D4 evaluation from scratch
└── README.md                                  # Project documentation & verified results
```

---

## 6. Ingredient Normalization Rules

Implemented in [`src/normalize.py`](src/normalize.py) and shared identically across network construction, the Fingerprint Scorer, and the Naive Bayes baseline.

- **Full-Dataset Audit (All 39,774 Recipes)**: Rerunning the preliminary audit on `data/raw/train.json` confirms **39,774** recipes, **428,275** ingredient occurrences, **6,714** distinct raw strings, **6,703** distinct strings after basic lowercase/whitespace cleanup, **499** uppercase/mixed-case names, **7** names with repeated internal whitespace, **11** formatting-equivalent collision groups, and **5** recipes with duplicate ingredients after basic cleanup.
- **Five Conservative, Idempotent Stages (Applied to the 17,279 Target-Cuisine Recipes)**:
  1. `unicode`: Curly-to-ASCII apostrophes/quotes and removal of trademark/copyright glyphs (`®`, `™`, `©`) **prior** to Unicode `NFKC` normalization (so `™` `\u2122` is replaced by a space rather than decomposing into ASCII `"TM"`), followed by `NFKD` accent folding (`é` $\to$ `e`, `í` $\to$ `i`, `ú` $\to$ `u`). Reduces `4,416` raw distinct names $\to$ `4,415` (`62` names changed across `336` recipes).
  2. `lowercase`: Case folding. Reduces `4,415` $\to$ `4,409` (`295` names changed across `1,726` recipes).
  3. `whitespace`: Strip leading/trailing whitespace and collapse repeated internal whitespace. Reduces `4,409` $\to$ `4,408` (`51` names changed across `131` recipes; collapsing whitespace left by `™` removal merges `Old El Paso™ taco seasoning mix` with `Old El Paso Taco Seasoning Mix`).
  4. `singularize`: Singularizes the **final token only** when it appears in an explicit **71-word** count-noun whitelist (`PLURAL_WHITELIST`, e.g., `tomatoes` $\to$ `tomato`, `onions` $\to$ `onion`, `tortillas` $\to$ `tortilla`, `scallions` $\to$ `scallion`, `sea scallops` $\to$ `sea scallop`), subject to a narrowly scoped phrase exception (`SINGULARIZE_EXCEPTIONS = {"veal scallops"}`) so that the Italian veal cut (*scaloppine*) is not singularized by the bivalve mollusk rule `"scallops": "scallop"`. Ambiguous or mass/part nouns (`cloves`, `leaves`, `greens`, `beans`, `peas`, `chilies`/`chiles`, `noodles`, `asparagus`, `molasses`) are untouched. Reduces `4,408` $\to$ `4,407` (`421` names changed across `13,275` recipes).
  5. `synonym`: Applies a 5-entry whole-name synonym dictionary (`SYNONYMS`), each with an explicit culinary/botanical rationale:
     - `scallion` $\to$ `green onion` (after `scallions` $\to$ `scallion` and `green onions` $\to$ `green onion`)
     - `garbanzo beans` $\to$ `chickpea` (after `chickpeas` $\to$ `chickpea`)
     - `asafetida` $\to$ `asafoetida`
     - `cuminseed` $\to$ `cumin seed`
     - `chat masala` $\to$ `chaat masala`
     Reduces `4,407` $\to$ `4,402` final distinct names (`5` names changed across `382` recipes).
  6. **Recipe-level deduplication**: Deduplicates normalized ingredients within each recipe (`4` target-cuisine recipes had `17` duplicate occurrences removed).
- **Auditability**: Every raw name whose normalized form differs is logged in `outputs/d1_dataset/merge_log.csv` (**720** rows, **14** multi-name collision groups covering **28** raw strings). Meaningful distinctions (fresh vs. dried, ground vs. whole seed, cheese types, oils, peppers, digits/percentages like `1% low-fat milk`) are strictly preserved.

---

## 7. PMI and NPMI Equations

For a given cuisine with $N$ recipes and eligible node set $V_c = \{a : n(a) \ge 20\}$, let $n(a)$ be the number of recipes containing ingredient $a$, and $n(a,b)$ be the number of recipes containing both $a$ and $b$ ($a < b$ alphabetically, $n(a,b) \ge 1$). Using binary recipe-level presence:

$$P(a) = \frac{n(a)}{N}, \qquad P(b) = \frac{n(b)}{N}, \qquad P(a,b) = \frac{n(a,b)}{N}$$

$$\text{PMI}(a,b) = \log_2 \left( \frac{P(a,b)}{P(a)\,P(b)} \right)$$

$$\text{NPMI}(a,b) = \frac{\text{PMI}(a,b)}{-\log_2 P(a,b)} \in [-1, 1]$$

- **Degenerate Convention**: When $P(a,b) = 1$ (so $P(a)=P(b)=1$ and $\text{PMI}=0 / 0$), we define $\text{NPMI}(a,b) = 1.0$, its limiting value under perfect co-occurrence (Bouma, 2009).
- **Negative Values**: Raw edges with negative PMI/NPMI are preserved in `edges_<cuisine>.csv` (`10,656` in Indian, `27,896` in Italian, `24,132` in Mexican).

---

## 8. Null-Model Algorithm and Limitations

- **Algorithm**: In [`src/null_model.py`](src/null_model.py), we construct the **full** binary recipe-by-ingredient incidence matrix $B \in \{0,1\}^{N \times M}$ for each cuisine and run **one Curveball Markov chain** (Strona et al., 2014; Carstens, 2015) compiled with `numba`:
  - Each trade selects two distinct recipes $i \ne j$ uniformly at random, identifies their disjoint ingredient sets $A = R_i \setminus R_j$ and $C = R_j \setminus R_i$, randomly permutes $A \cup C$ via Fisher–Yates shuffle whenever $|A| > 0$ and $|C| > 0$, and reassigns $|A|$ ingredients to $i$ and $|C|$ to $j$.
  - Every trade exactly preserves: (1) each recipe's ingredient count (row degrees), (2) each ingredient's recipe frequency (column degrees), and (3) binary incidence.
- **Chain Schedule**:
  - **Burn-in**: $10 \times N$ trades ($30,030$ for Indian; $78,380$ for Italian; $64,380$ for Mexican), exceeding the required $\ge 5 \times N$.
  - **Samples**: $1,000$ randomized matrices per cuisine, with **thinning** of $2 \times N$ trades between successive samples ($6,036,030$ total trades for Indian; $15,754,380$ for Italian; $12,940,380$ for Mexican).
  - **Streaming Accumulation**: Pairwise co-occurrences among eligible ingredients ($B_e^\top B_e$) are accumulated into exact 64-bit integer sums $\sum n_{\text{rand}}(a,b)$ and sums of squares $\sum n_{\text{rand}}(a,b)^2$ without storing all 1,000 matrices.
- **Measured Mixing Diagnostics (Descriptive Chains, `outputs/d2_backbone/null_model_metadata.json`)**:
  - Summary statistic $S = \sum_{a<b} n(a,b)^2$:
    - **Indian**: lag-1 autocorrelation $\rho_1(S) = 0.182$, effective sample size $\text{ESS}(S) = 655 / 1000$, Geweke $z(S) = -2.52$ (tracked pair Geweke $z = -1.83$, $\text{ESS} = 720$).
    - **Italian**: lag-1 autocorrelation $\rho_1(S) = 0.143$, effective sample size $\text{ESS}(S) = 667 / 1000$, Geweke $z(S) = -0.25$ (tracked pair Geweke $z = 1.25$, $\text{ESS} = 803$).
    - **Mexican**: lag-1 autocorrelation $\rho_1(S) = 0.188$, effective sample size $\text{ESS}(S) = 691 / 1000$, Geweke $z(S) = -1.03$ (tracked pair Geweke $z = -0.98$, $\text{ESS} = 741$).
- **Limitations**: Fixed-degree bipartite randomization conditions on marginal totals, so ingredients that appear in almost every recipe have limited swap flexibility; moreover, recipes are treated as unstructured sets without culinary role constraints (e.g., dessert vs. savory dish strata).

---

## 9. Statistical Interpretation of $z > 3$

For each observed eligible pair $(a,b)$, let $\mu_{\text{null}}(a,b)$ and $\sigma_{\text{null}}(a,b)$ (sample standard deviation with $ddof = 1$) be computed across the 1,000 Curveball samples:

$$z(a,b) = \frac{n(a,b) - \mu_{\text{null}}(a,b)}{\sigma_{\text{null}}(a,b)}$$

- **Zero-Variance Policy**: Pairs with $\sigma_{\text{null}}(a,b) = 0$ have undefined $z$-scores and are **conservatively excluded** from the backbone (`0` observed eligible pairs had zero variance in Indian, Italian, or Mexican).
- **Multiple-Testing Caveat**: The threshold $z > 3$ is an **uncorrected single-pair threshold** applied across $22,284$–$48,488$ observed pairs per cuisine. Because null co-occurrence counts are discrete hypergeometric-like variables and tens of thousands of pairs are tested simultaneously, $z > 3$ does **not** guarantee a formal family-wise error rate (FWER) or false discovery rate (FDR) guarantee.

---

## 10. Backbone Filtering Rules and Summary Results

An observed edge between eligible ingredients ($\ge 20$ recipes) is retained in the validated backbone if and only if **both** conditions hold:
1. `z_score > 3.0` (strict inequality; $z = 3.0$ is excluded)
2. `co_count >= 5`

### Descriptive Backbone Summary (`outputs/d2_backbone/backbone_summary.csv`)

| Cuisine | Recipes | Eligible Nodes ($\ge 20$) | Backbone Nodes | Raw Edges (`edges_before`) | Backbone Edges (`edges_after`) | Retained (`pct_retained`) | Louvain Communities | Modularity $Q$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Indian** | 3,003 | 268 | 261 | 22,284 | **1,829** | **8.21%** | 7 | 0.4882 |
| **Italian** | 7,838 | 487 | 473 | 48,488 | **3,758** | **7.75%** | 6 | 0.4727 |
| **Mexican** | 6,438 | 423 | 408 | 42,703 | **2,847** | **6.67%** | 7 | 0.4974 |

### Predictive Training-Only Backbone Summary (`outputs/d4_scorer/training_artifacts/backbone_summary_train.csv`)

| Cuisine | Train Recipes (80%) | Eligible Nodes ($\ge 20$) | Backbone Nodes | Raw Edges (`edges_before`) | Backbone Edges (`edges_after`) | Retained (`pct_retained`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Indian** | 2,403 | 237 | 237 | 18,296 | **1,466** | **8.01%** |
| **Italian** | 6,270 | 432 | 413 | 39,815 | **2,973** | **7.47%** |
| **Mexican** | 5,150 | 371 | 350 | 35,062 | **2,223** | **6.34%** |

---

## 11. Hub and Bridge Definitions and Empirical Findings

- **Weighted Degree (Strength)**: $\text{Strength}_c(a) = \sum_{b \in N_c(a)} \max(\text{NPMI}_c(a,b), 0)$ on cuisine $c$'s validated backbone.
- **Weighted Betweenness Centrality**: Normalized shortest-path betweenness on the undirected backbone using edge distance $d(a,b) = 1 / \max(\text{NPMI}_c(a,b), 0)$ (zero-strength edges excluded).
- **Bridge Score**: For an ingredient $a$ present as a node in both validated backbones $A$ and $B$:
  $$\text{Bridge}(a; A, B) = \min\big(\text{Strength}_A(a),\, \text{Strength}_B(a)\big)$$
  *(Measures shared structural centrality in the two backbones, not historical cultural exchange or causal influence.)*

### Top Weighted-Degree and Weighted-Betweenness Hubs (`outputs/d3_report/hubs_*.csv`)

- **Indian**:
  - Top-5 by Weighted Degree: `clove` ($12.77$), `garlic paste` ($12.28$), `green cardamom` ($12.18$), `mace` ($10.75$), `green chilies` ($10.44$).
  - Top-5 by Weighted Betweenness: `peeled fresh ginger` ($0.0540$), `heavy cream` ($0.0397$), `active dry yeast` ($0.0341$), `curry leaves` ($0.0308$), `ground cardamom` ($0.0304$).
- **Italian**:
  - Top-5 by Weighted Degree: `vanilla extract` ($18.24$), `carrot` ($18.11$), `sugar` ($15.50$), `tomato paste` ($14.90$), `all-purpose flour` ($14.82$).
  - Top-5 by Weighted Betweenness: `carrot` ($0.0527$), `cooking spray` ($0.0466$), `arborio rice` ($0.0426$), `ground pork` ($0.0383$), `fresh parmesan cheese` ($0.0312$).
- **Mexican**:
  - Top-5 by Weighted Degree: `ancho chile pepper` ($13.80$), `ground cinnamon` ($13.04$), `dried oregano` ($11.51$), `all-purpose flour` ($11.46$), `garlic cloves` ($11.43$).
  - Top-5 by Weighted Betweenness: `cooking spray` ($0.0460$), `baking powder` ($0.0425$), `ancho chile pepper` ($0.0387$), `radish` ($0.0371$), `ground cinnamon` ($0.0370$).

### Universal vs. Cuisine-Specific Hubs (`outputs/d3_report/universal_vs_specific_hubs.csv`)

Across the **26** distinct ingredients appearing in any cuisine's top-10 weighted-degree list:
- **Universal hubs (top-10 in all 3 cuisines)**: **0**
- **Two-cuisine hubs (top-10 in 2 cuisines)**: **4** — `clove` (Indian & Mexican), `all-purpose flour` (Italian & Mexican), `baking powder` (Italian & Mexican), and `sugar` (Italian & Mexican)
- **Cuisine-specific hubs (top-10 in 1 cuisine)**: **22** (9 Indian, 7 Italian, 6 Mexican)

### Top-5 Structural Bridge Ingredients (`outputs/d3_report/bridges.csv`)

- **Indian–Italian** (124 shared backbone nodes):
  1. `bay leaf` (bridge score **8.21**; strengths $9.54$ IN vs $8.21$ IT)
  2. `heavy cream` (bridge score **6.79**; strengths $6.79$ IN vs $7.04$ IT)
  3. `garlic cloves` (bridge score **6.52**; strengths $6.52$ IN vs $10.54$ IT)
  4. `garlic` (bridge score **6.37**; strengths $6.37$ IN vs $10.26$ IT)
  5. `onion` (bridge score **5.87**; strengths $5.87$ IN vs $14.80$ IT)
- **Indian–Mexican** (134 shared backbone nodes):
  1. `clove` (bridge score **9.60**; strengths $12.77$ IN vs $9.60$ MX)
  2. `cinnamon sticks` (bridge score **8.26**; strengths $9.97$ IN vs $8.26$ MX)
  3. `black peppercorn` (bridge score **7.79**; strengths $7.79$ IN vs $9.07$ MX)
  4. `cinnamon` (bridge score **7.24**; strengths $7.70$ IN vs $7.24$ MX)
  5. `garlic cloves` (bridge score **6.52**; strengths $6.52$ IN vs $11.43$ MX)
- **Italian–Mexican** (195 shared backbone nodes):
  1. `dried oregano` (bridge score **11.51**; strengths $11.83$ IT vs $11.51$ MX)
  2. `all-purpose flour` (bridge score **11.46**; strengths $14.82$ IT vs $11.46$ MX)
  3. `baking powder` (bridge score **11.22**; strengths $14.53$ IT vs $11.22$ MX)
  4. `garlic cloves` (bridge score **10.54**; strengths $10.54$ IT vs $11.43$ MX)
  5. `sugar` (bridge score **9.95**; strengths $15.50$ IT vs $9.95$ MX)

---

## 12. Fingerprint Scorer Formula and Spice-Test Investigation

Given an input ingredient list, [`src/fingerprint.py`](src/fingerprint.py) applies `normalize_recipe()` (deterministic normalization + deduplication + alphabetical sorting) and scores each cuisine $c \in \{\text{indian}, \text{italian}, \text{mexican}\}$ using its **training-split** backbone $E_c^{\text{train}}$. Let $M_c(R) = \{(a,b) : a < b,\; a,b \in R,\; (a,b) \in E_c^{\text{train}}\}$ denote the set of matched training-backbone edges in recipe $R$:

$$\text{Score}(c) = \sum_{(a,b) \in M_c(R)} \text{NPMI}_c(a,b)$$

- **Normalized Ablation Formula**:
  $$\text{Score}_{\text{norm}}(c) = \begin{cases} \dfrac{1}{|M_c(R)|} \sum\limits_{(a,b) \in M_c(R)} \text{NPMI}_c(a,b) & \text{if } |M_c(R)| > 0 \\[6pt] 0 & \text{if } |M_c(R)| = 0 \end{cases}$$
- **Tie & Zero-Score Fallback**: Ties in maximum score are broken deterministically by training-split class frequency (`italian`: 6,270 $>$ `mexican`: 5,150 $>$ `indian`: 2,403).
- **Garam Masala / Cumin / Turmeric Acceptance Investigation (`outputs/d4_scorer/garam_masala_diagnostic.json`)**:
  - **Training Backbone Vocabulary Check**:
    - `"garam masala"`: in Indian backbone (`n=706` train recipes); absent from Italian (`n=1`) and Mexican (`n=0`).
    - `"cumin"`: in Indian (`n=146`) and Mexican (`n=484`) backbones; absent from Italian (`n=3`).
    - `"ground cumin"`: in Indian (`n=549`), Italian (`n=28`), and Mexican (`n=1,075`) backbones.
    - `"cumin seed"`: in Indian (`n=565`) and Mexican (`n=71`) backbones; absent from Italian (`n=4`).
    - `"turmeric"`: `0` occurrences in the entire Kaggle dataset (the dataset uses `"ground turmeric"`, which occurs `n=590` times in Indian training recipes and is in the Indian backbone).
  - **Literal Acceptance Call `predict(["garam masala", "cumin", "turmeric"])`**:
    - Matched pair in Indian training backbone: `("cumin", "garam masala")` with $\text{NPMI} = 0.191835$.
    - Matched pairs in Italian / Mexican: none ($0.0$).
    - Resulting scores: `{'indian': 0.1918347594320284, 'italian': 0.0, 'mexican': 0.0}` $\to$ returns **`"indian"`** directly from the general scoring formula without fallback or special-casing.
  - **Dataset-String Variants**:
    - `predict(["garam masala", "ground cumin", "ground turmeric"])` matches all 3 pairs in Indian (`{'indian': 0.444653, 'italian': 0.0, 'mexican': 0.0}`) $\to$ **`"indian"`**.
    - `predict(["garam masala", "cumin seed", "ground turmeric"])` matches 2 pairs in Indian (`{'indian': 0.334114, 'italian': 0.0, 'mexican': 0.0}`) $\to$ **`"indian"`**.

---

## 13. Naive Bayes Methodology

Implemented in `BinaryMultinomialNB` ([`src/evaluate.py`](src/evaluate.py)) using `scikit-learn`'s `MultinomialNB(alpha=1.0, fit_prior=True)`:
- Fits a sorted vocabulary of all **4,105** normalized ingredients appearing in the **13,823** training recipes (`outputs/d4_scorer/evaluation_metadata.json`).
- Represents each recipe as a sparse binary ingredient-presence row vector $x \in \{0,1\}^{4105}$.
- Unseen test ingredients are safely ignored during CSR matrix construction.

---

## 14. Train/Test Leakage Prevention

1. **Pre-Modeling Split on Recipe IDs**: `src/split_data.py` splits recipe IDs (`13,823` train / `3,456` test, stratified by cuisine, seed `42`) before any ingredient counting, node filtering, PMI/NPMI computation, or Curveball randomization.
2. **Explicit Guard Boundary**: `assert_training_only(train_recipes, train_ids, test_ids)` in [`src/build_networks.py`](src/build_networks.py) is invoked inside both `build_predictive_networks()` and `BinaryMultinomialNB.fit()`, raising an `AssertionError` if even a single test recipe ID is present.
3. **Separate Artifacts**: Descriptive artifacts live in `outputs/d1_dataset/` and `outputs/d2_backbone/`; predictive training-only artifacts live in `outputs/d4_scorer/training_artifacts/` and are verified by [`tests/test_no_leakage.py`](tests/test_no_leakage.py) to have strictly smaller node/edge counts matching exact training-set frequencies.

---

## 15. Exact Commands for Running the Pipeline

Run from the repository root (`PYTHONPATH` is configured automatically inside each script):

```powershell
# Full end-to-end pipeline (preprocess, D1, benchmark, D2, D3, D4, poster, pytest & validation)
.\.venv\Scripts\python.exe run_all.py

# Recompute from scratch without null-model cache
.\.venv\Scripts\python.exe run_all.py --no-cache

# Run with custom seed or skip PDF/MD report generation
.\.venv\Scripts\python.exe run_all.py --seed 42
.\.venv\Scripts\python.exe run_all.py --skip-report

# Validate existing outputs only
.\.venv\Scripts\python.exe run_all.py --validate-only

# Individual stage scripts
.\.venv\Scripts\python.exe src/preprocess.py
.\.venv\Scripts\python.exe src/split_data.py
.\.venv\Scripts\python.exe src/build_networks.py
.\.venv\Scripts\python.exe src/null_model.py --check --benchmark
.\.venv\Scripts\python.exe src/backbone.py
.\.venv\Scripts\python.exe src/visualize.py
.\.venv\Scripts\python.exe src/analysis.py
.\.venv\Scripts\python.exe src/report.py
.\.venv\Scripts\python.exe src/poster.py
```

---

## 16. Commands for Evaluating Models and Running Tests

```powershell
# Reproduce Deliverable 4 evaluation from scratch
.\.venv\Scripts\python.exe evaluate.py

# Score a custom recipe from the command line
.\.venv\Scripts\python.exe src/fingerprint.py "garam masala" "cumin" "turmeric"

# Run the complete automated pytest suite (28 tests)
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 17. Output-File Descriptions

- `outputs/d1_dataset/nodes.csv`: `ingredient, cuisine, recipe_count, prevalence` for all 1,178 eligible (`recipe_count >= 20`) cuisine-ingredient pairs.
- `outputs/d1_dataset/edges_{indian,italian,mexican}.csv`: `ingredient_a, ingredient_b, co_count, PMI, NPMI` for all co-occurring eligible pairs.
- `outputs/d1_dataset/merge_log.csv`: `original_name, normalized_name, rule, occurrences, group_size` for all 720 changed ingredient strings.
- `outputs/d2_backbone/backbone_{indian,italian,mexican}.csv`: `ingredient_a, ingredient_b, co_count, expected_mean, expected_std, z_score, NPMI` for all validated edges ($z > 3$, $\text{co\_count} \ge 5$).
- `outputs/d2_backbone/backbone_summary.csv`: `cuisine, nodes, edges_before, edges_after, pct_retained, n_communities, modularity, eligible_nodes`.
- `outputs/d2_backbone/graph_{indian,italian,mexican}.html`: Self-contained interactive PyVis HTML network visualizations.
- `outputs/d3_report/report.pdf` & `report.md`: 4-page PDF and Markdown analytical reports with numbered Tables 1–7 and Figures 1–5.
- `outputs/d3_report/hubs_{indian,italian,mexican}.csv`, `universal_vs_specific_hubs.csv`, `communities_{indian,italian,mexican}.csv`, `bridges.csv`: Programmatically generated D3 tables.
- `outputs/d4_scorer/results.csv`, `confusion_matrix*.png`, `evaluation_metadata.json`, `garam_masala_diagnostic.json`, `training_artifacts/`: Reproducible D4 evaluation outputs.
- `outputs/poster_outline.md`: Academic poster outline populated with verified metrics.
- `outputs/validation_summary.json`: Final machine-readable PASS/FAIL report.

---

## 18. Computational Limitations and Measured Runtime

By compiling the Curveball Markov chain and upper-triangular co-occurrence accumulator with `numba` (`@njit`), the complete end-to-end clean pipeline (`python run_all.py --no-cache`)—including **both** the 1,000-sample descriptive null model ($34.7\text{M}$ total trades across 3 cuisines) and the 1,000-sample predictive training-only null model ($27.8\text{M}$ total trades across 3 cuisines)—executes in **59.58 seconds** on a single laptop CPU:

| Pipeline Phase (`python run_all.py --no-cache`) | Measured Wall-Clock Time (s) |
| --- | ---: |
| Phase 1: Dataset validation, normalization & split | `1.70 s` |
| Phase 2: Deliverable 1 descriptive network construction | `0.68 s` |
| Phase 3: Null-model degree checks & 20-sample benchmark | `1.30 s` |
| Phase 4: Deliverable 2 1,000-sample Curveball null model + PyVis HTMLs | `22.74 s` |
| Phase 5: Deliverable 3 hubs, communities, bridges, figures & 4-page PDF | `5.22 s` |
| Phase 6: Deliverable 4 training-only 1,000-sample null model + evaluation | `19.09 s` |
| Phase 7: Poster outline generation | `0.04 s` |
| Phase 8: Automated acceptance validation + `pytest` (28 tests) | `8.80 s` |
| **Total End-to-End Clean Pipeline Runtime** | **`59.58 s`** |

### Runtime Metadata Structure (`outputs/validation_summary.json`)

`run_all.py` separates full-pipeline timing records from standalone validation runs so they are never conflated:

- `invocation_mode`: `"full_pipeline"` or `"validate_only"`.
- `full_pipeline_runtimes_seconds`: Populated during full pipeline runs and preserved across subsequent `--validate-only` runs (`null` if no prior full-pipeline record was present in `validation_summary.json`, avoiding fabricated historical values).
- `validate_only_runtimes_seconds`: Latest standalone `python run_all.py --validate-only` runtime (`5.947 s` total wall time for Phase 8 acceptance checks + 28 `pytest` tests).
- `persisted_artifact_runtimes_seconds`: Stage-level runtimes persisted in the D2 and D4 metadata files:
  - `d2_descriptive_null_chains_seconds` (`outputs/d2_backbone/null_model_metadata.json`): `4.224 s` (Indian) + `8.308 s` (Italian) + `7.852 s` (Mexican) = `20.384 s`.
  - `d4_predictive_null_chains_seconds` (`outputs/d4_scorer/training_artifacts/training_metadata.json`): `3.031 s` (Indian) + `6.798 s` (Italian) + `6.504 s` (Mexican) = `16.333 s`.
  - `d4_evaluation_phases_seconds` (`outputs/d4_scorer/evaluation_metadata.json`): `19.048 s` total D4 evaluation (`16.724 s` for training networks and backbones).

---

## 19. Honest Discussion of Model Comparison

### Held-Out Test Set Performance (`outputs/d4_scorer/results.csv`, $N_{\text{test}} = 3,456$)

| Model | Accuracy | Macro-F1 | Precision (IN / IT / MX) | Recall (IN / IT / MX) | F1 (IN / IT / MX) | Zero-Score Fallbacks | Tied-Score Fallbacks |
| --- | ---: | ---: | :---: | :---: | :---: | ---: | ---: |
| **Fingerprint Scorer** | **87.67%** (`0.8767`) | **0.8712** | `0.918 / 0.872 / 0.867` | `0.800 / 0.917 / 0.863` | `0.855 / 0.894 / 0.865` | `139` (`4.02%`) | `0` |
| **Fingerprint (normalized ablation)** | **43.69%** (`0.4369`) | **0.4314** | `0.292 / 0.557 / 0.478` | `0.613 / 0.422 / 0.373` | `0.395 / 0.480 / 0.419` | `139` (`4.02%`) | `0` |
| **Naive Bayes** | **95.98%** (`0.9598`) | **0.9599** | `0.963 / 0.947 / 0.975` | `0.962 / 0.980 / 0.934` | `0.962 / 0.963 / 0.954` | `0` | `0` |

### Why Naive Bayes Outperforms the Fingerprint Scorer (and Why the Normalized Ablation Degrades)

1. **Vocabulary Coverage & Singleton Marginals**: Multinomial Naive Bayes uses all **4,105** training ingredients (including diagnostic rare ingredients with $<20$ recipes and single-ingredient marginals), whereas the Fingerprint Scorer is restricted to the **237–413** training backbone nodes per cuisine ($\ge 20$ training recipes) and requires **at least two** co-occurring backbone ingredients to produce a nonzero score.
2. **Zero-Score Fallbacks on Short/Sparse Recipes**: Exactly **139 of 3,456 test recipes (4.02%)** contain zero matched backbone edges in any cuisine and fall back to the majority training class (`italian`). Inspection of the confusion matrices (`confusion_matrix.png` vs `confusion_matrix_nb.png`) shows that these fallbacks contribute to the Fingerprint Scorer's false positives on Italian (`57` Indian and `155` Mexican recipes predicted as Italian, compared with `15` and `71` for Naive Bayes).
3. **Normalized Ablation Formula and Observed Limitations (`Score / |M_c(R)|` = 43.69% Accuracy)**:
   - **Mathematical Definition**: Whereas the primary Fingerprint Scorer sums NPMI over all matched training-backbone edges $M_c(R) = \binom{R}{2} \cap E_c^{\text{train}}$, the normalized ablation computes the sample mean $\text{Score}_{\text{norm}}(c) = \frac{1}{|M_c(R)|}\sum_{(a,b) \in M_c(R)} \text{NPMI}_c(a,b)$ for $|M_c(R)| > 0$ (and $0$ when $|M_c(R)| = 0$). Note that the denominator is the number of matched backbone edges $|M_c(R)|$ in cuisine $c$, which varies across cuisines for the same recipe $R$, rather than the cuisine-invariant total pair count $\binom{|R|}{2}$.
   - **Observed Confusion-Matrix Pattern**: In `outputs/d4_scorer/evaluation_metadata.json`, switching from the sum to the matched-edge mean reduces accuracy from `87.67%` to `43.69%` (`macro_f1 = 0.4314`). Specifically, `489` of `1,568` Italian test recipes (`31.2%`) and `405` of `1,288` Mexican test recipes (`31.4%`) are predicted as Indian under the normalized ablation (compared with `22` and `21` under the sum scorer), lowering Indian precision from `0.918` to `0.292`.
   - **Structural Limitation of Dividing by $|M_c(R)|$**: Because the denominator $|M_c(R)|$ depends on how many edges a recipe matches in cuisine $c$, dividing by $|M_c(R)|$ discards cumulative pair-count evidence. Mathematically, when a recipe matches many backbone edges in one cuisine (including moderate-NPMI pairs among common ingredients), its arithmetic mean is pulled toward the backbone average, whereas matching only one or two edges in another cuisine leaves that single edge's NPMI unaveraged. While this structural property of the $|M_c(R)|$ denominator is consistent with the observed shift toward the smaller Indian backbone (`1,466` training edges vs. `2,973` Italian and `2,223` Mexican), we report it as an analytical limitation of the matched-edge mean formulation rather than a proven causal explanation.

---

## 20. Deliverable-to-Acceptance-Check Table

Populated directly from `outputs/validation_summary.json` after executing `python run_all.py --no-cache`:

| Deliverable | Required Files | Acceptance Check | Status | Evidence or Reason |
| --- | --- | --- | :---: | --- |
| **D1: Ingredient Co-Occurrence Network Dataset** | `outputs/d1_dataset/nodes.csv`, `edges_indian.csv`, `edges_italian.csv`, `edges_mexican.csv`, `merge_log.csv` | Schemas valid; `recipe_count >= 20`; prevalence in `(0,100]`; alphabetical pairs; no self-loops/duplicates; finite PMI/NPMI in `[-1,1]` verified by full recomputation from recipes; authentic `merge_log.csv` & trademark regression check (`tests/test_d1.py`) | **PASS** | `nodes=1,178` rows; `edges={'indian': 22284, 'italian': 48488, 'mexican': 42703}`; `merge_log=720` rows; `8/8` D1 pytest checks passed. |
| **D2: Statistically Validated Backbones & PyVis Graphs** | `outputs/d2_backbone/backbone_indian.csv`, `backbone_italian.csv`, `backbone_mexican.csv`, `backbone_summary.csv`, `graph_indian.html`, `graph_italian.html`, `graph_mexican.html` | 1,000-sample Curveball null model (`run_mode=FULL`); degree/binarity/seed checks pass; `z_score` formula & exact completeness verified against `null_stats_*.csv`; all retained edges have `z_score > 3` and `co_count >= 5`; zero-variance excluded; `edges_after < edges_before`; self-contained PyVis HTMLs (`tests/test_d2.py`) | **PASS** | `mode=FULL (n_null=1000)`; `edges_after={'indian': 1829, 'italian': 3758, 'mexican': 2847}`; all `z>3 & co>=5`; `6/6` D2 pytest checks passed. |
| **D3: Hub, Community, Bridge Analysis & Report** | `outputs/d3_report/report.pdf`, `report.md`, `hubs_indian.csv`, `hubs_italian.csv`, `hubs_mexican.csv`, `universal_vs_specific_hubs.csv`, `communities_indian.csv`, `communities_italian.csv`, `communities_mexican.csv`, `bridges.csv`, `figures/*.png` | `report.pdf` is 3–4 pages; sections `(a)`–`(d)` and methodology present in MD & PDF; every finding cites numbered Table/Figure; hubs, Louvain communities & bridges match graph definitions (`tests/test_d3.py`) | **PASS** | `report.pdf` is `4` pages; `26` distinct top-10 hubs (`0` universal, `4` two-cuisine, `22` specific); `15` bridge rows (`5` per pair); `4/4` D3 pytest checks passed. |
| **D4: Fingerprint Scorer, Naive Bayes Baseline & Evaluation** | `outputs/d4_scorer/results.csv`, `confusion_matrix.png`, `confusion_matrix_nb.png`, `confusion_matrix_normalized.png`, `confusion_matrix_normalized_nb.png`, `training_artifacts/`, `evaluation_metadata.json` | Training-only network & NB fitting (`tests/test_no_leakage.py`); `predict(['garam masala','cumin','turmeric']) == 'indian'`; 3 models in `results.csv` with precision/recall/F1/Macro-F1 independently recomputed; confusion matrices sum to `3,456` (`tests/test_fingerprint.py`, `tests/test_evaluation.py`) | **PASS** | `mode=FULL`; Fingerprint Acc=`87.67%` (Macro-F1=`0.8712`), Ablation Acc=`43.69%` (Macro-F1=`0.4314`), Naive Bayes Acc=`95.98%` (Macro-F1=`0.9599`); spice test=`'indian'`; `10/10` D4 & leakage pytest checks passed. |
| **Poster Outline** | `outputs/poster_outline.md` | Contains all required sections, 3 actual figure references, and real computed metrics | **PASS** | `outputs/poster_outline.md` generated with verified metrics and figure paths. |
