# Deliverable 4 — Phase 3 Pre-Execution Design Document: Network-Informed Fingerprint Scorer v2 (Stacked)

> [!IMPORTANT]
> **Pre-Execution Specification (`STEP 0`)**: This document is written **before executing any Phase 3 computation**. It specifies the exact mathematical definitions of all D1–D3 network features, the nested cross-fitting procedure that prevents any target or graph leakage, the model hierarchy (`M0`–`M5`) and feature-group ablations (`A1`–`A6`), the primary metric hierarchy (**Macro-F1 primary**, accuracy secondary, per-cuisine recall regression guard $> 1.0\text{ pp}$), and the measured runtime estimate. The held-out test set (`3,456` recipes in `data/processed/split.json`) remains completely sealed and untouched.

---

## 1. Scientific Objective & Hypotheses

Phase 1 and Phase 2 demonstrated two key findings on the 13,823 training recipes (5-fold CV, `seed = 42`):
1. Adding ~4,865 raw binary pair-presence indicators directly to ~610 or ~3,834 unigram indicators in a high-dimensional L2 logistic regression (`ExpC`) does not improve over unigrams alone (`-0.21 pp` to `-0.35 pp`), because thousands of collinear binary pair columns force stronger L2 regularization that dilutes informative single-ingredient weights.
2. However, on pair-only models, replacing raw within-cuisine NPMI sums (`86.62%` accuracy, `85.82%` Macro-F1) with **contrastive backbone pair log-odds (`ExpB1_bb`)** jumped performance by **+4.01 pp accuracy (+4.25 pp Macro-F1)** to **90.63% accuracy (90.07% Macro-F1)** on the exact same validated backbone edges.

**Phase 3 Goal**: Test whether **compact, low-dimensional network-derived structural summaries** from Deliverables 1–3 (backbone pair log-odds, Louvain community coverage/concentration, D3 weighted-degree hubs, D3 cross-cuisine bridges, and backbone node coverage) add predictive value when combined in a low-dimensional stacked L2 logistic regression meta-model:
- **Question 1 (`M5 vs. M0`)**: Can a **pure network-only Fingerprint Scorer v2 (`M5`, 22 features, zero unigram NB features)** substantially outperform the original D4 Fingerprint Scorer (`M0`) by combining contrastive backbone pair log-odds with D2/D3 structural summaries (communities, hubs, bridges, backbone coverage)?
- **Question 2 (`M4 vs. M1` and `M4 vs. M2`)**: Does the full network-informed stack (`M4`, combining NB log-posteriors with all D2/D3 network features, 25 features total) improve **Macro-F1** (primary metric) over standalone Naive Bayes (`M1`) or calibrated Naive Bayes (`M2`), particularly on short or ambiguous recipes?

---

## 2. Exact Feature Definitions (Using D1–D3's Actual Definitions Only)

Let $\mathcal{C} = \{\text{indian}, \text{italian}, \text{mexican}\}$ be the three cuisines. For any training partition $\mathcal{D}_{\text{fit}}$ (either an inner cross-fit training split of ~7,372 recipes or a full outer training split of ~11,058 recipes), we first run the unmodified D1–D3 pipeline on $\mathcal{D}_{\text{fit}}$ **only**:
1. **Normalization & Eligible Network (`src/build_networks.py`)**: Normalize recipes via `normalize_recipe`; extract eligible nodes $V_c^{\text{elig}}$ with `min_ingredient_count = 20` in cuisine $c$ and co-occurrence edges with PMI and NPMI.
2. **Curveball Validated Backbone (`src/backbone.py`)**: Run Curveball null chains (`z > 3.0`, `co_count >= 5`, `zero_variance_policy = exclude`) on $\mathcal{D}_{\text{fit}}$ to obtain each cuisine's validated backbone edge table $E_c$ and undirected weighted backbone graph $G_c = (V_c, E_c)$ via `src/graph_utils.py:backbone_graph(bb_c, nodes_c)`, where edge weights are $w_c(a,b) = \max(\text{NPMI}_c(a,b), 0)$ and $V_c$ is the set of backbone nodes in $G_c$.
3. **D3 Hubs (`src/analysis.py:compute_hubs`)**:
   - In D3 (`src/analysis.py` line 8 & lines 55–61), a cuisine's **hubs** are the top-10 backbone nodes by weighted degree (strength $S_c(u) = \sum_{v \in N_{G_c}(u)} w_c(u,v)$), ranked by `rank_desc(strengths(G_c))` (`weighted_degree_rank <= 10`, ties broken alphabetically):
     $$H_c = \{u \in V_c : \text{rank\_desc}(S_c)(u) \le 10\}$$
4. **D3 Louvain Communities (`src/analysis.py:compute_communities` / `src/graph_utils.py:louvain`)**:
   - In D3 (`src/graph_utils.py` lines 52–61), Louvain community detection `louvain(G_c, seed=42, resolution=1.0)` partitions the backbone nodes $V_c$ into $K_c$ disjoint communities $\{C_{c,0}, C_{c,1}, \dots, C_{c,K_c-1}\}$ deterministically relabeled by size descending, then alphabetically by first member.
5. **D3 Bridges (`src/analysis.py:compute_bridges`)**:
   - In D3 (`src/analysis.py` lines 10 & 119–131), for every unordered pair of cuisines $(A, B) \in \{(\text{indian},\text{italian}), (\text{indian},\text{mexican}), (\text{italian},\text{mexican})\}$, candidate bridge nodes are shared backbone nodes $u \in V_A \cap V_B$ scored by:
     $$\text{Bridge}(u; A, B) = \min(S_A(u), S_B(u))$$
     and ranked descending (ties broken alphabetically) to select the top-5 bridge ingredients (`top_n_bridges = 5`) for pair $(A, B)$, denoted $B_{A,B}^{\text{top5}}$.
   - For each cuisine $c$, an ingredient $u$ is a **D3 bridge ingredient involving cuisine $c$** if it belongs to the top-5 bridge list for either of the two cuisine pairs involving $c$:
     $$B_c = \bigcup_{c' \in \mathcal{C} \setminus \{c\}} B_{\min(c,c'), \max(c,c')}^{\text{top5}}$$
     *(Note: Since each of the two pairs involving $c$ has 5 top bridges, $|B_c| \le 10$ per cuisine, strictly following `compute_bridges` in `src/analysis.py` without inventing any new bridge definition.)*

### Complete 25-Feature Vector $\mathbf{x}(R)$ for a Recipe $R$

Given a normalized deduplicated recipe $R = \{u_1, \dots, u_L\}$ of length $L = |R|$:

#### Group 1: Unigram Evidence (`nb_log_post`, 3 features — 1 per cuisine $c$)
1. **`nb_log_prob_{c}`**: Log-posterior $\log P_{\text{NB}}(c \mid R)$ from `BinaryMultinomialNB(alpha=1.0, fit_prior=True)` fitted strictly on $\mathcal{D}_{\text{fit}}$.

#### Group 2: Backbone Pair Log-Odds Score (`pair_log_odds`, 3 features — 1 per cuisine $c$)
2. **`pair_log_odds_{c}`**: Let $E_{\text{union}} = \bigcup_{c' \in \mathcal{C}} E_{c'}$ be the union of validated backbone pairs on $\mathcal{D}_{\text{fit}}$. Using the exact `ExpB1_bb` formula from Phase 2:
   $$w(c, p) = \log\left(\frac{P_{\mathcal{D}_{\text{fit}}}(p \mid c) + \varepsilon}{P_{\mathcal{D}_{\text{fit}}}(p \mid \neg c) + \varepsilon}\right), \quad \text{pair\_log\_odds}_c(R) = \sum_{p \in \binom{R}{2} \cap E_{\text{union}}} w(c, p)$$
   where $\varepsilon = 10^{-6}$ (the optimal smoothing value selected across all 5 folds in Phase 2 `ExpB1_bb`, verified inside inner CV).

#### Group 3: Louvain Community Features (`communities`, 9 features — 3 per cuisine $c$)
For cuisine $c$, let $m_{c,k}(R) = |R \cap C_{c,k}|$ be the number of recipe ingredients falling into community $k \in \{0, \dots, K_c-1\}$ of cuisine $c$'s backbone $G_c$, and let $M_c^{\text{comm}}(R) = \sum_{k=0}^{K_c-1} m_{c,k}(R) = |R \cap V_c|$ be the total number of recipe ingredients covered by cuisine $c$'s communities (note that because Louvain partitions all nodes of $G_c$, $\bigcup_{k} C_{c,k} = V_c$).
3. **`comm_cov_frac_{c}` (Fraction of recipe ingredients covered by cuisine $c$'s communities)**:
   $$\text{comm\_cov\_frac}_c(R) = \frac{\sum_{k=0}^{K_c-1} m_{c,k}(R)}{\max(L, 1)}$$
4. **`comm_max_cov_{c}` (Maximum single-community coverage in cuisine $c$)**:
   $$\text{comm\_max\_cov}_c(R) = \frac{\max_{0 \le k < K_c} m_{c,k}(R)}{\max(L, 1)}$$
   *(Captures whether a recipe's ingredients cluster tightly inside one cohesive culinary module of cuisine $c$, such as an Italian pasta/cheese community or an Indian spice-masala community.)*
5. **`comm_entropy_{c}` (Community concentration / normalized Shannon entropy of ingredient spread across cuisine $c$'s communities)**:
   Let $p_{c,k}(R) = m_{c,k}(R) / M_c^{\text{comm}}(R)$ for $m_{c,k}(R) > 0$. The Shannon entropy of the recipe's distribution across cuisine $c$'s communities is:
   $$H_{\text{comm}}(c, R) = \begin{cases} -\sum_{k : m_{c,k}(R) > 0} p_{c,k}(R) \log(p_{c,k}(R)) & \text{if } M_c^{\text{comm}}(R) > 0 \\[4pt] 0.0 & \text{if } M_c^{\text{comm}}(R) = 0 \end{cases}$$

#### Group 4: D3 Hub Feature (`hubs`, 3 features — 1 per cuisine $c$)
6. **`hub_count_{c}` (Hub ingredient count in cuisine $c$)**:
   $$\text{hub\_count}_c(R) = |R \cap H_c|$$
   where $H_c$ is the D3 top-10 weighted-degree hub set for cuisine $c$.

#### Group 5: D3 Bridge Feature (`bridges`, 3 features — 1 per cuisine $c$)
7. **`bridge_count_{c}` (Bridge ingredient count involving cuisine $c$)**:
   $$\text{bridge\_count}_c(R) = |R \cap B_c|$$
   where $B_c$ is the union of D3 top-5 bridge ingredients across the two cuisine pairs involving cuisine $c$.

#### Group 6: Backbone Ingredient Coverage (`backbone_cov`, 3 features — 1 per cuisine $c$)
8. **`backbone_ing_count_{c}` (Backbone ingredient coverage count in cuisine $c$, complementing the fraction `comm_cov_frac_{c}` so both count $|R \cap V_c|$ and fraction $|R \cap V_c|/L$ are represented)**:
   $$\text{backbone\_ing\_count}_c(R) = |R \cap V_c|$$
   *(Note: Since Louvain communities partition $V_c$, $|R \cap V_c|/L$ is identical to `comm_cov_frac_{c}`; providing $|R \cap V_c|$ as `backbone_ing_count_{c}` avoids exact linear collinearity while supplying both raw backbone node coverage and fractional coverage. In our group ablations, removing "Community structure" removes `comm_max_cov_{c}` and `comm_entropy_{c}`, while removing "Backbone coverage" removes both `backbone_ing_count_{c}` and `comm_cov_frac_{c}`.)*

#### Group 7: Recipe Length (`recipe_length`, 1 global feature)
9. **`recipe_length`**:
   $$L = |R|$$
   the number of normalized, deduplicated ingredients in recipe $R$.

**Total Features**: $3 + 3 + 9 + 3 + 3 + 3 + 1 = \mathbf{25\text{ features}}$ for `M4` (Full Stack), and $25 - 3 = \mathbf{22\text{ features}}$ for `M5` (Pure Network-Only Fingerprint Scorer v2).

---

## 3. Strict Cross-Fitting & Zero-Leakage Protocol

To guarantee zero data leakage when training the stacked meta-model on each outer fold $f \in \{0, 1, 2, 3, 4\}$:

1. **Outer Split ($13,823$ training recipes)**:
   - Uses the exact 5 outer folds from `outputs/d4_experiments/cv_folds.json` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
   - Let $\mathcal{T}_f$ ($N \approx 11,058$) be outer fold $f$'s training partition and $\mathcal{V}_f$ ($N \approx 2,765$) be outer fold $f$'s validation partition.
   - Guard: `assert_training_only` verifies $\mathcal{T}_f \cap (\mathcal{V}_f \cup \mathcal{D}_{\text{test}}) = \emptyset$.

2. **Inner 3-Fold Cross-Fitting for Training-Row Features ($\mathcal{T}_f$)**:
   - Split $\mathcal{T}_f$ using `StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + f)` into inner partitions $(\mathcal{T}_{f,m}^{\text{in}}, \mathcal{V}_{f,m}^{\text{in}})$ for $m \in \{0, 1, 2\}$ ($|\mathcal{T}_{f,m}^{\text{in}}| \approx 7,372$, $|\mathcal{V}_{f,m}^{\text{in}}| \approx 3,686$).
   - For each inner fold $m \in \{0, 1, 2\}$:
     - Using **only** $\mathcal{T}_{f,m}^{\text{in}}$ (strictly excluding $\mathcal{V}_{f,m}^{\text{in}}$, $\mathcal{V}_f$, and $\mathcal{D}_{\text{test}}$):
       1. Normalize $\mathcal{T}_{f,m}^{\text{in}}$ and build eligible networks (`min_ingredient_count = 20`).
       2. Run Curveball null chains (`n_randomizations = 50` for inner cross-fit builds — documented compute shortcut) and extract inner validated backbones $E_c^{(f,m)}$ ($z > 3.0, \text{co\_count} \ge 5$).
       3. Compute D3 graphs $G_c^{(f,m)}$, top-10 weighted-degree hubs $H_c^{(f,m)}$, Louvain communities $\{C_{c,k}^{(f,m)}\}$, and top-5 pairwise bridges $B_c^{(f,m)}$ on $\mathcal{T}_{f,m}^{\text{in}}$.
       4. Fit `BinaryMultinomialNB(alpha=1.0, fit_prior=True)` and backbone pair log-odds weights $w^{(f,m)}(c,p)$ ($\varepsilon = 10^{-6}$) on $\mathcal{T}_{f,m}^{\text{in}}$.
     - Extract the 25 out-of-fold feature values for every recipe $R \in \mathcal{V}_{f,m}^{\text{in}}$ using the components fitted on $\mathcal{T}_{f,m}^{\text{in}}$.
   - Assembling across $m \in \{0, 1, 2\}$ yields the cross-fitted training feature matrix $X_{\text{train}}^{(f)} \in \mathbb{R}^{|\mathcal{T}_f| \times 25}$, where **every single training row's features were generated by models and networks built without that recipe**.

3. **Full Outer-Training Fit for Outer-Validation Features ($\mathcal{V}_f$)**:
   - Fit the entire pipeline (eligible networks, Curveball backbones with `n_randomizations = 100`, D3 hubs, Louvain communities, D3 bridges, `BinaryMultinomialNB`, and backbone pair log-odds weights) on the full outer training partition $\mathcal{T}_f$.
   - Compute the 25 features for each outer validation recipe $R \in \mathcal{V}_f$ to form $X_{\text{val}}^{(f)} \in \mathbb{R}^{|\mathcal{V}_f| \times 25}$.

4. **Standardization & Inner-CV Meta-Model Tuning (Macro-F1 Primary)**:
   - **Standardization**: Inside inner CV (and on $\mathcal{T}_f$ for final fold fitting), standardize each selected feature column using `StandardScaler` (`mean` and `std` computed **strictly on the training matrix** $X_{\text{train}}^{(f)}$, never on $X_{\text{val}}^{(f)}$).
   - **Meta-Model**: Multi-class L2-regularized logistic regression `LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)`.
   - **Hyperparameter Selection**: Tune $C \in \{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0\}$ via 3-fold inner CV on the cross-fitted training matrix $X_{\text{train}}^{(f)}$ (with `StandardScaler` re-fitted inside each inner split), selecting the $C$ that maximizes **mean inner-CV Macro-F1** (ties broken by **mean inner-CV accuracy**, then smaller $C$, per the Metric Hierarchy).

---

## 4. Models (`M0`–`M5`) & Feature-Group Ablations (`A1`–`A6`)

### 4.1 Main Models
- **`M0`: Original Fingerprint Scorer** (unchanged reference from D4/Phase 1: sum of backbone NPMI with majority-class tie/zero-score fallback).
- **`M1`: Naive Bayes Alone** (unchanged reference from D4/Phase 1: `BinaryMultinomialNB(alpha=1.0, fit_prior=True)` over full training vocabulary).
- **`M2`: Calibrated NB Stack Alone (3 features)** — L2 logistic regression meta-model trained only on the 3 cross-fitted NB log-posterior features (`nb_log_prob_{c}`). Serves as a sanity check to separate the effect of logistic recalibration of NB scores from the effect of adding network features.
- **`M3`: NB + Backbone Pair Log-Odds (6 features)** — L2 logistic regression meta-model on `nb_log_prob_{c}` (3) + `pair_log_odds_{c}` (3).
- **`M4`: Full Stack: NB + All Network Features (25 features)** — L2 logistic regression meta-model on all 25 cross-fitted features.
- **`M5`: Network-Only Fingerprint Scorer v2 (22 features, NO NB features)** — L2 logistic regression meta-model on all 22 network features (`pair_log_odds`, `communities`, `hubs`, `bridges`, `backbone_cov`, `recipe_length`).

### 4.2 Feature-Group Ablations from `M4` (Leave-One-Group-Out)
To measure the marginal contribution of each feature group inside `M4`:
- **`Abl1 (-NB)`**: Remove `nb_log_prob_{c}` (3 features) from `M4` $\rightarrow$ identical to `M5` (22 features).
- **`Abl2 (-PairLogOdds)`**: Remove `pair_log_odds_{c}` (3 features) from `M4` (22 features).
- **`Abl3 (-Communities)`**: Remove Louvain community structure features `comm_max_cov_{c}` and `comm_entropy_{c}` (6 features) from `M4` (19 features).
- **`Abl4 (-Hubs)`**: Remove `hub_count_{c}` (3 features) from `M4` (22 features).
- **`Abl5 (-Bridges)`**: Remove `bridge_count_{c}` (3 features) from `M4` (22 features).
- **`Abl6 (-BackboneCov & Length)`**: Remove `backbone_ing_count_{c}`, `comm_cov_frac_{c}`, and `recipe_length` (7 features) from `M4` (18 features).
- *(Plus a supplementary ablation on `M5` removing community/hub/bridge/coverage features to compare `M5` against pair-log-odds-only).*

---

## 5. Pre-Registered Recipe-Length Bins & Metric Hierarchy

### 5.1 Pre-Registered Fixed Recipe-Length Bins
Defined prior to running Phase 3 based on normalized ingredient count $L = |R|$:
1. **Short recipes (`1–5` ingredients)**: Sparse recipes where pair matches are often zero or minimal.
2. **Medium-short recipes (`6–8` ingredients)**: Lower-middle recipe length.
3. **Medium-long recipes (`9–12` ingredients)**: Typical recipe length around the dataset median ($10$ ingredients).
4. **Long recipes (`13+` ingredients)**: Dense recipes with many active ingredient pairs and community members.

### 5.2 Metric Hierarchy & Regression Guard
1. **PRIMARY**: **Macro-F1** (unweighted arithmetic mean of per-cuisine $F_1$ across `indian`, `italian`, `mexican`), reported per outer fold, as 5-fold CV Mean ± SD, and as pooled OOF Macro-F1 with 95% paired bootstrap confidence intervals (`n_boot = 2000, seed = 42`). Used for all inner-CV hyperparameter selection ($C$).
2. **SECONDARY**: **Accuracy**, reported per outer fold, 5-fold CV Mean ± SD, and pooled OOF with 95% paired bootstrap CIs and paired McNemar tests (with exact discordant counts).
3. **TERTIARY**: Per-cuisine precision, recall, and $F_1$ on every outer fold and pooled OOF.
4. **QUATERNARY**: Pooled out-of-fold $3 \times 3$ confusion matrices for `M0`–`M5`.
5. **SUPPLEMENTARY (Appendix only)**: Weighted-$F_1$.
6. **Explicit Recall Regression Guard**: For every primary paired comparison (`M5 vs. M0`, `M4 vs. M1`, `M4 vs. M2`, `M3 vs. M1`), we automatically check whether a model that improves mean Macro-F1 lowers any single cuisine's recall by $> 1.0\text{ percentage point}$ (`0.0100`). Any violation is flagged explicitly in the report.
7. **Statistical Honesty Guard**: No claim of improvement is made if the 95% paired bootstrap CI for $\Delta\text{Macro-F1}$ includes zero.

---

## 6. Runtime Estimate & Documented Compute Shortcuts

- **Outer-fold Curveball backbones (5 folds $\times$ 3 cuisines, `n_randomizations = 100`)**: Already cached in `outputs/d4_experiments/cache/` from Phase 1/2 $\rightarrow$ **~0.5 seconds total** to load.
- **Inner cross-fitting Curveball backbones (5 outer folds $\times$ 3 inner folds $\times$ 3 cuisines = 45 backbone builds on ~7,372 recipes each)**:
  - At `n_randomizations = 100`, each 3-cuisine inner fold takes ~1.1s $\rightarrow$ $15 \times 1.1\text{s} \approx 16.5\text{s}$.
  - Applying the **documented inner-build shortcut `n_randomizations = 50`** (`burn_in = 10 * N_c`, `thinning = 2 * N_c`), each 3-cuisine inner fold takes **~0.6s** $\rightarrow$ $15 \times 0.6\text{s} \approx \mathbf{9\text{ seconds}}$ total (cached in `outputs/d4_experiments/phase3/cache/`).
- **D3 Graph Analysis (Strengths, Hubs, Louvain Communities, Bridges) across 5 outer + 15 inner builds (60 cuisine graphs total, each with 200–370 nodes)**:
  - Notice that our Phase 3 features use D3's **weighted-degree hubs (`top10_weighted_degree`)**, **Louvain communities**, and **weighted-degree bridges**, none of which require all-pairs shortest-path betweenness centrality! Computing `backbone_graph`, `strengths`, `louvain`, and `compute_bridges` takes `< 0.03s` per graph $\rightarrow$ **~2 seconds total** for all 20 builds (we also run full `compute_hubs` on the 5 outer folds for completeness, which takes `< 1.5s`).
- **Feature extraction (`13,823` cross-fitted training rows across 5 folds + `13,823` OOF validation rows) + 25-feature L2 Logistic Regression inner-CV tuning + 2,000-sample paired bootstrap**:
  - Feature extraction: **~4 seconds**.
  - L2 Logistic Regression tuning on 25 dense standardized features (5 outer folds $\times$ 3 inner folds $\times$ 9 $C$-values $\times$ 12 models/ablations): `< 0.01s` per fit $\rightarrow$ **~8 seconds**.
  - 2,000-iteration bootstrap CIs across all models: **~3 seconds**.
- **Total Estimated Runtime**: **~25–35 seconds** (well under 1 minute). This is completely reasonable and fast, so we proceed immediately with the full nested cross-fitting implementation.
