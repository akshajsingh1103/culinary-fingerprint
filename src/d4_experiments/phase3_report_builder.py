"""Generate outputs/d4_experiments/phase3/phase3_report.md from phase3_summary.json and results.csv."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import ROOT, read_json

PHASE3_DIR = ROOT / "outputs" / "d4_experiments" / "phase3"


def format_phase3_report() -> str:
    summary = read_json(PHASE3_DIR / "phase3_summary.json")
    models = summary["model_summaries"]
    paired = summary["paired_comparisons"]
    ablations = summary["ablation_comparisons_vs_M4"]
    coefs = summary["standardized_coefficients"]
    nb_changes = summary["nb_prediction_change_analysis"]
    length_bins = summary["recipe_length_bins_table"]
    verif = summary["verification_log"]
    folds = summary["fold_reports"]

    main_keys = [
        "M0: Original Fingerprint Scorer (reference)",
        "M1: Naive Bayes Alone (reference)",
        "M2: Calibrated NB Stack Alone (3 features)",
        "M3: NB + Backbone Pair Log-Odds Stack (6 features)",
        "M4: Full Stack — NB + All Network Features (25 features)",
        "M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)",
    ]

    lines = [
        "# Deliverable 4 — Phase 3 Report: Network-Informed Fingerprint Scorer v2 (Stacked)",
        "",
        "> [!IMPORTANT]",
        "> **Held-Out Test Set Sealed & Metric Hierarchy Enforced**:",
        "> 1. **Test Set Untouched**: The 3,456 held-out test recipes (`test_ids` in `data/processed/split.json`) were **never loaded, inspected, or evaluated**. All results below come strictly from nested 5-fold cross-validation (`outputs/d4_experiments/cv_folds.json`, `seed = 42`) across the 13,823 training recipes, with 3-fold inner cross-fitting (`n_randomizations = 50` documented inner shortcut; `n_randomizations = 100` outer builds) so that every training row's 25 features are computed from D1 networks, D2 backbones, D3 hubs/communities/bridges, and Naive Bayes models fitted **without that recipe**.",
        "> 2. **Primary Metric Hierarchy**: **Macro-F1** (unweighted mean of per-cuisine $F_1$) is the **primary metric** for all model comparisons and inner-CV hyperparameter selection ($C$), with **Accuracy** secondary, per-cuisine Precision/Recall/$F_1$ tertiary, pooled OOF confusion matrices quaternary, and **Weighted-$F_1$** reported solely in the Appendix. Every paired comparison is checked against the **Recall Regression Guard** (flagging any $> 1.0\\text{ pp}$ drop in any single cuisine's recall) and the **Bootstrap CI Honesty Guard** (making no claim of improvement if the 95% paired bootstrap CI for $\\Delta\\text{Macro-F1}$ includes zero).",
        "",
        "---",
        "",
        "## Executive Summary & Plain-Spoken Findings",
        "",
        "### 1. Does the Pure Network-Only Fingerprint Scorer v2 (`M5`) Improve Upon the Original Fingerprint Scorer (`M0`)?",
        "**Yes — massively and unambiguously across every metric and every cuisine.**",
        "- **`M5` (Network-Only Fingerprint Scorer v2, 22 D2/D3 network features, zero unigram NB features)** achieves **93.84% ± 0.46% CV Macro-F1** (**93.84% pooled OOF Macro-F1**) and **94.04% ± 0.47% CV Accuracy**, compared to **`M0` (Original Fingerprint Scorer)** at **85.82% ± 0.92% CV Macro-F1** and **86.62% ± 0.77% CV Accuracy**.",
        "- **Pooled Paired Gain (`M5 minus M0`)**:",
        "  - **$\\Delta$ Macro-F1 (Primary)**: **+8.01 percentage points** (`95% paired bootstrap CI: [+7.44 pp, +8.61 pp]` — strictly excludes zero).",
        "  - **$\\Delta$ Accuracy (Secondary)**: **+7.42 percentage points** (`95% paired bootstrap CI: [+6.87 pp, +7.96 pp]`).",
        "  - **McNemar Discordant Counts**: `M5` is correct on **1,291** recipes where `M0` errs, while `M0` is correct on only **266** recipes where `M5` errs (a net gain of **+1,025 recipes** out of 13,823; $\\chi^2_{\\text{CC}} = 673.46$, exact binomial $p = 1.97 \\times 10^{-161}$).",
        "  - **Recall Regression Guard**: **PASSED** (`M5` improves recall in **all three cuisines simultaneously**: `indian` recall **+13.52 pp** [`78.36% -> 91.89%`], `italian` recall **+4.55 pp** [`91.80% -> 96.35%`], `mexican` recall **+8.06 pp** [`84.17% -> 92.23%`]).",
        "- **Where does `M5`'s gain come from?** Even compared against a stacked logistic regression using *only* the 3 backbone pair log-odds scores (`Abl_M5a: 90.19%` Macro-F1, `90.49%` Accuracy), adding the 19 D3 structural features (Louvain community coverage/max-coverage/entropy, D3 top-10 weighted-degree hubs, D3 top-5 pairwise bridges, backbone node counts, and recipe length) adds a further **+3.65 pp Macro-F1** (`95% CI: [+3.20 pp, +4.13 pp]`, $+490$ net recipes, $p = 2.41 \\times 10^{-58}$) — and remarkably, **D3 structural summaries alone without any pair log-odds (`Abl_M5b`) achieve 92.82% ± 0.75% Macro-F1 (93.13% Accuracy)**, outperforming `M0` by **+7.00 pp Macro-F1**!",
        "",
        "### 2. Do Compact Network Features Improve Over Naive Bayes When Stacked Together (`M4 vs. M1` and `M4 vs. M2`)?",
        "**Yes —unlike the 4,865 raw binary pair indicators in Phase 2 (which hurt unigram models by -0.21 to -0.35 pp), compact low-dimensional D1–D3 network summaries (`M4`, 25 features) produce a statistically significant improvement in Macro-F1 and Accuracy over both standalone Naive Bayes (`M1`) and Calibrated Naive Bayes (`M2`), with zero recall regressions:**",
        "- **`M4` (Full Stack: NB + All Network Features, 25 features)**: **95.82% ± 0.44% CV Macro-F1** (**95.82% pooled OOF Macro-F1**) and **95.88% ± 0.46% CV Accuracy**.",
        "- **`M1` (Naive Bayes Alone)**: **95.53% ± 0.62% CV Macro-F1** and **95.66% ± 0.58% CV Accuracy**.",
        "- **`M2` (Calibrated NB Stack Alone, 3 features — sanity check)**: **95.50% ± 0.63% CV Macro-F1** and **95.64% ± 0.63% CV Accuracy** (confirming that simply feeding NB log-posteriors into an L2 logistic regression meta-model provides **0.00 pp gain** on its own: $\\Delta\\text{Macro-F1} = -0.03\\text{ pp}$, $95\\%\\text{ CI } [-0.19, +0.13]$, McNemar $p = 0.824$).",
        "- **Pooled Paired Gain (`M4 minus M1`)**:",
        "  - **$\\Delta$ Macro-F1 (Primary)**: **+0.28 percentage points** (`95% paired bootstrap CI: [+0.05 pp, +0.52 pp]` — **excludes zero**).",
        "  - **$\\Delta$ Accuracy (Secondary)**: **+0.22 percentage points** (`95% paired bootstrap CI: [+0.01 pp, +0.42 pp]`).",
        "  - **McNemar Test**: `116` recipes fixed vs. `86` broken (**+30 net recipes**, exact binomial $p = 0.0410$).",
        "  - **Recall Regression Guard**: **PASSED** (`indian` recall **+0.37 pp**, `mexican` recall **+0.95 pp**, `italian` recall **-0.45 pp** — well within the $1.0\\text{ pp}$ guard threshold).",
        "- **Pooled Paired Gain (`M4 minus M2`, isolating the pure contribution of the 22 network features inside the same L2 meta-learner)**:",
        "  - **$\\Delta$ Macro-F1 (Primary)**: **+0.31 percentage points** (`95% paired bootstrap CI: [+0.07 pp, +0.57 pp]` — **excludes zero**).",
        "  - **$\\Delta$ Accuracy (Secondary)**: **+0.24 percentage points** (`95% paired bootstrap CI: [+0.03 pp, +0.46 pp]`).",
        "  - **McNemar Test**: `121` recipes fixed vs. `88` broken (**+33 net recipes**, exact binomial $p = 0.0266$).",
        "  - **Recall Regression Guard**: **PASSED** (`indian` recall **+1.29 pp**, `mexican` recall **+0.41 pp**, `italian` recall **-0.30 pp**).",
        "- **Honest Nuance on `M3` (`NB + Backbone Pair Log-Odds Only`, 6 features)**:",
        "  - Adding *only* the 3 backbone pair log-odds scores to NB (`M3: 95.56% ± 0.65%` Macro-F1) yields a tiny **+0.03 pp** Macro-F1 difference vs. `M1` (`95% CI: [-0.14 pp, +0.20 pp]`, $p = 0.714$) and **+0.06 pp** vs. `M2` (`95% CI: [-0.07 pp, +0.19 pp]`, $p = 0.403$). Because these CIs include zero, **backbone pair log-odds alone do not improve upon Naive Bayes**; the statistically significant gain in `M4` comes specifically from the **D2/D3 structural graph features (hubs, communities, bridges, and backbone node coverage)** correcting Naive Bayes's systematic bias on multi-ingredient recipes.",
        "",
        "---",
        "",
        "## 1. Cross-Fitting & Zero-Leakage Verification Log",
        "",
        "Every outer fold matches `outputs/d4_experiments/cv_folds.json` identically. Inside each outer fold $f \\in \\{0,1,2,3,4\\}$, every single training recipe's 25 features in $X_{\\text{train}}^{(f)}$ were generated out-of-fold by an inner 3-fold `PartitionNetworkFeatureExtractor` fitted strictly on the other $2/3$ of outer-training recipes (`~7,372` recipes, rebuilding ingredient normalization, eligible nodes, 50-randomization Curveball backbones, D3 hubs, Louvain communities, D3 bridges, Naive Bayes, and pair log-odds from scratch without that recipe):",
        "",
        "| Outer Fold | Outer Train $N$ | Outer Val $N$ | Train SHA-256 (16 hex) | Val SHA-256 (16 hex) | All Train Rows Cross-Fitted Out-of-Fold | Inner Folds / Curveball $N_{\\text{rand}}$ | Outer Curveball $N_{\\text{rand}}$ | Train $\\cap$ Val | Train $\\cap$ Test | Val $\\cap$ Test | Selected Meta-Model $C$ (`M2 / M3 / M4 / M5`) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for f_i, v in enumerate(verif):
        sc = folds[f_i]["selected_meta_C"]
        c_str = (
            f"`{sc['M2: Calibrated NB Stack Alone (3 features)']}` / "
            f"`{sc['M3: NB + Backbone Pair Log-Odds Stack (6 features)']}` / "
            f"`{sc['M4: Full Stack — NB + All Network Features (25 features)']}` / "
            f"`{sc['M5: Network-Only Fingerprint Scorer v2 (22 features, no NB)']}`"
        )
        lines.append(
            f"| {v['fold']} | {v['n_outer_train']:,} | {v['n_outer_val']:,} | `{v['train_ids_sha256'][:16]}...` | "
            f"`{v['val_ids_sha256'][:16]}...` | `{v['all_train_rows_cross_fitted_out_of_fold']}` | "
            f"{v['inner_folds_count']} / `{v['inner_curveball_n_randomizations']}` | "
            f"`{v['outer_curveball_n_randomizations']}` | `{v['train_val_overlap']}` | "
            f"`{v['train_test_overlap']}` | `{v['val_test_overlap']}` | {c_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Primary Results Table: Macro-F1 (Primary) & Accuracy (Secondary) Across Outer Folds",
        "",
        "### 2.1 Summary Across 5 Outer Folds and Pooled Out-of-Fold (`13,823` Training Recipes)",
        "",
        "| Model | CV Macro-F1 (Mean ± SD) [PRIMARY] | Pooled OOF Macro-F1 [95% Bootstrap CI] | CV Accuracy (Mean ± SD) [SECONDARY] | Pooled OOF Accuracy [95% Bootstrap CI] | F1 `indian` (Mean) | F1 `italian` (Mean) | F1 `mexican` (Mean) | Zero-Score Fallbacks |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    boot_models = summary["bootstrap_ci"]["models"]
    for m_name in main_keys:
        ms = models[m_name]
        bm = boot_models[m_name]
        f1_ci = bm["macro_f1_ci95"]
        acc_ci = bm["accuracy_ci95"]
        lines.append(
            f"| **{m_name}** | **{ms['cv_macro_f1_mean']*100:.2f}% ± {ms['cv_macro_f1_std']*100:.2f}%** | "
            f"**{ms['oof_pooled_macro_f1']*100:.2f}%** [{f1_ci[0]*100:.2f}%, {f1_ci[1]*100:.2f}%] | "
            f"{ms['cv_accuracy_mean']*100:.2f}% ± {ms['cv_accuracy_std']*100:.2f}% | "
            f"{ms['oof_pooled_accuracy']*100:.2f}% [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%] | "
            f"{ms['cv_f1_indian_mean']*100:.2f}% | {ms['cv_f1_italian_mean']*100:.2f}% | "
            f"{ms['cv_f1_mexican_mean']*100:.2f}% | {ms['zero_score_fallback_count']} |"
        )

    lines.extend([
        "",
        "### 2.2 Fold-by-Fold Macro-F1 (Primary) and Accuracy (Secondary) for `M0`–`M5`",
        "",
        "| Model | Fold 0 Macro-F1 / Acc | Fold 1 Macro-F1 / Acc | Fold 2 Macro-F1 / Acc | Fold 3 Macro-F1 / Acc | Fold 4 Macro-F1 / Acc | 5-Fold Mean ± SD Macro-F1 | 5-Fold Mean ± SD Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_name in main_keys:
        ms = models[m_name]
        lines.append(
            f"| **{m_name}** | {ms['fold_0_macro_f1']*100:.2f}% / {ms['fold_0_acc']*100:.2f}% | "
            f"{ms['fold_1_macro_f1']*100:.2f}% / {ms['fold_1_acc']*100:.2f}% | "
            f"{ms['fold_2_macro_f1']*100:.2f}% / {ms['fold_2_acc']*100:.2f}% | "
            f"{ms['fold_3_macro_f1']*100:.2f}% / {ms['fold_3_acc']*100:.2f}% | "
            f"{ms['fold_4_macro_f1']*100:.2f}% / {ms['fold_4_acc']*100:.2f}% | "
            f"**{ms['cv_macro_f1_mean']*100:.2f}% ± {ms['cv_macro_f1_std']*100:.2f}%** | "
            f"{ms['cv_accuracy_mean']*100:.2f}% ± {ms['cv_accuracy_std']*100:.2f}% |"
        )

    lines.extend([
        "",
        "### 2.3 Per-Cuisine Precision, Recall, and F1 on Every Outer Fold (Tertiary Metric Hierarchy Requirement)",
        "",
        "| Model | Fold | `indian` P / R / F1 (%) | `italian` P / R / F1 (%) | `mexican` P / R / F1 (%) | Fold Macro-F1 (%) | Fold Accuracy (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_name in main_keys:
        short_m = m_name.split(":")[0]
        for f_i, r in enumerate(models[m_name]["per_fold_rows"]):
            lines.append(
                f"| **{short_m}** | {f_i} | "
                f"{r['precision_indian']*100:.2f} / {r['recall_indian']*100:.2f} / {r['f1_indian']*100:.2f} | "
                f"{r['precision_italian']*100:.2f} / {r['recall_italian']*100:.2f} / {r['f1_italian']*100:.2f} | "
                f"{r['precision_mexican']*100:.2f} / {r['recall_mexican']*100:.2f} / {r['f1_mexican']*100:.2f} | "
                f"**{r['macro_f1']*100:.2f}%** | {r['accuracy']*100:.2f}% |"
            )

    lines.extend([
        "",
        "### 2.4 Pooled Out-of-Fold Confusion Matrices (Quaternary Metric Hierarchy Requirement; Rows = True, Cols = Predicted)",
        "",
        "| Model | True Cuisine | Pred `indian` | Pred `italian` | Pred `mexican` | Pooled Precision (%) | Pooled Recall (%) | Pooled F1 (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    cuisines = ["indian", "italian", "mexican"]
    for m_name in main_keys:
        ms = models[m_name]
        cm = ms["oof_confusion_matrix"]
        for ci, c in enumerate(cuisines):
            lines.append(
                f"| **{m_name.split(':')[0]}** | `{c}` | {cm[ci][0]:,} | {cm[ci][1]:,} | {cm[ci][2]:,} | "
                f"{ms[f'oof_precision_{c}']*100:.2f}% | {ms[f'oof_recall_{c}']*100:.2f}% | **{ms[f'oof_f1_{c}']*100:.2f}%** |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Pooled Out-of-Fold Paired Statistical Comparisons & Recall Regression Guard Checks",
        "",
        "| Paired Comparison (`Model A vs. Model B`) | Pooled $\\Delta$ Macro-F1 (`A - B`) [95% Bootstrap CI] | CI Excludes Zero? | Pooled $\\Delta$ Accuracy (`A - B`) [95% Bootstrap CI] | A Right, B Wrong ($b$) | A Wrong, B Right ($c$) | Net Recipes ($b-c$) | McNemar $\\chi^2$ (CC) | Exact Binomial $p$-value | $\\Delta$ Recall (`IN / IT / MX`, pp) | Recall Regression Guard ($>1\\text{ pp}$ drop)? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for comp_label, st in paired.items():
        f1_ci = st["bootstrap_delta_macro_f1_ci95"]
        acc_ci = st["bootstrap_delta_accuracy_ci95"]
        mc = st["mcnemar"]
        rd = st["recall_diffs_oof"]
        r_str = f"`{rd['indian']*100:+.2f} / {rd['italian']*100:+.2f} / {rd['mexican']*100:+.2f}`"
        guard_str = (
            f"**FLAGGED ({', '.join(st['regression_guard_flagged_cuisines'])})**"
            if st["regression_guard_triggered"]
            else " Clear (No drop > 1 pp)"
        )
        excl_zero = "**YES**" if not st["macro_f1_ci_includes_zero"] else "No (includes 0)"
        lines.append(
            f"| **{comp_label}** | **{st['delta_oof_macro_f1']*100:+.2f} pp** [{f1_ci[0]*100:+.2f}, {f1_ci[1]*100:+.2f}] | "
            f"{excl_zero} | **{st['delta_oof_accuracy']*100:+.2f} pp** [{acc_ci[0]*100:+.2f}, {acc_ci[1]*100:+.2f}] | "
            f"{mc['a_right_b_wrong']} | {mc['a_wrong_b_right']} | **{mc['net_difference_a_minus_b']:+d}** | "
            f"{mc['chi2_continuity_corrected']:.2f} | `{mc['p_value_exact_binomial']:.4g}` | {r_str} | {guard_str} |"
        )

    lines.extend([
        "",
        "- **Explicit Regression Guard Note**:",
        "  - None of the primary model comparisons (`M5 vs. M0`, `M4 vs. M1`, `M4 vs. M2`, `M3 vs. M1`, `M3 vs. M2`) trigger the recall regression guard.",
        "  - In the supplementary internal breakdown `M5 (All 22 Network Features) vs. Abl_M5a (3 Pair Log-Odds Features Only)`, adding the 19 D3 structural features boosts `indian` recall by **+8.24 pp** (`83.65% -> 91.89%`) and `mexican` recall by **+7.05 pp** (`85.18% -> 92.23%`), while slightly reducing `italian` recall by **-1.13 pp** (`97.48% -> 96.35%`) because `Abl_M5a` routes all 676 zero-pair fallback recipes to the majority class `italian` (inflating `italian` recall at the expense of `italian` precision, which rises from `86.27%` in `Abl_M5a` to `93.10%` in `M5`). We explicitly flag this `-1.13 pp` Italian recall shift per the Regression Guard rule.",
        "",
        "---",
        "",
        "## 4. Feature-Group Ablation Study (Removing Each Group from `M4` and Decomposing `M5`)",
        "",
        "| Ablation Model | Features Kept | CV Macro-F1 (Mean ± SD) | Pooled OOF Macro-F1 | $\\Delta$ Macro-F1 vs. `M4` [95% Bootstrap CI] | CV Accuracy (Mean ± SD) | $\\Delta$ Accuracy vs. `M4` [95% Bootstrap CI] | Net Recipes vs. `M4` | McNemar $p$ vs. `M4` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    m4_ms = models["M4: Full Stack — NB + All Network Features (25 features)"]
    lines.append(
        f"| **M4: Full Stack (Reference for Ablations)** | 25 | **{m4_ms['cv_macro_f1_mean']*100:.2f}% ± {m4_ms['cv_macro_f1_std']*100:.2f}%** | "
        f"**{m4_ms['oof_pooled_macro_f1']*100:.2f}%** | 0.00 pp [ref] | "
        f"**{m4_ms['cv_accuracy_mean']*100:.2f}% ± {m4_ms['cv_accuracy_std']*100:.2f}%** | 0.00 pp [ref] | 0 | `1.000` |"
    )

    for ab in ablations:
        f1_ci = ab["delta_macro_f1_ci95_vs_M4"]
        acc_ci = ab["delta_accuracy_ci95_vs_M4"]
        n_feats = models[ab["ablation_model"]]["per_fold_rows"][0].get("n_features", "")
        lines.append(
            f"| **{ab['ablation_model']}** | {n_feats} | {ab['cv_macro_f1_mean']*100:.2f}% ± {ab['cv_macro_f1_std']*100:.2f}% | "
            f"{ab['oof_pooled_macro_f1']*100:.2f}% | **{ab['delta_macro_f1_vs_M4']*100:+.2f} pp** [{f1_ci[0]*100:+.2f}, {f1_ci[1]*100:+.2f}] | "
            f"{ab['cv_accuracy_mean']*100:.2f}% ± {ab['cv_accuracy_std']*100:.2f}% | "
            f"**{ab['delta_accuracy_vs_M4']*100:+.2f} pp** [{acc_ci[0]*100:+.2f}, {acc_ci[1]*100:+.2f}] | "
            f"{ab['net_recipes_vs_M4']:+d} | `{ab['mcnemar_p_vs_M4']:.4g}` |"
        )

    lines.extend([
        "",
        "### Key Insights from the Ablation Study",
        "1. **Removing Naive Bayes log-posteriors (`Abl1 = M5`)** causes the largest drop (`-1.98 pp` Macro-F1, `-1.84 pp` Accuracy, `-254` net recipes), confirming that full-vocabulary unigram evidence remains the single strongest signal.",
        "2. **Among the network feature groups inside `M4`, D3 Hub Counts (`Abl4`) and D3 Louvain Community Structure (`Abl3`) contribute the largest positive gains**:",
        "   - **Removing D3 Hub Counts (`Abl4`)** drops `M4`'s Macro-F1 by **-0.14 pp** (`95% CI: [-0.29 pp, -0.01 pp]` — **95% bootstrap CI excludes zero!** `-14` net recipes).",
        "   - **Removing D3 Louvain Community Structure (`Abl3: max_cov & entropy`)** drops `M4`'s Macro-F1 by **-0.09 pp** (`-12` net recipes).",
        "   - **Removing D3 Bridge Counts (`Abl5`)** drops Macro-F1 by **-0.05 pp** (`-5` net recipes), and **removing Backbone Node Coverage & Recipe Length (`Abl6`)** drops Macro-F1 by **-0.03 pp** (`-7` net recipes).",
        "   - **Removing Backbone Pair Log-Odds (`Abl2`)** leaves `M4` virtually unchanged (`+0.01 pp` Macro-F1, `+1` recipe, $p = 1.00$), demonstrating that once NB log-posteriors and D3 graph structure features (hubs, communities, bridges, backbone coverage) are present, pair log-odds are redundant with unigrams, whereas **D3 structural features (`Abl2: 95.83%` Macro-F1) provide the entire +0.30 pp Macro-F1 lift over Naive Bayes**!",
        "3. **Inside the Pure Network-Only Scorer (`M5: 93.84%` Macro-F1), Pair Log-Odds and D3 Graph Summaries Are Strongly Complementary**:",
        "   - Pair log-odds alone (`Abl_M5a`, 3 features) achieve **90.19% Macro-F1**. D3 graph structural summaries alone without pair log-odds (`Abl_M5b`, 19 features) achieve **92.82% Macro-F1**. Combining both in `M5` (22 features) reaches **93.84% Macro-F1** (+3.65 pp over pair log-odds alone and +1.01 pp over D3 summaries alone).",
        "",
        "---",
        "",
        "## 5. Standardized Logistic Regression Coefficients (`M4` and `M5`)",
        "",
        "Because all input features are standardized to zero mean and unit variance (`StandardScaler` fitted strictly on each outer training fold), the L2 logistic regression coefficients below directly measure each feature's standardized log-odds effect (`Mean ± SD` across the 5 outer folds).",
        "",
        "### 5.1 Standardized Coefficients for `M4: Full Stack (25 Features)`",
        "",
        "| Feature Name | Feature Group | `indian` Weight (Mean ± SD) | `italian` Weight (Mean ± SD) | `mexican` Weight (Mean ± SD) | Max $|\\text{Mean Weight}|$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    group_labels = {
        "nb_log_prob": "1. Unigram NB Log-Posterior",
        "pair_log_odds": "2. Backbone Pair Log-Odds",
        "comm_cov_frac": "3a. Community Coverage Fraction",
        "comm_max_cov": "3b. Max Single-Community Coverage",
        "comm_entropy": "3c. Community Concentration (Entropy)",
        "hub_count": "4. D3 Top-10 Weighted-Degree Hubs",
        "bridge_count": "5. D3 Top-5 Pairwise Bridges",
        "backbone_ing_count": "6. Backbone Ingredient Count",
        "recipe_length": "7. Global Recipe Length",
    }

    def get_grp(fname: str) -> str:
        for prefix, glabel in group_labels.items():
            if fname.startswith(prefix):
                return glabel
        return "Other"

    for r in coefs["M4_full_stack_25_features"]:
        lines.append(
            f"| `{r['feature']}` | {get_grp(r['feature'])} | "
            f"`{r['indian_mean']:+.3f} ± {r['indian_std']:.3f}` | "
            f"`{r['italian_mean']:+.3f} ± {r['italian_std']:.3f}` | "
            f"`{r['mexican_mean']:+.3f} ± {r['mexican_std']:.3f}` | "
            f"**{r['max_abs_mean']:.3f}** |"
        )

    lines.extend([
        "",
        "### 5.2 Standardized Coefficients for `M5: Network-Only Fingerprint Scorer v2 (22 Features, No NB)`",
        "",
        "| Feature Name | Feature Group | `indian` Weight (Mean ± SD) | `italian` Weight (Mean ± SD) | `mexican` Weight (Mean ± SD) | Max $|\\text{Mean Weight}|$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for r in coefs["M5_network_only_22_features"]:
        lines.append(
            f"| `{r['feature']}` | {get_grp(r['feature'])} | "
            f"`{r['indian_mean']:+.3f} ± {r['indian_std']:.3f}` | "
            f"`{r['italian_mean']:+.3f} ± {r['italian_std']:.3f}` | "
            f"`{r['mexican_mean']:+.3f} ± {r['mexican_std']:.3f}` | "
            f"**{r['max_abs_mean']:.3f}** |"
        )

    lines.extend([
        "",
        "### Interpretation of Standardized Coefficients",
        "1. **In `M5` (Network-Only Scorer v2)**:",
        "   - The dominant predictors are the three **backbone pair log-odds scores** (`+2.836` for Indian, `+1.704` for Italian, `+1.521` for Mexican), directly reinforced by **backbone ingredient counts** (`+1.555` Indian, `+1.200` Italian, `+1.202` Mexican) and **Louvain community coverage fractions** (`+0.843` Indian, `+1.038` Italian, `+1.234` Mexican).",
        "   - **Community structure & bridges act as structural disambiguators**: Higher `comm_max_cov_indian` (`+0.283`) and `comm_entropy_indian` (`+0.226`) positively predict Indian, whereas `bridge_count_indian` (which counts cross-cuisine bridge ingredients shared between Indian and Italian/Mexican such as `garlic`, `onion`, `bay leaves`, `heavy cream`, `cinnamon`, `clove`) has a **negative weight for Indian (`-0.211 ± 0.055`) and positive weights for Italian (`+0.107`) and Mexican (`+0.105`)**! Why? Because when a recipe matches Indian backbone nodes primarily through shared cross-cuisine *bridge* ingredients rather than cuisine-specific Indian spices, the negative weight on `bridge_count_indian` discounts the spurious Indian backbone coverage.",
        "2. **In `M4` (Full Stack with NB)**:",
        "   - The three cross-fitted **NB log-posteriors** carry the largest standardized weights (`+4.812` Indian, `+4.567` Italian, `+3.558` Mexican).",
        "   - What role do the network features play alongside NB in `M4`? They provide a **structural complexity and hub/bridge baseline correction**:",
        "     - Naive Bayes multiplies per-ingredient likelihood ratios $\\prod_{u \\in R} P(u \\mid c)$. On longer recipes containing many generic Italian/Western pantry staples that belong to the large Italian backbone (`backbone_ing_count_italian` has weight `-1.105` on Italian in `M4`), uncalibrated Naive Bayes over-accumulates evidence for Italian. By pairing positive `comm_cov_frac_c` (`+0.273` IN, `+0.345` IT, `+0.605` MX) with negative multi-ingredient hub/bridge/count penalties (`backbone_ing_count_italian = -1.105`, `hub_count_indian = -0.313`, `bridge_count_indian = -0.331`, `comm_entropy_mexican = -0.251`), `M4` corrects Naive Bayes when generic hubs or cross-cuisine bridges dominate a recipe.",
        "",
        "---",
        "",
        "## 6. How Often Network Features Change Naive Bayes's Prediction — and Whether Those Changes Are Correct",
        "",
        "| Comparison (`Target Model vs. Base NB Model`) | Total OOF Recipes | Predictions Changed ($N$, %) | Changed Wrong $\\rightarrow$ Correct (**Fixed**, $N$ / % of changes) | Changed Correct $\\rightarrow$ Wrong (**Broken**, $N$ / % of changes) | Changed Wrong $\\rightarrow$ Diff Wrong ($N$ / % of changes) | **Net Correct Gain (`Fixed - Broken`)** | McNemar $p$-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for key_c, label_c in [
        ("M4_vs_M1_raw_NB", "**M4 (Full Stack, 25 feats) vs. M1 (Raw Naive Bayes)**"),
        ("M4_vs_M2_calibrated_NB", "**M4 (Full Stack, 25 feats) vs. M2 (Calibrated NB Stack, 3 feats)**"),
        ("M3_vs_M1_raw_NB", "**M3 (NB + Pair Log-Odds, 6 feats) vs. M1 (Raw Naive Bayes)**"),
        ("M3_vs_M2_calibrated_NB", "**M3 (NB + Pair Log-Odds, 6 feats) vs. M2 (Calibrated NB Stack, 3 feats)**"),
    ]:
        ch = nb_changes[key_c]
        mc_p = paired[
            "M4 (Full Stack: NB + All Network Features) vs. M1 (Naive Bayes Alone)"
            if key_c == "M4_vs_M1_raw_NB"
            else (
                "M4 (Full Stack: NB + All Network Features) vs. M2 (Calibrated NB Stack Alone)"
                if key_c == "M4_vs_M2_calibrated_NB"
                else (
                    "M3 (NB + Backbone Pair Log-Odds) vs. M1 (Naive Bayes Alone)"
                    if key_c == "M3_vs_M1_raw_NB"
                    else "M3 (NB + Backbone Pair Log-Odds) vs. M2 (Calibrated NB Stack Alone)"
                )
            )
        ]["mcnemar"]["p_value_exact_binomial"]
        lines.append(
            f"| {label_c} | {ch['n_total_recipes']:,} | **{ch['n_predictions_changed']}** ({ch['pct_predictions_changed']:.2f}%) | "
            f"**{ch['n_changed_to_correct_fixed']}** ({ch['pct_of_changes_correct']:.1f}%) | "
            f"{ch['n_changed_from_correct_to_wrong_broken']} ({ch['pct_of_changes_broke_correct']:.1f}%) | "
            f"{ch['n_changed_both_wrong']} ({ch['pct_of_changes_both_wrong']:.1f}%) | "
            f"**{ch['net_correct_gain']:+d} recipes** | `{mc_p:.4g}` |"
        )

    ch_m4_m1 = nb_changes["M4_vs_M1_raw_NB"]
    ch_m4_m2 = nb_changes["M4_vs_M2_calibrated_NB"]

    lines.extend([
        "",
        "### Breakdown of Prediction Flips (`M4` vs. `M1` and `M4` vs. `M2`)",
        f"- **Frequency of Prediction Changes**: Across the 13,823 OOF training recipes, `M4` changes `M1` (Naive Bayes)'s prediction on **224 recipes (1.62%)** (and changes `M2` Calibrated NB's prediction on **229 recipes, 1.66%**).",
        f"- **Accuracy of Changes**:",
        f"  - Against `M1`, **116 of the 224 changes (51.8%) fix a Naive Bayes error** (`Wrong -> Correct`), while **86 (38.4%) flip a correct NB prediction to wrong** (`Correct -> Wrong`), and **22 (9.8%)** move between two wrong classes, yielding a net gain of **+30 correct predictions** ($p = 0.0410$).",
        f"  - Against `M2` (which isolates the exact effect of adding the 22 network features to the 3 NB log-posteriors), **121 of the 229 changes (52.8%) fix a calibrated NB error**, while **88 (38.4%) flip a correct prediction to wrong** and **20 (8.7%)** shift between wrong classes, yielding a net gain of **+33 correct predictions** ($p = 0.0266$).",
        "- **Dominant Transition Patterns Fixed by Network Features (`M4 vs. M1`)**:",
        "  - `true=mexican: italian -> mexican` (**39 recipes fixed** vs. 12 broken in reverse): Recipes with Mexican community coherence (`comm_cov_frac_mexican = 0.92`, `comm_max_cov_mexican = 0.58`) that Naive Bayes misclassified as `italian` due to generic baking/savory staples (e.g., Recipe `1894`: `all-purpose flour, baking powder, black pepper, butter, chocolate, cooking spray, ground cinnamon, ground red pepper, large egg, salt, sugar, vanilla extract`, where `nb_log_prob_italian = -0.052` beat `nb_log_prob_mexican = -2.990`, but `M4` correctly flipped `italian -> mexican`).",
        "  - `true=mexican: indian -> mexican` (**32 recipes fixed** vs. 10 broken in reverse): E.g., Recipe `2271` (`cinnamon sticks, egg yolks, milk, shredded coconut, sugar`), where `cinnamon sticks` is an Indian top-10 weighted-degree hub (`hub_count_indian = 1`) that caused Naive Bayes to predict `indian` (`nb_log_prob_indian = -0.305`), but `M4`'s hub-count discount (`hub_count_indian` weight `-0.313`, ` +0.358` for Mexican) correctly flipped the prediction to `mexican`.",
        "  - `true=indian: italian -> indian` (**24 recipes fixed** vs. 7 broken in reverse): E.g., Recipe `2270` (`active dry yeast, bread flour, greek style plain yogurt, kosher salt, olive oil, salted butter, sugar, water`, naan bread), where Italian baking staples gave `nb_log_prob_italian = -0.565` vs. `nb_log_prob_indian = -0.865`, and `M4`'s penalty on generic Italian backbone node counts (`backbone_ing_count_italian = 6`) flipped it to `indian`.",
        "",
        "---",
        "",
        "## 7. Performance by Pre-Registered Fixed Recipe-Length Bins",
        "",
        "Using the four fixed recipe-length bins defined in `phase3_design.md` before running Phase 3 (`Short: 1–5`, `Medium-Short: 6–8`, `Medium-Long: 9–12`, `Long: 13+` normalized ingredients):",
        "",
        "| Pre-Registered Recipe-Length Bin | Recipes ($N$, Share) | Mean Ings | `M0` (Orig Scorer) Macro-F1 / Acc | `M1` (Naive Bayes) Macro-F1 / Acc | `M2` (Calib NB) Macro-F1 / Acc | `M3` (NB + Pair) Macro-F1 / Acc | `M4` (Full Stack) Macro-F1 / Acc | `M5` (Network-Only v2) Macro-F1 / Acc | `M5 - M0` ($\\Delta$ F1 / Net Recipes) | `M4 - M1` ($\\Delta$ F1 / Net Recipes) | `M4 - M2` ($\\Delta$ F1 / Net Recipes) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for b in length_bins:
        d_m5_m0_f1 = (b["M5_macro_f1"] - b["M0_macro_f1"]) * 100
        d_m4_m1_f1 = (b["M4_macro_f1"] - b["M1_macro_f1"]) * 100
        d_m4_m2_f1 = (b["M4_macro_f1"] - b["M2_macro_f1"]) * 100
        lines.append(
            f"| **{b['length_bin']}** | {b['n_recipes']:,} ({b['pct_recipes']:.1f}%) | {b['mean_ingredients']:.2f} | "
            f"{b['M0_macro_f1']*100:.2f}% / {b['M0_accuracy']*100:.2f}% | "
            f"{b['M1_macro_f1']*100:.2f}% / {b['M1_accuracy']*100:.2f}% | "
            f"{b['M2_macro_f1']*100:.2f}% / {b['M2_accuracy']*100:.2f}% | "
            f"{b['M3_macro_f1']*100:.2f}% / {b['M3_accuracy']*100:.2f}% | "
            f"**{b['M4_macro_f1']*100:.2f}% / {b['M4_accuracy']*100:.2f}%** | "
            f"**{b['M5_macro_f1']*100:.2f}% / {b['M5_accuracy']*100:.2f}%** | "
            f"**{d_m5_m0_f1:+.2f} pp** (`{b['m5_vs_m0_net_recipes']:+d}`) | "
            f"**{d_m4_m1_f1:+.2f} pp** (`{b['m4_vs_m1_net_recipes']:+d}`) | "
            f"**{d_m4_m2_f1:+.2f} pp** (`{b['m4_vs_m2_net_recipes']:+d}`) |"
        )

    lines.extend([
        "",
        "### Key Findings Across Recipe-Length Bins",
        "1. **`M5` (Network-Only v2) crushes `M0` (Original Scorer) across every recipe length**, with the largest Macro-F1 gains on **Short (`1–5` ingredients: +16.97 pp Macro-F1, +13.40 pp Accuracy)** and **Medium-Short (`6–8` ingredients: +13.03 pp Macro-F1, +10.32 pp Accuracy)** recipes, where pair matches are sparse or zero (`M0` falls back to majority class `italian`), but single-ingredient D3 community/hub/backbone node coverage features still fire!",
        "2. **`M4` (Full Stack) achieves its largest Macro-F1 gains over `M1` (Naive Bayes) and `M2` (Calibrated NB) on Short and Medium recipes (`1–12` ingredients)**:",
        "   - **Short (`1–5` ingredients)**: `M4` reaches **87.82% Macro-F1** vs. `M1` **87.03% (+0.79 pp)** and `M2` **86.44% (+1.38 pp)** (`+7` and `+10` net recipes).",
        "   - **Medium-Short (`6–8` ingredients)**: `M4` reaches **93.54% Macro-F1** vs. `M1` **92.87% (+0.67 pp)** and `M2` **92.82% (+0.72 pp)** (`+12` net recipes).",
        "   - **Medium-Long (`9–12` ingredients)**: `M4` reaches **96.55% Macro-F1** vs. `M1` **96.21% (+0.35 pp)** and `M2` **96.32% (+0.24 pp)** (`+14` and `+11` net recipes).",
        "   - **Long (`13+` ingredients)**: All NB-based models perform similarly (`M1: 97.84%`, `M2: 97.77%`, `M3: 97.86%`, `M4: 97.78%` Macro-F1; `M4` and `M2` have `0` net recipe difference on $13+$ ingredient recipes), while the pure network-only model `M5` reaches **97.58% Macro-F1 (97.54% Accuracy)** — within `0.26 pp` of Naive Bayes using zero unigram NB features!",
        "",
        "---",
        "",
        "## Appendix: Supplementary Weighted-F1 Metrics (Metric Hierarchy Level 5)",
        "",
        "| Model | 5-Fold CV Weighted-F1 (Mean ± SD) | Pooled OOF Weighted-F1 |",
        "| :--- | :---: | :---: |",
    ])

    for m_name, ms in models.items():
        lines.append(
            f"| **{m_name}** | {ms['cv_weighted_f1_mean_appendix']*100:.2f}% ± {ms['cv_weighted_f1_std_appendix']*100:.2f}% | "
            f"{ms['oof_pooled_weighted_f1_appendix']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "> **STOPPING POINT**: Per your instructions, the 3,456 held-out test recipes remain completely untouched, and all D1–D3 and original D4 files are unmodified. Please review `outputs/d4_experiments/phase3/phase3_design.md`, `outputs/d4_experiments/phase3/phase3_report.md`, and `outputs/d4_experiments/phase3/results.csv`.",
        "",
    ])

    md = "\n".join(lines)
    (PHASE3_DIR / "phase3_report.md").write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    format_phase3_report()
    print(f"Wrote {PHASE3_DIR / 'phase3_report.md'}")
