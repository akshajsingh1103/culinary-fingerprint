# Culinary Fingerprint — Interactive Network-Based Cuisine Classifier

**Culinary Fingerprint** is a self-contained Streamlit web application that predicts whether a recipe belongs to **Indian**, **Italian**, or **Mexican** cuisine using our **Network-informed (v2)** Fingerprint Scorer (22-feature stacked L2 logistic regression over validated co-occurrence backbones, hubs, Louvain communities, and bridges), and benchmarks **Baseline (v1)** vs. **Network-informed (v2)** on the 3,456 held-out test recipes.

> **Supported Cuisines Notice**: Only **Indian**, **Italian**, and **Mexican** are supported; every recipe is forced into one of these three cuisines.

---

## Quick Start (From Repository Root)

1. **(Optional) Export or refresh the self-contained runtime artifacts**:
   ```bash
   python outputs/app/export_artifacts.py
   ```
   *(All exported artifacts are already included in `outputs/app/artifacts/`.)*

2. **Launch the Streamlit app**:
   ```bash
   streamlit run outputs/app/app.py
   ```

---

## Directory Structure

```text
outputs/app/
  app.py                  # Streamlit application entry point (Predict | Results | How it works)
  export_artifacts.py     # One-time artifact exporter from the training-only D1-D4 pipeline
  scorer_runtime.py       # Self-contained runtime engine (normalization, Scorer v1 & Scorer v2)
  assets/                 # Offline hand-drawn ingredient line-art background SVG pattern
  artifacts/              # Exported backbones, hubs, communities, bridges, v2 weights, vocab, test results
  .streamlit/config.toml  # Custom dark theme configuration
  README.md               # App documentation
  requirements_app.txt    # Python package requirements for the app
```

---

## Features

- **Search-as-You-Type Ingredient Multiselect**: Ordered by training recipe frequency so common What's Cooking staples appear first.
- **Paste an Ingredient List**: Paste comma- or newline-separated ingredient strings; automatically normalizes using the pipeline's 5-stage deterministic normalizer and shows which items were recognized vs. unrecognized.
- **"Surprise me" Button**: Loads a random real recipe from the What's Cooking dataset and hides its true cuisine until you click **Predict**, then reveals the **Actual cuisine** badge above the result card.
- **Centered Network-Informed Result Card**: Displays the predicted cuisine and plain-text confidence (formatted as `>99%` above `99.9%`) in a single centered card with a soft cuisine-coloured glow (saffron Indian, emerald Italian, crimson Mexican) and per-cuisine confidence bars.
- **"Why This Prediction" Panel**: Inspect matched backbone pairs per cuisine, Top-10 weighted-degree hub ingredients found, Louvain community coverage & maximum single-community concentration, cross-cuisine bridge ingredients, and unrecognized items.
- **Interactive Offline Network Graph**: Renders an interactive force-directed HTML graph of the top matched backbone edges for the predicted cuisine (completely offline with zero CDN dependencies).
- **Held-Out Test Results & Pipeline Overview Tabs**: Compare **Baseline (v1)** vs. **Network-informed (v2)** on the 3,456 held-out test recipes (Macro-F1, accuracy, per-cuisine recall, paired McNemar/bootstrap tests, and confusion matrices) and read a plain-language walkthrough of Deliverables 1–4.
