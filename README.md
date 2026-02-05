# Agentic Ranking Systems for Skewed Engagement Data

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Runs in ~5-10 seconds. Produces console output from all 7 agents plus 3 visualizations in `output/`.

---

## 1. Overview & Architecture

### What This Project Proves

A single preprocessing decision — how you bin continuous engagement data — cascades through feature engineering, model training, and ranking quality in ways that are invisible until you measure them. This project builds a complete recommendation pipeline, runs it twice with different binning strategies (equal-width vs equal-height), and quantifies exactly where and why one fails.

**The core claim:** Equal-width binning on heavy-tailed data creates a cascading failure that degrades Precision@K, destabilizes cross-validation, and introduces structural fairness violations — all before the model even trains.

### Why This Exists

Interview projects need to demonstrate more than code. They need to show that you understand *why* you made each decision, that you can trace a metric change back to a root cause, and that you think about downstream consequences. This project is designed to support a 30-60 minute technical walkthrough where every layer — from data generation to evaluation — has a defensible rationale.

### Pipeline Architecture

```
┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│ DataAgent  │──>│ BinningAgent │──>│ FeatureAgent │──>│ ModelAgent │
└────────────┘   └──────────────┘   └──────────────┘   └────────────┘
                                                              │
┌────────────────┐   ┌──────────────────┐   ┌─────────────────┘
│ ExplainerAgent │<──│ EvaluationAgent  │<──│ RankingAgent    │
└────────────────┘   └──────────────────┘   └─────────────────┘
```

Seven agents, sequential execution, shared state via dictionary passthrough. Each agent receives `**kwargs` from its predecessor, does its work, and returns an enriched dictionary (`{**kwargs, ...new_keys}`) for the next agent.

### Why This Pattern

**Why agents instead of a monolithic script?** Three reasons:

1. **Debuggability.** When Precision@K drops, you can isolate the problem to a specific agent. Was the data generated wrong? Were bins miscalculated? Did feature engineering lose information? Each agent prints its own diagnostics (`utils/printing.py`), making the pipeline self-documenting at runtime.

2. **Testability.** Each agent has a single `run(**kwargs) -> dict` contract (`agents/base_agent.py:12-14`). You can unit test any agent by constructing its expected input dictionary. No global state, no side effects beyond print statements.

3. **Extensibility.** Adding a third binning strategy (e.g., KDE-based or Bayesian) means modifying `BinningAgent` and propagating one new column — not rewriting the pipeline. The downstream agents don't care *how* bins were created, only that `ew_bin` and `eh_bin` columns exist.

**Why sequential, not parallel?** The pipeline has linear data dependencies — you can't bin data that hasn't been generated, can't encode features without bins, can't train without features. The `main.py:39-41` loop makes this explicit:

```python
results = {}
for agent in agents:
    results = agent.run(**results)
```

This is intentionally simple. In production, independent branches (e.g., feature engineering for two strategies) could parallelize, but for a demo the clarity of sequential execution is worth more than the ~2 seconds you'd save.

### Project Structure

```
recommendation-engine/
├── main.py                     # Orchestrator: runs 7 agents, generates plots
├── requirements.txt            # numpy, pandas, scikit-learn, scipy, matplotlib
├── agents/
│   ├── base_agent.py           # Abstract base: name + run() contract
│   ├── data_agent.py           # Synthetic lognormal engagement data
│   ├── binning_agent.py        # Equal-width (np.linspace) vs equal-height (pd.qcut)
│   ├── feature_agent.py        # One-hot encoding with sparsity analysis
│   ├── model_agent.py          # Logistic regression, coefficient tables
│   ├── ranking_agent.py        # Per-entity top-K ranking, Precision@K
│   ├── evaluation_agent.py     # Cross-val, entropy, Spearman, summary table
│   └── explainer_agent.py      # Plain-language narrative for stakeholders
├── utils/
│   ├── constants.py            # All hyperparameters in one place
│   └── printing.py             # Formatted tables, bars, distribution summaries
└── output/
    ├── engagement_distribution.png
    ├── bin_comparison.png
    └── coefficient_comparison.png
```

### Configuration at a Glance

All hyperparameters live in `utils/constants.py` — no magic numbers buried in agent code:

| Constant | Value | Why This Value |
|----------|-------|----------------|
| `RANDOM_SEED` | 42 | Reproducibility across all agents |
| `NUM_ENTITIES` | 500 | Large enough for statistical stability, small enough to run in seconds  |
| `NUM_ITEMS` | 1,000 | 2:1 item-to-entity ratio mimics real catalogs |
| `INTERACTIONS_PER_ENTITY` | 20 | 10,000 total interactions — enough for reliable model training |
| `NUM_BINS` | 10 | Standard discretization granularity; enough to show the failure clearly |
| `TOP_K` | 5 | Typical recommendation slate size |
| `TEST_SIZE` | 0.2 | Standard 80/20 split |
| `POSITIVE_QUANTILE` | 0.90 | Creates ~10% positive rate (realistic imbalanced classification) |
| `MU` | 3.0 | Lognormal mean → mode ~5s, median ~20s, mean ~41s |
| `SIGMA` | 1.2 | Controls tail heaviness (skewness ~4.7) |
| `ENGAGEMENT_CLIP_MAX` | 3,600 | 1-hour cap prevents extreme outliers from distorting bins |
| `CV_FOLDS` | 5 | Standard for model stability assessment |

---
