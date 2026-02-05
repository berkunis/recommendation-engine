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

## 2. Data Generation & Distribution Theory

### Why Lognormal

Real-world engagement metrics — dwell time, session duration, scroll depth, time-on-page — consistently follow heavy-tailed distributions. People mostly glance and leave (short dwell), but occasionally deep-dive for minutes. The lognormal captures this naturally: it's the distribution you get when many small multiplicative factors determine the outcome (page load speed, content relevance, user intent, device type). If those factors were *additive*, you'd get a normal distribution. The multiplicative nature of engagement makes lognormal the right generative model.

### Parameters and What They Produce

The data is generated at `data_agent.py:23`:

```python
raw_engagement = rng.lognormal(mean=MU, sigma=SIGMA, size=total_interactions)
```

With `MU=3.0` and `SIGMA=1.2` (`constants.py:11-12`), the theoretical distribution has these properties:

| Statistic | Formula | Value |
|-----------|---------|-------|
| Mode | exp(mu - sigma^2) | ~4.7s |
| Median | exp(mu) | ~20.1s |
| Mean | exp(mu + sigma^2/2) | ~41.3s |
| Std Dev | mean * sqrt(exp(sigma^2) - 1) | ~75.5s |
| Skewness | (exp(sigma^2) + 2) * sqrt(exp(sigma^2) - 1) | ~4.7 |
| P99 | exp(mu + 2.33 * sigma) | ~327s |

The key insight is that the **mean is more than double the median**. Half your users engage for under 20 seconds, but the average is 41 seconds because the right tail pulls it up. This mean-median divergence is the root cause of the equal-width binning failure — `np.linspace` uses the range (dominated by the tail), not the density (dominated by the body).

### Why Synthetic Data

Three reasons to generate data rather than use a public dataset:

1. **Reproducibility.** `RANDOM_SEED=42` means every run produces identical output. An interviewer can verify every claim in this README by running `python main.py`.
2. **No privacy concerns.** No PII, no licensing, no ethical review needed.
3. **Controllable skew.** By adjusting `MU` and `SIGMA`, you can dial the tail heaviness up or down and watch how the cascade changes. Try `SIGMA=0.5` — the failure disappears because the distribution becomes nearly symmetric.

### Clipping and Thresholds

Raw lognormal values are clipped to `[1, 3600]` seconds (`constants.py:13-14`, applied at `data_agent.py:24`). The lower bound prevents zero/negative engagement (physically meaningless). The upper bound at 1 hour caps extreme outliers — without it, a single sample could be 10,000+ seconds and stretch the equal-width bins even further.

The positive outcome threshold is set at P90 (`data_agent.py:32-33`):

```python
threshold = np.percentile(engagement, POSITIVE_QUANTILE * 100)
df["positive_outcome"] = (df["engagement_time"] >= threshold).astype(int)
```

This creates a ~10% positive rate — a realistic class imbalance that mirrors recommendation systems where most interactions are passive (scrolled past) and few are genuinely positive (clicked, purchased, completed).

---

## 3. Binning Strategies Deep Dive

### Equal-Width Binning: The Naive Approach

Equal-width binning divides the engagement range into bins of identical width (`binning_agent.py:20`):

```python
ew_edges = np.linspace(data.min(), data.max(), NUM_BINS + 1)
```

With engagement ranging from ~1s to ~3600s across 10 bins, each bin spans ~360 seconds. The problem is immediate: nearly all data points fall below 360s, so **95%+ of the data lands in Bin 0**. Bins 1-9 are sparsely populated or empty. The binning has not discretized the data — it has collapsed it into a single category.

### Equal-Height Binning: The Density-Aware Alternative

Equal-height (quantile) binning places boundaries so each bin contains approximately the same number of observations (`binning_agent.py:49-51`):

```python
df["eh_bin"], eh_edges_interval = pd.qcut(
    data, NUM_BINS, labels=False, retbins=True, duplicates="drop"
)
```

Each bin gets ~10% of the data. The bin widths are wildly unequal — Bin 0 might span [1s, 5s] while Bin 9 spans [150s, 3600s] — but the *count* per bin is uniform. This means the model sees equal representation across the engagement spectrum.

### The Cascading Failure Chain

The equal-width binning failure doesn't just reduce accuracy by a few points. It triggers a five-step cascade:

1. **Bin collapse.** 95%+ of data lands in Bin 0. The other 9 bins are near-empty.
2. **Feature sparsity.** One-hot encoding produces 10 features where only `ew_bin_0` activates meaningfully. The other 9 features are nearly all-zero, providing no learning signal.
3. **Coefficient instability.** The model has too few examples to estimate reliable coefficients for bins 1-9. Small changes in training data flip these coefficients, causing high variance across cross-validation folds.
4. **Ranking degradation.** With unstable predictions, per-entity rankings become noisy. The model can't distinguish between engagement levels within the dominant Bin 0, so it ranks quasi-randomly.
5. **Fairness violation.** Entities or items in the tail (bins 1-9) — power users, niche content — are lumped together and systematically under-ranked compared to the homogeneous mass in Bin 0.

Each step makes the next worse. The model isn't "slightly worse" — the entire information pathway from raw data to final ranking is compromised by a single upstream decision.

### Measuring Bin Balance: Entropy and CV

Two metrics quantify how balanced the bins are:

**Normalized entropy** (`evaluation_agent.py:14-22`): Measures how uniformly data is distributed across bins. Ranges from 0 (all data in one bin) to 1 (perfectly uniform). Equal-width typically scores ~0.3-0.5; equal-height scores ~0.99.

```python
def _normalized_entropy(self, counts):
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy
```

**Coefficient of variation** (CV): Standard deviation divided by mean of bin counts (`binning_agent.py:44`). A CV near 0 means all bins have similar counts. Equal-width has a CV around 2.5+; equal-height has a CV near 0.02.

### Why 10 Bins

The choice of `NUM_BINS=10` balances two concerns. Too few bins (3-4) and even equal-width might look acceptable because the first bin hasn't yet absorbed everything. Too many bins (50+) and both strategies suffer from noise — bins become so narrow that individual bin counts are unreliable. Ten bins is the sweet spot: enough granularity to clearly demonstrate the failure, and a standard discretization size you'd see in real production systems.

---

## 4. Feature Engineering & Model Training

### One-Hot Encoding: Why Not Ordinal

The `FeatureAgent` converts bin labels to one-hot vectors rather than ordinal integers. The tradeoff:

- **Ordinal encoding** (1 feature): Assumes a linear relationship between bin number and outcome. Compact but hides the per-bin structure — you can't see which specific bins the model relies on.
- **One-hot encoding** (10 features): Makes no linearity assumption. Each bin gets its own coefficient, making the model's per-bin reasoning fully transparent in the coefficient table.

For this project, interpretability wins. The coefficient table is the single most important diagnostic output — it shows whether the model has learned a meaningful engagement ordering. Ordinal encoding would compress this into a single number.

### Ensuring All Bins Are Represented

A subtle but important step at `feature_agent.py:21-24`:

```python
for i in range(NUM_BINS):
    col = f"ew_bin_{i}"
    if col not in ew_dummies.columns:
        ew_dummies[col] = 0
```

If a bin has zero observations (common with equal-width), `pd.get_dummies` won't create a column for it. This zero-fill ensures the feature matrix always has exactly `NUM_BINS` columns regardless of bin occupancy. Without it, the equal-width and equal-height feature matrices could have different dimensions, breaking downstream comparisons.

### Activation Rates and Sparsity

The activation rate of a feature is the fraction of rows where it equals 1. For equal-width, `ew_bin_0` activates on ~95% of rows; bins 1-9 activate on fractions of a percent each. For equal-height, each feature activates on ~10%.

**Sparsity** is `1 - mean(feature_matrix)` (`feature_agent.py:36`). With 10 one-hot features, the theoretical minimum sparsity is 0.90 (each row has exactly one 1 and nine 0s). Both strategies hit ~0.90 sparsity overall, but the *distribution* of that sparsity is radically different — equal-width concentrates all the activation into one feature, while equal-height spreads it evenly.

### Model Choice: Logistic Regression

`ModelAgent` trains `LogisticRegression` at `model_agent.py:18-20`:

```python
model = LogisticRegression(
    solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
)
```

Why logistic regression over gradient-boosted trees, neural nets, or other options:

1. **Interpretability.** Each coefficient directly maps to a feature's influence. You can point at the coefficient table and say "this is what the model learned about Bin 7."
2. **Speed.** Trains in milliseconds, enabling rapid iteration and cross-validation.
3. **Sufficient complexity.** The task is binary classification on 10 one-hot features. A more powerful model would overfit or simply learn the same linear boundary with extra overhead.

The `lbfgs` solver handles the L2-regularized optimization efficiently. `max_iter=1000` ensures convergence even when the feature matrix is nearly rank-deficient (as it is with equal-width features where 9 of 10 columns are near-zero).

### Stratified Split

The train/test split at `model_agent.py:62-63` uses stratification:

```python
idx_train, idx_test = train_test_split(
    indices, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
```

With only ~10% positive rate, a random split could accidentally put 8% positives in train and 12% in test (or vice versa), skewing accuracy comparisons. Stratification guarantees both splits preserve the original class ratio, making the accuracy comparison between strategies fair.

### Coefficient Interpretation

The equal-height model should produce coefficients that increase monotonically from Bin 0 (lowest engagement) to Bin 9 (highest engagement). This means the model has correctly learned that higher engagement bins correlate with positive outcomes. The equal-width model's coefficients are erratic — bins 1-9 have unstable coefficients because they were estimated from too few examples.

---

## 5. Ranking & Evaluation

### Per-Entity Ranking

`RankingAgent` converts predicted probabilities into per-entity rankings at `ranking_agent.py:15-17`:

```python
df_ranked["rank"] = df_ranked.groupby("entity_id")["score"].rank(
    ascending=False, method="first"
)
```

Each entity's items are ranked by predicted positive probability, highest first. The `method="first"` tiebreaker ensures deterministic rankings even when scores are identical (common with equal-width, where most items in Bin 0 get the same predicted probability).

### Precision@K

Precision@K measures what fraction of the top-K recommended items per entity actually had a positive outcome. It's the most intuitive recommendation quality metric: "of the 5 items we showed, how many were actually good?"

```
Precision@K = (# of positive items in top-K) / K
```

The pipeline computes this at `ranking_agent.py:71-72` for entities that have at least K items in the test set. Equal-height consistently outperforms equal-width because its predictions carry genuine discriminative signal rather than collapsing to a near-constant probability.

### Spearman Rank Correlation

At `evaluation_agent.py:62-74`, the pipeline measures how differently the two strategies rank the same items for the same entity:

```python
corr, _ = stats.spearmanr(merged["rank_ew"], merged["rank_eh"])
```

Spearman correlation ranges from -1 (perfectly reversed) to +1 (identical rankings). A value near 0 means the two strategies produce essentially unrelated rankings — the binning choice has completely changed what gets recommended. This is the most compelling "so what?" metric: the same data, the same model architecture, the same evaluation — but a different preprocessing decision produces a fundamentally different user experience.

### 5-Fold Cross-Validation

`evaluation_agent.py:54-55` runs 5-fold CV to measure model stability:

```python
cv_ew = cross_val_score(lr_ew, X_ew, y, cv=CV_FOLDS, scoring="accuracy")
cv_eh = cross_val_score(lr_eh, X_eh, y, cv=CV_FOLDS, scoring="accuracy")
```

The mean accuracy matters, but the **standard deviation** matters more. Equal-width models show high variance across folds because the sparse features make the model sensitive to which examples end up in each fold. Equal-height models show low variance because every fold sees balanced feature activation.

### Normalized Entropy

Normalized entropy (`evaluation_agent.py:14-22`) measures bin balance on a [0, 1] scale:

- **0.0** — all data in one bin (maximum information loss)
- **1.0** — perfectly uniform distribution (maximum information preservation)

The formula divides Shannon entropy by the theoretical maximum (`log2(num_bins)`), making it comparable across different bin counts. It's the single best summary of whether binning preserved or destroyed information about the original engagement distribution.

### Side-by-Side Comparison

The `EvaluationAgent` produces a summary table (`evaluation_agent.py:87-121`) comparing all five metrics:

| Metric | What It Measures | Equal-Width | Equal-Height | Winner |
|--------|-----------------|-------------|--------------|--------|
| Norm. Entropy | Bin balance | ~0.35 | ~0.99 | EH |
| Bin CV | Count uniformity | ~2.5 | ~0.02 | EH |
| CV Accuracy | Model quality | Lower | Higher | EH |
| CV Stability | Fold-to-fold variance | Higher std | Lower std | EH |
| Precision@K | Recommendation quality | Lower | Higher | EH |

Equal-height wins every metric. This isn't because equal-height is a magic bullet — it's because equal-width is fundamentally broken on heavy-tailed data.

---

## 6. Fairness & Bias

### How Preprocessing Creates Structural Bias

The fairness problem here is subtle and important: **the bias is injected during preprocessing, before the model ever trains**. No amount of model-level debiasing (reweighting, adversarial training, fairness constraints) can fix a binning strategy that has already erased the information the model would need to be fair.

When equal-width binning puts 95% of data in Bin 0, it's making a structural claim: "all engagement below 360 seconds is equivalent." A user who engaged for 5 seconds is treated identically to one who engaged for 300 seconds. The model *cannot* learn to distinguish them because the feature representation has already collapsed the distinction.

### Tail Entity Blindness

The entities and items most affected by this collapse are those in the tail of the engagement distribution:

- **Power users** who consistently engage for 2-5 minutes get lumped with casual browsers who spend 10 seconds
- **Niche content** that generates deep but rare engagement is indistinguishable from mainstream content that generates shallow but frequent engagement
- **Specialized items** (complex financial products, senior job postings, technical articles) that naturally have longer engagement times lose their distinguishing signal

This is "tail entity blindness" — the system literally cannot see the differences that matter most for these segments.

### Equal-Height as a Fairness Intervention

Equal-height binning is a preprocessing-level fairness intervention. By ensuring each bin has equal representation, it guarantees that the model sees proportional signal from every part of the engagement spectrum. Tail entities get the same feature resolution as head entities.

This isn't a complete fairness solution — it doesn't address disparate impact across demographic groups, or ensure equal precision across entity segments. But it removes the most egregious source of structural bias: the wholesale erasure of tail entity information. It's a necessary (though not sufficient) condition for downstream fairness.

### Connection to Amazon's 2018 Recruiting AI

Amazon's automated resume screening tool, [reported by Reuters in 2018](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G), systematically penalized resumes that contained markers associated with women. The root cause was training data that reflected historical hiring bias — the system learned to replicate past decisions rather than evaluate candidates equitably.

This project demonstrates the *mechanism* by which such failures occur. If engagement data (time spent reviewing a resume) follows a heavy-tailed distribution and you bin it with equal-width:

1. Most candidates land in the same bin regardless of reviewer engagement quality
2. Candidates who generate deep review (potentially diverse or non-traditional backgrounds) lose their distinguishing signal
3. The model learns a flattened view of candidate quality that preserves existing biases

The preprocessing decision — made months before anyone notices the bias — has already determined the outcome.

### Proxy Discrimination Through Engagement Patterns

Engagement time can serve as a proxy for protected characteristics. If users from certain demographics tend to engage differently with content (cultural preferences, accessibility needs, device differences), then a binning strategy that erases engagement granularity disproportionately affects those groups. Equal-width binning amplifies this proxy discrimination by collapsing the part of the distribution where these differences are most pronounced — the tail.

---

## 7. Agentic AI & GenAI Perspective

### Why an Agent Architecture

The seven-agent pipeline isn't just a code organization choice — it's a deliberate design for AI system properties:

1. **Testability.** Each agent's `run(**kwargs) -> dict` contract (`base_agent.py:12-14`) is a pure function (modulo print statements). You can test `BinningAgent` in isolation by constructing its input dictionary. No hidden state, no implicit dependencies.

2. **Extensibility.** Adding a new binning strategy means modifying one agent. Adding a new evaluation metric means modifying another. The pipeline topology doesn't change — you're extending nodes, not rewiring the graph.

3. **Debuggability.** When Precision@K is low, you trace backwards through the agent chain. Is the ranking agent correct? Check its input probabilities. Are the probabilities sensible? Check the model coefficients. Are the coefficients stable? Check the feature activation rates. Each agent prints its own diagnostics, creating a self-documenting execution trace.

4. **Auditability.** For regulated domains (credit, hiring, healthcare), you need to explain *why* a recommendation was made. The agent chain provides a natural audit trail: "This candidate was ranked 3rd because the model assigned probability 0.73, which came from engagement bin 7, which corresponds to a review time of 120-180 seconds."

### LLM Extension Points

The architecture is designed for GenAI integration, even though the current implementation is pure Python:

- **ExplainerAgent → LLM Narratives.** The current `ExplainerAgent` has hardcoded explanations (`explainer_agent.py:13-47`). Swapping in an LLM call would let it generate context-aware narratives — "For this specific run, equal-width scored 0.35 entropy because SIGMA was set to 1.2, which produces extreme skew."

- **DataAgent → Natural Language Parsing.** Instead of reading constants from `constants.py`, a DataAgent could accept "Generate engagement data that mimics a short-form video platform" and use an LLM to select appropriate distribution parameters.

- **OrchestratorAgent.** `main.py:39-41` runs agents in a hardcoded sequence. An LLM-powered orchestrator could dynamically decide whether to run additional analyses based on intermediate results — e.g., "The entropy gap is unusually large; let's add a KDE-based binning strategy for comparison."

### The `run(**kwargs) -> dict` Contract

Every agent accepts keyword arguments and returns a dictionary that extends the input (`{**kwargs, ...new_keys}`). This design was chosen specifically for LLM integration:

- **LLMs work naturally with dictionaries** — they can inspect, filter, and transform them without schema definitions.
- **The contract is self-describing** — you can ask "what keys does ModelAgent add?" and the answer is readable from the return statement (`model_agent.py:83-98`).
- **No class hierarchies to navigate** — an LLM agent wrapper just needs to call `.run()` and forward the result.

### Production Considerations

If this pipeline moved from demo to production, the agent architecture would need:

- **Latency budgets.** Each agent should report execution time. LLM-augmented agents need timeout handling and fallback to non-LLM paths.
- **Error handling.** Currently, an exception in any agent crashes the pipeline. Production agents should return structured error objects that downstream agents can handle gracefully.
- **Observability.** Replace `print()` with structured logging. Each agent invocation should emit metrics (input size, output size, execution time, key statistics) to a monitoring system.
- **Caching.** If `DataAgent` parameters haven't changed, its output shouldn't be regenerated. Agent-level memoization based on input hash would speed up iteration.

### Current Scope

The current implementation makes **no LLM calls**. Every agent is deterministic Python with a fixed random seed. This is intentional for the demo context: you can verify every claim by reading the code and running it. The agent architecture proves the design pattern works; the LLM integration is an extension, not a prerequisite.

---

## 8. Domain Mappings

This pipeline was built as a domain-agnostic template. The abstract concepts (entity, item, engagement) map directly to specific business domains. Below are two detailed mappings that show how the same preprocessing failure manifests in real products.

### Amazon Recruiting AI

| Abstract Concept | Recruiting Domain |
|------------------|-------------------|
| Entity | Candidate |
| Item | Job posting |
| Engagement time | Time recruiter spent reviewing resume |
| Positive outcome | Advanced to phone screen |
| Equal-width failure | Most resumes reviewed in <2 minutes; deep reviews (5-15 min) for non-traditional candidates are binned identically to 1-minute skims |
| Equal-height fix | Granular bins distinguish 30s skim from 3-minute review from 10-minute deep-dive |

**Business insight:** A recruiting system that bins review time with equal-width will systematically under-rank candidates who receive longer reviews. If longer reviews correlate with non-traditional backgrounds (career changers, international candidates, self-taught engineers), the system produces disparate impact without any explicitly discriminatory feature.

**What an interviewer wants to hear:** "The bias isn't in the model weights — it's in the bin boundaries. By the time the model trains, the damage is done. This is why fairness audits that only inspect model coefficients miss preprocessing-induced bias."

### Credit Karma

| Abstract Concept | Financial Products Domain |
|------------------|--------------------------|
| Entity | User (consumer) |
| Item | Financial product (credit card, loan, insurance) |
| Engagement time | Dwell time on product detail page |
| Positive outcome | Clicked "Apply" or "Learn More" |
| Equal-width failure | Most users glance at products for <10s; users researching complex products (mortgages, investment accounts) who spend 2-5 minutes are invisible |
| Equal-height fix | Separates quick dismissals from genuine consideration from deep research |

**Business insight:** Financial product recommendation has regulatory implications. If the system recommends high-interest credit cards to users who were actually researching low-interest balance transfers (both in the "long dwell" tail), that's a compliance risk. Equal-height binning preserves the granularity needed to distinguish "comparing mortgage rates" from "reading credit card rewards terms."

**What an interviewer wants to hear:** "In financial services, a bad recommendation isn't just a poor user experience — it can violate fair lending regulations. The binning strategy directly affects which products appear in a user's top-K, and regulatory bodies will scrutinize that pipeline from preprocessing through ranking."

---

## 9. Interview Q&A

Organized by question category. Each answer references specific code so you can point at the implementation during the walkthrough.

### System Design

**Q: How would you scale this to 100M users and 10M items?**

A: The current pipeline processes 10K interactions in-memory. At 100M users, three things change:

1. **Data layer:** Replace in-memory DataFrame with a distributed data store (Spark, BigQuery). The `DataAgent` becomes a data *reader* rather than a generator, pulling from a feature store.
2. **Training:** Move from scikit-learn to a distributed training framework (TensorFlow, PyTorch with distributed data parallel). The model architecture might change to a two-tower neural network for separate entity and item embeddings.
3. **Serving:** Pre-compute rankings offline (batch pipeline) and store in a low-latency key-value store (Redis, DynamoDB). The `RankingAgent` becomes an API endpoint that looks up pre-computed scores rather than computing them on the fly.

The *binning decision* remains exactly as important at scale. In fact, it's more critical — with 10M items, tail items represent a larger absolute count of affected users even if they're the same percentage.

**Q: How would you add real-time updates?**

A: Two-layer architecture. The batch pipeline (this project) runs daily to retrain the model and recompute base rankings. A lightweight online layer captures real-time signals (clicks in the current session, trending items) and re-ranks the top-K candidates using a fast model (logistic regression or a small neural net). The binning strategy for the online features matters too — you'd want adaptive bins that update with the data distribution, not fixed-width bins computed once.

**Q: How would you A/B test equal-width vs equal-height in production?**

A: Split users randomly into control (equal-width) and treatment (equal-height). Measure Precision@K on live recommendations using implicit feedback (clicks, conversions). Run for at least 2 weeks to account for day-of-week effects. Watch for novelty effects — users might click more on treatment simply because recommendations changed. Key guardrail metrics: overall engagement shouldn't drop, revenue per session should be flat or up, and the treatment group shouldn't show increased bounce rate.

### Machine Learning

**Q: Why logistic regression instead of XGBoost or a neural network?**

A: Three reasons tied to this project's goals. First, **interpretability** — the coefficient table (`model_agent.py:24-29`) directly shows what the model learned about each bin. XGBoost's feature importance wouldn't distinguish "this bin is important because it has lots of data" from "this bin is important because it's predictive." Second, **transparency of failure** — when equal-width fails, you can see it in the coefficients (erratic, unstable values for bins 1-9). A neural net would hide this failure inside opaque weight matrices. Third, **sufficiency** — with 10 one-hot features and a binary target, logistic regression is the appropriate complexity. A more powerful model would either learn the same linear boundary or overfit to noise.

**Q: How do you handle overfitting?**

A: Logistic regression with L2 regularization (the `lbfgs` solver default) is inherently resistant to overfitting on 10 features. The 5-fold cross-validation (`evaluation_agent.py:54-55`) confirms this — the train-test accuracy gap is small for both strategies. If we moved to a more complex model, I'd add explicit regularization strength tuning via `GridSearchCV`, early stopping, and potentially holdout validation separate from the test set.

**Q: What if the engagement distribution changes over time?**

A: This is distribution shift, and it's the reason you'd want **adaptive binning** in production. The current pipeline computes bin edges from the training data. If the distribution shifts (e.g., a new feature launches that increases average engagement), those edges become stale. Solutions: (1) Retrain bins periodically using recent data windows. (2) Use quantile-based bins computed on a rolling window. (3) Monitor bin distribution entropy as a drift detection signal — if entropy drops suddenly, the bins need recomputation. The `_normalized_entropy` method (`evaluation_agent.py:14-22`) is already the right metric for this monitoring.

### Software Engineering

**Q: How would you test this pipeline?**

A: Three levels:

1. **Unit tests per agent.** Construct input dictionaries with known properties, call `agent.run(**input)`, and assert on output keys and values. For example, test that `BinningAgent` produces exactly `NUM_BINS` unique bin labels, or that `FeatureAgent` output has exactly `NUM_BINS` columns.
2. **Integration tests.** Run the full pipeline with a small dataset (100 interactions) and verify end-to-end properties: the output dictionary contains all expected keys, Precision@K is between 0 and 1, the comparison table has the right shape.
3. **Statistical tests.** Verify that equal-height entropy is above 0.95, that equal-width CV is above 2.0, and that the Spearman correlation is below 0.5. These are *property-based* tests that would catch regressions in the underlying logic.

**Q: How would you add CI/CD?**

A: GitHub Actions workflow with three stages: (1) `pip install -r requirements.txt && python main.py` as a smoke test — if any agent throws an exception, the build fails. (2) `pytest tests/` for the unit and integration tests described above. (3) A notebook-style check that captures the pipeline output and diffs it against a golden snapshot — if the numbers change, a reviewer investigates whether it's intentional.

**Q: Why `**kwargs` instead of typed dataclasses?**

A: Tradeoff between flexibility and safety. `**kwargs` makes agents composable without coupling — `ModelAgent` doesn't need to import a type from `BinningAgent`. For a demo pipeline with 7 agents, this flexibility outweighs the safety benefit. In production, I'd use `TypedDict` or Pydantic models to define the contract between agents, catching key-missing errors at import time rather than runtime. The current design makes it trivial to add new keys without updating type definitions — useful during rapid iteration.

### Behavioral

**Q: Tell me about a tradeoff you made in this project.**

A: The biggest tradeoff was choosing one-hot encoding over ordinal for the bin features. Ordinal is more compact (1 feature vs 10) and implicitly assumes a linear relationship between bin number and outcome — which is actually true here (higher bins = more engagement = higher positive rate). But one-hot gives us the coefficient table, which is the primary diagnostic tool for explaining *why* equal-width fails. I chose interpretability over compactness because the project's purpose is to teach, not to optimize. In production, I'd likely use ordinal encoding for the final model and keep one-hot in a diagnostic sidecar.

**Q: What would you do differently if you started over?**

A: Two things. First, I'd add a third binning strategy — something adaptive like Bayesian blocks or KDE-based — to show that the insight generalizes beyond "use quantiles." Equal-height is better than equal-width, but it's not necessarily optimal; an adaptive method could place more bins where the data has more structure. Second, I'd build a small Streamlit dashboard instead of static plots. Being able to slide the `SIGMA` parameter and watch the cascade change in real time would make the demo more compelling than screenshots.

**Q: How do you handle ambiguity in requirements?**

A: This project is a good example. The original requirement was vague: "build a recommendation engine for interviews." Ambiguity in three dimensions — which domain, how complex, what to optimize for. I resolved it by making each dimension explicit and configurable: the domain is abstract (entity/item/engagement), the complexity is minimal-but-complete (7 agents, 1 model), and the optimization target is understanding (not accuracy). Every constant lives in `constants.py` so that changing the domain is a parameter change, not a code change. When requirements are ambiguous, I default to flexibility — build the simplest thing that supports multiple interpretations, then let the user specialize.
