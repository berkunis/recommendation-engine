import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from agents.base_agent import BaseAgent
from utils.printing import print_subsection, print_table
from utils.constants import RANDOM_SEED, CV_FOLDS, NUM_BINS


class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvaluationAgent — Strategy Comparison")

    def _normalized_entropy(self, counts):
        counts = np.array(counts, dtype=float)
        counts = counts[counts > 0]
        if len(counts) <= 1:
            return 0.0
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def run(self, **kwargs) -> dict:
        self.print_header()

        ew_counts = kwargs["ew_counts"]
        eh_counts = kwargs["eh_counts"]
        model_ew = kwargs["model_ew"]
        model_eh = kwargs["model_eh"]
        X_ew = kwargs["X_ew"]
        X_eh = kwargs["X_eh"]
        y = kwargs["y"]
        ranked_ew = kwargs["ranked_ew"]
        ranked_eh = kwargs["ranked_eh"]
        eligible_entities = kwargs["eligible_entities"]
        precision_at_k_ew = kwargs["precision_at_k_ew"]
        precision_at_k_eh = kwargs["precision_at_k_eh"]

        # 1. Bin Balance
        print_subsection("Bin Balance")
        ew_entropy = self._normalized_entropy(ew_counts.values)
        eh_entropy = self._normalized_entropy(eh_counts.values)
        ew_cv = ew_counts.std() / ew_counts.mean() if ew_counts.mean() > 0 else 0
        eh_cv = eh_counts.std() / eh_counts.mean() if eh_counts.mean() > 0 else 0
        print(f"  Equal-Width  — Norm. Entropy: {ew_entropy:.4f}, CV: {ew_cv:.4f}")
        print(f"  Equal-Height — Norm. Entropy: {eh_entropy:.4f}, CV: {eh_cv:.4f}")

        # 2. Model Stability (5-fold CV)
        print_subsection("Model Stability (5-Fold Cross-Validation)")
        lr_ew = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr_eh = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)

        cv_ew = cross_val_score(lr_ew, X_ew, y, cv=CV_FOLDS, scoring="accuracy")
        cv_eh = cross_val_score(lr_eh, X_eh, y, cv=CV_FOLDS, scoring="accuracy")

        print(f"  Equal-Width  — Mean: {cv_ew.mean():.4f} +/- {cv_ew.std():.4f}")
        print(f"  Equal-Height — Mean: {cv_eh.mean():.4f} +/- {cv_eh.std():.4f}")

        # 3. Ranking Consistency (Spearman)
        print_subsection("Ranking Consistency (Spearman Correlation)")
        correlations = []
        for eid in eligible_entities:
            ew_entity = ranked_ew[ranked_ew["entity_id"] == eid].sort_values("item_id")
            eh_entity = ranked_eh[ranked_eh["entity_id"] == eid].sort_values("item_id")
            if len(ew_entity) >= 2 and len(eh_entity) >= 2:
                merged = ew_entity[["item_id", "rank"]].merge(
                    eh_entity[["item_id", "rank"]],
                    on="item_id", suffixes=("_ew", "_eh"),
                )
                if len(merged) >= 2:
                    corr, _ = stats.spearmanr(merged["rank_ew"], merged["rank_eh"])
                    if not np.isnan(corr):
                        correlations.append(corr)

        mean_corr = np.mean(correlations) if correlations else 0
        std_corr = np.std(correlations) if correlations else 0
        print(f"  Mean Spearman correlation: {mean_corr:.4f} +/- {std_corr:.4f}")
        print(f"  Entities compared:         {len(correlations)}")

        # 4. Outcome Quality (Precision@K)
        print_subsection("Outcome Quality (Precision@K)")
        print(f"  Equal-Width  Precision@K: {precision_at_k_ew:.4f}")
        print(f"  Equal-Height Precision@K: {precision_at_k_eh:.4f}")

        # Summary comparison table
        print_subsection("Side-by-Side Comparison")
        headers = ["Metric", "Equal-Width", "Equal-Height", "Winner"]
        rows = [
            [
                "Norm. Entropy",
                f"{ew_entropy:.4f}",
                f"{eh_entropy:.4f}",
                "EH" if eh_entropy > ew_entropy else "EW",
            ],
            [
                "Bin CV",
                f"{ew_cv:.4f}",
                f"{eh_cv:.4f}",
                "EH" if eh_cv < ew_cv else "EW",
            ],
            [
                "CV Accuracy",
                f"{cv_ew.mean():.4f}+/-{cv_ew.std():.4f}",
                f"{cv_eh.mean():.4f}+/-{cv_eh.std():.4f}",
                "EH" if cv_eh.mean() > cv_ew.mean() else "EW",
            ],
            [
                "CV Stability",
                f"{cv_ew.std():.4f}",
                f"{cv_eh.std():.4f}",
                "EH" if cv_eh.std() < cv_ew.std() else "EW",
            ],
            [
                "Precision@K",
                f"{precision_at_k_ew:.4f}",
                f"{precision_at_k_eh:.4f}",
                "EH" if precision_at_k_eh > precision_at_k_ew else "EW",
            ],
        ]
        print_table(headers, rows)

        return {
            **kwargs,
            "ew_entropy": ew_entropy,
            "eh_entropy": eh_entropy,
            "cv_ew": cv_ew,
            "cv_eh": cv_eh,
            "mean_spearman": mean_corr,
        }
