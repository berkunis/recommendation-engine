import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from utils.printing import print_subsection, print_table
from utils.constants import NUM_BINS


class FeatureAgent(BaseAgent):
    def __init__(self):
        super().__init__("FeatureAgent — One-Hot Encoding of Bin Labels")

    def run(self, **kwargs) -> dict:
        self.print_header()
        df = kwargs["df"]
        outcome_col = kwargs["outcome_col"]

        # --- Equal-Width Features ---
        print_subsection("Equal-Width One-Hot Features")
        ew_dummies = pd.get_dummies(df["ew_bin"], prefix="ew_bin")
        # Ensure all bins represented
        for i in range(NUM_BINS):
            col = f"ew_bin_{i}"
            if col not in ew_dummies.columns:
                ew_dummies[col] = 0
        ew_cols = sorted([c for c in ew_dummies.columns if c.startswith("ew_bin_")])
        ew_dummies = ew_dummies[ew_cols]

        headers = ["Feature", "Activation Rate", "Non-Zero Count"]
        rows = []
        for col in ew_cols:
            rate = ew_dummies[col].mean()
            count = ew_dummies[col].sum()
            rows.append([col, f"{rate:.4f}", count])
        print_table(headers, rows)

        ew_sparsity = 1 - ew_dummies.values.mean()
        print(f"\n  Overall sparsity: {ew_sparsity:.4f}")

        # --- Equal-Height Features ---
        print_subsection("Equal-Height One-Hot Features")
        eh_dummies = pd.get_dummies(df["eh_bin"], prefix="eh_bin")
        eh_cols = sorted([c for c in eh_dummies.columns if c.startswith("eh_bin_")])
        eh_dummies = eh_dummies[eh_cols]

        headers = ["Feature", "Activation Rate", "Non-Zero Count"]
        rows = []
        for col in eh_cols:
            rate = eh_dummies[col].mean()
            count = eh_dummies[col].sum()
            rows.append([col, f"{rate:.4f}", count])
        print_table(headers, rows)

        eh_sparsity = 1 - eh_dummies.values.mean()
        print(f"\n  Overall sparsity: {eh_sparsity:.4f}")

        print_subsection("Sparsity Comparison")
        print(f"  Equal-Width sparsity:  {ew_sparsity:.4f}")
        print(f"  Equal-Height sparsity: {eh_sparsity:.4f}")
        print("  Key insight: Equal-width features are nearly all-zero for bins 1-9,")
        print("  providing almost no learning signal for the model.")

        X_ew = ew_dummies.values.astype(np.float64)
        X_eh = eh_dummies.values.astype(np.float64)
        y = df[outcome_col].values

        return {
            **kwargs,
            "X_ew": X_ew,
            "X_eh": X_eh,
            "y": y,
            "ew_feature_names": ew_cols,
            "eh_feature_names": eh_cols,
        }
