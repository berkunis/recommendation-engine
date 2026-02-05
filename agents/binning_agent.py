import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from utils.printing import print_subsection, print_table, print_bar
from utils.constants import NUM_BINS


class BinningAgent(BaseAgent):
    def __init__(self):
        super().__init__("BinningAgent — Equal-Width vs Equal-Height Binning")

    def run(self, **kwargs) -> dict:
        self.print_header()
        df = kwargs["df"].copy()
        engagement_col = kwargs["engagement_col"]
        data = df[engagement_col].values

        # --- Equal-Width Binning ---
        print_subsection("Equal-Width Binning")
        ew_edges = np.linspace(data.min(), data.max(), NUM_BINS + 1)
        df["ew_bin"] = np.clip(
            np.digitize(data, ew_edges[1:-1]), 0, NUM_BINS - 1
        )

        ew_counts = df["ew_bin"].value_counts().sort_index()
        print(f"  Bin width: {(data.max() - data.min()) / NUM_BINS:.1f}s each")
        print(f"  Edges: {[f'{e:.1f}' for e in ew_edges]}")

        headers = ["Bin", "Range", "Count", "Pct"]
        rows = []
        max_count = ew_counts.max()
        for i in range(NUM_BINS):
            lo, hi = ew_edges[i], ew_edges[i + 1]
            count = ew_counts.get(i, 0)
            pct = count / len(df) * 100
            rows.append([f"Bin {i}", f"[{lo:.0f}, {hi:.0f})", count, f"{pct:.1f}%"])
        print_table(headers, rows)

        print("\n  Distribution:")
        for i in range(NUM_BINS):
            count = ew_counts.get(i, 0)
            print_bar(f"Bin {i}", count, max_count)

        ew_cv = ew_counts.std() / ew_counts.mean() if ew_counts.mean() > 0 else 0
        print(f"\n  Coefficient of Variation: {ew_cv:.3f}")

        # --- Equal-Height (Quantile) Binning ---
        print_subsection("Equal-Height (Quantile) Binning")
        df["eh_bin"], eh_edges_interval = pd.qcut(
            data, NUM_BINS, labels=False, retbins=True, duplicates="drop"
        )

        eh_counts = df["eh_bin"].value_counts().sort_index()
        actual_bins = len(eh_counts)
        print(f"  Target bins: {NUM_BINS}, Actual bins: {actual_bins}")
        print(f"  Edges: {[f'{e:.1f}' for e in eh_edges_interval]}")

        headers = ["Bin", "Range", "Count", "Pct"]
        rows = []
        max_count_eh = eh_counts.max()
        for i in range(actual_bins):
            lo = eh_edges_interval[i]
            hi = eh_edges_interval[i + 1]
            count = eh_counts.get(i, 0)
            pct = count / len(df) * 100
            rows.append([f"Bin {i}", f"[{lo:.1f}, {hi:.1f})", count, f"{pct:.1f}%"])
        print_table(headers, rows)

        print("\n  Distribution:")
        for i in range(actual_bins):
            count = eh_counts.get(i, 0)
            print_bar(f"Bin {i}", count, max_count_eh)

        eh_cv = eh_counts.std() / eh_counts.mean() if eh_counts.mean() > 0 else 0
        print(f"\n  Coefficient of Variation: {eh_cv:.3f}")

        print_subsection("Comparison")
        print(f"  Equal-Width CV:  {ew_cv:.3f}  (higher = more imbalanced)")
        print(f"  Equal-Height CV: {eh_cv:.3f}  (lower = more balanced)")

        return {
            **kwargs,
            "df": df,
            "ew_edges": ew_edges,
            "eh_edges": eh_edges_interval,
            "ew_counts": ew_counts,
            "eh_counts": eh_counts,
        }
