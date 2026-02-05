"""
Agentic Ranking Systems for Skewed Engagement Data
===================================================
Demonstrates how binning strategy choice affects recommendation quality
when engagement data follows a heavy-tailed (lognormal) distribution.

Usage: python main.py
"""

import os
import numpy as np
from utils.printing import print_section_header
from agents import (
    DataAgent,
    BinningAgent,
    FeatureAgent,
    ModelAgent,
    RankingAgent,
    EvaluationAgent,
    ExplainerAgent,
)


def run_pipeline():
    print_section_header("Agentic Ranking Systems for Skewed Engagement Data")
    print("  Pipeline: DataAgent -> BinningAgent -> FeatureAgent -> ModelAgent")
    print("            -> RankingAgent -> EvaluationAgent -> ExplainerAgent\n")

    agents = [
        DataAgent(),
        BinningAgent(),
        FeatureAgent(),
        ModelAgent(),
        RankingAgent(),
        EvaluationAgent(),
        ExplainerAgent(),
    ]

    results = {}
    for agent in agents:
        results = agent.run(**results)

    # Optional visualizations
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Engagement time histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        engagement = results["df"]["engagement_time"].values
        ax.hist(engagement, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Engagement Time (seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Engagement Time (Lognormal)")
        ax.axvline(np.median(engagement), color="red", linestyle="--", label=f"Median: {np.median(engagement):.1f}s")
        ax.axvline(np.mean(engagement), color="orange", linestyle="--", label=f"Mean: {np.mean(engagement):.1f}s")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "engagement_distribution.png"), dpi=150)
        plt.close()

        # 2. Bin comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ew_counts = results["ew_counts"]
        axes[0].bar(range(len(ew_counts)), ew_counts.values, color="salmon", edgecolor="black")
        axes[0].set_xlabel("Bin")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Equal-Width Binning")
        axes[0].set_xticks(range(len(ew_counts)))

        eh_counts = results["eh_counts"]
        axes[1].bar(range(len(eh_counts)), eh_counts.values, color="mediumseagreen", edgecolor="black")
        axes[1].set_xlabel("Bin")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Equal-Height (Quantile) Binning")
        axes[1].set_xticks(range(len(eh_counts)))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bin_comparison.png"), dpi=150)
        plt.close()

        # 3. Coefficient comparison
        model_ew = results["model_ew"]
        model_eh = results["model_eh"]
        ew_names = results["ew_feature_names"]
        eh_names = results["eh_feature_names"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].barh(range(len(ew_names)), model_ew.coef_[0], color="salmon", edgecolor="black")
        axes[0].set_yticks(range(len(ew_names)))
        axes[0].set_yticklabels(ew_names)
        axes[0].set_xlabel("Coefficient")
        axes[0].set_title("Equal-Width Model Coefficients")

        axes[1].barh(range(len(eh_names)), model_eh.coef_[0][:len(eh_names)], color="mediumseagreen", edgecolor="black")
        axes[1].set_yticks(range(len(eh_names)))
        axes[1].set_yticklabels(eh_names)
        axes[1].set_xlabel("Coefficient")
        axes[1].set_title("Equal-Height Model Coefficients")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "coefficient_comparison.png"), dpi=150)
        plt.close()

        print_section_header("Visualizations Saved")
        print(f"  Plots saved to: {output_dir}/")
        print("    - engagement_distribution.png")
        print("    - bin_comparison.png")
        print("    - coefficient_comparison.png")

    except ImportError:
        print("\n  [Note] matplotlib not installed — skipping visualizations.")

    print_section_header("Pipeline Complete")
    print("  All 7 agents executed successfully.")
    print("  See README.md for domain mappings and interview talking points.\n")


if __name__ == "__main__":
    run_pipeline()
