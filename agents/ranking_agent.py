import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from utils.printing import print_subsection, print_table
from utils.constants import TOP_K, RANDOM_SEED


class RankingAgent(BaseAgent):
    def __init__(self):
        super().__init__("RankingAgent — Top-K Item Ranking per Entity")

    def _rank_items(self, df_test, probas, strategy_name):
        df_ranked = df_test.copy()
        df_ranked["score"] = probas
        df_ranked["rank"] = df_ranked.groupby("entity_id")["score"].rank(
            ascending=False, method="first"
        )
        return df_ranked

    def run(self, **kwargs) -> dict:
        self.print_header()

        df_test = kwargs["df_test"].copy()
        proba_ew = kwargs["proba_ew"]
        proba_eh = kwargs["proba_eh"]
        outcome_col = kwargs["outcome_col"]

        # Rank items per entity for each strategy
        ranked_ew = self._rank_items(df_test, proba_ew, "Equal-Width")
        ranked_eh = self._rank_items(df_test, proba_eh, "Equal-Height")

        # Filter to entities with >= K items
        entity_counts = df_test["entity_id"].value_counts()
        eligible_entities = entity_counts[entity_counts >= TOP_K].index.tolist()

        print_subsection("Ranking Overview")
        print(f"  Total test entities:    {df_test['entity_id'].nunique()}")
        print(f"  Entities with >= {TOP_K} items: {len(eligible_entities)}")

        # Get top-K for eligible entities
        topk_ew = ranked_ew[
            ranked_ew["entity_id"].isin(eligible_entities) & (ranked_ew["rank"] <= TOP_K)
        ]
        topk_eh = ranked_eh[
            ranked_eh["entity_id"].isin(eligible_entities) & (ranked_eh["rank"] <= TOP_K)
        ]

        # Show sample rankings
        rng = np.random.RandomState(RANDOM_SEED)
        sample_entities = rng.choice(eligible_entities, size=min(3, len(eligible_entities)), replace=False)

        for eid in sample_entities:
            print_subsection(f"Sample Rankings — Entity {eid}")
            for label, ranked_df in [("Equal-Width", ranked_ew), ("Equal-Height", ranked_eh)]:
                entity_items = ranked_df[ranked_df["entity_id"] == eid].sort_values("rank")
                top_items = entity_items.head(TOP_K)
                print(f"\n  {label} Top-{TOP_K}:")
                headers = ["Rank", "Item", "Score", "Outcome"]
                rows = []
                for _, row in top_items.iterrows():
                    rows.append([
                        int(row["rank"]),
                        int(row["item_id"]),
                        f"{row['score']:.4f}",
                        int(row[outcome_col]),
                    ])
                print_table(headers, rows)

        # Precision@K
        print_subsection("Precision@K Summary")
        prec_ew = topk_ew[outcome_col].mean() if len(topk_ew) > 0 else 0
        prec_eh = topk_eh[outcome_col].mean() if len(topk_eh) > 0 else 0
        print(f"  Equal-Width  Precision@{TOP_K}: {prec_ew:.4f}")
        print(f"  Equal-Height Precision@{TOP_K}: {prec_eh:.4f}")

        return {
            **kwargs,
            "ranked_ew": ranked_ew,
            "ranked_eh": ranked_eh,
            "topk_ew": topk_ew,
            "topk_eh": topk_eh,
            "eligible_entities": eligible_entities,
            "precision_at_k_ew": prec_ew,
            "precision_at_k_eh": prec_eh,
        }
