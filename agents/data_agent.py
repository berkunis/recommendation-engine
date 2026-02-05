import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from utils.printing import print_distribution_summary, print_subsection, print_table
from utils.constants import (
    RANDOM_SEED, NUM_ENTITIES, NUM_ITEMS, INTERACTIONS_PER_ENTITY,
    MU, SIGMA, ENGAGEMENT_CLIP_MIN, ENGAGEMENT_CLIP_MAX, POSITIVE_QUANTILE,
)


class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataAgent — Synthetic Engagement Data Generator")

    def run(self, **kwargs) -> dict:
        self.print_header()
        rng = np.random.RandomState(RANDOM_SEED)

        total_interactions = NUM_ENTITIES * INTERACTIONS_PER_ENTITY
        entity_ids = rng.randint(0, NUM_ENTITIES, size=total_interactions)
        item_ids = rng.randint(0, NUM_ITEMS, size=total_interactions)

        raw_engagement = rng.lognormal(mean=MU, sigma=SIGMA, size=total_interactions)
        engagement = np.clip(raw_engagement, ENGAGEMENT_CLIP_MIN, ENGAGEMENT_CLIP_MAX)

        df = pd.DataFrame({
            "entity_id": entity_ids,
            "item_id": item_ids,
            "engagement_time": engagement,
        })

        threshold = np.percentile(engagement, POSITIVE_QUANTILE * 100)
        df["positive_outcome"] = (df["engagement_time"] >= threshold).astype(int)

        print_subsection("Dataset Overview")
        print(f"  Total interactions: {len(df)}")
        print(f"  Unique entities:    {df['entity_id'].nunique()}")
        print(f"  Unique items:       {df['item_id'].nunique()}")

        print_distribution_summary(df["engagement_time"].values, "Engagement Time (seconds)")

        print_subsection("Positive Outcome Rate")
        pos_rate = df["positive_outcome"].mean()
        print(f"  Threshold (P{int(POSITIVE_QUANTILE*100)}): {threshold:.2f}s")
        print(f"  Positive rate:      {pos_rate:.2%}")
        print(f"  Positive count:     {df['positive_outcome'].sum()}")
        print(f"  Negative count:     {(1 - df['positive_outcome']).sum()}")

        return {
            "df": df,
            "engagement_col": "engagement_time",
            "outcome_col": "positive_outcome",
            "threshold": threshold,
        }
