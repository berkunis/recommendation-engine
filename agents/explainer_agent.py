from agents.base_agent import BaseAgent
from utils.printing import print_subsection


class ExplainerAgent(BaseAgent):
    def __init__(self):
        super().__init__("ExplainerAgent — Plain-Language Insights")

    def run(self, **kwargs) -> dict:
        self.print_header()

        print_subsection("1. Data Density — Why Equal-Width Fails")
        print("""
  Engagement data follows a heavy-tailed distribution where the vast majority
  of interactions are short (a few seconds), while a small fraction are very
  long (minutes or even an hour). When we divide this range into equal-width
  bins, the first bin captures nearly all the data because the range is
  dominated by rare extreme values. Bins 1 through 9 sit almost empty,
  wasting the model's resolution on ranges where no data exists. Equal-height
  binning adapts to the data's actual density, placing bin boundaries where
  the data lives and giving the model meaningful variation to learn from.
""")

        print_subsection("2. Learning Stability — Empty Bins Hurt Models")
        print("""
  When most one-hot features are almost always zero, the model has very little
  signal to estimate their coefficients. The logistic regression assigns near-
  zero or arbitrary weights to features that activate on only a handful of
  examples, making predictions unstable across different data splits. Cross-
  validation confirms this: equal-width models show higher variance in accuracy
  because a few training examples shifting between folds can flip the learned
  weights entirely. Equal-height features activate uniformly, giving the
  optimizer a stable gradient landscape and consistent coefficients.
""")

        print_subsection("3. Fairness & Coverage — Tail Entities Are Ignored")
        print("""
  Entities or items whose engagement falls in the tail of the distribution —
  think of power users, niche products, or specialized job postings — all
  get lumped into a single undifferentiated bin under equal-width binning. The
  model treats them identically despite potentially important differences in
  behavior. This creates systematic blind spots: the ranking system cannot
  distinguish between moderately engaged and highly engaged entities,
  leading to lower precision in top-K recommendations and unfair treatment
  of minority segments. Equal-height binning preserves granularity across
  the entire range, ensuring that tail behavior is captured and acted upon.
""")

        return {**kwargs}
