import subprocess
import sys

import numpy as np
import pytest

from agents import (
    DataAgent,
    BinningAgent,
    FeatureAgent,
    ModelAgent,
    RankingAgent,
    EvaluationAgent,
    ExplainerAgent,
)
from utils.constants import (
    NUM_ENTITIES,
    INTERACTIONS_PER_ENTITY,
    NUM_BINS,
    ENGAGEMENT_CLIP_MIN,
    ENGAGEMENT_CLIP_MAX,
    POSITIVE_QUANTILE,
    CV_FOLDS,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def data_result():
    return DataAgent().run()


@pytest.fixture(scope="module")
def binning_result(data_result):
    return BinningAgent().run(**data_result)


@pytest.fixture(scope="module")
def feature_result(binning_result):
    return FeatureAgent().run(**binning_result)


@pytest.fixture(scope="module")
def model_result(feature_result):
    return ModelAgent().run(**feature_result)


@pytest.fixture(scope="module")
def ranking_result(model_result):
    return RankingAgent().run(**model_result)


@pytest.fixture(scope="module")
def evaluation_result(ranking_result):
    return EvaluationAgent().run(**ranking_result)


@pytest.fixture(scope="module")
def explainer_result(evaluation_result):
    return ExplainerAgent().run(**evaluation_result)


# ── TestDataAgent ───────────────────────────────────────────────────────────


class TestDataAgent:
    def test_output_keys(self, data_result):
        for key in ("df", "engagement_col", "outcome_col", "threshold"):
            assert key in data_result

    def test_shape(self, data_result):
        df = data_result["df"]
        expected_rows = NUM_ENTITIES * INTERACTIONS_PER_ENTITY
        assert len(df) == expected_rows
        assert df.shape[1] == 4  # entity_id, item_id, engagement_time, positive_outcome

    def test_engagement_range(self, data_result):
        engagement = data_result["df"]["engagement_time"].values
        assert engagement.min() >= ENGAGEMENT_CLIP_MIN
        assert engagement.max() <= ENGAGEMENT_CLIP_MAX

    def test_positive_rate(self, data_result):
        pos_rate = data_result["df"]["positive_outcome"].mean()
        expected = 1 - POSITIVE_QUANTILE  # ~0.10
        assert abs(pos_rate - expected) < 0.02


# ── TestBinningAgent ────────────────────────────────────────────────────────


class TestBinningAgent:
    def test_bin_columns_exist(self, binning_result):
        df = binning_result["df"]
        assert "ew_bin" in df.columns
        assert "eh_bin" in df.columns

    def test_ew_bin_count(self, binning_result):
        ew_counts = binning_result["ew_counts"]
        assert len(ew_counts) <= NUM_BINS

    def test_eh_bin_balance(self, binning_result):
        eh_counts = binning_result["eh_counts"]
        cv = eh_counts.std() / eh_counts.mean()
        assert cv < 0.1

    def test_counts_in_output(self, binning_result):
        assert "ew_counts" in binning_result
        assert "eh_counts" in binning_result


# ── TestFeatureAgent ────────────────────────────────────────────────────────


class TestFeatureAgent:
    def test_feature_dimensions(self, feature_result):
        assert feature_result["X_ew"].shape[1] == NUM_BINS
        assert feature_result["X_eh"].shape[1] == NUM_BINS

    def test_one_hot_row_sums(self, feature_result):
        for matrix_key in ("X_ew", "X_eh"):
            row_sums = feature_result[matrix_key].sum(axis=1)
            np.testing.assert_array_equal(row_sums, np.ones(len(row_sums)))

    def test_y_length(self, feature_result):
        assert len(feature_result["y"]) == feature_result["X_ew"].shape[0]


# ── TestModelAgent ──────────────────────────────────────────────────────────


class TestModelAgent:
    def test_output_keys(self, model_result):
        for key in ("model_ew", "model_eh", "proba_ew", "proba_eh"):
            assert key in model_result

    def test_probabilities_range(self, model_result):
        for key in ("proba_ew", "proba_eh"):
            proba = model_result[key]
            assert proba.min() >= 0.0
            assert proba.max() <= 1.0

    def test_lengths_match(self, model_result):
        assert len(model_result["y_test"]) == len(model_result["proba_ew"])
        assert len(model_result["y_test"]) == len(model_result["proba_eh"])


# ── TestRankingAgent ────────────────────────────────────────────────────────


class TestRankingAgent:
    def test_precision_range(self, ranking_result):
        assert 0 <= ranking_result["precision_at_k_ew"] <= 1
        assert 0 <= ranking_result["precision_at_k_eh"] <= 1

    def test_rank_column(self, ranking_result):
        assert "rank" in ranking_result["ranked_ew"].columns
        assert "rank" in ranking_result["ranked_eh"].columns

    def test_eligible_entities_is_list(self, ranking_result):
        assert isinstance(ranking_result["eligible_entities"], list)


# ── TestEvaluationAgent ─────────────────────────────────────────────────────


class TestEvaluationAgent:
    def test_entropy_range(self, evaluation_result):
        assert 0 <= evaluation_result["ew_entropy"] <= 1
        assert 0 <= evaluation_result["eh_entropy"] <= 1

    def test_eh_entropy_greater(self, evaluation_result):
        assert evaluation_result["eh_entropy"] > evaluation_result["ew_entropy"]

    def test_cv_arrays(self, evaluation_result):
        assert isinstance(evaluation_result["cv_ew"], np.ndarray)
        assert isinstance(evaluation_result["cv_eh"], np.ndarray)
        assert len(evaluation_result["cv_ew"]) == CV_FOLDS
        assert len(evaluation_result["cv_eh"]) == CV_FOLDS


# ── TestFullPipeline (integration) ──────────────────────────────────────────


class TestFullPipeline:
    def test_all_keys_present(self, explainer_result):
        expected_keys = [
            "df",
            "engagement_col",
            "outcome_col",
            "threshold",
            "ew_counts",
            "eh_counts",
            "X_ew",
            "X_eh",
            "y",
            "model_ew",
            "model_eh",
            "proba_ew",
            "proba_eh",
            "y_test",
            "ranked_ew",
            "ranked_eh",
            "eligible_entities",
            "precision_at_k_ew",
            "precision_at_k_eh",
            "ew_entropy",
            "eh_entropy",
            "cv_ew",
            "cv_eh",
            "mean_spearman",
        ]
        for key in expected_keys:
            assert key in explainer_result, f"Missing key: {key}"

    def test_main_exits_cleanly(self):
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"main.py failed:\n{result.stderr}"
