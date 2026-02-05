import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from agents.base_agent import BaseAgent
from utils.printing import print_subsection, print_table
from utils.constants import RANDOM_SEED, TEST_SIZE


class ModelAgent(BaseAgent):
    def __init__(self):
        super().__init__("ModelAgent — Logistic Regression Training")

    def _train_and_report(self, X_train, X_test, y_train, y_test, feature_names, label):
        print_subsection(f"{label} Model")

        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
        )
        model.fit(X_train, y_train)

        # Coefficients
        headers = ["Feature", "Coefficient"]
        rows = []
        for name, coef in zip(feature_names, model.coef_[0]):
            rows.append([name, f"{coef:.4f}"])
        rows.append(["(intercept)", f"{model.intercept_[0]:.4f}"])
        print_table(headers, rows)

        # Accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"\n  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")

        # Classification report
        print(f"\n  Classification Report ({label}):")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        for line in report.split("\n"):
            print(f"    {line}")

        y_proba = model.predict_proba(X_test)[:, 1]
        return model, y_pred, y_proba, test_acc

    def run(self, **kwargs) -> dict:
        self.print_header()

        # Suppress expected numerical warnings from sparse equal-width features
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        X_ew = kwargs["X_ew"]
        X_eh = kwargs["X_eh"]
        y = kwargs["y"]
        df = kwargs["df"]
        ew_feature_names = kwargs["ew_feature_names"]
        eh_feature_names = kwargs["eh_feature_names"]

        # Shared train/test split by index
        indices = np.arange(len(y))
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
        )

        X_ew_train, X_ew_test = X_ew[idx_train], X_ew[idx_test]
        X_eh_train, X_eh_test = X_eh[idx_train], X_eh[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        model_ew, pred_ew, proba_ew, acc_ew = self._train_and_report(
            X_ew_train, X_ew_test, y_train, y_test, ew_feature_names, "Equal-Width"
        )
        model_eh, pred_eh, proba_eh, acc_eh = self._train_and_report(
            X_eh_train, X_eh_test, y_train, y_test, eh_feature_names, "Equal-Height"
        )

        print_subsection("Model Comparison")
        print(f"  Equal-Width Test Accuracy:  {acc_ew:.4f}")
        print(f"  Equal-Height Test Accuracy: {acc_eh:.4f}")

        df_test = df.iloc[idx_test].copy()

        return {
            **kwargs,
            "model_ew": model_ew,
            "model_eh": model_eh,
            "proba_ew": proba_ew,
            "proba_eh": proba_eh,
            "y_test": y_test,
            "idx_train": idx_train,
            "idx_test": idx_test,
            "df_test": df_test,
            "X_ew_train": X_ew_train,
            "X_ew_test": X_ew_test,
            "X_eh_train": X_eh_train,
            "X_eh_test": X_eh_test,
            "y_train": y_train,
        }
