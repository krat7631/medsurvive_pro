"""Predictive comparison: how well do patient characteristics rank and calibrate survival?

Models
- Kaplan-Meier baseline: the same survival curve for everyone. Any model must beat it.
- Cox proportional hazards (scikit-survival, small ridge penalty for stability).
- Random survival forest: no proportional-hazards or linearity assumption.

Metrics, all on held-out folds
- Harrell's C-index: how often the model ranks the patient who has the event first as higher
  risk (0.5 = chance, 1.0 = perfect ordering).
- Uno's C-index: the same, re-weighted for censoring, so heavy censoring does not inflate it.
- Integrated Brier score: squared error of the predicted survival probabilities over follow-up
  (lower is better; it checks calibration, which C-indices ignore).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv

from .data import EVENT, TIME, covariate_columns

MODEL_NAMES = {"km": "Kaplan-Meier baseline", "cox": "Cox proportional hazards", "rsf": "Random survival forest"}


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = df[covariate_columns(df) + [TIME, EVENT]].dropna()
    X = frame[covariate_columns(df)].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")
    y = Surv.from_arrays(event=frame[EVENT].astype(bool).to_numpy(), time=frame[TIME].to_numpy())
    return X, y


def build_model(name: str, seed: int = 42):
    if name == "cox":
        return make_pipeline(OneHotEncoder(), StandardScaler(), CoxPHSurvivalAnalysis(alpha=0.1))
    if name == "rsf":
        return make_pipeline(OneHotEncoder(), RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=seed))
    raise ValueError(f"Unknown model {name!r}")


@dataclass
class KMBaseline:
    """Predicts the training-set Kaplan-Meier curve for every patient."""

    times_: np.ndarray | None = None
    survival_: np.ndarray | None = None

    def fit(self, X, y):
        self.times_, self.survival_ = kaplan_meier_estimator(y["event"], y["time"])
        return self

    def predict(self, X) -> np.ndarray:
        return np.zeros(len(X))

    def survival_at(self, X, times: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.times_, times, side="right") - 1
        curve = np.where(idx >= 0, self.survival_[np.clip(idx, 0, None)], 1.0)
        return np.tile(curve, (len(X), 1))


def evaluation_times(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Times inside both follow-up windows, between the 10th and 80th percentile of test event times."""
    event_times = y_test["time"][y_test["event"]]
    lo, hi = np.percentile(event_times, [10, 80])
    hi = min(hi, y_train["time"].max() * 0.99, y_test["time"].max() * 0.99)
    return np.linspace(lo, hi, 25)


def survival_matrix(model, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    if isinstance(model, KMBaseline):
        return model.survival_at(X, times)
    return np.vstack([fn(times) for fn in model.predict_survival_function(X)])


def score_fold(model, X_train, y_train, X_test, y_test) -> dict:
    risk = model.predict(X_test)
    times = evaluation_times(y_train, y_test)
    tau = times[-1]
    # The Brier score's censoring weights are estimated on the training fold, so they are undefined
    # for test patients followed beyond the longest training follow-up. Those few are excluded from
    # the Brier score only; both C-indices use every test patient.
    within = y_test["time"] < y_train["time"].max()
    return {
        "harrell_c": concordance_index_censored(y_test["event"], y_test["time"], risk)[0],
        "uno_c": concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0],
        "integrated_brier": integrated_brier_score(
            y_train, y_test[within], survival_matrix(model, X_test[within], times), times),
    }


def cross_validate(df: pd.DataFrame, folds: int = 5, repeats: int = 3, seed: int = 42) -> dict:
    """Repeated stratified k-fold (stratified on the event indicator). Returns mean and std per metric."""
    X, y = to_xy(df)
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    scores: dict[str, list[dict]] = {name: [] for name in MODEL_NAMES}

    for train_idx, test_idx in splitter.split(X, y["event"]):
        X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
        for name in MODEL_NAMES:
            model = KMBaseline() if name == "km" else build_model(name, seed)
            model.fit(X_tr, y_tr)
            scores[name].append(score_fold(model, X_tr, y_tr, X_te, y_te))

    summary = {}
    for name, rows in scores.items():
        frame = pd.DataFrame(rows)
        summary[name] = {metric: {"mean": round(float(frame[metric].mean()), 4),
                                  "std": round(float(frame[metric].std()), 4)} for metric in frame.columns}
    return {"folds": folds, "repeats": repeats, "patients": len(X), "events": int(y["event"].sum()),
            "models": summary}
