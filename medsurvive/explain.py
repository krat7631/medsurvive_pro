"""Per-patient explanations of predicted risk.

- Random survival forest: SHAP's permutation explainer attributes each patient's risk score to
  their covariates relative to a background sample. Accurate for a non-linear model but slow
  (seconds per patient), so the bundled datasets' values are precomputed by scripts/evaluate.py.
- Cox model: the log-hazard is linear in the standardised covariates, so the exact SHAP value of
  covariate j for a patient is coef_j * (x_j - mean_j). Instant, so it works for uploaded data.

Positive values mean "this characteristic raised predicted risk for this patient". They describe
the model, not causal effects of the characteristics.
"""

import numpy as np
import pandas as pd
import shap

from .modeling import build_model, to_xy

BACKGROUND_SIZE = 100


def explain_cox(df: pd.DataFrame) -> dict:
    """Exact SHAP values for the Cox model's log-hazard, for every patient."""
    X, y = to_xy(df)
    model = build_model("cox").fit(X, y)
    encoder, scaler, cox = model[0], model[1], model[-1]
    X_encoded = encoder.transform(X)
    standardized = pd.DataFrame(scaler.transform(X_encoded), columns=X_encoded.columns, index=X.index)
    # The scaler centres every column, so mean_j = 0 and the contribution is coef_j * x_j.
    values = standardized * cox.coef_
    importance = values.abs().mean().sort_values(ascending=False)
    return {"feature_importance": importance.round(4).to_dict(), "shap_values": values, "feature_values": X_encoded}


def explain_risk(df: pd.DataFrame, explain_n: int = 150, seed: int = 42) -> dict:
    """Fit the forest on all rows, then explain a random sample of patients."""
    X, y = to_xy(df)
    model = build_model("rsf", seed).fit(X, y)
    encoded_columns = model[0].transform(X.iloc[:1]).columns
    encoder, forest = model[0], model[-1]
    X_encoded = encoder.transform(X)

    rng = np.random.default_rng(seed)
    background = X_encoded.iloc[rng.choice(len(X_encoded), min(BACKGROUND_SIZE, len(X_encoded)), replace=False)]
    sample = X_encoded.iloc[rng.choice(len(X_encoded), min(explain_n, len(X_encoded)), replace=False)]

    explainer = shap.PermutationExplainer(forest.predict, background, seed=seed)
    values = explainer(sample)

    importance = pd.Series(np.abs(values.values).mean(axis=0), index=encoded_columns).sort_values(ascending=False)
    return {
        "feature_importance": importance.round(4).to_dict(),
        "shap_values": pd.DataFrame(values.values, columns=encoded_columns, index=sample.index),
        "feature_values": sample,
    }
