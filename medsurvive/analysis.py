"""Descriptive and inferential survival statistics: Kaplan-Meier, log-rank, Cox hazard ratios
and the proportional-hazards check that a Cox model's hazard ratios depend on."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

from .data import EVENT, TIME, covariate_columns

MIN_GROUP_SIZE = 5


@dataclass(frozen=True)
class KMCurve:
    group: str
    n: int
    events: int
    median: float | None
    timeline: list[float]
    survival: list[float]
    lower: list[float]
    upper: list[float]
    censor_times: list[float]


def kaplan_meier(df: pd.DataFrame, group: str | None = None) -> list[KMCurve]:
    """One curve per group (or one overall), with 95% CIs and censoring times for tick marks."""
    groups = [("All patients", df)] if group is None else [
        (str(name), sub) for name, sub in df.groupby(group, observed=True) if len(sub) >= MIN_GROUP_SIZE
    ]
    curves = []
    for name, sub in groups:
        kmf = KaplanMeierFitter().fit(sub[TIME], sub[EVENT], label=name)
        ci = kmf.confidence_interval_survival_function_
        median = kmf.median_survival_time_
        curves.append(KMCurve(
            group=name,
            n=len(sub),
            events=int(sub[EVENT].sum()),
            median=None if not np.isfinite(median) else float(median),
            timeline=kmf.survival_function_.index.tolist(),
            survival=kmf.survival_function_[name].tolist(),
            lower=ci.iloc[:, 0].tolist(),
            upper=ci.iloc[:, 1].tolist(),
            censor_times=sub.loc[sub[EVENT] == 0, TIME].tolist(),
        ))
    return curves


def logrank(df: pd.DataFrame, group: str) -> dict:
    """Test whether survival differs between groups (two or more)."""
    sub = df.groupby(group, observed=True).filter(lambda g: len(g) >= MIN_GROUP_SIZE)
    result = multivariate_logrank_test(sub[TIME], sub[group].astype(str), sub[EVENT])
    return {"test_statistic": float(result.test_statistic), "p_value": float(result.p_value),
            "degrees_of_freedom": int(result.degrees_of_freedom)}


def design_matrix(df: pd.DataFrame, covariates: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categoricals (first level as reference) and drop rows with missing covariates."""
    covariates = covariates if covariates is not None else covariate_columns(df)
    frame = df[covariates + [TIME, EVENT]].dropna()
    return pd.get_dummies(frame, drop_first=True, dtype=float)


def fit_cox(df: pd.DataFrame, covariates: list[str] | None = None, penalizer: float = 0.0) -> CoxPHFitter:
    return CoxPHFitter(penalizer=penalizer).fit(design_matrix(df, covariates), TIME, EVENT)


def hazard_ratio_table(cox: CoxPHFitter) -> pd.DataFrame:
    s = cox.summary
    return pd.DataFrame({
        "hazard_ratio": s["exp(coef)"],
        "ci_lower": s["exp(coef) lower 95%"],
        "ci_upper": s["exp(coef) upper 95%"],
        "p_value": s["p"],
    })


def treatment_effect(df: pd.DataFrame, treatment: str, reference: str) -> dict:
    """Crude and covariate-adjusted hazard ratio of each treatment arm against the reference arm."""
    frame = df.copy()
    levels = [reference] + sorted(str(v) for v in frame[treatment].dropna().unique() if str(v) != reference)
    frame[treatment] = pd.Categorical(frame[treatment].astype(str), categories=levels)

    crude = hazard_ratio_table(fit_cox(frame, [treatment]))
    adjusted = hazard_ratio_table(fit_cox(frame, covariate_columns(frame)))
    arms = [f"{treatment}_{level}" for level in levels[1:]]
    return {
        "reference": reference,
        "crude": crude.loc[arms].to_dict("index"),
        "adjusted": adjusted.loc[arms].to_dict("index"),
        "adjusted_for": [c for c in covariate_columns(frame) if c != treatment],
    }


def proportional_hazards_check(df: pd.DataFrame, covariates: list[str] | None = None,
                               alpha: float = 0.05) -> pd.DataFrame:
    """Schoenfeld-residual test per model term. A small p-value means that term's hazard ratio
    changes over follow-up, so a single hazard ratio misdescribes it."""
    design = design_matrix(df, covariates)
    cox = CoxPHFitter().fit(design, TIME, EVENT)
    result = proportional_hazard_test(cox, design, time_transform="rank").summary
    out = pd.DataFrame({"test_statistic": result["test_statistic"], "p_value": result["p"]})
    out["violates_assumption"] = out["p_value"] < alpha
    return out
