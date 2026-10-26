"""Datasets and the contract every analysis relies on.

Every dataset is normalised to one frame with a `time` column (follow-up time) and an `event`
column (1 = event observed, 0 = censored). Censored patients are kept: dropping them, as the
first version of this app did through a blanket `dropna()`, biases every survival estimate
towards earlier events.
"""

from dataclasses import dataclass, field

import pandas as pd
from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer

TIME, EVENT = "time", "event"


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    title: str
    description: str
    time_unit: str
    event_meaning: str
    treatment: str | None
    reference_arm: str | None
    # True only when the source documents a randomised comparison of this column; None = not asserted.
    randomized_treatment: bool | None
    source: str
    column_labels: dict[str, str] = field(default_factory=dict)


DATASETS: dict[str, DatasetInfo] = {
    "gbsg2": DatasetInfo(
        key="gbsg2",
        title="German Breast Cancer Study Group 2 (GBSG2)",
        description="686 women with primary node-positive breast cancer, followed for recurrence-free survival.",
        time_unit="days",
        event_meaning="recurrence or death",
        treatment="horTh",
        reference_arm="no",
        randomized_treatment=None,
        source="German Breast Cancer Study Group 2 trial (Schumacher et al., 1994); distributed with scikit-survival.",
        column_labels={
            "age": "Age (years)", "estrec": "Estrogen receptor (fmol/l)", "horTh": "Hormonal therapy",
            "menostat": "Menopausal status", "pnodes": "Positive lymph nodes", "progrec": "Progesterone receptor (fmol/l)",
            "tgrade": "Tumour grade", "tsize": "Tumour size (mm)",
        },
    ),
    "veterans": DatasetInfo(
        key="veterans",
        title="Veterans' Administration Lung Cancer Trial",
        description="137 men with advanced inoperable lung cancer, randomised to standard or test chemotherapy.",
        time_unit="days",
        event_meaning="death",
        treatment="Treatment",
        reference_arm="standard",
        randomized_treatment=True,
        source="Kalbfleisch & Prentice (1980), The Statistical Analysis of Failure Time Data; distributed with scikit-survival.",
        column_labels={
            "Age_in_years": "Age (years)", "Celltype": "Tumour cell type", "Karnofsky_score": "Karnofsky performance score",
            "Months_from_Diagnosis": "Months from diagnosis", "Prior_therapy": "Prior therapy", "Treatment": "Chemotherapy arm",
        },
    ),
}

_LOADERS = {"gbsg2": load_gbsg2, "veterans": load_veterans_lung_cancer}


class DatasetError(ValueError):
    """Raised when an uploaded dataset does not satisfy the survival-data contract."""


def load_dataset(key: str) -> pd.DataFrame:
    X, y = _LOADERS[key]()
    event_field, time_field = y.dtype.names
    df = X.copy()
    df[TIME] = y[time_field].astype(float)
    df[EVENT] = y[event_field].astype(int)
    return df.reset_index(drop=True)


def prepare_uploaded(raw: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    """Validate a user CSV and map its outcome columns onto `time` / `event`."""
    missing = [c for c in (time_col, event_col) if c not in raw.columns]
    if missing:
        raise DatasetError(f"Columns not found: {', '.join(missing)}.")

    df = raw.rename(columns={time_col: TIME, event_col: EVENT}).copy()
    df[TIME] = pd.to_numeric(df[TIME], errors="coerce")
    df[EVENT] = pd.to_numeric(df[EVENT], errors="coerce")
    # Only rows without an outcome are unusable; missing covariates are handled per analysis.
    df = df.dropna(subset=[TIME, EVENT])

    if df.empty:
        raise DatasetError("No rows have both a numeric time and a numeric event value.")
    if (df[TIME] < 0).any():
        raise DatasetError("Follow-up time cannot be negative.")
    if not set(df[EVENT].unique()) <= {0, 1}:
        raise DatasetError("The event column must contain only 0 (censored) and 1 (event).")
    if df[EVENT].sum() == 0:
        raise DatasetError("The dataset has no observed events, so survival cannot be estimated.")

    df[EVENT] = df[EVENT].astype(int)
    return df.reset_index(drop=True)


def covariate_columns(df: pd.DataFrame) -> list[str]:
    """Every column except the outcome. Outcome columns never enter a model."""
    return [c for c in df.columns if c not in (TIME, EVENT)]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in covariate_columns(df)
            if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object]
