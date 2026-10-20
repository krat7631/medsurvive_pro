import pandas as pd
import pytest

from medsurvive.data import DATASETS, EVENT, TIME, DatasetError, covariate_columns, load_dataset, prepare_uploaded


@pytest.mark.parametrize("key", list(DATASETS))
def test_bundled_datasets_keep_censored_patients(key):
    df = load_dataset(key)
    assert {TIME, EVENT} <= set(df.columns)
    assert set(df[EVENT].unique()) == {0, 1}
    # Regression test: the first version dropped every censored patient via dropna().
    assert (df[EVENT] == 0).sum() > 0


@pytest.mark.parametrize("key", list(DATASETS))
def test_registry_matches_data(key):
    info, df = DATASETS[key], load_dataset(key)
    assert info.treatment in df.columns
    assert info.reference_arm in set(df[info.treatment].astype(str))
    assert set(info.column_labels) <= set(df.columns)


def test_outcome_columns_are_never_covariates():
    df = load_dataset("gbsg2")
    assert TIME not in covariate_columns(df)
    assert EVENT not in covariate_columns(df)


def test_prepare_uploaded_maps_columns_and_keeps_missing_covariates():
    raw = pd.DataFrame({"days": [10, 20, 30, None], "died": [1, 0, 1, 1], "age": [50, None, 70, 60]})
    df = prepare_uploaded(raw, "days", "died")
    assert len(df) == 3  # only the row without a time is dropped
    assert df[EVENT].tolist() == [1, 0, 1]
    assert df["age"].isna().sum() == 1


@pytest.mark.parametrize(
    "raw, message",
    [
        (pd.DataFrame({"t": [1, 2], "e": [0, 2]}), "only 0"),
        (pd.DataFrame({"t": [-1, 2], "e": [1, 0]}), "negative"),
        (pd.DataFrame({"t": [1, 2], "e": [0, 0]}), "no observed events"),
        (pd.DataFrame({"t": ["a", "b"], "e": [1, 0]}), "No rows"),
    ],
)
def test_prepare_uploaded_rejects_invalid_outcomes(raw, message):
    with pytest.raises(DatasetError, match=message):
        prepare_uploaded(raw, "t", "e")


def test_prepare_uploaded_reports_missing_columns():
    with pytest.raises(DatasetError, match="not found"):
        prepare_uploaded(pd.DataFrame({"t": [1]}), "time", "event")
