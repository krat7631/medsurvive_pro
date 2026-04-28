import pandas as pd

REQUIRED_COLUMNS = {
    "patient_id",
    "age",
    "sex",
    "diagnosis_code",
    "treatment_type",
    "duration",
    "event",
}


def test_dataset_contains_required_columns():
    df = pd.read_csv("medsurvive_updated_synthetic_data.csv")
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"


def test_event_is_binary():
    df = pd.read_csv("medsurvive_updated_synthetic_data.csv")
    assert set(df["event"].dropna().unique()).issubset({0, 1})
