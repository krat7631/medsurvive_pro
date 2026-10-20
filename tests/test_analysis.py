import numpy as np
import pandas as pd
import pytest

from medsurvive.analysis import kaplan_meier, logrank, proportional_hazards_check, treatment_effect
from medsurvive.data import load_dataset
from medsurvive.explain import explain_cox
from medsurvive.modeling import cross_validate


@pytest.fixture(scope="module")
def gbsg2():
    return load_dataset("gbsg2")


def test_kaplan_meier_curves_are_monotone_and_bounded(gbsg2):
    for curve in kaplan_meier(gbsg2, "horTh"):
        s = np.array(curve.survival)
        assert (np.diff(s) <= 1e-12).all()
        assert s.min() >= 0 and s.max() <= 1
        assert curve.n == len(gbsg2[gbsg2["horTh"].astype(str) == curve.group])


def test_gbsg2_reproduces_hormonal_therapy_effect(gbsg2):
    # Hormonal therapy is associated with longer recurrence-free survival in this dataset
    # (hazard ratio of about 0.7); a change here means the pipeline changed, not the data.
    effect = treatment_effect(gbsg2, "horTh", "no")
    crude = effect["crude"]["horTh_yes"]
    assert crude["hazard_ratio"] == pytest.approx(0.69, abs=0.02)
    assert crude["ci_upper"] < 1
    assert logrank(gbsg2, "horTh")["p_value"] < 0.01


def test_veterans_trial_shows_no_treatment_effect():
    df = load_dataset("veterans")
    effect = treatment_effect(df, "Treatment", "standard")["crude"]["Treatment_test"]
    assert effect["ci_lower"] < 1 < effect["ci_upper"]


def test_proportional_hazards_check_flags_karnofsky_score():
    ph = proportional_hazards_check(load_dataset("veterans"))
    assert ph.loc["Karnofsky_score", "violates_assumption"]


def test_cox_shap_values_are_centred(gbsg2):
    result = explain_cox(gbsg2)
    assert result["shap_values"].shape == result["feature_values"].shape
    assert np.allclose(result["shap_values"].mean(), 0, atol=1e-8)


def test_models_beat_the_baseline_out_of_sample(gbsg2):
    cv = cross_validate(gbsg2, folds=3, repeats=1)
    assert cv["models"]["km"]["harrell_c"]["mean"] == pytest.approx(0.5)
    for name in ("cox", "rsf"):
        assert cv["models"][name]["harrell_c"]["mean"] > 0.6
        assert cv["models"][name]["integrated_brier"]["mean"] < cv["models"]["km"]["integrated_brier"]["mean"]


def test_analyses_handle_an_uploaded_dataset_without_treatment():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"time": rng.exponential(100, 80).round(1), "event": rng.integers(0, 2, 80),
                       "age": rng.integers(30, 80, 80), "group": pd.Categorical(rng.choice(["a", "b"], 80))})
    assert len(kaplan_meier(df)) == 1
    assert "p_value" in logrank(df, "group")
    assert not proportional_hazards_check(df).empty
