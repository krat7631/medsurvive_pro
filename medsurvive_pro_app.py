"""MedSurvive Pro: survival analysis and treatment-effect estimation on real clinical trial data.

Run with:  streamlit run medsurvive_pro_app.py
"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from medsurvive.analysis import (
    fit_cox,
    hazard_ratio_table,
    kaplan_meier,
    logrank,
    proportional_hazards_check,
    treatment_effect,
)
from medsurvive.data import (
    DATASETS,
    EVENT,
    TIME,
    DatasetError,
    categorical_columns,
    covariate_columns,
    load_dataset,
    prepare_uploaded,
)
from medsurvive.explain import explain_cox
from medsurvive.modeling import MODEL_NAMES, cross_validate
from medsurvive.plots import cv_figure, forest_figure, km_figure, shap_strip_figure

REPORTS = Path(__file__).parent / "reports"
UPLOAD = "upload"

st.set_page_config(page_title="MedSurvive Pro", layout="wide")


@st.cache_data
def bundled(key: str) -> pd.DataFrame:
    return load_dataset(key)


@st.cache_data
def precomputed_report(key: str) -> dict | None:
    path = REPORTS / f"{key}_report.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def precomputed_shap(key: str) -> pd.DataFrame | None:
    path = REPORTS / f"{key}_rsf_shap.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Running 5-fold cross-validation…")
def cached_cv(df: pd.DataFrame) -> dict:
    return cross_validate(df, repeats=1)


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def mean_std(scores: dict, metric: str) -> str:
    return f"{scores[metric]['mean']:.3f} ± {scores[metric]['std']:.3f}"


def describe_hr(hr: float, lo: float, hi: float, p: float, arm: str, reference: str) -> str:
    if lo < 1 < hi:
        return (f"**No clear difference** between *{arm}* and *{reference}*: HR {hr:.2f} "
                f"(95% CI {lo:.2f}–{hi:.2f}, p = {fmt_p(p)}). The interval includes 1.")
    direction = "lower" if hr < 1 else "higher"
    return (f"*{arm}* is associated with **{abs(1 - hr):.0%} {direction} hazard** than *{reference}*: "
            f"HR {hr:.2f} (95% CI {lo:.2f}–{hi:.2f}, p = {fmt_p(p)}).")


# ── Sidebar: dataset ──────────────────────────────────────────────────────────
st.sidebar.title("MedSurvive Pro")
options = {**{k: v.title for k, v in DATASETS.items()}, UPLOAD: "Upload your own CSV"}
choice = st.sidebar.radio("Dataset", list(options), format_func=options.get)

info = DATASETS.get(choice)
if choice == UPLOAD:
    uploaded = st.sidebar.file_uploader("CSV file", type="csv")
    if uploaded is None:
        st.title("MedSurvive Pro")
        st.info("Upload a CSV with one row per patient, a follow-up time column and a 0/1 event column.")
        st.stop()
    raw = pd.read_csv(uploaded)
    time_col = st.sidebar.selectbox("Follow-up time column", raw.columns)
    event_col = st.sidebar.selectbox("Event column (1 = event, 0 = censored)", raw.columns, index=min(1, len(raw.columns) - 1))
    try:
        data = prepare_uploaded(raw, time_col, event_col)
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].astype("category")
    group_options = categorical_columns(data)
    treatment = st.sidebar.selectbox("Treatment column (optional)", [None] + group_options)
    reference = (st.sidebar.selectbox("Reference arm", sorted(map(str, data[treatment].dropna().unique())))
                 if treatment else None)
    title, time_unit, event_meaning = uploaded.name, "time units", "event"
else:
    data = bundled(choice)
    treatment, reference = info.treatment, info.reference_arm
    title, time_unit, event_meaning = info.title, info.time_unit, info.event_meaning

labels = info.column_labels if info else {}
label = lambda c: labels.get(c, c)  # noqa: E731


def term_label(term: str) -> str:
    """Readable name for a model term: 'tgrade_II' or 'tgrade=II' -> 'Tumour grade: II'."""
    if term in data.columns:
        return label(term)
    for col in sorted(covariate_columns(data), key=len, reverse=True):
        for sep in ("_", "="):
            if term.startswith(f"{col}{sep}"):
                return f"{label(col)}: {term.removeprefix(f'{col}{sep}')}"
    return term

# ── Header ────────────────────────────────────────────────────────────────────
st.title(title)
if info:
    st.caption(f"{info.description} Source: {info.source}")
st.warning("Educational and research use only. These analyses are not clinical decision support.", icon="⚕️")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(data):,}")
c2.metric(f"Events ({event_meaning})", f"{int(data[EVENT].sum()):,}")
c3.metric("Censored", f"{1 - data[EVENT].mean():.0%}")
c4.metric("Longest follow-up", f"{data[TIME].max():,.0f} {time_unit}")

tabs = st.tabs(["Survival curves", "Treatment effect", "Prediction models", "Explanations", "Assumptions", "Data"])

# ── Survival curves ───────────────────────────────────────────────────────────
with tabs[0]:
    group_choices = [None] + categorical_columns(data)
    default = group_choices.index(treatment) if treatment in group_choices else 0
    group = st.selectbox("Compare groups by", group_choices, index=default,
                         format_func=lambda c: "No grouping" if c is None else label(c))
    curves = kaplan_meier(data, group)
    st.plotly_chart(km_figure(curves, time_unit), use_container_width=True)
    st.caption("Kaplan–Meier estimates with 95% confidence bands. Vertical ticks mark censored patients, "
               "who left follow-up without an event and still count until then.")

    medians = pd.DataFrame([{"Group": c.group, "Patients": c.n, "Events": c.events,
                             f"Median time ({time_unit})": "not reached" if c.median is None else f"{c.median:,.0f}"}
                            for c in curves])
    st.dataframe(medians, hide_index=True, use_container_width=True)
    if group and len(curves) >= 2:
        lr = logrank(data, group)
        st.markdown(f"**Log-rank test:** χ² = {lr['test_statistic']:.2f}, df = {lr['degrees_of_freedom']}, "
                    f"p = {fmt_p(lr['p_value'])}"
                    + (" — survival differs between groups." if lr["p_value"] < 0.05
                       else " — no significant difference between groups."))

# ── Treatment effect ──────────────────────────────────────────────────────────
with tabs[1]:
    if not treatment:
        st.info("Choose a treatment column in the sidebar to estimate treatment effects.")
    else:
        effect = treatment_effect(data, treatment, reference)
        for arm, crude in effect["crude"].items():
            adjusted = effect["adjusted"][arm]
            arm_name = arm.removeprefix(f"{treatment}_")
            arm_text, ref_text = f"{label(treatment)}: {arm_name}", f"{label(treatment)}: {reference}"
            st.subheader(f"{label(treatment)}: {arm_name} vs {reference}")
            left, right = st.columns(2)
            left.markdown("**Unadjusted**\n\n" + describe_hr(crude["hazard_ratio"], crude["ci_lower"],
                          crude["ci_upper"], crude["p_value"], arm_text, ref_text))
            right.markdown("**Adjusted** for " + ", ".join(label(c) for c in effect["adjusted_for"]) + "\n\n"
                           + describe_hr(adjusted["hazard_ratio"], adjusted["ci_lower"], adjusted["ci_upper"],
                                         adjusted["p_value"], arm_text, ref_text))
        if info and info.randomized_treatment:
            st.caption("Treatment was randomised in this trial, so the unadjusted estimate is a fair comparison; "
                       "adjustment can still improve precision.")
        else:
            st.caption("Read these as associations. Without documented randomisation, patients who received the "
                       "treatment may differ in ways the adjusted model does not capture.")

        st.markdown("#### Every covariate in the adjusted Cox model")
        table = hazard_ratio_table(fit_cox(data))
        st.plotly_chart(forest_figure(table.rename(index=term_label)), use_container_width=True)
        st.caption("Hazard ratio above 1 = higher risk of the event; below 1 = lower risk. Orange and blue mark "
                   "intervals that exclude 1. Continuous variables show the ratio per one-unit increase (per year, "
                   "per node, per mm), so small-looking ratios can add up; categorical variables are compared with "
                   "their first level.")
        st.download_button("Download Cox summary (CSV)", table.to_csv().encode(), "cox_summary.csv", "text/csv")

# ── Prediction models ─────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("How well do patient characteristics **predict** who has an event first? Every number below "
                "is measured on patients held out from training.")
    report = precomputed_report(choice) if info else None
    cv = report["cross_validation"] if report else None
    if cv is None:
        if st.button("Run cross-validation (about a minute)"):
            cv = cached_cv(data)
    if cv:
        st.plotly_chart(cv_figure(cv, MODEL_NAMES), use_container_width=True)
        rows = [{"Model": MODEL_NAMES[m], "Harrell's C": mean_std(s, "harrell_c"), "Uno's C": mean_std(s, "uno_c"),
                 "Integrated Brier (lower is better)": mean_std(s, "integrated_brier")}
                for m, s in cv["models"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        repeats = f"{cv['repeats']} repeat{'s' if cv['repeats'] != 1 else ''}"
        st.caption(f"{cv['folds']}-fold cross-validation × {repeats}, stratified by event. "
                   "C-index: 0.5 = chance ordering, 1.0 = perfect. The Kaplan–Meier baseline gives everyone the "
                   "same curve, so any useful model must beat it. Differences smaller than the ± spread are not "
                   "meaningful.")

# ── Explanations ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("Which characteristics push an individual patient's predicted risk up or down?")
    cox_shap = explain_cox(data)
    st.markdown("#### Cox model (exact, every patient)")
    st.plotly_chart(shap_strip_figure(cox_shap["shap_values"].rename(columns=term_label),
                                      cox_shap["feature_values"].rename(columns=term_label)), use_container_width=True)
    rsf = precomputed_shap(choice) if info else None
    if rsf is not None:
        sample = "every patient" if len(rsf) >= len(data) else f"{len(rsf)}-patient sample"
        st.markdown(f"#### Random survival forest (SHAP, {sample}, precomputed)")
        shap_cols = [c for c in rsf.columns if c.startswith("shap_")]
        values = rsf[shap_cols].rename(columns=lambda c: c.removeprefix("shap_"))
        st.plotly_chart(shap_strip_figure(values.rename(columns=term_label), rsf[values.columns].rename(columns=term_label)),
                        use_container_width=True)
    st.caption("Each dot is a patient; colour runs from low (blue) to high (orange) values of that feature. "
               "Attributions explain the models' predictions, not causal effects.")

# ── Assumptions ───────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("A Cox model reports one hazard ratio per covariate, which is only valid if that ratio stays "
                "roughly constant over follow-up (*proportional hazards*). This checks each term with a "
                "Schoenfeld-residual test.")
    ph = proportional_hazards_check(data).rename(index=term_label)
    flagged = ph[ph["violates_assumption"]]
    display = ph.assign(p_value=ph["p_value"].map(fmt_p),
                        violates_assumption=ph["violates_assumption"].map({True: "⚠ yes", False: "no"}))
    st.dataframe(display.rename(columns={"test_statistic": "Test statistic", "p_value": "p-value",
                                         "violates_assumption": "Violates (p < 0.05)"}), use_container_width=True)
    if flagged.empty:
        st.success("No term shows evidence against proportional hazards.")
    else:
        st.warning("Hazard ratios for " + ", ".join(flagged.index) + " change over follow-up. Treat their single "
                   "hazard ratio as an average; stratifying on them or adding a time interaction would model "
                   "them properly. The random survival forest does not make this assumption.")

# ── Data ──────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.dataframe(data, use_container_width=True, height=420)
    buffer = io.StringIO()
    data.to_csv(buffer, index=False)
    st.download_button("Download data (CSV)", buffer.getvalue(), f"{choice}.csv", "text/csv")
    st.caption(f"`{TIME}` = follow-up in {time_unit}; `{EVENT}` = 1 if {event_meaning} was observed, 0 if censored. "
               f"Covariates: {', '.join(label(c) for c in covariate_columns(data))}.")
