import io

import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter

st.set_page_config(
    page_title="MedSurvive Pro",
    page_icon="🩺",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "diagnosis_code",
    "treatment_type",
    "duration",
    "event",
]

TREATMENT_LABELS = {
    "A": "Medical Management",
    "B": "Procedure-Forward Care",
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna()


data = load_data("medsurvive_updated_synthetic_data.csv")

missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
if missing_columns:
    st.error(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
        + ". Please use the documented schema."
    )
    st.stop()

# Sidebar filters
st.sidebar.title("🔍 Filter Patients")
age_range = st.sidebar.slider("Age Range", int(data.age.min()), int(data.age.max()), (30, 70))
sex_filter = st.sidebar.multiselect("Sex", options=data.sex.unique(), default=list(data.sex.unique()))
diag_filter = st.sidebar.multiselect(
    "Diagnosis Code", options=data.diagnosis_code.unique(), default=list(data.diagnosis_code.unique())
)
treatment_filter = st.sidebar.multiselect(
    "Treatment Type", options=data.treatment_type.unique(), default=list(data.treatment_type.unique())
)

# Layman Explanation Section
st.sidebar.markdown("---")
st.sidebar.header("📖 Code & Treatment Info")

diag_info = {
    "I10": "Essential (primary) hypertension: high blood pressure with no identifiable cause.",
    "E11": "Type 2 diabetes mellitus: chronic condition affecting the way the body processes blood sugar.",
    "J44": "Chronic obstructive pulmonary disease (COPD): a group of lung conditions that cause breathing difficulties.",
    "K21": "Gastroesophageal reflux disease (GERD): acid reflux that irritates the esophagus.",
    "N18": "Chronic kidney disease: gradual loss of kidney function over time.",
    "F41": "Anxiety disorders: a group of mental health disorders characterized by excessive fear or anxiety.",
    "M54": "Back pain: pain in the back often due to musculoskeletal issues.",
    "R51": "Headache: pain in any region of the head."
}

selected_diag_explain = st.sidebar.selectbox("❓ What does this diagnosis code mean?", list(diag_info.keys()))
st.sidebar.info(f"**{selected_diag_explain}** — {diag_info[selected_diag_explain]}")

treatment_info = {
    "A (Medical Management)": "Primarily medicine-led, guideline-based treatment without major invasive procedures.",
    "B (Procedure-Forward Care)": "Care pathway with higher use of procedures, interventions, or operation-oriented management."
}

selected_treatment_type = st.sidebar.selectbox("💊 What does this treatment type mean?", list(treatment_info.keys()))
st.sidebar.success(f"**{selected_treatment_type}** — {treatment_info[selected_treatment_type]}")

# Data use disclaimer
st.warning(
    "This app uses synthetic sample data for educational and prototyping purposes. "
    "It is not clinical decision support and should not be used for patient care decisions."
)

# Apply filters
filtered_data = data[
    (data.age.between(age_range[0], age_range[1])) &
    (data.sex.isin(sex_filter)) &
    (data.diagnosis_code.isin(diag_filter)) &
    (data.treatment_type.isin(treatment_filter))
]

# Title
st.title("🩺 MedSurvive Pro: Survival & Risk Insights")

# Raw data toggle
if st.checkbox("Show filtered data"):
    table_view = filtered_data.copy()
    table_view["treatment_type"] = table_view["treatment_type"].map(TREATMENT_LABELS).fillna(table_view["treatment_type"])
    st.dataframe(table_view)

if filtered_data.empty:
    st.info("No records match the current filters. Widen the filters to continue analysis.")
    st.stop()

# Download CSV
csv_buffer = io.StringIO()
filtered_data.to_csv(csv_buffer, index=False)
st.sidebar.download_button("⬇️ Download Filtered CSV", csv_buffer.getvalue(), "filtered_data.csv", "text/csv")

# Kaplan-Meier Plot with Plotly
st.subheader("📈 Kaplan–Meier Survival Curve")
group_by = st.selectbox("Group curves by:", [None, "sex", "treatment_type", "diagnosis_code"])
kmf = KaplanMeierFitter()
fig_km = go.Figure()

if group_by:
    for name, grouped_df in filtered_data.groupby(group_by):
        if grouped_df.empty:
            continue
        kmf.fit(grouped_df["duration"], grouped_df["event"], label=str(name))
        fig_km.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[str(name)],
                                    mode='lines', name=str(name)))
else:
    kmf.fit(filtered_data["duration"], filtered_data["event"], label="All Patients")
    fig_km.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_["All Patients"],
                                mode='lines', name="All Patients"))

fig_km.update_layout(title="Kaplan–Meier Curve", xaxis_title="Days", yaxis_title="Survival Probability")
st.plotly_chart(fig_km)

# Cox Proportional Hazards Model
st.subheader("📊 Cox Proportional Hazards Model")
cox_model = None
data_encoded = None
try:
    data_encoded = pd.get_dummies(filtered_data.drop(columns=["patient_id"]), drop_first=True)
    if len(data_encoded) < 20:
        st.info("Cox model skipped: fewer than 20 filtered records often leads to unstable estimates.")
    else:
        cox_model = CoxPHFitter(penalizer=0.1)
        cox_model.fit(data_encoded, duration_col="duration", event_col="event")
        st.write(cox_model.summary)

        fig_cox = cox_model.plot()
        st.pyplot(fig_cox.figure)

        # Download Cox summary
        cox_summary_buffer = io.StringIO()
        cox_model.summary.to_csv(cox_summary_buffer)
        st.sidebar.download_button("⬇️ Download Cox Summary", cox_summary_buffer.getvalue(), "cox_summary.csv", "text/csv")

except Exception as e:
    st.error(
        f"Error fitting Cox model: {e}\n\n"
        "Tip: broaden filters, reduce sparse category levels, or use a larger dataset to improve convergence."
    )

# SHAP Summary Plot
if st.checkbox("Show SHAP summary plot"):
    st.subheader("🧠 Feature Importance (SHAP Values)")
    try:
        if cox_model is None or data_encoded is None:
            st.info("SHAP is available after a successful Cox model fit.")
        else:
            with st.spinner("Computing SHAP values..."):
                explainer = shap.Explainer(cox_model.predict_partial_hazard, data_encoded)
                shap_values = explainer(data_encoded)
                shap.summary_plot(shap_values, data_encoded, show=False)
                st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"SHAP error: {e}")

# Suggested Treatments
st.subheader("💊 Suggested Treatments")

diagnosis_map = {
    "I10": ("Hypertension", ["ACE Inhibitors", "Beta Blockers"], ["Blood Pressure Monitoring"]),
    "E11": ("Type 2 Diabetes", ["Insulin", "Metformin"], ["HbA1c Test", "Retinal Screening"]),
    "J44": ("COPD", ["Bronchodilators", "Steroids"], ["Pulmonary Function Test"]),
    "K21": ("GERD", ["Antacids", "PPIs"], ["Endoscopy"]),
    "N18": ("Chronic Kidney Disease", ["Dialysis", "ACE Inhibitors"], ["Creatinine Test", "GFR Measurement"]),
    "F41": ("Anxiety Disorders", ["SSRIs", "CBT"], ["Psychiatric Evaluation"]),
    "M54": ("Back Pain", ["NSAIDs", "Physical Therapy"], ["X-ray", "MRI"]),
    "R51": ("Headache", ["Analgesics", "Triptans"], ["Neurological Exam"]),
}

selected_code = st.selectbox("Select diagnosis code:", sorted(data["diagnosis_code"].unique()))
if selected_code in diagnosis_map:
    name, treatments, procedures = diagnosis_map[selected_code]
    st.markdown(f"**Condition:** {name}")
    st.markdown(f"**Recommended Treatments:** {', '.join(treatments)}")
    st.markdown(f"**Suggested Procedures:** {', '.join(procedures)}")

st.caption(
    "Treatment cohorts are coded as A/B in this synthetic dataset: "
    "A = Medical Management, B = Procedure-Forward Care."
)
