import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import shap
import io

# Load and clean data
data = pd.read_csv("medsurvive_updated_synthetic_data.csv")
data = data.dropna()  # Clean NaNs

# Sidebar filters
st.sidebar.title("🔍 Filter Patients")
age_range = st.sidebar.slider("Age Range", int(data.age.min()), int(data.age.max()), (30, 70))
sex_filter = st.sidebar.multiselect("Sex", options=data.sex.unique(), default=list(data.sex.unique()))
diag_filter = st.sidebar.multiselect("Diagnosis Code", options=data.diagnosis_code.unique(), default=list(data.diagnosis_code.unique()))
treatment_filter = st.sidebar.multiselect("Treatment Type", options=data.treatment_type.unique(), default=list(data.treatment_type.unique()))

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
    "surgical": "Involves physical interventions such as operations or procedures.",
    "medical": "Uses drugs or medications to treat conditions.",
    "combined": "Uses both surgical and medical treatments for comprehensive care.",
    "non-invasive": "Treatments that do not require entering the body or breaking the skin (e.g., therapy, lifestyle changes)."
}

selected_treatment_type = st.sidebar.selectbox("💊 What does this treatment type mean?", list(treatment_info.keys()))
st.sidebar.success(f"**{selected_treatment_type}** — {treatment_info[selected_treatment_type]}")

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
    st.dataframe(filtered_data)

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
try:
    data_encoded = pd.get_dummies(filtered_data.drop(columns=["patient_id"]), drop_first=True)
    cph = CoxPHFitter()
    cph.fit(data_encoded, duration_col="duration", event_col="event")
    st.write(cph.summary)

    fig_cox = cph.plot()
    st.pyplot(fig_cox.figure)

    # Download Cox summary
    cox_summary_buffer = io.StringIO()
    cph.summary.to_csv(cox_summary_buffer)
    st.sidebar.download_button("⬇️ Download Cox Summary", cox_summary_buffer.getvalue(), "cox_summary.csv", "text/csv")

except Exception as e:
    st.error(f"Error fitting Cox model: {e}")

# SHAP Summary Plot
if st.checkbox("Show SHAP summary plot"):
    st.subheader("🧠 Feature Importance (SHAP Values)")
    try:
        explainer = shap.Explainer(cph.predict_partial_hazard, data_encoded)
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
