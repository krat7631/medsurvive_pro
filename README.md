# 🩺 MedSurvive Pro: Survival & Risk Insights Web App

**MedSurvive Pro** is an interactive, explainable web application that enables medical professionals, researchers, and data enthusiasts to explore survival outcomes, assess treatment effectiveness, and visualize patient risk using real or synthetic clinical data.

Built using **Streamlit**, **Lifelines**, **SHAP**, and **Plotly**, the app supports in-depth survival analysis via Kaplan–Meier plots, Cox Proportional Hazards modeling, and feature importance interpretation.

---

## 🚀 Features

- 📊 **Kaplan–Meier Survival Curves** – interactive and grouped by sex, diagnosis code, or treatment type
- 📈 **Cox Proportional Hazards Model** – statistical insights into survival risk factors
- 🧠 **SHAP Feature Importance** – model transparency and feature explanations
- 🧰 **Sidebar Filters** – filter patients by age, sex, diagnosis, and treatment
- 📥 **Download CSVs** – export filtered data or Cox regression summaries
- 💊 **Diagnosis Explanations** – understand medical codes and suggested treatments

---

## 🧪 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medsurvive-pro.git
cd medsurvive-pro
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Or (macOS/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Launch the App

```bash
streamlit run main.py
```
alternative
python -m streamlit run medsurvive_pro_app.py

Visit [http://localhost:8501](http://localhost:8501) in your browser to start using the app.

---

## 📁 Using Your Own CSV

To use your own clinical dataset:

### Step 1 – Replace the CSV File

Replace the example dataset with your own:

```
/medsurvive_pro/medsurvive_updated_synthetic_data.csv
```

### Step 2 – Required Columns

Ensure your CSV contains the following columns:

| Column           | Description                                      |
|------------------|--------------------------------------------------|
| `patient_id`     | Unique identifier for each patient               |
| `age`            | Patient’s age (numeric)                          |
| `sex`            | Categorical – e.g., "Male", "Female"             |
| `diagnosis_code` | ICD-10 diagnosis codes – e.g., "I10", "E11"      |
| `treatment_type` | Treatment category – e.g., "Medication", "Surgery" |
| `duration`       | Time in days patient was followed                |
| `event`          | 1 if event occurred (death, relapse), else 0     |

### Step 3 – Remove Missing Values

```python
df = df.dropna()
```

---

## ⚠️ Cox Model: Common Error

> ❗ **Error fitting Cox model: Convergence halted due to matrix inversion problems. Suspicion is high collinearity.**

### Cause:
This is common with small datasets or datasets with many one-hot encoded columns. Lifelines fails to fit the Cox model when:

- There’s **high collinearity** (correlated variables)
- The **matrix is singular** due to insufficient data
- Too many categorical variables with few samples

### Solutions:

✅ Recommended for larger datasets:

```python
cph = CoxPHFitter(penalizer=0.1)
```

✅ Drop or combine sparse categories  
✅ Ensure sample size > number of encoded features  
✅ Use this error as an **indicator** that the dataset needs enrichment

🔗 More info in the [Lifelines Troubleshooting Docs](https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model)

---

## 🧠 Diagnosis Code Reference

The app allows users to select an ICD-10 diagnosis code and displays:

- ✅ **Condition name**
- 💊 **Suggested treatments**
- 🧪 **Recommended procedures**

To add more codes, modify the `diagnosis_map` dictionary inside `main.py`.

```python
diagnosis_map = {
    "I10": ("Hypertension", ["ACE Inhibitors", "Beta Blockers"], ["BP Monitoring"]),
    "E11": ("Type 2 Diabetes", ["Insulin", "Metformin"], ["HbA1c Test", "Eye Exam"]),
    # Add more codes as needed
}
```

---

## 📦 Downloadable Reports

From the sidebar, users can:

- ⬇️ Download **filtered dataset** as CSV
- ⬇️ Download **Cox model summary** if the model fits

---

## 📚 File Overview

```
medsurvive_pro/
├── main.py                     # Streamlit app code
├── medsurvive_updated_synthetic_data.csv  # Sample dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
```

## 🙌 Credits

- [Streamlit](https://streamlit.io/)
- [Lifelines](https://lifelines.readthedocs.io/)
- [Plotly](https://plotly.com/)
- [SHAP](https://github.com/slundberg/shap)
