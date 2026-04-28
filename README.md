# MedSurvive Pro

MedSurvive Pro is a beginner-friendly Streamlit project for exploring survival analysis in healthcare-style datasets.
It combines visual analytics (Kaplan-Meier curves), statistical risk modeling (Cox Proportional Hazards), and model explainability (SHAP) in a single web app.

---

## Project Aim

The goal of this project is to make survival analysis understandable and interactive for:

- students learning medical or health data analytics
- researchers prototyping ideas quickly
- data science beginners who want a practical end-to-end project

Instead of writing long scripts and plotting manually, users can filter data, run models, and download outputs from the app interface.

---

## Why This Project Matters

Many beginners struggle with survival analysis because it combines:

- time-to-event data concepts
- statistical assumptions
- multiple libraries and preprocessing steps

This project simplifies that workflow by packaging everything into one app where you can:

- compare survival probabilities across groups
- estimate risk effects of variables
- inspect feature influence in a transparent way

It is built for learning and prototyping, not clinical decision-making.

---

## Important Data Notice

This repository includes a **synthetic sample dataset** (`medsurvive_updated_synthetic_data.csv`) for demonstration and prototyping.

- It is **not real patient data**.
- It is **not validated for clinical use**.
- Do **not** use this app to make patient care decisions.

---

## What You Can Do in the App

- Filter patients by age, sex, diagnosis, and treatment cohort
- Plot Kaplan-Meier survival curves for selected groups
- Fit a Cox Proportional Hazards model and inspect coefficients
- Generate SHAP-based feature importance (when model fit is successful)
- Export filtered data and Cox summary CSV files
- Read diagnosis and treatment guidance in the sidebar

---

## Tech Stack

- Python 3.10+
- Streamlit
- Pandas
- Lifelines
- Plotly
- Matplotlib
- SHAP

---

## Project Structure

```text
.
├── medsurvive_pro_app.py                 # Main Streamlit application
├── medsurvive_updated_synthetic_data.csv # Demo dataset (synthetic)
├── requirements.txt                      # Python dependencies
├── tests/
│   └── test_data_contract.py             # Basic dataset contract tests
├── pyproject.toml                        # Tooling config (pytest + ruff)
├── .gitignore
├── LICENSE
└── README.md
```

---

## How to Download and Run (Step-by-Step)

### 1) Clone the repository

```bash
git clone https://github.com/krat7631/medsurvive_pro.git
cd medsurvive_pro
```

### 2) Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
python -m streamlit run medsurvive_pro_app.py
```

### 5) Open in browser

Go to [http://localhost:8501](http://localhost:8501)

---

## One-Command Quick Run (macOS/Linux)

If you prefer a single command:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python -m streamlit run medsurvive_pro_app.py
```

---

## Using Your Own Dataset

You can replace the sample CSV with your own file, but it must include these required columns:

- `patient_id`
- `age`
- `sex`
- `diagnosis_code`
- `treatment_type`
- `duration`
- `event`

The app validates required columns at startup and stops with a clear error if columns are missing.

---

## Troubleshooting for Beginners

### `python` or `pip` command not found

- Try `python3` and `pip3` instead.
- Make sure Python is installed and added to PATH.

### Port already in use

Run Streamlit on another port:

```bash
python -m streamlit run medsurvive_pro_app.py --server.port 8502
```

### App opens but model fails

This can happen when filtered data is too small or highly collinear.
Try:

- broadening filters
- increasing sample size
- reducing sparse categories

---

## Testing and Code Quality

Run tests:

```bash
pytest -q
```

Run lint checks:

```bash
ruff check .
```

---

## Limitations and Ethics

- Synthetic datasets may hide real-world bias and complexity.
- Survival models require domain validation and assumption checking.
- Results are exploratory and educational, not diagnostic.

---

## License

This project is licensed under the MIT License. See `LICENSE`.
