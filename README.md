# MedSurvive Pro

Interactive survival analysis on real clinical trial data: Kaplan–Meier curves, treatment-effect
estimation with Cox models, a check of the assumption those models rely on, and an honest comparison of
Cox regression against a random survival forest on held-out patients.

- **Data:** two published trials bundled with scikit-survival, plus upload of any CSV with a time and an event column.
- **Statistics:** Kaplan–Meier with 95% bands and censoring marks, log-rank tests, crude and adjusted hazard
  ratios, Schoenfeld-residual tests of proportional hazards.
- **Prediction:** Cox vs random survival forest vs a no-information baseline, repeated cross-validation,
  Harrell's and Uno's C-index, integrated Brier score.
- **Explanation:** exact SHAP values for the Cox model, permutation SHAP for the forest.
- **Stack:** Python · Streamlit · lifelines · scikit-survival · SHAP · Plotly · pytest · GitHub Actions.

> Educational and research use only. Nothing here is clinical decision support.

## Results

### Treatment effects

| Trial | Comparison | Log-rank p | Hazard ratio (95% CI) | Adjusted HR (95% CI) |
|---|---|---|---|---|
| GBSG2 breast cancer, 686 patients | Hormonal therapy vs none | 0.003 | **0.69** (0.54–0.89) | **0.71** (0.55–0.91) |
| Veterans lung cancer, 137 patients (randomised) | Test vs standard chemotherapy | 0.93 | 1.02 (0.71–1.45) | 1.34 (0.89–2.02) |

Hormonal therapy is associated with roughly 30% lower hazard of recurrence or death, and adjusting for
age, tumour size and grade, nodes, receptor status and menopausal status barely moves the estimate.
The test chemotherapy shows no detectable effect: both intervals include 1.

### Prediction on held-out patients

5-fold cross-validation × 3 repeats, stratified by event (mean ± standard deviation across folds).

| Trial | Model | Harrell's C | Uno's C | Integrated Brier ↓ |
|---|---|---|---|---|
| GBSG2 | Kaplan–Meier baseline | 0.500 | 0.500 | 0.175 ± 0.007 |
| GBSG2 | Cox | 0.679 ± 0.040 | 0.679 ± 0.044 | 0.157 ± 0.009 |
| GBSG2 | Random survival forest | **0.691 ± 0.025** | **0.690 ± 0.027** | **0.152 ± 0.009** |
| Veterans | Kaplan–Meier baseline | 0.500 | 0.500 | 0.224 ± 0.013 |
| Veterans | Cox | **0.708 ± 0.037** | **0.706 ± 0.036** | **0.173 ± 0.018** |
| Veterans | Random survival forest | 0.685 ± 0.057 | 0.682 ± 0.056 | 0.195 ± 0.015 |

Both models beat the baseline on both trials. The forest is slightly ahead on the larger trial and the Cox
model on the smaller one, which is the usual pattern for a flexible model with little data, but the C-index
gaps are within one standard deviation, so neither is clearly better at ranking patients. On the 137-patient
trial the Cox model is clearly better calibrated (Brier 0.173 vs 0.195).

### Proportional hazards

A Cox hazard ratio assumes the ratio is constant over follow-up. The app tests every term:

- **Veterans:** Karnofsky performance score (p < 0.001), age and prior therapy violate it. Their single
  hazard ratios are averages over an effect that changes during follow-up; stratifying on them or adding a
  time interaction would model them properly. The random survival forest makes no such assumption, which is
  one reason to compare it with the Cox model rather than trust either alone.
- **GBSG2:** only tumour grade III shows evidence of non-proportional hazards (p = 0.015).

Full numbers: [`reports/gbsg2_report.json`](reports/gbsg2_report.json),
[`reports/veterans_report.json`](reports/veterans_report.json).

## Why version 2 changed the data

Version 1 analysed a synthetic CSV and had two errors that made its output unreliable:

1. **Censored patients were dropped.** `df.dropna()` removed every row with a missing death time — all 285
   censored patients (28.5%). Survival analysis exists to handle censoring; without those patients every
   curve and hazard ratio was biased towards early events.
2. **The outcome leaked into the model.** `death_time_days` and post-baseline ICU dates were Cox covariates.

Fixing both revealed a third problem: the synthetic data carried no signal. Covariates were unrelated to
survival (e.g. age vs follow-up time, r = −0.04), it contained impossible values (oxygen saturation up to
102.6%, ICU discharge after the end of follow-up), and with the leaks removed the models scored a C-index
of 0.49–0.51, indistinguishable from chance. Version 2 therefore uses real trial data, and its tests guard
against both original errors.

## How it is built

```
medsurvive/
  data.py       dataset registry, CSV validation, the time/event contract (censored rows are never dropped)
  analysis.py   Kaplan–Meier, log-rank, Cox hazard ratios, proportional-hazards tests
  modeling.py   KM baseline / Cox / random survival forest, repeated stratified CV, C-index and Brier score
  explain.py    exact Cox SHAP values; permutation SHAP for the forest
  plots.py      Plotly figures
medsurvive_pro_app.py   Streamlit UI (survival curves, treatment effect, prediction, explanations, assumptions, data)
scripts/evaluate.py     reproduces every number above and precomputes the forest's SHAP values
```

Decisions worth knowing:

- **Outcome columns can never become covariates** — `covariate_columns()` excludes them and a test checks it.
- **Missing covariates are handled per analysis, not by dropping rows up front,** so one missing lab value
  does not remove a patient from the Kaplan–Meier curve.
- **Brier scores skip test patients followed longer than anyone in the training fold,** because the
  censoring weights are undefined there; both C-indices use every test patient.
- **Forest SHAP is precomputed.** The permutation explainer takes about 3 s per patient on a laptop CPU, too
  slow for an interactive page; Cox attributions are exact and instant, so uploads use those.

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run medsurvive_pro_app.py      # http://localhost:8501
```

Reproduce the results (about 10 minutes on a laptop CPU):

```bash
python scripts/evaluate.py
```

Tests and lint:

```bash
pip install -r requirements-dev.txt
pytest -q && ruff check .
```

### Using your own data

Upload a CSV with one row per patient, a follow-up time column and a 0/1 event column (1 = event observed,
0 = censored). Any other columns become covariates; text columns are treated as categories. Pick a
treatment column in the sidebar to get hazard ratios against a reference arm.

## Limitations

- Hazard ratios from non-randomised comparisons are associations; unmeasured differences between groups can
  bias them however many covariates are adjusted for.
- Both trials are decades old and small by modern standards; the models are illustrations of method, not
  validated prognostic tools.
- SHAP values explain the models' predictions, not causal effects of patient characteristics.

## Data sources

- GBSG2: German Breast Cancer Study Group 2 trial (Schumacher et al., 1994), via `sksurv.datasets.load_gbsg2`.
- Veterans' Administration Lung Cancer Trial (Kalbfleisch & Prentice, 1980), via
  `sksurv.datasets.load_veterans_lung_cancer`.

## License

MIT — see [LICENSE](LICENSE).
