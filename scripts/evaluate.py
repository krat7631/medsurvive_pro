"""Reproduce every number in the README and precompute the app's slow artefacts.

For each bundled dataset this writes reports/<dataset>_report.json with:
- Kaplan-Meier medians and the log-rank test for the treatment arms;
- crude and covariate-adjusted treatment hazard ratios;
- the proportional-hazards check;
- repeated cross-validated C-indices and integrated Brier scores (KM baseline, Cox, RSF);
and reports/<dataset>_rsf_shap.csv with random-forest SHAP values, which take minutes to compute
and are therefore loaded by the app rather than computed on demand.

Usage:
    python scripts/evaluate.py            # both datasets, ~10 minutes on a laptop CPU
    python scripts/evaluate.py --quick    # fewer CV repeats and SHAP samples, for a pipeline check
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import lifelines
import sksurv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsurvive.analysis import kaplan_meier, logrank, proportional_hazards_check, treatment_effect  # noqa: E402
from medsurvive.data import DATASETS, load_dataset  # noqa: E402
from medsurvive.explain import explain_risk  # noqa: E402
from medsurvive.modeling import cross_validate  # noqa: E402

REPORTS = ROOT / "reports"


def evaluate(key: str, quick: bool) -> dict:
    info = DATASETS[key]
    df = load_dataset(key)
    started = time.time()

    curves = kaplan_meier(df, info.treatment)
    ph = proportional_hazards_check(df)
    report = {
        "dataset": info.title,
        "source": info.source,
        "patients": len(df),
        "events": int(df["event"].sum()),
        "censored_share": round(float(1 - df["event"].mean()), 4),
        "time_unit": info.time_unit,
        "event_meaning": info.event_meaning,
        "treatment": {
            "column": info.treatment,
            "reference": info.reference_arm,
            "randomized": info.randomized_treatment,
            "arms": [{"group": c.group, "patients": c.n, "events": c.events, "median_time": c.median} for c in curves],
            "logrank": logrank(df, info.treatment),
            "hazard_ratios": treatment_effect(df, info.treatment, info.reference_arm),
        },
        "proportional_hazards": ph.round(4).to_dict("index"),
        "cross_validation": cross_validate(df, repeats=1 if quick else 3),
    }

    shap = explain_risk(df, explain_n=30 if quick else 150)
    report["rsf_shap_importance"] = shap["feature_importance"]
    shap_rows = shap["shap_values"].add_prefix("shap_").join(shap["feature_values"])
    shap_rows.round(4).to_csv(REPORTS / f"{key}_rsf_shap.csv", index=False)

    report["runtime_minutes"] = round((time.time() - started) / 60, 1)
    report["environment"] = {"python": platform.python_version(), "lifelines": lifelines.__version__,
                             "scikit-survival": sksurv.__version__}
    report["quick"] = quick
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASETS), action="append")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    for key in args.dataset or list(DATASETS):
        print(f"Evaluating {key}…", flush=True)
        report = evaluate(key, args.quick)
        (REPORTS / f"{key}_report.json").write_text(json.dumps(report, indent=2))
        models = report["cross_validation"]["models"]
        print(f"  C-index  Cox {models['cox']['harrell_c']['mean']:.3f} | RSF {models['rsf']['harrell_c']['mean']:.3f} "
              f"| written in {report['runtime_minutes']} min", flush=True)


if __name__ == "__main__":
    main()
