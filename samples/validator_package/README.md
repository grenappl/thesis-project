# Validator package

For: Sabrinah Yonell Yap (data-science validation, thesis Section 4.9.3(C)). Read `../data_scientist_review_guide.md` first; this folder is the concrete material that accompanies it.

| File | What it is |
|---|---|
| `model/xgboost_alignment_augmented.json` | The XGBoost Alignment-Augmented model, as XGBoost's version-stable JSON. **518 trees = best early-stopping round 517 + 1**, i.e. exactly the model the thesis reports; plain `Booster.predict` needs no `iteration_range`. |
| `model/model_metadata.json` | Feature order (25 columns), selected hyperparameters, test metrics, training provenance. |
| `sample_output.csv` | 200 held-out test tracks drawn uniformly at random (seed 42): inputs, actual popularity, the served prediction, notebook 7's saved prediction, and the three largest SHAP drivers. |
| `sample_responses.json` | Three complete `POST /predict` request/response pairs (tracks at the 10th/50th/90th percentile of actual popularity), including the per-feature TreeSHAP breakdown. |
| `bias_report.md`, `bias_summary.csv` | Residual bias across popularity, genre, year, alignment gap and more, over all 37,632 test tracks. |

## How to check it (about 5 minutes)

1. **Model identity.** SHA-256 of what you were sent:

```
30a7eb803e5253ce21c582e6c17fe186b59804e0e52eb17a085b71ff0112dfe1  model/xgboost_alignment_augmented.json
b998784330f2674b141238cc08e71ccd238b37f87b0bc39b3098f1248a701b12  model/model_metadata.json
```

2. **It is the reported model.** `uv run python scripts/export_popularity_model.py` retrains from the frozen configuration and must print `test R2 +0.06783  RMSE 15.0927  MAE 12.1144` and a maximum difference from notebook 7 of ~1e-6.
3. **The sample is honest.** In `sample_output.csv`, `served_minus_notebook7` should be ~0 for every row (max here: 4.2e-05); recompute RMSE/MAE of `served_prediction` vs `actual_popularity` — they will be near the test-set figures (RMSE ≈ 15), not better.
4. **Bias.** Every number in `bias_report.md` is computed by `scripts/make_validator_package.py` from `notebooks/artifacts/`; nothing is hand-edited. You do not need to run it (it loads the web app's prediction service, so it needs the full project environment, `uv sync`, not the light modeling one). To check the report independently, recompute the residuals from `notebooks/artifacts/test_predictions.csv` and the corpus labels, as `scripts/verify_reference_figures.py` does for the metrics.

## Read the sample correctly

Errors of 10–20 points on individual tracks are normal: the model's R² is ≈ 0.07 (`reviewer_reference.md`, Section A), so it predicts near the average and cannot separate hits from misses. The sample is meant to show *what the model outputs and how it fails*, not to demonstrate accuracy. Do not compare it against hand-picked famous songs — those are the model's worst case by construction.

## Not included

The Random Forest models (several GB; not committed). Notebook 7 retrains all nine models if you want them.
