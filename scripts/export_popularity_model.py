"""Exports the trained popularity model for the demo app's POST /predict.

Retrains the XGBoost Alignment-Augmented model -- the model Section 4.2.2
names as "the primary model for all interpretability reporting and for the
deployed demo" -- from the frozen configuration notebook 6 selected, on the
split notebook 5 produced, exactly as notebook 7 trains it. Then verifies
the result against the test-set predictions notebook 7 saved, and refuses to
write anything if they disagree:

    uv run python scripts/export_popularity_model.py

Why retrain rather than copy the notebook's .joblib: notebooks/artifacts/models/
is not committed (the Random Forest models are several GB), and a .joblib
pickle is tied to the exact XGBoost version that wrote it. XGBoost's own JSON
format is version-stable. The verification step is what makes retraining
safe: if the retrained model reproduces notebook 7's saved predictions, it
*is* the model the thesis reports on, not an approximation of it.

Writes:
    models/popularity/xgboost_alignment_augmented.json   the booster
    models/popularity/model_metadata.json                feature order, config,
                                                         test metrics, provenance

models/ is gitignored, like every other fetched-or-trained model in this repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / 'notebooks'
sys.path.insert(0, str(NOTEBOOKS))
import modeling_config as mc  # noqa: E402  (lives beside the notebooks, not in api/)

MODEL_NAME = 'alignment_augmented'
OUT_DIR = ROOT / 'models' / 'popularity'
MODEL_FILE = OUT_DIR / 'xgboost_alignment_augmented.json'
META_FILE = OUT_DIR / 'model_metadata.json'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='max allowed |retrained - notebook 7| test prediction '
                             'difference, in popularity points (default 1e-4)')
    args = parser.parse_args()

    config = mc.load_json('config_xgb.json')
    splits = pd.read_csv(mc.artifact('splits.csv'), usecols=['id', 'partition'])
    saved = pd.read_csv(mc.artifact('test_predictions.csv'),
                        usecols=['id', f'xgboost__{MODEL_NAME}'])

    # Identical construction to notebook 7, so the row order and the column
    # order are the ones the reported model was trained on.
    df = mc.load_corpus().merge(splits, on='id', how='inner')
    df = df[df['partition'].isin(['train', 'val', 'test'])].reset_index(drop=True)
    X_all, blocks = mc.build_feature_frame(df)
    mc.assert_clean(X_all)
    y = df[mc.TARGET].astype(float)
    cols = mc.columns_for(MODEL_NAME, blocks)
    idx = {p: (df['partition'] == p).to_numpy() for p in ['train', 'val', 'test']}

    model = xgb.XGBRegressor(
        eval_metric='rmse',
        early_stopping_rounds=config['early_stopping_rounds'],
        n_jobs=-1,
        **config['params'],
    )
    model.fit(X_all.loc[idx['train'], cols], y[idx['train']],
              eval_set=[(X_all.loc[idx['val'], cols], y[idx['val']])], verbose=False)

    test_ids = df.loc[idx['test'], 'id'].to_numpy()
    retrained = pd.Series(model.predict(X_all.loc[idx['test'], cols]), index=test_ids)
    reference = saved.set_index('id')[f'xgboost__{MODEL_NAME}']
    if set(reference.index) != set(retrained.index):
        print('FAIL: the test set here is not the test set notebook 7 evaluated on. '
              'Re-run notebook 5 so splits.csv matches, then retry.')
        return 1
    max_diff = float((retrained - reference.loc[retrained.index]).abs().max())
    metrics = mc.regression_metrics(y[idx['test']], retrained.to_numpy())

    print(f'trained on {int(idx["train"].sum()):,} rows, {len(cols)} features, '
          f'stopped at round {model.best_iteration}')
    print(f'test R2 {metrics["r2"]:+.5f}  RMSE {metrics["rmse"]:.4f}  MAE {metrics["mae"]:.4f}')
    print(f'max |retrained - notebook 7| over {len(retrained):,} test tracks: {max_diff:.3e}')
    if max_diff > args.tolerance:
        print(f'FAIL: exceeds tolerance {args.tolerance:g}. The retrained model is not '
              'the model the thesis reports on; nothing written.')
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_FILE)
    manifest = mc.load_json('manifest.json')
    metadata = {
        'model': f'xgboost__{MODEL_NAME}',
        'purpose': 'Demo-app inference only (Objective 6). Predictions for new songs '
                   'use approximated audio descriptors and are illustrative (Section 1.4).',
        'feature_names': cols,
        'target': mc.TARGET,
        'xgboost_version': xgb.__version__,
        'config': config,
        'best_iteration': int(model.best_iteration),
        'trained_rows': int(idx['train'].sum()),
        'test_rows': int(idx['test'].sum()),
        'test_metrics': metrics,
        'verified_against_notebook_7': {'max_abs_prediction_diff': max_diff,
                                        'tolerance': args.tolerance},
        'split': {'method': manifest.get('split_method'),
                  'random_seed': manifest.get('random_seed'),
                  'partition_counts': manifest.get('partition_counts')},
        'exported_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    }
    META_FILE.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'wrote {MODEL_FILE.relative_to(ROOT)}')
    print(f'wrote {META_FILE.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
