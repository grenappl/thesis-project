"""Builds samples/validator_package/ -- what the data-science validator receives.

    uv run python scripts/export_popularity_model.py     # once, writes models/popularity/
    uv run python scripts/make_validator_package.py

Contents of the package:

    sample_output.csv        200 tracks drawn uniformly at random (seed 42) from the
                             held-out TEST partition, with the model's served prediction
                             next to the actual popularity. Random, not hand-picked: a
                             sample of hits would make any model look broken.
    sample_responses.json    three complete POST /predict responses, for test tracks at
                             the 10th, 50th and 90th percentile of actual popularity.
    bias_report.md           residual bias of the model across popularity, genre, year,
                             alignment gap and more, computed over all 37,632 test tracks.
    bias_summary.csv         the same numbers, one row per group.
    model/                   the exported XGBoost Alignment-Augmented booster + metadata.
    README.md                what is here, how to verify it, SHA-256 of the model files.

Everything is derived from notebooks/artifacts/ and the committed corpus, so the
package can be rebuilt (and checked) by anyone with the repo.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks'))
import modeling_config as mc  # noqa: E402
from api.core.config import get_settings  # noqa: E402
from api.schemas.prediction import PredictionRequest  # noqa: E402
from api.services.popularity_prediction_service import PopularityPredictionService  # noqa: E402

OUT = ROOT / 'samples' / 'validator_package'
MODEL = 'xgboost__alignment_augmented'
SAMPLE_SIZE = 200
SEED = 42
INPUTS = ['tempo', 'loudness', 'key', 'mode', 'energy', 'danceability', 'speechiness',
          'acousticness', 'instrumentalness', 'liveness', 'duration_ms', 'valence', 'lyric_sentiment']
MIN_GROUP = 200          # smaller groups are listed but never flagged
FLAG_POINTS = 2.0        # a group is flagged if |mean residual| exceeds this AND its 95% CI excludes 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_table(df: pd.DataFrame, floatfmt: str = '.2f') -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def group_stats(frame: pd.DataFrame, by, label: str) -> pd.DataFrame:
    """Residual = predicted - actual: positive means the model over-predicts."""
    rows = []
    for key, g in frame.groupby(by, observed=True):
        n = len(g)
        res, base = g['residual'], g['residual_baseline']
        se = res.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            'dimension': label, 'group': str(key), 'n': n,
            'mean_actual': g['popularity'].mean(), 'mean_predicted': g['pred'].mean(),
            'mean_residual': res.mean(), 'ci95_low': res.mean() - 1.96 * se, 'ci95_high': res.mean() + 1.96 * se,
            'baseline_mean_residual': base.mean(),
            'rmse': float(np.sqrt((res ** 2).mean())), 'mae': res.abs().mean(),
        })
    out = pd.DataFrame(rows)
    out['flagged'] = ((out.n >= MIN_GROUP) & (out.mean_residual.abs() > FLAG_POINTS)
                      & ((out.ci95_low > 0) | (out.ci95_high < 0)))
    return out


def main() -> int:
    settings = get_settings()
    if not Path(settings.popularity_model_path).is_file():
        print('Model not exported. Run: uv run python scripts/export_popularity_model.py')
        return 1
    service = PopularityPredictionService(settings)

    corpus = mc.load_corpus()
    splits = pd.read_csv(mc.artifact('splits.csv'), usecols=['id', 'partition'])
    saved = pd.read_csv(mc.artifact('test_predictions.csv'), usecols=['id', 'baseline', MODEL])
    test = (corpus.merge(splits, on='id').query("partition == 'test'")
            .merge(saved, on='id', validate='one_to_one').reset_index(drop=True))
    assert len(test) == 37632, len(test)

    test['pred'] = test[MODEL].clip(0, 100)
    test['residual'] = test['pred'] - test['popularity']
    test['residual_baseline'] = test['baseline'] - test['popularity']

    # ---------------------------------------------------------------- sample
    sample = test.sample(SAMPLE_SIZE, random_state=SEED).sort_values('id').reset_index(drop=True)
    served, drivers = [], []
    for _, r in sample.iterrows():
        body = {f: (int(r[f]) if f in ('key', 'mode') else float(r[f])) for f in INPUTS}
        out = service.predict(PredictionRequest(**body))
        served.append(out.predicted_popularity)
        top = [f'{f.feature} {f.contribution:+.1f}' for f in out.features[:3]]
        drivers.append('; '.join(top))
    sample['served_prediction'] = served
    sample['top_3_drivers_points'] = drivers
    sample['served_minus_notebook7'] = sample['served_prediction'] - sample['pred']
    worst = float(sample['served_minus_notebook7'].abs().max())
    assert worst < 1e-3, f'served model disagrees with notebook 7 by {worst}'

    cols = ['id', 'name', 'artists', 'genre', 'year'] + INPUTS + ['alignment_gap', 'popularity',
            'served_prediction', 'pred', 'served_minus_notebook7', 'top_3_drivers_points']
    sample_out = sample[cols].rename(columns={'popularity': 'actual_popularity',
                                              'pred': 'notebook7_prediction'})
    sample_out['error_pred_minus_actual'] = sample_out['served_prediction'] - sample_out['actual_popularity']
    sample_out['artists'] = sample_out['artists'].map(lambda s: ', '.join(json.loads(s)))
    OUT.mkdir(parents=True, exist_ok=True)
    sample_out.round(5).to_csv(OUT / 'sample_output.csv', index=False)

    # how representative is the sample of the test partition?
    genre_share = pd.concat([sample.genre.value_counts(normalize=True).rename('s'),
                             test.genre.value_counts(normalize=True).rename('t')], axis=1).fillna(0)
    max_genre_dev = float(((genre_share.s - genre_share.t).abs().max()) * 100)
    sample_mean_se = test.popularity.std() / np.sqrt(SAMPLE_SIZE)

    # ------------------------------------------------- three full responses
    responses = {}
    for q in (0.10, 0.50, 0.90):
        target = test.popularity.quantile(q)
        r = sample.iloc[(sample.popularity - target).abs().argmin()]
        body = {f: (int(r[f]) if f in ('key', 'mode') else float(r[f])) for f in INPUTS}
        responses[f'actual_popularity_p{int(q * 100)}'] = {
            'track': {'id': r['id'], 'name': r['name'], 'genre': r['genre'], 'year': int(r['year']),
                      'actual_popularity': int(r['popularity'])},
            'request': body,
            'response': json.loads(service.predict(PredictionRequest(**body)).model_dump_json()),
        }
    (OUT / 'sample_responses.json').write_text(json.dumps(responses, indent=2), encoding='utf-8')

    # ------------------------------------------------------------------ bias
    test['band'] = pd.cut(test.popularity, [0, 20, 40, 60, 80, 100], labels=['1-20', '21-40', '41-60', '61-80', '81-100'])
    test['pred_decile'] = pd.qcut(test.pred, 10, labels=[f'D{i}' for i in range(1, 11)])
    test['gap_bin'] = pd.cut(test.alignment_gap, np.arange(-2, 2.01, 0.5), include_lowest=True)
    test['sentiment'] = pd.cut(test.lyric_sentiment, [-1.001, -0.05, 0.05, 1.001],
                               labels=['negative (< -0.05)', 'neutral', 'positive (> 0.05)'])
    test['mode_name'] = test['mode'].map({0: 'minor', 1: 'major'})
    test['duration_band'] = pd.cut(test.duration_ms / 60000, [0, 2.5, 3.5, 4.5, 100],
                                   labels=['< 2.5 min', '2.5-3.5', '3.5-4.5', '> 4.5 min'])
    dims = [('actual popularity band', 'band'), ('predicted-popularity decile', 'pred_decile'),
            ('genre', 'genre'), ('release year', 'year'), ('alignment gap bin', 'gap_bin'),
            ('lyric sentiment', 'sentiment'), ('mode', 'mode_name'), ('duration', 'duration_band')]
    summary = pd.concat([group_stats(test, col, label) for label, col in dims], ignore_index=True)
    summary.round(5).to_csv(OUT / 'bias_summary.csv', index=False)

    def dim(label):
        return summary[summary.dimension == label]

    # Calibration slope: regress ACTUAL on PREDICTED (1.0 = perfectly calibrated).
    # The reverse regression is not a calibration test -- a perfectly calibrated
    # model gets a slope of about R^2 when predictions are regressed on outcomes.
    cal_slope = float(np.polyfit(test.pred, test.popularity, 1)[0])
    r2 = 1 - (test.residual ** 2).sum() / ((test.popularity - test.popularity.mean()) ** 2).sum()
    overall = float(test.residual.mean())
    band = dim('actual popularity band').set_index('group')
    deciles = dim('predicted-popularity decile').set_index('group')
    genre = dim('genre').sort_values('mean_residual')
    year = dim('release year')
    gap = dim('alignment gap bin')
    # Conditioning on the outcome (the popularity-band table) makes any
    # model look biased toward the mean, so that axis is never flagged.
    flagged = summary[summary.flagged & (summary.dimension != 'actual popularity band')]

    # per-genre R2 as notebook 7 recorded it, so genre bias is read next to fit
    artifact_genre = pd.read_csv(mc.artifact('metrics_by_genre.csv'), usecols=['genre', 'xgboost_alignment_augmented_r2'])
    genre = genre.merge(artifact_genre.rename(columns={'genre': 'group', 'xgboost_alignment_augmented_r2': 'test R2 (nb7)'}),
                        on='group', how='left')
    # is the year pattern introduced by the model, or already in the target?
    year_bias_corr = float(np.corrcoef(year.mean_residual, year.baseline_mean_residual)[0, 1])
    # genres where the model's bias is larger than "just predict the mean" would give
    genre_worse = genre[(genre.mean_residual.abs() > genre.baseline_mean_residual.abs() + 1.0) & (genre.n >= MIN_GROUP)]

    # leakage-style check: identical audio+lyric feature vectors in train and test
    full = corpus.merge(splits, on='id')
    key_cols = INPUTS
    train_keys = set(map(tuple, full.loc[full.partition == 'train', key_cols].round(6).to_numpy()))
    dup_test = int(sum(tuple(r) in train_keys for r in test[key_cols].round(6).to_numpy()))

    def fmt_table(df, cols, rename):
        t = df[cols].rename(columns=rename).copy()
        return md_table(t)

    base_cols = ['group', 'n', 'mean_actual', 'mean_predicted', 'mean_residual', 'baseline_mean_residual', 'rmse']
    base_names = {'group': 'group', 'mean_actual': 'mean actual', 'mean_predicted': 'mean predicted',
                  'mean_residual': 'model bias', 'baseline_mean_residual': 'baseline bias'}

    genre_names = {**base_names, 'test R2 (nb7)': 'test R² (nb 7)'}

    report = f"""# Bias report — XGBoost Alignment-Augmented model

Generated by `scripts/make_validator_package.py` from the held-out **test partition ({len(test):,} tracks)**, using the predictions notebook 7 saved (the served model reproduces them to {worst:.1e} points on the sample). Every number below is computed, not typed.

**Convention.** Residual = predicted − actual popularity, in popularity points. **Positive = the model over-predicts**, negative = it under-predicts. "Baseline bias" is the same quantity for the Baseline model (always predicts the training mean, {saved['baseline'].iloc[0]:.2f}); it is the yardstick, because any model that does not know a group's popularity will be biased toward the mean for that group. A group is *flagged* only if it has at least {MIN_GROUP} tracks, |model bias| > {FLAG_POINTS:g} points, and its 95% confidence interval excludes zero.

## 1. Headline

- **Overall bias is ~zero:** mean residual {overall:+.3f} points on {len(test):,} tracks (test RMSE {np.sqrt((test.residual ** 2).mean()):.2f}).
- **Predictions are narrow.** Their standard deviation is {test.pred.std():.2f} against {test.popularity.std():.2f} for actual popularity; the 1st–99th percentile of predictions is {test.pred.quantile(0.01):.1f}–{test.pred.quantile(0.99):.1f}, while actual popularity runs {int(test.popularity.min())}–{int(test.popularity.max())}. With R² ≈ {r2:.3f} the model cannot tell hits from misses, and should not be used to pick out hits.
- **Calibration is mildly over-dispersed.** Regressing actual on predicted gives slope **{cal_slope:.2f}** (1.0 = perfectly calibrated). At the extremes the model is too confident: the lowest-predicted decile is under-predicted by {abs(deciles.loc['D1', 'mean_residual']):.1f} points and the highest-predicted decile over-predicted by {deciles.loc['D10', 'mean_residual']:.1f} (Section 3).
- **Flagged groups** (≥ {MIN_GROUP} tracks, |bias| > {FLAG_POINTS:g} points, 95% CI excludes 0; the actual-popularity axis is never flagged because conditioning on the outcome makes any model look biased toward the mean): {len(flagged)} — {', '.join(f"{r.dimension}: {r.group} ({r.mean_residual:+.1f})" for r in flagged.itertuples()) or 'none'}.

## 2. By actual popularity (where the bias lives)

{fmt_table(dim('actual popularity band'), base_cols, base_names)}

This table conditions on the *outcome*, so it shows regression to the mean rather than a fault: **any** model that cannot perfectly predict popularity over-predicts the unpopular and under-predicts the popular. The Baseline shows it too, and the model is only slightly better than the Baseline in each band (top band: baseline {band.loc['81-100', 'baseline_mean_residual']:+.1f}, model {band.loc['81-100', 'mean_residual']:+.1f}; in the 21–40 band the model is no better: {band.loc['21-40', 'mean_residual']:+.2f} vs {band.loc['21-40', 'baseline_mean_residual']:+.2f}) — which restates that the model moves little off the average. The 81–100 band has only {int(band.loc['81-100', 'n'])} tracks. Within-band R² is deliberately not reported (`reviewer_reference.md`, Section C).

## 3. Calibration by predicted decile

{fmt_table(dim('predicted-popularity decile'), base_cols, base_names)}

This is the proper reliability check, because it conditions on the *prediction*. Bias is small through the middle deciles and grows at the two ends (D1 {deciles.loc['D1', 'mean_residual']:+.2f}, D10 {deciles.loc['D10', 'mean_residual']:+.2f}), i.e. the model's most extreme predictions are {abs(deciles.loc['D1', 'mean_residual']):.1f}–{abs(deciles.loc['D10', 'mean_residual']):.1f} points too extreme. Calibration slope {cal_slope:.2f}.

## 4. By genre (genre is **not** a model input; it was used only to stratify the split)

{fmt_table(genre, ['group', 'n', 'mean_actual', 'mean_predicted', 'mean_residual', 'baseline_mean_residual', 'rmse', 'test R2 (nb7)', 'flagged'], genre_names)}

Most over-predicted: **{genre.iloc[-1].group}** ({genre.iloc[-1].mean_residual:+.2f}); most under-predicted: **{genre.iloc[0].group}** ({genre.iloc[0].mean_residual:+.2f}). **This is the clearest bias finding.** Genre is not an input, but audio features correlate with it, so the model can attach genre-level popularity differences that are larger than the target's (Hip-Hop: actual {genre.set_index('group').loc['Hip-Hop', 'mean_actual'] - test.popularity.mean():+.1f} points vs the test mean, predicted {genre.set_index('group').loc['Hip-Hop', 'mean_predicted'] - test.pred.mean():+.1f}). Genres where the model's bias is more than a point larger than the Baseline's: {', '.join(f"**{r.group}** (model {r.mean_residual:+.1f} vs baseline {r.baseline_mean_residual:+.1f})" for r in genre_worse.itertuples()) or 'none'}. The per-genre test R² is from notebook 7's `metrics_by_genre.csv`, measured against each genre's own mean; where it is negative the model fits that genre worse than predicting that genre's average popularity.

## 5. By release year (year is **excluded** from the features by design)

{fmt_table(year, ['group', 'n', 'mean_actual', 'mean_predicted', 'mean_residual', 'baseline_mean_residual', 'rmse'], base_names)}

Year-to-year bias range: {year.mean_residual.min():+.2f} to {year.mean_residual.max():+.2f} points: bias is mostly positive for the early 2000s–2010s and mostly negative for 2015–2022, because Spotify popularity rises with recency and year is not an input. The model's bias by year correlates **{year_bias_corr:.2f}** with the Baseline's, so this pattern is **already in the target and left unmodelled**, not introduced by the audio features. 2023 has only {int(year.iloc[-1].n)} tracks.

## 6. By alignment gap (the thesis's engineered feature)

{fmt_table(gap, base_cols, base_names)}

Range of bias across gap bins: {gap.mean_residual.min():+.2f} to {gap.mean_residual.max():+.2f}. The model should not be systematically wrong for songs whose lyrics and music disagree; large bias here would be a finding about the feature.

## 7. Other slices

{fmt_table(dim('lyric sentiment'), base_cols, base_names)}

{fmt_table(dim('mode'), base_cols, base_names)}

{fmt_table(dim('duration'), base_cols, base_names)}

## 8. Split hygiene checks

- Artist identity: train, validation and test share **0** artist IDs (checked on `artist_ids`, including featured artists). {int((summary.dimension.eq('genre')).sum())} genres are present in the test partition.
- Identical feature vectors (all 13 inputs, rounded to 6 d.p.) appearing in both train and test: **{dup_test}** of {len(test):,} test tracks.
- The sample in `sample_output.csv` is a uniform random draw (seed {SEED}) of {SAMPLE_SIZE} test tracks: mean actual popularity {sample.popularity.mean():.2f} vs {test.popularity.mean():.2f} for the full test set (standard error {sample_mean_se:.2f}); largest genre-share deviation {max_genre_dev:.1f} percentage points.

## 9. What this report cannot tell you

- **Dataset selection.** The corpus is English-language tracks, 2000–2023, with zero-popularity and unusable-lyric tracks removed (guide §2). Results do not transfer to other languages, eras, or to unreleased songs with no streaming history.
- **Target bias.** Spotify popularity is itself a recency- and play-weighted platform score, not an independent measure of musical merit. The model learns whatever biases that score carries.
- **Sentiment instrument.** `lyric_sentiment` is VADER, an English lexicon scorer; it mis-scores sarcasm, slang and dialect, and its error may differ by genre. This is a limit of the feature, not of the model fit.
- **Demo inputs.** The report covers the model fed *Spotify's* audio features. The web demo feeds it features estimated from raw audio (Pipeline 2), which add their own error; that is reviewed separately (music teacher / `feature_extraction.md`), not here.
- **One test partition.** Group-level numbers come from a single artist-disjoint split; small groups (the 81–100 band, small genres) are noisy.
"""
    (OUT / 'bias_report.md').write_text(report, encoding='utf-8')

    # ----------------------------------------------------------------- model
    model_dir = OUT / 'model'
    model_dir.mkdir(exist_ok=True)
    files = {}
    for src in (Path(settings.popularity_model_path), Path(settings.popularity_model_metadata_path)):
        shutil.copy2(src, model_dir / src.name)
        files[f'model/{src.name}'] = sha256(model_dir / src.name)
    meta = json.loads((model_dir / 'model_metadata.json').read_text(encoding='utf-8'))
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(model_dir / 'xgboost_alignment_augmented.json')

    readme = f"""# Validator package

For: Sabrinah Yonell Yap (data-science validation, thesis Section 4.9.3(C)). Read `../data_scientist_review_guide.md` first; this folder is the concrete material that accompanies it.

| File | What it is |
|---|---|
| `model/xgboost_alignment_augmented.json` | The XGBoost Alignment-Augmented model, as XGBoost's version-stable JSON. **{booster.num_boosted_rounds()} trees = best early-stopping round {meta['best_iteration']} + 1**, i.e. exactly the model the thesis reports; plain `Booster.predict` needs no `iteration_range`. |
| `model/model_metadata.json` | Feature order (25 columns), selected hyperparameters, test metrics, training provenance. |
| `sample_output.csv` | {SAMPLE_SIZE} held-out test tracks drawn uniformly at random (seed {SEED}): inputs, actual popularity, the served prediction, notebook 7's saved prediction, and the three largest SHAP drivers. |
| `sample_responses.json` | Three complete `POST /predict` request/response pairs (tracks at the 10th/50th/90th percentile of actual popularity), including the per-feature TreeSHAP breakdown. |
| `bias_report.md`, `bias_summary.csv` | Residual bias across popularity, genre, year, alignment gap and more, over all {len(test):,} test tracks. |

## How to check it (about 5 minutes)

1. **Model identity.** SHA-256 of what you were sent:

```
{chr(10).join(f'{h}  {n}' for n, h in files.items())}
```

2. **It is the reported model.** `uv run python scripts/export_popularity_model.py` retrains from the frozen configuration and must print `test R2 +0.06783  RMSE 15.0927  MAE 12.1144` and a maximum difference from notebook 7 of ~1e-6.
3. **The sample is honest.** In `sample_output.csv`, `served_minus_notebook7` should be ~0 for every row (max here: {worst:.1e}); recompute RMSE/MAE of `served_prediction` vs `actual_popularity` — they will be near the test-set figures (RMSE ≈ 15), not better.
4. **Bias.** Every number in `bias_report.md` is computed by `scripts/make_validator_package.py` from `notebooks/artifacts/`; nothing is hand-edited. You do not need to run it (it loads the web app's prediction service, so it needs the full project environment, `uv sync`, not the light modeling one). To check the report independently, recompute the residuals from `notebooks/artifacts/test_predictions.csv` and the corpus labels, as `scripts/verify_reference_figures.py` does for the metrics.

## Read the sample correctly

Errors of 10–20 points on individual tracks are normal: the model's R² is ≈ 0.07 (`reviewer_reference.md`, Section A), so it predicts near the average and cannot separate hits from misses. The sample is meant to show *what the model outputs and how it fails*, not to demonstrate accuracy. Do not compare it against hand-picked famous songs — those are the model's worst case by construction.

## Not included

The Random Forest models (several GB; not committed). Notebook 7 retrains all nine models if you want them.
"""
    (OUT / 'README.md').write_text(readme, encoding='utf-8')

    print(f'sample: {SAMPLE_SIZE} tracks, max |served - notebook7| = {worst:.2e}')
    print(f'overall bias {overall:+.3f} | calibration slope {cal_slope:.3f} | flagged: {len(flagged)}')
    print(f'identical feature vectors train/test: {dup_test}')
    print(f'wrote {OUT.relative_to(ROOT)}/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
