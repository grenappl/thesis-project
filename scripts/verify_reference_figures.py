"""One-command check of the modeling results, for the data-science validator.

    uv sync --only-group modeling
    uv run --only-group modeling python scripts/verify_reference_figures.py
    uv run --only-group modeling python scripts/verify_reference_figures.py --compare-reference

Run it WITHOUT --compare-reference first. That pass recomputes everything from
raw test-set predictions and the dataset's own labels (it does not read the
metrics the notebooks computed), runs the leakage and integrity checks, and
prints the numbers it measured -- without showing the figures the authors
report. Write down your own conclusions, then add --compare-reference to check
the reported figures in samples/reviewer_reference.md against what was measured.

What it checks
  1. Reproducibility log: versions, git commit, SHA-256 of the inputs.
  2. Test metrics (R2, RMSE, MAE) for all nine models, recomputed from
     test_predictions.csv and the corpus labels, and compared with the
     notebooks' own ladder_metrics.csv.
  3. The split: partition sizes and mean popularity, and that no artist ID is
     shared between train, validation and test.
  4. Feature isolation: Control has no alignment_gap, Augmented adds exactly
     that one column, no forbidden column reaches any model.
  5. The engineered feature: alignment_gap == lyric_sentiment - (2*valence - 1)
     on every row.
  6. An independent paired bootstrap of the Control vs Augmented delta R2.
  7. Selected hyperparameters match the tuning grid's best validation RMSE.

Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'notebooks'))
import modeling_config as mc  # noqa: E402

ALGOS = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest'}
MODELS = {'audio_only': 'Audio-Only', 'lyric_only': 'Lyric-Only', 'control': 'Control',
          'alignment_augmented': 'Alignment-Augmented'}

results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> bool:
    results.append((bool(ok), message))
    print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
    return bool(ok)


def section(title: str) -> None:
    print(f'\n== {title}')


def sha256_of(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).name.encode())
        h.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return h.hexdigest()


def r2_rmse_mae(y, p):
    e = y - p
    return (1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()))


def paired_bootstrap(y, p_ctrl, p_aug, n_boot, seed):
    """Delta R2 (augmented - control), R2 recomputed inside every resample."""
    rng = np.random.default_rng(seed)
    n = len(y)
    y2, e0, e1 = y ** 2, (y - p_ctrl) ** 2, (y - p_aug) ** 2
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        ss_tot = y2[i].sum() - y[i].sum() ** 2 / n
        deltas[b] = (1 - e1[i].sum() / ss_tot) - (1 - e0[i].sum() / ss_tot)
    return np.percentile(deltas, [2.5, 97.5])


def log_environment(data_files, art) -> None:
    section('1. Reproducibility log (include this in your review)')
    pkgs = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'shap', 'networkx']
    versions = {p: metadata.version(p) for p in pkgs if _has(p)}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'not a git checkout', False
    print(f'  python      {platform.python_version()} on {platform.platform()}')
    print(f'  packages    ' + ', '.join(f'{k} {v}' for k, v in versions.items()))
    print(f'  git commit  {commit}{" (working tree has uncommitted changes)" if dirty else ""}')
    print(f'  data        {len(data_files)} yearly CSVs, sha256 {sha256_of(data_files)[:16]}...')
    for name in ('splits.csv', 'test_predictions.csv', 'config_xgb.json', 'config_rf.json', 'ladder_metrics.csv'):
        print(f'  {name:22s} sha256 {sha256_of([art / name])[:16]}...')


def _has(pkg: str) -> bool:
    try:
        metadata.version(pkg)
        return True
    except metadata.PackageNotFoundError:
        return False


def main() -> int:
    # Windows consoles default to a legacy codepage that cannot print the
    # reference tables' "−" (U+2212); never let output encoding stop a review.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--compare-reference', action='store_true',
                        help='after measuring, check the reference figures in the reviewer reference document')
    parser.add_argument('--reference', default=str(ROOT / 'samples' / 'reviewer_reference.md'))
    parser.add_argument('--bootstrap', type=int, default=10000, help='resamples for the independent bootstrap')
    args = parser.parse_args()

    art = ROOT / 'notebooks' / 'artifacts'
    data_files = sorted(glob.glob(str(ROOT / 'notebooks' / 'data_filtered' / 'songs_*.csv')))
    log_environment(data_files, art)

    corpus = mc.load_corpus()
    splits = pd.read_csv(mc.artifact('splits.csv'), usecols=['id', 'partition'])
    preds = pd.read_csv(mc.artifact('test_predictions.csv'))
    full = corpus.merge(splits, on='id', validate='one_to_one')
    test = full[full.partition == 'test'].merge(preds.drop(columns='y_true'), on='id', validate='one_to_one')
    y = test['popularity'].to_numpy(float)

    # ---------------------------------------------------------------- metrics
    section('2. Test-set metrics, recomputed from raw predictions and corpus labels')
    check(len(test) == len(preds) and set(test.id) == set(preds.id), f'prediction file covers exactly the {len(test):,} test tracks')
    check(np.allclose(y, preds.set_index('id').loc[test.id, 'y_true'].to_numpy()), 'y_true in the prediction file equals the corpus popularity')
    measured: dict[tuple[str, str], tuple[float, float, float]] = {}
    measured[('shared', 'baseline')] = r2_rmse_mae(y, test['baseline'].to_numpy(float))
    for a in ALGOS:
        for m in MODELS:
            measured[(a, m)] = r2_rmse_mae(y, test[f'{a}__{m}'].to_numpy(float))
    print(f'  {"algorithm":14s} {"model":22s} {"R2":>9s} {"RMSE":>9s} {"MAE":>9s}')
    for (a, m), (r2, rmse, mae) in measured.items():
        print(f'  {a:14s} {m:22s} {r2:9.5f} {rmse:9.4f} {mae:9.4f}')
    ladder = pd.read_csv(art / 'ladder_metrics.csv').set_index(['algorithm', 'model'])
    worst = max(abs(ladder.loc[k, c] - v) for k, vals in measured.items() for c, v in zip(('r2', 'rmse', 'mae'), vals))
    check(worst < 1e-6, f"recomputed metrics equal the notebooks' ladder_metrics.csv (max difference {worst:.1e})")

    # ------------------------------------------------------------------ split
    section('3. The split')
    counts = splits.partition.value_counts()
    for part in ('train', 'val', 'test', 'dropped_cross_partition_collaboration'):
        mean = full.loc[full.partition == part, 'popularity'].mean()
        print(f'  {part:40s} {counts[part]:>8,d} tracks   mean popularity {mean:6.2f}')
    ids = {}
    for part in ('train', 'val', 'test'):
        s: set = set()
        for lst in mc.artist_lists(full.loc[full.partition == part, 'artist_ids']):
            s.update(lst)
        ids[part] = s
    overlap = {f'{a}&{b}': len(ids[a] & ids[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    check(all(v == 0 for v in overlap.values()), f'no artist ID appears in two partitions {overlap}')
    names: dict[str, set] = {}
    for lst, part in zip(mc.artist_lists(full.artists), full.partition):
        if part in ids:
            for n in lst:
                names.setdefault(n, set()).add(part)
    shared_names = sum(len(v) > 1 for v in names.values())
    print(f'  note: {shared_names} artist NAMES appear in more than one partition; each is a different artist ID sharing a name.')
    feat_cols = ['tempo', 'loudness', 'key', 'mode', 'energy', 'danceability', 'speechiness', 'acousticness',
                 'instrumentalness', 'liveness', 'duration_ms', 'valence', 'lyric_sentiment']
    train_vectors = set(map(tuple, full.loc[full.partition == 'train', feat_cols].round(6).to_numpy()))
    dup = sum(tuple(r) in train_vectors for r in test[feat_cols].round(6).to_numpy())
    check(dup == 0, f'no test track has a feature vector identical to a training track ({dup} found)')

    # --------------------------------------------------------- feature checks
    section('4. Feature isolation and the engineered feature')
    X, blocks = mc.build_feature_frame(corpus)
    try:
        mc.assert_clean(X)
        clean = True
    except AssertionError as exc:
        clean = False
        print(f'  {exc}')
    check(clean, 'no forbidden column and no NaN in the feature matrix')
    cols = {name: mc.columns_for(name, blocks) for name in mc.LADDER}
    check('alignment_gap' not in cols['control'], 'Control does not contain alignment_gap')
    check(sorted(set(cols['alignment_augmented']) - set(cols['control'])) == ['alignment_gap']
          and set(cols['control']) <= set(cols['alignment_augmented']),
          'Alignment-Augmented = Control + exactly one column (alignment_gap)')
    check(not any(set(c) & set(mc.FORBIDDEN_COLS) for c in cols.values()), 'no ladder model uses a forbidden column')
    expected_gap = corpus.lyric_sentiment - (2 * corpus.valence - 1)
    gap_err = float(np.abs(corpus.alignment_gap - expected_gap).max())
    check(gap_err < 1e-9, f'alignment_gap == lyric_sentiment - (2*valence - 1) on all {len(corpus):,} rows (max error {gap_err:.1e})')
    check(not corpus[['popularity', 'valence', 'lyric_sentiment', 'alignment_gap']].isna().any().any() and (corpus.popularity > 0).all(),
          'no missing values; no zero-popularity rows')

    # -------------------------------------------------------------- bootstrap
    section(f'5. Independent paired bootstrap of delta R2 (Augmented - Control), {args.bootstrap:,} resamples')
    central = json.loads((art / 'central_comparison.json').read_text())
    boot = {}
    for a in ALGOS:
        d_obs = measured[(a, 'alignment_augmented')][0] - measured[(a, 'control')][0]
        lo, hi = paired_bootstrap(y, test[f'{a}__control'].to_numpy(float), test[f'{a}__alignment_augmented'].to_numpy(float),
                                  args.bootstrap, seed=20240601)
        boot[a] = (d_obs, lo, hi)
        print(f'  {ALGOS[a]:14s} delta R2 {d_obs:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   contains zero: {lo <= 0 <= hi}')
        check(abs(d_obs - central[a]['delta_r2']) < 1e-6, f'{ALGOS[a]}: observed delta R2 equals central_comparison.json')
        check(abs(lo - central[a]['ci_low']) < 1e-4 and abs(hi - central[a]['ci_high']) < 1e-4,
              f'{ALGOS[a]}: independent CI agrees with the notebook\'s to 1e-4 (Monte Carlo error is ~1e-5)')

    # ----------------------------------------------------------------- tuning
    section('6. Selected hyperparameters')
    tuning = pd.read_csv(art / 'tuning_results.csv')
    for a, cfg_file in (('xgboost', 'config_xgb.json'), ('random_forest', 'config_rf.json')):
        best = tuning[tuning.algorithm == a].sort_values('val_rmse').iloc[0]
        cfg = json.loads((art / cfg_file).read_text())['params']
        keys = ('max_depth', 'learning_rate') if a == 'xgboost' else ('n_estimators', 'min_samples_leaf')
        same = all(float(best[k]) == float(cfg[k]) for k in keys)
        print(f'  {ALGOS[a]:14s} best of {int((tuning.algorithm == a).sum())} grid runs: ' + ', '.join(f'{k}={cfg[k]}' for k in keys)
              + f'  (validation RMSE {best.val_rmse:.4f})')
        check(same, f'{ALGOS[a]}: the configuration used is the grid\'s lowest validation RMSE')

    # ------------------------------------------------------------- reference
    if args.compare_reference:
        compare_reference(Path(args.reference), measured, full, art, central)

    failed = [m for ok, m in results if not ok]
    print(f'\n{len(results) - len(failed)} of {len(results)} checks passed.')
    if failed:
        print('FAILED:')
        for m in failed:
            print('  -', m)
    return 1 if failed else 0


# ------------------------------------------------------------------------ reference parsing
def num(s: str) -> float:
    return float(s.replace('−', '-').replace('+', '').replace(',', '').replace('%', '').strip())


def decimals(s: str) -> int:
    s = s.replace('−', '-').strip()
    return len(s.split('.')[1]) if '.' in s else 0


def close(measured: float, shown: str) -> bool:
    """True if `measured` rounds to the figure as printed (e.g. 0.0678 at 4 dp)."""
    return abs(measured - num(shown)) <= 0.5 * 10 ** -decimals(shown) + 1e-9


def tables(text: str) -> dict[str, list[list[str]]]:
    """Heading -> table rows (cells), for every '### ' block of the reference document."""
    out: dict[str, list[list[str]]] = {}
    for block in re.split(r'(?m)^### ', text)[1:]:
        title, _, body = block.partition('\n')
        rows = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in body.splitlines()
                if ln.startswith('|') and not re.match(r'^\|[\s\-|]+\|$', ln)]
        out[title.strip()] = rows[1:] if rows else []
        out[title.strip() + '\0text'] = body  # type: ignore[assignment]
    return out


def compare_reference(path: Path, measured, full, art, central) -> None:
    section(f'7. Reference figures in {path.name} vs what was measured')
    if not path.is_file():
        check(False, f'reference document not found: {path}')
        return
    text = path.read_text(encoding='utf-8')
    tabs = tables(text)

    def find(prefix):
        key = next((k for k in tabs if k.startswith(prefix) and '\0' not in k), None)
        if key is None:
            check(False, f'reference document has no section starting "{prefix}"')
        return key

    # the nine models
    key = find('The nine models')
    if key:
        by_algo = {v: k for k, v in ALGOS.items()}
        by_model = {v: k for k, v in MODELS.items()}
        by_model['Baseline'] = 'baseline'
        n = 0
        for algo, model, _nf, r2, rmse, mae in tabs[key]:
            k = ('shared' if algo == 'shared' else by_algo[algo], by_model[model])
            ok = all(close(m_, s) for m_, s in zip(measured[k], (r2, rmse, mae)))
            n += ok
            if not ok:
                check(False, f'{algo} {model}: reference ({r2}, {rmse}, {mae}) vs measured ({measured[k][0]:.4f}, {measured[k][1]:.3f}, {measured[k][2]:.3f})')
        check(n == len(tabs[key]) == 9, f'ladder table: {n} of {len(tabs[key])} rows match the measured R2 / RMSE / MAE at the printed precision')

    # the split
    key = find('The split')
    if key:
        counts = full.partition.value_counts()
        pmap = {'Train': 'train', 'Validation': 'val', 'Test': 'test'}
        n = ok_rows = 0
        for row in tabs[key]:
            if row[0] in pmap:
                n += 1
                part = pmap[row[0]]
                mean = full.loc[full.partition == part, 'popularity'].mean()
                ok_rows += int(counts[part] == int(num(row[1])) and close(mean, row[3]))
        check(n == ok_rows == 3, f'split table: {ok_rows} of {n} partitions match on size and mean popularity')
        excl = next((r for r in tabs[key] if r[0].startswith('Excluded')), None)
        if excl:
            check(counts['dropped_cross_partition_collaboration'] == int(num(excl[1])), 'excluded-collaboration count matches')

    # tuning
    key = find('Tuning')
    if key:
        tuning = pd.read_csv(art / 'tuning_results.csv')
        for algo, _cfg, rmse in tabs[key]:
            a = 'xgboost' if algo == 'XGBoost' else 'random_forest'
            check(close(float(tuning[tuning.algorithm == a].val_rmse.min()), rmse), f'{algo}: reference validation RMSE {rmse} matches the grid\'s best')

    # central comparison
    key = find('The central comparison')
    if key:
        for algo, delta, ci, _excl in tabs[key]:
            a = 'xgboost' if algo == 'XGBoost' else 'random_forest'
            lo, hi = re.findall(r'[+−-]?\d+\.\d+', ci)
            check(close(central[a]['delta_r2'], delta) and close(central[a]['ci_low'], lo) and close(central[a]['ci_high'], hi),
                  f'{algo}: reference delta R2 {delta} and CI {ci} match central_comparison.json')

    # robustness
    key = find('Robustness')
    if key:
        rob = json.loads((art / 'robustness_summary.json').read_text())
        for check_name, result in tabs[key]:
            nums = re.findall(r'[+−-]\d+\.\d+', result)
            if check_name.startswith('5-fold'):
                folds = [f['delta_r2'] for f in rob['cross_validation']['per_fold']]
                check(len(nums) >= 5 and all(close(f, s) for f, s in zip(folds, nums[:5])), 'robustness: per-fold CV delta R2 values match')
            elif check_name.startswith('Hyperparameter'):
                hp = [s['delta_r2'] for s in rob['hyperparameter_robustness']['settings']]
                check(len(nums) >= 4 and all(close(f, s) for f, s in zip(hp, nums[:4])), 'robustness: hyperparameter-setting delta R2 values match')
            elif check_name.startswith('Seed'):
                mean, std = rob['seed_sensitivity']['mean_delta_r2'], rob['seed_sensitivity']['r2_std_across_seeds']
                check(any(close(mean, s) for s in nums) and bool(re.search(rf'{std:.5f}', result)), 'robustness: seed mean delta R2 and R2 standard deviation match')

    # SHAP
    body = tabs.get(next((k for k in tabs if k.startswith('SHAP attribution') and '\0' in k), ''), '')
    if body:
        groups = pd.read_csv(art / 'shap_group_contributions.csv')
        shares = [float(s) for s in groups.share_pct]
        shown = [float(x) for x in re.findall(r'(\d+\.\d)%', body)][:3]
        check(len(shown) == 3 and all(abs(a - b) <= 0.05 + 1e-9 for a, b in zip(shares, shown)), f'SHAP group shares match (measured {[round(s, 1) for s in shares]}, shown {shown})')
        ranks = pd.read_csv(art / 'shap_feature_ranks.csv')
        rank = int(ranks.loc[ranks.feature == 'alignment_gap', 'rank'].iloc[0])
        stated = re.search(r'alignment_gap` ranks (\d+)(?:st|nd|rd|th) of (\d+)', body)
        check(bool(stated) and (int(stated.group(1)), int(stated.group(2))) == (rank, len(ranks)),
              f'alignment_gap rank: measured {rank} of {len(ranks)}, stated {stated.group(0) if stated else "nothing parseable"}')


if __name__ == '__main__':
    raise SystemExit(main())
