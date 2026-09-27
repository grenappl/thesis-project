"""Fits the regression heads used by api.pipelines.demo.regression_heads.

Every target shares one input: the mean-pooled TF-Hub VGGish embedding (128)
concatenated with the PANNs CNN14 embedding compressed to 256 dims by a
fitted PCA (`models/panns/panns_pca256.joblib`, built by streaming the 4 GB
Kaggle PANNs file — see scripts/build_panns_pca.py). Both
come from the same sources the pipeline computes at inference time, which is
what makes Kaggle-trained heads transfer to new audio (local-vs-Kaggle
cosine similarity: VGGish ~0.97, PANNs ~0.96).

Run from WSL2 (memory: ~3 GB peak):

    uv run python scripts/train_vggish_ridge.py \\
        "/mnt/c/Users/User/Downloads/songs(1).csv"

Writes models/regression_heads/{target}.joblib. If
data/validation_songs/feature_cache.npz exists, also scores each model on
the 66 real validation songs (same inputs the pipeline would compute), so a
training run doubles as an end-to-end-equivalent validation.

How this design was arrived at (each step tested head-to-head, kept only if
it won on both held-out Kaggle rows and the real songs):
- GBM beat Ridge on every target tried (Ridge's L2 shrinkage under-predicts
  extreme values — e.g. acousticness's extreme-tail MAE 0.108 -> 0.073).
- More capacity kept helping: defaults (100 iterations) -> 1500 -> 3000
  all improved, and models were still improving at each cap.
- Adding PANNs to VGGish improved valence further (held-out corr 0.817 ->
  0.830). MERT and mel-spectrogram-stats embeddings lost to VGGish alone.
- Liveness moved off the old heuristic (mean PANNs probability over the
  applause/cheering/crowd classes), which ranked songs well but sat on the
  wrong scale — real-song MAE 0.224 -> 0.097 with a GBM head.
- `tempo` is deliberately NOT here: a VGGish regression only "wins" by
  compressing every prediction into a narrow BPM band, and using it to pick
  Librosa's octave made correlation worse (0.182 -> 0.142). See
  feature_extraction.md's "Tempo" section.
- `mode` is deliberately NOT here either: a VGGish classifier was worse on
  the real songs (0.667) than chroma-based detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

TARGETS = ("valence", "acousticness", "danceability", "energy", "speechiness",
           "instrumentalness", "loudness", "liveness")

GBM_CONFIG = dict(max_iter=5000, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=50, random_state=42)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("songs_csv", help="Path to the raw Spotify dataset CSV (songs(1).csv)")
    parser.add_argument("--vggish-npz", default="data/kaggle_embeddings/vggish_embeddings.npz")
    parser.add_argument("--panns-pca-npz", default="data/kaggle_embeddings/panns_pca256.npz")
    parser.add_argument("--pca", default="models/panns/panns_pca256.joblib")
    parser.add_argument("--output-dir", default="models/regression_heads")
    parser.add_argument("--targets", nargs="*", default=list(TARGETS))
    args = parser.parse_args()

    vg = np.load(args.vggish_npz, allow_pickle=True)
    pn = np.load(args.panns_pca_npz, allow_pickle=True)
    if not (vg["id"] == pn["id"]).all():
        raise ValueError("VGGish and PANNs-PCA embedding files are not row-aligned")
    X = np.hstack([vg["features"], pn["features"]]).astype(np.float32)
    Y = pd.read_csv(args.songs_csv, usecols=["id", *args.targets]).set_index("id").loc[vg["id"]]

    val = None
    cache_path = Path("data/validation_songs/feature_cache.npz")
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        pca = joblib.load(args.pca)
        val_X = np.hstack([cache["vggish"], pca.transform(cache["panns_emb"])]).astype(np.float32)
        manifest = pd.read_csv("data/validation_songs/manifest.csv").set_index("id")
        val = (val_X, manifest.loc[list(cache["id"])])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'target':17}{'held-out corr':>14}{'real MAE':>10}{'real corr':>10}{'iters':>7}", flush=True)
    for target in args.targets:
        y = Y[target].to_numpy()
        rows = np.flatnonzero(~np.isnan(y))
        tr, te = train_test_split(rows, test_size=0.2, random_state=42)  # index split: no array copies
        model = HistGradientBoostingRegressor(**GBM_CONFIG).fit(X[tr], y[tr])
        held_out = np.corrcoef(model.predict(X[te]), y[te])[0, 1]
        line = f"{target:17}{held_out:14.3f}"
        if val is not None:
            pred, real = model.predict(val[0]), val[1][target].to_numpy()
            line += f"{np.abs(pred - real).mean():10.4f}{np.corrcoef(pred, real)[0, 1]:10.3f}"
        print(f"{line}{model.n_iter_:7d}", flush=True)
        joblib.dump(model, output_dir / f"{target}.joblib")


if __name__ == "__main__":
    main()
