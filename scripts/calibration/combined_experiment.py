"""VGGish-only vs VGGish + PANNs-PCA256 features, same GBM config, same
split — scored on held-out Kaggle rows and on the 66 real validation songs
(cached local embeddings, with the fitted PCA applied to the local PANNs
embedding exactly as inference would).

Usage: python combined_experiment.py valence [acousticness ...]
Saves each winning combined model to data/calibration_outputs/combined_<target>.joblib.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

SCRATCH = "data/calibration_outputs"
targets = sys.argv[1:]

cache = np.load("data/validation_songs/feature_cache.npz", allow_pickle=True)
val_ids = list(cache["id"])
manifest = pd.read_csv("data/validation_songs/manifest.csv").set_index("id")
pca = joblib.load("models/panns/panns_pca256.joblib")
val_V = cache["vggish"]
val_C = np.hstack([val_V, pca.transform(cache["panns_emb"]).astype(np.float32)])

vg = np.load("data/kaggle_embeddings/vggish_embeddings.npz", allow_pickle=True)
pn = np.load("data/kaggle_embeddings/panns_pca256.npz", allow_pickle=True)
assert (vg["id"] == pn["id"]).all(), "embedding files not row-aligned"
V = vg["features"]
C = np.hstack([V, pn["features"]])
songs = pd.read_csv(os.environ.get("SONGS_CSV", "data/songs.csv"), usecols=["id", *targets]).set_index("id")
Y = songs.loc[vg["id"]]

cfg = dict(max_iter=3000, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
           validation_fraction=0.1, n_iter_no_change=50, random_state=42)


def stats(pred, real):
    return np.abs(pred - real).mean(), np.corrcoef(pred, real)[0, 1]


print(f"{'target':17}{'features':14}{'held-out corr':>14}{'real MAE':>10}{'real corr':>10}{'iters':>7}", flush=True)
for t in targets:
    y = Y[t].to_numpy()
    rows = np.flatnonzero(~np.isnan(y))
    tr, te = train_test_split(rows, test_size=0.2, random_state=42)  # index split: no full-array copies
    real = manifest.loc[val_ids, t].to_numpy()
    for name, X, Xval in (("vggish", V, val_V), ("vggish+panns", C, val_C)):
        m = HistGradientBoostingRegressor(**cfg).fit(X[tr], y[tr])
        _, ho = stats(m.predict(X[te]), y[te])
        mae, corr = stats(m.predict(Xval), real)
        print(f"{t:17}{name:14}{ho:14.3f}{mae:10.4f}{corr:10.3f}{m.n_iter_:7d}", flush=True)
        joblib.dump(m, f"{SCRATCH}/{name.replace('+', '_')}_{t}.joblib")
