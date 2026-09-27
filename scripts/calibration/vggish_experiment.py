"""VGGish-side calibration experiments, scored both on held-out Kaggle rows
and on the 66 real validation songs (cached VGGish embeddings — the exact
input the pipeline's regression heads see, so this is end-to-end-equivalent
without the ~70-minute full pipeline run).

1. Sanity: production models on cached embeddings should reproduce the
   per_song_results.csv numbers (proves the cache matches the pipeline).
2. Capacity: production GBM uses sklearn defaults (max_iter=100). With
   ~390k training rows that's likely under-fit — try a larger, early-
   stopped model.
3. Tempo octave selection, ground-truth-free: Librosa's tempo has strong
   periodicity detection but picks the wrong octave 26% of the time. A
   VGGish tempo regression is compressed but knows roughly "slow vs fast" —
   use it only to choose among {t, 2t, t/2} for Librosa's estimate t.
"""

import os
import sys


import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

TARGETS = ["valence", "acousticness", "danceability", "loudness",
           "instrumentalness", "energy", "speechiness"]
SONGS_CSV = os.environ.get("SONGS_CSV", "data/songs.csv")

cache = np.load("data/validation_songs/feature_cache.npz", allow_pickle=True)
val_ids = list(cache["id"])
V = cache["vggish"]
manifest = pd.read_csv("data/validation_songs/manifest.csv").set_index("id")

emb = np.load("data/kaggle_embeddings/vggish_embeddings.npz", allow_pickle=True)
ids = emb["id"]
X = emb["features"]
songs = pd.read_csv(SONGS_CSV, usecols=["id", *TARGETS, "tempo"]).set_index("id")
Y = songs.loc[ids]
ok = Y.notna().all(axis=1).to_numpy()
X, Y = X[ok], Y[ok]


def stats(pred, real):
    return np.abs(pred - real).mean(), np.corrcoef(pred, real)[0, 1]


print(f"{'target':17}{'prod real-MAE':>14}{'prod corr':>10}{'tuned held-out corr':>21}"
      f"{'tuned real-MAE':>15}{'tuned corr':>11}{'iters':>7}")
tuned_models = {}
for t in TARGETS:
    real = manifest.loc[val_ids, t].to_numpy()
    prod = joblib.load(f"models/vggish_ridge/{t}.joblib")
    p_mae, p_corr = stats(prod.predict(V), real)

    y = Y[t].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = HistGradientBoostingRegressor(
        max_iter=1500, learning_rate=0.1, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=42,
    ).fit(Xtr, ytr)
    _, ho_corr = stats(m.predict(Xte), yte)
    t_mae, t_corr = stats(m.predict(V), real)
    tuned_models[t] = m
    print(f"{t:17}{p_mae:14.4f}{p_corr:10.3f}{ho_corr:21.3f}{t_mae:15.4f}{t_corr:11.3f}{m.n_iter_:7d}", flush=True)

joblib.dump(tuned_models, "data/calibration_outputs/tuned_vggish_models.joblib")

# --- tempo octave selection ---
yt = Y["tempo"].to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, yt, test_size=0.2, random_state=42)
tm = HistGradientBoostingRegressor(
    max_iter=1500, learning_rate=0.1, max_leaf_nodes=63,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=42,
).fit(Xtr, ytr)
joblib.dump(tm, "data/calibration_outputs/tempo_guide.joblib")

real_t = manifest.loc[val_ids, "tempo"].to_numpy()
lib = cache["librosa_tempo"]
guide = tm.predict(V)
cands = np.stack([lib, lib * 2, lib / 2], axis=1)


def octave_pick(ref):
    # compare in log space: octave errors are multiplicative
    return cands[np.arange(len(lib)), np.abs(np.log(cands) - np.log(ref)[:, None]).argmin(axis=1)]


picked = octave_pick(guide)
oracle = cands[np.arange(len(lib)), np.abs(cands - real_t[:, None]).argmin(axis=1)]
print("\n=== tempo (66 real songs) ===")
for name, pred in [("librosa (current)", lib), ("VGGish GBM alone", guide),
                   ("librosa, octave picked by VGGish", picked), ("oracle octave (ceiling)", oracle)]:
    mae, corr = stats(pred, real_t)
    print(f"  {name:34} MAE={mae:6.2f} corr={corr:.3f}")
print(f"  octave changed on {np.sum(picked != lib)}/{len(lib)} songs; "
      f"matches oracle choice on {np.sum(picked == oracle)}/{len(lib)}")
