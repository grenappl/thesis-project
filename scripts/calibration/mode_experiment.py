"""Mode detection from chroma profiles is worse than always answering
"major" on the 66 validation songs (0.64-0.71 vs a 0.74 majority baseline).
Test a VGGish-embedding classifier trained on the 491k real Spotify mode
labels instead, and combine it with Temperley key profiles: mode from the
classifier, tonic = best-correlating Temperley profile *within that mode*.
Also a 12-class tonic classifier from VGGish, for completeness (VGGish's
64-band log-mel front end is coarse on pitch, so not expected to work).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

cache = np.load("data/validation_songs/feature_cache.npz", allow_pickle=True)
val_ids = list(cache["id"])
manifest = pd.read_csv("data/validation_songs/manifest.csv").set_index("id")
real = manifest.loc[val_ids, ["key", "mode"]].to_numpy()

emb = np.load("data/kaggle_embeddings/vggish_embeddings.npz", allow_pickle=True)
songs = pd.read_csv(os.environ.get("SONGS_CSV", "data/songs.csv"), usecols=["id", "key", "mode"]).set_index("id")
Y = songs.loc[emb["id"]]
ok = Y.notna().all(axis=1).to_numpy() & (Y["key"].to_numpy() >= 0)
X, Y = emb["features"][ok], Y[ok]

clf_kw = dict(max_iter=1000, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=30, random_state=42)

# --- mode ---
ym = Y["mode"].to_numpy().astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, ym, test_size=0.2, random_state=42)
mode_clf = HistGradientBoostingClassifier(**clf_kw).fit(Xtr, ytr)
p = mode_clf.predict_proba(Xte)[:, 1]
print(f"held-out Kaggle mode: acc={np.mean((p > 0.5) == yte):.3f} "
      f"majority baseline={max(yte.mean(), 1 - yte.mean()):.3f} AUC={roc_auc_score(yte, p):.3f}")

val_mode = mode_clf.predict(cache["vggish"])
print(f"66 real songs mode: VGGish clf acc={np.mean(val_mode == real[:, 1]):.3f} "
      f"(always-major 0.742, current chroma 0.71)")

# --- combined: mode from classifier, tonic from Temperley within that mode ---
def tonic_within(chroma, mode):
    prof = TEMPERLEY_MAJOR if mode == 1 else TEMPERLEY_MINOR
    return int(np.argmax([np.corrcoef(chroma, np.roll(prof, t))[0, 1] for t in range(12)]))


def mirex(pk, pm, rk, rm):
    if pk == rk and pm == rm:
        return 1.0
    if pm == rm and (pk - rk) % 12 in (5, 7):
        return 0.5
    if pm != rm:
        rel = (rk + 3) % 12 if rm == 0 else (rk - 3) % 12
        return 0.3 if pk == rel else (0.2 if pk == rk else 0.0)
    return 0.0


C = cache["chroma_cqt"]
preds = [(tonic_within(c, m), m) for c, m in zip(C, val_mode)]
exact = np.mean([p == (r[0], r[1]) for p, r in zip(preds, real)])
key = np.mean([p[0] == r[0] for p, r in zip(preds, real)])
mode = np.mean([p[1] == r[1] for p, r in zip(preds, real)])
mx = np.mean([mirex(p[0], p[1], r[0], r[1]) for p, r in zip(preds, real)])
print(f"\ncombined (VGGish mode + Temperley tonic-within-mode): exact={exact:.2f} key={key:.2f} "
      f"mode={mode:.2f} MIREX={mx:.3f}")
print("reference: Temperley alone exact=0.47 key=0.48 mode=0.67 MIREX=0.602; current KK 0.36/0.38/0.71/0.530")
