"""Liveness calibration experiment. The current pipeline output (mean PANNs
probability over applause/cheering/crowd classes) ranks songs reasonably but
sits on a completely different scale from Spotify liveness (pred std 0.008
vs real 0.200). Candidates, all trained on the Kaggle PANNs embeddings
(streamed + subsampled — the full array is 4 GB and has OOM'd WSL2 before):

  A. current heuristic, raw (baseline)
  B. heuristic -> isotonic regression (monotonic rescale only)
  C. GBM on all 527 PANNs class probabilities
  D. GBM on the 2048-dim PANNs embedding

Then — the real test — every candidate is scored on the 66 real validation
songs using PANNs outputs computed locally (feature_cache.npz), which also
checks that local PANNs embeddings match Kaggle's before trusting any model
trained on Kaggle's.
"""

import os
import io
import sys
import zipfile


import joblib
import numpy as np
import pandas as pd
import torch
from numpy.lib import format as npformat
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from api.pipelines.demo.liveness_panns import _crowd_noise_label_indices
from panns_inference import labels as panns_labels

SAMPLE = 100_000
SONGS_CSV = os.environ.get("SONGS_CSV", "data/songs.csv")
NPZ = "data/kaggle_embeddings/panns_embeddings.npz"

cache = np.load("data/validation_songs/feature_cache.npz", allow_pickle=True)
manifest = pd.read_csv("data/validation_songs/manifest.csv").set_index("id")
val_ids = list(cache["id"])

z = zipfile.ZipFile(NPZ)
ids = np.load(io.BytesIO(z.read("id.npy")), allow_pickle=True)
n = len(ids)
rng = np.random.default_rng(42)
wanted_idx = np.sort(rng.choice(n, SAMPLE, replace=False))
slot = {int(g): k for k, g in enumerate(wanted_idx)}  # global row -> position in X
id_to_idx = {v: i for i, v in enumerate(ids)}
val_rows = {id_to_idx[v]: v for v in val_ids if v in id_to_idx}

print(f"streaming {n} rows, keeping {SAMPLE} + {len(val_rows)} validation-song rows", flush=True)
with z.open("features.npy") as f:
    version = npformat.read_magic(f)
    reader = npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0
    shape, _, dtype = reader(f)
    dim = shape[1]
    row_bytes = dim * np.dtype(dtype).itemsize
    X = np.empty((SAMPLE, dim), dtype=np.float32)  # fill in place: no list/stack/cast copies
    val_emb = {}
    chunk = 4096
    for start in range(0, n, chunk):
        count = min(chunk, n - start)
        block = np.frombuffer(f.read(count * row_bytes), dtype=dtype).reshape(count, dim)
        for j in range(count):
            gi = start + j
            if gi in slot:
                X[slot[gi]] = block[j]
            if gi in val_rows:
                val_emb[val_rows[gi]] = block[j].copy()
songs = pd.read_csv(SONGS_CSV, usecols=["id", "liveness"]).set_index("id")
y = songs.loc[ids[wanted_idx], "liveness"].to_numpy()
assert not np.isnan(y).any()

ckpt = torch.load("models/panns/Cnn14_mAP=0.431.pth", map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt)
W = state["fc_audioset.weight"].numpy()
b = state["fc_audioset.bias"].numpy()
crowd = _crowd_noise_label_indices(list(panns_labels))


def probs(E):
    return 1 / (1 + np.exp(-(E @ W.T + b)))


def heuristic(P):
    return P[:, crowd].mean(axis=1)


P = probs(X)
H = heuristic(P)
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)  # index split, no copies
Pte, Hte, yte = P[te], H[te], y[te]

iso = IsotonicRegression(out_of_bounds="clip").fit(H[tr], y[tr])
gbm_p = HistGradientBoostingRegressor(max_iter=300, random_state=42).fit(P[tr], y[tr])
gbm_e = HistGradientBoostingRegressor(max_iter=300, random_state=42).fit(X[tr], y[tr])
Xte = X[te]


def report(name, pred, real):
    print(f"  {name:28} MAE={np.abs(pred - real).mean():.4f} corr={np.corrcoef(pred, real)[0, 1]:.3f}")


print("\n=== held-out Kaggle (n=%d) ===" % len(yte))
report("A raw heuristic", Hte, yte)
report("B heuristic + isotonic", iso.predict(Hte), yte)
report("C GBM on 527 class probs", gbm_p.predict(Pte), yte)
report("D GBM on 2048 embedding", gbm_e.predict(Xte), yte)

# --- real songs: local PANNs vs Kaggle PANNs, then candidates on local ---
loc_E = cache["panns_emb"]
loc_P = cache["panns_prob"]
sims = [np.dot(loc_E[i], val_emb[v]) / (np.linalg.norm(loc_E[i]) * np.linalg.norm(val_emb[v]))
        for i, v in enumerate(val_ids) if v in val_emb]
print(f"\nlocal vs Kaggle PANNs embedding cosine sim (n={len(sims)}): "
      f"mean={np.mean(sims):.3f} min={np.min(sims):.3f}")
print(f"local clipwise vs fc_audioset(local emb) max abs diff: {np.abs(probs(loc_E) - loc_P).max():.2e}")

real = manifest.loc[val_ids, "liveness"].to_numpy()
locH = heuristic(loc_P)
print("\n=== 66 real validation songs (local PANNs) ===")
report("A raw heuristic", locH, real)
report("B heuristic + isotonic", iso.predict(locH), real)
report("C GBM on 527 class probs", gbm_p.predict(loc_P), real)
report("D GBM on 2048 embedding", gbm_e.predict(loc_E), real)

joblib.dump({"iso": iso, "gbm_probs": gbm_p, "gbm_emb": gbm_e},
            "data/calibration_outputs/liveness_candidates.joblib")
