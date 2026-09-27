"""Key/mode have never been checked against real Spotify values. Score the
current method (Krumhansl-Kessler profiles on raw chroma_cqt) and standard
alternatives on the 66 validation songs, from cached chroma.

Metrics: exact key+mode, key-only (pitch class), mode-only, and the MIREX
weighted score (exact=1, perfect fifth=0.5, relative major/minor=0.3,
parallel major/minor=0.2).
"""

import numpy as np
import pandas as pd

PROFILES = {
    "krumhansl_kessler (current)": (
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    ),
    "temperley": (
        [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    ),
    "albrecht_shanahan": (
        [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081],
        [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052],
    ),
    "shaath": (
        [6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4],
        [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2],
    ),
}

SCRATCH = "data/calibration_outputs"
cache = np.load("data/validation_songs/feature_cache.npz", allow_pickle=True)
manifest = pd.read_csv("data/validation_songs/manifest.csv").set_index("id")
real = manifest.loc[list(cache["id"]), ["key", "mode"]].to_numpy()
mask = real[:, 0] >= 0  # Spotify uses -1 for "no key detected"
real = real[mask]


def detect(chroma, major, minor):
    best = (-np.inf, 0, 1)
    for mode, prof in ((1, np.array(major)), (0, np.array(minor))):
        for tonic in range(12):
            s = np.corrcoef(chroma, np.roll(prof, tonic))[0, 1]
            if s > best[0]:
                best = (s, tonic, mode)
    return best[1], best[2]


def mirex(pk, pm, rk, rm):
    if pk == rk and pm == rm:
        return 1.0
    if pm == rm and (pk - rk) % 12 in (5, 7):
        return 0.5
    if pm != rm:
        rel = (rk + 3) % 12 if rm == 0 else (rk - 3) % 12
        if pk == rel:
            return 0.3
        if pk == rk:
            return 0.2
    return 0.0


print(f"n={len(real)} songs\n")
print(f"{'chroma':16}{'profile':30}{'exact':>7}{'key':>7}{'mode':>7}{'MIREX':>7}")
for chroma_name in ("chroma_cqt", "chroma_cqt_harm", "chroma_cens"):
    C = cache[chroma_name][mask]
    for pname, (maj, mnr) in PROFILES.items():
        preds = [detect(c, maj, mnr) for c in C]
        exact = np.mean([p == (r[0], r[1]) for p, r in zip(preds, real)])
        key = np.mean([p[0] == r[0] for p, r in zip(preds, real)])
        mode = np.mean([p[1] == r[1] for p, r in zip(preds, real)])
        mx = np.mean([mirex(p[0], p[1], r[0], r[1]) for p, r in zip(preds, real)])
        print(f"{chroma_name:16}{pname:30}{exact:7.2f}{key:7.2f}{mode:7.2f}{mx:7.3f}")
print(f"\nmajority-class mode baseline: {max(np.mean(real[:, 1]), 1 - np.mean(real[:, 1])):.2f}")
