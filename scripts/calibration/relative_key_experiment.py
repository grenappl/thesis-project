"""Does the bass register resolve relative major/minor confusion?

Reads bass_chroma_cache.npz (bass_chroma_cache.py). Tuned settings are
scored leave-one-out so the 66 songs aren't used to both pick and grade.

    uv run python scripts/calibration/relative_key_experiment.py
"""
import numpy as np
import pandas as pd

MA = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
MI = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
PROFILES = [(t, 1, np.roll(MA, t)) for t in range(12)] + [(t, 0, np.roll(MI, t)) for t in range(12)]

c = np.load("data/validation_songs/bass_chroma_cache.npz", allow_pickle=True)
m = pd.read_csv("data/validation_songs/manifest.csv").set_index("id").loc[list(c["id"])]
rk, rm = m["key"].to_numpy(), m["mode"].to_numpy()
N = len(rk)


def scores(x):
    return np.array([np.corrcoef(x, p)[0, 1] for _, _, p in PROFILES])


S_full = np.array([scores(x) for x in c["chroma"]])
S_bass = np.array([scores(x) for x in c["bass"]])
KEYS = np.array([t for t, _, _ in PROFILES]); MODES = np.array([md for _, md, _ in PROFILES])


def relative(i):  # index of the relative major/minor of profile i
    t, md = KEYS[i], MODES[i]
    return 12 + (t + 9) % 12 if md == 1 else (t + 3) % 12


def predict_weighted(w):
    best = np.argmax(S_full + w * S_bass, axis=1)
    return KEYS[best], MODES[best]


def predict_tiebreak(delta, how):
    ks, ms = [], []
    for s in range(N):
        b = int(np.argmax(S_full[s])); r = relative(b)
        if S_full[s, b] - S_full[s, r] < delta:
            maj, mnr = (b, r) if MODES[b] == 1 else (r, b)
            if how == "tonic":  # which tonic carries more bass energy
                pick = maj if c["bass"][s][KEYS[maj]] >= c["bass"][s][KEYS[mnr]] else mnr
            else:  # which profile the bass chroma matches better
                pick = maj if S_bass[s, maj] >= S_bass[s, mnr] else mnr
            b = pick
        ks.append(KEYS[b]); ms.append(MODES[b])
    return np.array(ks), np.array(ms)


def report(name, k, md):
    print(f"{name:42} key {np.mean(k == rk):.3f}  mode {np.mean(md == rm):.3f}  exact {np.mean((k == rk) & (md == rm)):.3f}")


def loo(predict, grid, metric):
    """Leave-one-out: pick the grid value on 65 songs, predict the 66th."""
    table = [predict(g) for g in grid]
    ks, ms = np.zeros(N, int), np.zeros(N, int)
    for i in range(N):
        tr = np.arange(N) != i
        g = int(np.argmax([metric(k[tr], md[tr], tr) for k, md in table]))
        ks[i], ms[i] = table[g][0][i], table[g][1][i]
    return ks, ms


exact = lambda k, md, tr: np.mean((k == rk[tr]) & (md == rm[tr]))
report("baseline (Temperley, full chroma)", *predict_weighted(0.0))
report(f"always-major", np.where(True, predict_weighted(0.0)[0], 0), np.ones(N, int))
for how in ("tonic", "profile"):
    report(f"tiebreak[{how}] always (no tuning)", *predict_tiebreak(np.inf, how))
    report(f"tiebreak[{how}] delta LOO-tuned", *loo(lambda d: predict_tiebreak(d, how), [0.02, 0.05, 0.1, 0.2, 0.3, np.inf], exact))
report("full + bass, w=1 (no tuning)", *predict_weighted(1.0))
report("full + bass, w LOO-tuned", *loo(predict_weighted, [0, 0.25, 0.5, 1, 2, 4], exact))
report("bass only", *predict_weighted(1e6))

# confidence: margin between the winning profile and the runner-up
srt = np.sort(S_full, axis=1); margin = srt[:, -1] - srt[:, -2]
k0, m0 = predict_weighted(0.0)
q = margin >= np.median(margin)
print(f"baseline mode acc, high-margin half {np.mean(m0[q] == rm[q]):.3f}, low half {np.mean(m0[~q] == rm[~q]):.3f}; median margin {np.median(margin):.4f}")
