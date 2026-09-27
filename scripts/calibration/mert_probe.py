"""Feasibility probe: do the Kaggle MERT embeddings carry key/mode/tempo info
that VGGish/chroma don't? Held-out Kaggle rows only (no local MERT yet)."""
import os
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

t0 = time.time()
z = np.load("data/kaggle_embeddings/mert_embeddings.npz", allow_pickle=True)
ids = z["id"]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(ids), 220_000, replace=False))
X = z["features"][idx]
ids = ids[idx]
print("loaded", X.shape, f"{time.time()-t0:.0f}s", flush=True)
df = pd.read_csv(os.environ.get("SONGS_CSV", "data/songs.csv"), usecols=["id", "key", "mode", "tempo"]).drop_duplicates("id").set_index("id").loc[ids]
ok = (df["key"].to_numpy() >= 0) & (df["tempo"].to_numpy() > 30)
X, df = X[ok], df[ok]
n = len(X); tr, te = np.arange(0, n - 20_000), np.arange(n - 20_000, n)
Xs = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
mode = df["mode"].to_numpy(); key = df["key"].to_numpy(); tempo = df["tempo"].to_numpy()
print(f"always-major held-out: {(mode[te]==1).mean():.3f}", flush=True)

m = HistGradientBoostingClassifier(max_iter=500, early_stopping=True, random_state=0).fit(Xs[tr], mode[tr])
print(f"MERT mode GBM acc: {(m.predict(Xs[te])==mode[te]).mean():.3f}  {time.time()-t0:.0f}s", flush=True)

k24 = key * 2 + mode
lr = LogisticRegression(max_iter=300, C=0.5).fit(Xs[tr][:120_000], k24[tr][:120_000])
p = lr.predict(Xs[te])
print(f"MERT 24-class key+mode exact: {(p==k24[te]).mean():.3f}  tonic: {((p//2)==key[te]).mean():.3f}  mode: {((p%2)==mode[te]).mean():.3f}  {time.time()-t0:.0f}s", flush=True)

r = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True, random_state=0).fit(Xs[tr], np.log2(tempo[tr]))
pt = 2 ** r.predict(Xs[te])
oct_err = np.abs(np.log2(pt / tempo[te]))
print(f"MERT tempo corr: {np.corrcoef(pt, tempo[te])[0,1]:.3f}  within-octave(|log2|<0.5): {(oct_err<0.5).mean():.3f}  within 8%: {(np.abs(pt/tempo[te]-1)<0.08).mean():.3f}  {time.time()-t0:.0f}s", flush=True)
