"""Spotify-style features from the VGGish + PANNs embeddings.

One gradient-boosted regression head per target, all fed the same input:
the 128-dim VGGish embedding concatenated with the 2048-dim PANNs embedding
compressed to 256 dims by a fitted PCA. Trained against real Spotify values
for ~491k tracks by scripts/train_vggish_ridge.py; see that script's
docstring for how this design was arrived at, and feature_extraction.md's
"Real-song validation" section for accuracy on real audio.

Runs in the main pipeline process — no TensorFlow here, only scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TARGETS = ("valence", "acousticness", "danceability", "energy", "speechiness",
           "instrumentalness", "loudness", "liveness")

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HEADS_DIR = _ROOT / "models" / "regression_heads"
DEFAULT_PCA_PATH = _ROOT / "models" / "panns" / "panns_pca256.joblib"

_UNBOUNDED = {"loudness"}  # dB; every other target is on Spotify's [0, 1] scale


def load_heads(heads_dir: str | Path = DEFAULT_HEADS_DIR, pca_path: str | Path = DEFAULT_PCA_PATH):
    import joblib

    missing = [p for p in [Path(pca_path), *(Path(heads_dir) / f"{t}.joblib" for t in TARGETS)] if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing trained model file(s): {', '.join(str(p) for p in missing)}. "
            "Run scripts/train_vggish_ridge.py first (see feature_extraction.md)."
        )
    heads = {t: joblib.load(Path(heads_dir) / f"{t}.joblib") for t in TARGETS}
    return joblib.load(pca_path), heads


def build_features(vggish_embedding: np.ndarray, panns_embedding: np.ndarray, pca) -> np.ndarray:
    panns_reduced = pca.transform(np.asarray(panns_embedding, dtype=np.float32).reshape(1, -1))
    return np.hstack([np.asarray(vggish_embedding, dtype=np.float32).reshape(1, -1), panns_reduced]).astype(np.float32)


def predict_features(
    vggish_embedding: np.ndarray,
    panns_embedding: np.ndarray,
    heads_dir: str | Path = DEFAULT_HEADS_DIR,
    pca_path: str | Path = DEFAULT_PCA_PATH,
) -> dict[str, float]:
    pca, heads = load_heads(heads_dir, pca_path)
    x = build_features(vggish_embedding, panns_embedding, pca)
    preds = {t: float(model.predict(x)[0]) for t, model in heads.items()}
    # boosted trees can overshoot slightly past the training range
    return {t: v if t in _UNBOUNDED else min(max(v, 0.0), 1.0) for t, v in preds.items()}
