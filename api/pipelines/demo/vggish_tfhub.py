"""Valence, acousticness, instrumentalness, danceability, energy,
speechiness, and loudness via Google's official TF-Hub VGGish embedding +
a regression head per target trained on the Kaggle precomputed-embeddings
dataset (see scripts/train_vggish_ridge.py — Ridge for some targets,
gradient-boosted trees for others; this module doesn't care which, it just
loads and predicts). Danceability, energy, speechiness, and loudness moved
here from essentia_features.py/librosa_features.py/speechiness.py after
real-song validation (scripts/validate_pipeline_accuracy.py) showed this
approach clearly beats their old hand-built implementations — see
feature_extraction.md's "Real-song validation" section for the before/after
numbers.

This has to run as its own subprocess, never imported into the same process
as api.pipelines.demo.essentia_features. Essentia bundles its own TensorFlow
runtime; loading a standalone `tensorflow`/`tensorflow_hub` in the same
process crashes with `Check failed: ... ALREADY_EXISTS: Op with name
Bitcast` (a C++ op-registry collision, not something catchable in Python).

Why not just use Essentia's own VGGish model (audioset-vggish-3.pb)? It's
trained independently of Google's official release — same architecture,
different weights (cosine similarity ~0.56 against the Kaggle dataset's
embeddings, vs. ~0.97 for this module's embeddings). Ridge models trained
on the Kaggle embeddings only transfer to real inference if inference uses
embeddings from the same source. See feature_extraction.md for the full
investigation.

Run standalone with:

    uv run python -m api.pipelines.demo.vggish_tfhub <audio_path> [--models-dir <dir>]
"""

from __future__ import annotations

import os
from pathlib import Path

_VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
_VGGISH_SAMPLE_RATE = 16000

# Persisted under models/ (already gitignored) so the ~280MB module is
# downloaded once, not re-fetched into a temp dir on every pipeline run.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "models" / "tfhub_cache"
os.environ.setdefault("TFHUB_CACHE_DIR", str(_DEFAULT_CACHE_DIR))

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "vggish_ridge"
_TARGETS = ("valence", "acousticness", "instrumentalness", "danceability", "energy", "speechiness", "loudness")

_model = None  # lazily-loaded TF-Hub VGGish module, cached across calls in-process


def _load_vggish_model():
    global _model
    if _model is None:
        import tensorflow_hub as hub

        _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _model = hub.load(_VGGISH_MODEL_URL)
    return _model


def compute_vggish_embedding(path: str | Path) -> "np.ndarray":
    """Mean-pooled 128-dim VGGish embedding for the whole track."""
    import librosa
    import numpy as np
    import tensorflow as tf

    audio, _sr = librosa.load(str(path), sr=_VGGISH_SAMPLE_RATE, mono=True)
    model = _load_vggish_model()
    embeddings = model(tf.constant(audio, dtype=tf.float32))
    embeddings_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.array(embeddings)
    return embeddings_np.mean(axis=0)


def load_ridge_models(models_dir: str | Path = _DEFAULT_MODELS_DIR) -> dict:
    import joblib

    models_dir = Path(models_dir)
    models = {}
    for target in _TARGETS:
        model_path = models_dir / f"{target}.joblib"
        if not model_path.exists():
            raise RuntimeError(
                f"No trained model found for {target!r} at {model_path}. "
                "Run scripts/train_vggish_ridge.py first (see feature_extraction.md)."
            )
        models[target] = joblib.load(model_path)
    return models


def predict_vggish_features(path: str | Path, models_dir: str | Path = _DEFAULT_MODELS_DIR) -> dict[str, float]:
    embedding = compute_vggish_embedding(path)
    models = load_ridge_models(models_dir)
    return {target: float(model.predict(embedding.reshape(1, -1))[0]) for target, model in models.items()}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute valence/acousticness/instrumentalness/danceability/energy/speechiness/loudness via TF-Hub VGGish + a regression head per target."
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--models-dir", default=str(_DEFAULT_MODELS_DIR), help="Directory with the fitted .joblib Ridge models")
    args = parser.parse_args()

    result = predict_vggish_features(args.audio_path, args.models_dir)
    print(json.dumps(result, indent=2))
