"""Mean-pooled 128-dim embedding from Google's official TF-Hub VGGish.

This module only produces the embedding; the per-target regression heads
that turn it into Spotify-style features live in
api.pipelines.demo.regression_heads (they also use the PANNs embedding,
which is computed in the main pipeline process).

This has to run as its own subprocess, never imported into the same process
as api.pipelines.demo.essentia_features. Essentia bundles its own TensorFlow
runtime; loading a standalone `tensorflow`/`tensorflow_hub` in the same
process crashes with `Check failed: ... ALREADY_EXISTS: Op with name
Bitcast` (a C++ op-registry collision, not something catchable in Python).
Nothing calls Essentia today, but the isolation stays in case it returns.

Why not Essentia's own VGGish model (audioset-vggish-3.pb)? It's trained
independently of Google's release — same architecture, different weights
(cosine similarity ~0.56 against the Kaggle training embeddings, vs ~0.97
for this module). Heads trained on Kaggle embeddings only transfer if
inference uses embeddings from the same source.

Run standalone (prints a JSON list of 128 floats):

    uv run python -m api.pipelines.demo.vggish_tfhub <audio_path>
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

_model = None  # lazily-loaded TF-Hub VGGish module, cached across calls in-process


def _load_vggish_model():
    global _model
    if _model is None:
        import tensorflow as tf
        import tensorflow_hub as hub

        # VGGish always runs on CPU. It's small, and in the GPU calibration
        # container PyTorch (PANNs) owns the CUDA libraries — letting
        # TensorFlow load a second, differently-versioned cuDNN in the same
        # machine is a common crash. Must happen before any TF op runs.
        tf.config.set_visible_devices([], "GPU")
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


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Print the mean-pooled TF-Hub VGGish embedding as JSON.")
    parser.add_argument("audio_path", help="Path to the audio file")
    args = parser.parse_args()
    print(json.dumps(compute_vggish_embedding(args.audio_path).tolist()))
