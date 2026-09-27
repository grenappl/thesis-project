"""PANNs CNN14 (AudioSet-pretrained) clip embedding, 2048-dim.

Feeds api.pipelines.demo.regression_heads, alongside the VGGish embedding.
This replaced an earlier liveness heuristic that averaged CNN14's predicted
probability over the applause/cheering/crowd classes: it ranked songs well
(real-song corr 0.694) but sat on the wrong scale entirely (predictions
~0.01 against real Spotify liveness of ~0.1-0.8, MAE 0.224). A regression
head over the embedding brought MAE to ~0.10 — see feature_extraction.md.

The model itself (PyTorch) has no platform restriction, but the
`panns_inference` package shells out to `wget` on first import to fetch its
small AudioSet label CSV — which isn't a standard Windows command, so a
fresh install fails there with a raw FileNotFoundError, not a clean
ImportError. `_load_model` catches that broadly so it still fails with a
clear message. In practice this runs under WSL2, where `wget` is present.
"""

from __future__ import annotations

import numpy as np
import librosa

PANNS_SAMPLE_RATE = 32000  # the rate CNN14 was trained on

_model = None


def _load_model(checkpoint_path: str | None = None):
    global _model
    if _model is not None:
        return _model

    try:
        from panns_inference import AudioTagging
    except Exception as exc:  # not just ImportError — see module docstring
        raise RuntimeError(
            "panns-inference is not usable in this environment (either it isn't "
            "installed, or its first-import label download failed — that download "
            "shells out to `wget`, which isn't available on stock Windows). Run this "
            "under WSL2, or install it with `uv add panns-inference` and download the "
            "Cnn14_mAP=0.431.pth checkpoint (see feature_extraction.md)."
        ) from exc

    kwargs = {"checkpoint_path": checkpoint_path} if checkpoint_path else {}
    _model = AudioTagging(**kwargs)
    return _model


def extract_panns_embedding(y: np.ndarray, sr: int, checkpoint_path: str | None = None) -> np.ndarray:
    """2048-dim CNN14 embedding for the whole clip."""
    if sr != PANNS_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SAMPLE_RATE)
    _clipwise, embedding = _load_model(checkpoint_path).inference(y[None, :])
    return embedding[0]
