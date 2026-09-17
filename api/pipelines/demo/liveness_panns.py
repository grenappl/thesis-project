"""Liveness proxy: PANNs CNN14 (AudioSet-pretrained) crowd-noise probability.

Liveness in Spotify's schema estimates "was an audience present when this
was recorded." There's no DSP shortcut for that, so it's approximated with
a general-purpose audio tagger (PANNs CNN14, trained on AudioSet) and
averaging the predicted probability across its applause/cheering/crowd
classes.

The model itself (PyTorch) has no platform restriction, but the
`panns_inference` package shells out to `wget` on first import to fetch its
small AudioSet label CSV — which isn't a standard Windows command, so a
fresh install fails there with a raw FileNotFoundError, not a clean
ImportError. `_load_model` below catches that broadly (not just
ImportError) so it still fails with a clear message instead of a confusing
stack trace. In practice this module runs as part of the rest of Pipeline 2
under WSL2, where `wget` is present.
"""

from __future__ import annotations

import numpy as np
import librosa

PANNS_SAMPLE_RATE = 32000  # the rate CNN14 was trained on

# Substrings matched case-insensitively against PANNs' AudioSet label set.
CROWD_NOISE_LABEL_KEYWORDS = ["applause", "cheering", "crowd"]

_model = None
_labels: list[str] | None = None


def _load_model(checkpoint_path: str | None = None):
    global _model, _labels
    if _model is not None:
        return _model, _labels

    try:
        from panns_inference import AudioTagging, labels as panns_labels
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
    _labels = list(panns_labels)
    return _model, _labels


def _crowd_noise_label_indices(labels: list[str]) -> list[int]:
    lowered = [label.lower() for label in labels]
    indices = [
        i
        for i, label in enumerate(lowered)
        if any(keyword in label for keyword in CROWD_NOISE_LABEL_KEYWORDS)
    ]
    if not indices:
        raise RuntimeError(
            "None of the expected crowd-noise AudioSet labels "
            f"({CROWD_NOISE_LABEL_KEYWORDS}) were found in the PANNs label set. "
            "The label list may have changed — inspect `labels` from panns_inference."
        )
    return indices


def extract_liveness(y: np.ndarray, sr: int, checkpoint_path: str | None = None) -> float:
    """Average predicted probability across AudioSet's crowd-noise classes."""
    if sr != PANNS_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SAMPLE_RATE)

    model, labels = _load_model(checkpoint_path)
    clipwise_output, _ = model.inference(y[None, :])
    probs = clipwise_output[0]

    indices = _crowd_noise_label_indices(labels)
    return float(np.mean(probs[indices]))
