"""Shared audio loading for the demo pipeline, with graceful edge-case handling.

Every extractor in this pipeline takes an already-decoded (y, sr) waveform
rather than a file path, so "can this file even be read" is handled in one
place instead of once per library.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

MIN_DURATION_SEC = 0.5


class UnsupportedAudioError(ValueError):
    """Raised when a file can't be decoded as audio at all."""


class AudioTooShortError(ValueError):
    """Raised when a clip is too short to extract meaningful features from."""


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Decode an audio file to a mono waveform.

    Raises UnsupportedAudioError for unreadable/corrupt/unsupported formats
    and AudioTooShortError for empty or sub-MIN_DURATION_SEC clips, so callers
    can fail gracefully (e.g. return a 4xx to the demo-app user) instead of
    crashing on a stack trace from inside librosa/soundfile.
    """
    path = Path(path)
    if not path.exists():
        raise UnsupportedAudioError(f"Audio file not found: '{path}'")

    try:
        y, actual_sr = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:  # librosa/soundfile/audioread raise varied types
        raise UnsupportedAudioError(
            f"Could not decode '{path}' as audio (unsupported or corrupt format)"
        ) from exc

    if y.size == 0:
        raise AudioTooShortError(f"Audio file '{path}' contains no audio samples")

    duration_sec = len(y) / actual_sr
    if duration_sec < MIN_DURATION_SEC:
        raise AudioTooShortError(
            f"Audio clip '{path}' is {duration_sec:.2f}s, "
            f"shorter than the {MIN_DURATION_SEC}s minimum needed to extract features"
        )

    return y, actual_sr
