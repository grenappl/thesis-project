"""Librosa-based descriptors: tempo, spectral centroid, key, mode, duration.

These are the descriptors that map onto plain DSP (no ML model needed) via
Librosa. Runs natively on Windows.

Energy used to live here too (a hand-weighted RMS/brightness/onset-density
composite, never fit against real data) — real-song validation
(scripts/validate_pipeline_accuracy.py) showed a VGGish-embedding Ridge
regression against real Spotify energy beats it. It's now computed by
api.pipelines.demo.vggish_tfhub — see feature_extraction.md's "Real-song
validation" section.
"""

from __future__ import annotations

import numpy as np
import librosa
from librosa.feature.rhythm import tempo as _librosa_tempo

# Krumhansl-Kessler key profiles (Krumhansl, 1990) — empirically measured
# perceived "fit" of each pitch class to a major/minor tonic. Index 0 is the
# tonic itself (highest weight); index i is the pitch class i semitones above
# the tonic. Matches librosa's chroma convention (index 0 = C) and Spotify's
# key encoding (0=C, 1=C#/Db, ..., 11=B), so no reindexing is needed.
_MAJOR_KEY_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_KEY_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


_TEMPO_STD_BPM = 1.5  # librosa's default (1.0) makes the tempo estimate hug
# its internal ~120 BPM prior too tightly — confirmed on real validation
# songs, several unrelated tracks collapsed to near-identical BPM outputs
# under the default. Loosening it is a real, replicated improvement (0.025
# corr at n=18 and default 1.0; 0.059 at n=66 and default 1.0), but the
# specific best value is NOT well-determined: a clean sweep at n=18 picked
# 1.5 as a clear peak, but re-swept at n=66 the 1.0-3.0 range was noisy with
# no stable peak (1.5-2.5 all landed 0.14-0.25, no consistent winner). Kept
# at 1.5 — inside that plateau, not chasing an unstable "best" value from
# either sample. Still a genuinely weak feature (corr 0.18-0.39 depending on
# sample) — this narrows the failure mode, it doesn't fix it. See
# scripts/validate_pipeline_accuracy.py and feature_extraction.md.


def extract_tempo(y: np.ndarray, sr: int) -> float:
    """Tempo (BPM) via onset-strength autocorrelation, with a loosened
    tempo-estimation prior (see `_TEMPO_STD_BPM`).

    Silent or near-silent audio produces a flat onset-strength envelope with
    no clear periodicity; librosa still returns a (essentially arbitrary)
    tempo estimate rather than raising, so callers should treat a tempo
    reading on very-low-energy audio as low-confidence.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if not np.any(onset_env):
        return 0.0
    tempo = _librosa_tempo(onset_envelope=onset_env, sr=sr, std_bpm=_TEMPO_STD_BPM)[0]
    return float(tempo)


def extract_spectral_centroid(y: np.ndarray, sr: int) -> float:
    """Spectral centroid (Hz), averaged across frames."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.mean(centroid))


def extract_duration_ms(y: np.ndarray, sr: int) -> float:
    """Clip duration in milliseconds — just arithmetic, already have y/sr."""
    return float(len(y) / sr * 1000.0)


def extract_key_and_mode(y: np.ndarray, sr: int) -> tuple[int, int]:
    """Musical key (0=C, 1=C#/Db, ..., 11=B) and mode (1=major, 0=minor) via
    chroma-profile correlation — the standard Krumhansl-Schmuckler technique:
    average the track's pitch-class energy (chroma) over time, then find
    which of the 24 major/minor key profiles it correlates with best.

    Silence (zero-variance chroma, correlation undefined) defaults to
    (key=0, mode=1) rather than raising or returning NaN.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    if np.std(chroma_mean) == 0:
        return 0, 1

    best_score = -np.inf
    best_key = 0
    best_mode = 1
    for mode, profile in ((1, _MAJOR_KEY_PROFILE), (0, _MINOR_KEY_PROFILE)):
        for tonic in range(12):
            rotated = np.roll(profile, tonic)
            score = float(np.corrcoef(chroma_mean, rotated)[0, 1])
            if score > best_score:
                best_score = score
                best_key = tonic
                best_mode = mode

    return best_key, best_mode


def extract_librosa_features(y: np.ndarray, sr: int) -> dict[str, float]:
    key, mode = extract_key_and_mode(y, sr)
    return {
        "tempo": extract_tempo(y, sr),
        "spectral_centroid": extract_spectral_centroid(y, sr),
        "duration_ms": extract_duration_ms(y, sr),
        "key": key,
        "mode": mode,
    }
