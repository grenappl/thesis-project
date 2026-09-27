"""Librosa-based descriptors: tempo, spectral centroid, key, mode, duration.

These are the descriptors that map onto plain DSP (no ML model needed) via
Librosa. Runs natively on Windows.

Energy used to live here too (a hand-weighted RMS/brightness/onset-density
composite, never fit against real data) — real-song validation
(scripts/validate_pipeline_accuracy.py) showed an embedding-based
regression against real Spotify energy beats it. It is now computed by
api.pipelines.demo.regression_heads — see feature_extraction.md, "Real-song
validation" section.
"""

from __future__ import annotations

import numpy as np
import librosa
from librosa.feature.rhythm import tempo as _librosa_tempo

# Temperley (1999) key profiles — the same values Essentia ships as its
# "temperley" key profile. Index 0 is the tonic; index i is the pitch class
# i semitones above it. Matches librosa's chroma convention (index 0 = C)
# and Spotify's key encoding (0=C, ..., 11=B), so no reindexing is needed.
# Replaced Krumhansl-Kessler (1990) after checking both against 66 real
# songs: exact key+mode 0.36 -> 0.47, tonic 0.38 -> 0.48, MIREX weighted
# score 0.530 -> 0.602 — consistent across three chroma variants, not one
# lucky configuration. Mode alone is weak for every profile tried (0.64-0.71,
# below the 0.74 you'd get by always answering "major"); a VGGish mode
# classifier trained on 491k real labels didn't beat that either.
_MAJOR_KEY_PROFILE = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
_MINOR_KEY_PROFILE = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])


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


# Winning-profile correlation minus the runner-up's. On the 66 validation
# songs, mode was right 0.73 of the time when the margin was at or above the
# median (~0.10) and 0.61 below it, so a small margin is a real, useful "not
# sure" signal. The runner-up is very often the relative major/minor (same
# notes, different home note): 10 of the 22 mode errors were exactly that
# swap. Resolving it from the bass register was tried and didn't help (mode
# 0.667 -> 0.682 at best, below always-major 0.742) — see
# scripts/calibration/relative_key_experiment.py.
LOW_KEY_CONFIDENCE_MARGIN = 0.10


def estimate_key(y: np.ndarray, sr: int) -> dict[str, float]:
    """Musical key (0=C, 1=C#/Db, ..., 11=B) and mode (1=major, 0=minor) via
    chroma-profile correlation — the standard Krumhansl-Schmuckler technique
    (with Temperley's profiles, see above): average the track's pitch-class
    energy (chroma) over time, then find which of the 24 major/minor key
    profiles it correlates with best.

    Also returns the runner-up profile (`key_alternative`/`mode_alternative`)
    and `key_confidence`, the correlation margin between the two (see
    `LOW_KEY_CONFIDENCE_MARGIN`).

    Silence (zero-variance chroma, correlation undefined) defaults to
    C major with zero confidence rather than raising or returning NaN.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    if np.std(chroma_mean) == 0:
        return {"key": 0, "mode": 1, "key_alternative": 9, "mode_alternative": 0, "key_confidence": 0.0}

    candidates = sorted(
        (
            (float(np.corrcoef(chroma_mean, np.roll(profile, tonic))[0, 1]), tonic, mode)
            for mode, profile in ((1, _MAJOR_KEY_PROFILE), (0, _MINOR_KEY_PROFILE))
            for tonic in range(12)
        ),
        reverse=True,
    )
    (best_score, key, mode), (second_score, alt_key, alt_mode) = candidates[0], candidates[1]
    return {
        "key": key,
        "mode": mode,
        "key_alternative": alt_key,
        "mode_alternative": alt_mode,
        "key_confidence": best_score - second_score,
    }


def extract_key_and_mode(y: np.ndarray, sr: int) -> tuple[int, int]:
    estimate = estimate_key(y, sr)
    return estimate["key"], estimate["mode"]


def extract_librosa_features(y: np.ndarray, sr: int) -> dict[str, float]:
    return {
        "tempo": extract_tempo(y, sr),
        "spectral_centroid": extract_spectral_centroid(y, sr),
        "duration_ms": extract_duration_ms(y, sr),
        **estimate_key(y, sr),
    }
