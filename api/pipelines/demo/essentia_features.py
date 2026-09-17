"""Essentia-based descriptors: loudness.

Librosa has no equivalent, so this module is the reason the whole demo
pipeline runs under WSL2 (see feature_extraction.md for why Essentia
specifically needs it, and why the pipeline isn't split across two
environments). The import is deferred to call time so importing this module
(or the rest of the demo pipeline) on Windows doesn't fail — only calling
these functions does, with a message pointing at PIPELINE_SETUP.md.

Valence, acousticness, instrumentalness, danceability, energy, and
speechiness used to have Essentia/Librosa-based implementations here or in
librosa_features.py, but real-song validation
(scripts/validate_pipeline_accuracy.py) showed VGGish-embedding Ridge
regression against real Spotify targets beats all of them — see
feature_extraction.md's "Real-song validation" section. They're now
computed by api.pipelines.demo.vggish_tfhub, which has to run as its own
subprocess (Essentia's bundled TensorFlow and a standalone tensorflow/
tensorflow_hub install crash if imported in the same process).
"""

from __future__ import annotations

from pathlib import Path


def _import_essentia():
    try:
        import essentia.standard as es
    except ImportError as exc:
        raise RuntimeError(
            "essentia is not installed in this environment. Essentia has no "
            "Windows wheel — the demo pipeline (including this call) must run "
            "under WSL2. See PIPELINE_SETUP.md for the WSL2 venv setup."
        ) from exc
    return es


_REPLAYGAIN_REFERENCE_DB = -19.686  # NOT Essentia's own documented ~-31dB
# reference (that's SMPTE/ReplayGain-1.0-spec calibration, unrelated to
# Spotify's loudness metering) — fit directly against real songs' ReplayGain
# output vs. real Spotify loudness (scripts/validate_pipeline_accuracy.py).
# ReplayGain's raw output is a GAIN (dB needed to reach its reference
# level), inversely related to actual loudness — confirmed empirically: raw
# output correlated at -0.52 against real loudness; negating flips that to
# +0.52 (correlation is invariant to an additive constant, so the exact
# reference only affects absolute scale, not direction). Checked a full
# 2-parameter linear fit (slope+intercept) against this constant-shift-only
# approach: nearly identical MAE, so the extra degree of freedom isn't
# buying anything — the relationship really is close to a pure sign flip.
# Fit at n=18, re-checked at n=66 (refit value -19.541, MAE barely moved) —
# stable, not overfit to the small sample, unlike the tempo fix below it.


def extract_loudness(path: str | Path) -> float:
    """Integrated loudness in dB, derived from Essentia's `ReplayGain`
    algorithm (plain DSP, no model download needed). ReplayGain's raw output
    is a gain adjustment, not loudness — negated and re-based around its
    -31dB reference level to approximate actual loudness (same direction/
    units as Spotify's `loudness` column, not independently calibrated to
    Spotify's exact metering).
    """
    es = _import_essentia()
    audio = es.MonoLoader(filename=str(path))()
    raw_gain = float(es.ReplayGain()(audio))
    return _REPLAYGAIN_REFERENCE_DB - raw_gain


def extract_essentia_features(path: str | Path) -> dict[str, float]:
    return {
        "loudness": extract_loudness(path),
    }
