import numpy as np
import pytest

from api.pipelines.demo.audio_io import load_audio
from api.pipelines.demo.librosa_features import (
    extract_duration_ms,
    extract_key_and_mode,
    extract_librosa_features,
    extract_spectral_centroid,
    extract_tempo,
)

SR = 22050


def _chord_wav(freqs, duration=5.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (0.3 * y / len(freqs)).astype(np.float32), sr


def test_extract_features_on_tone(tone_wav):
    y, sr = load_audio(tone_wav)
    features = extract_librosa_features(y, sr)

    assert features["tempo"] >= 0
    assert features["spectral_centroid"] > 0
    assert features["duration_ms"] > 0
    assert 0 <= features["key"] <= 11
    assert features["mode"] in (0, 1)


def test_duration_ms_matches_clip_length():
    y = np.zeros(SR * 3, dtype=np.float32)
    assert extract_duration_ms(y, SR) == pytest.approx(3000.0)


def test_key_and_mode_detect_c_major_chord():
    y, sr = _chord_wav([261.63, 329.63, 392.00])  # C4, E4, G4
    key, mode = extract_key_and_mode(y, sr)
    assert key == 0  # C
    assert mode == 1  # major


def test_key_and_mode_detect_a_minor_chord():
    y, sr = _chord_wav([220.00, 261.63, 329.63])  # A3, C4, E4
    key, mode = extract_key_and_mode(y, sr)
    assert key == 9  # A
    assert mode == 0  # minor


def test_key_and_mode_default_on_silence(silent_wav):
    y, sr = load_audio(silent_wav)
    key, mode = extract_key_and_mode(y, sr)
    assert key == 0
    assert mode == 1


def test_silent_audio_tempo_does_not_crash(silent_wav):
    y, sr = load_audio(silent_wav)
    tempo = extract_tempo(y, sr)
    assert tempo == 0.0


def test_silent_audio_spectral_centroid_does_not_crash(silent_wav):
    y, sr = load_audio(silent_wav)
    centroid = extract_spectral_centroid(y, sr)
    assert centroid == 0.0
