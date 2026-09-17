import sys

import numpy as np
import pytest

import api.pipelines.demo.liveness_panns as liveness_panns
from api.pipelines.demo.audio_io import load_audio
from api.pipelines.demo.liveness_panns import (
    PANNS_SAMPLE_RATE,
    _crowd_noise_label_indices,
    extract_liveness,
)

FAKE_LABELS = [
    "Speech",
    "Music",
    "Applause",
    "Cheering",
    "Crowd",
    "Silence",
]


class _FakeModel:
    """Stands in for panns_inference.AudioTagging without loading real weights."""

    def __init__(self, probs_by_label):
        self.probs_by_label = probs_by_label
        self.last_input_length = None

    def inference(self, batch):
        self.last_input_length = batch.shape[1]
        probs = np.array([self.probs_by_label.get(label, 0.0) for label in FAKE_LABELS])
        return probs[None, :], None


@pytest.fixture(autouse=True)
def _reset_module_cache(monkeypatch):
    monkeypatch.setattr(liveness_panns, "_model", None, raising=False)
    monkeypatch.setattr(liveness_panns, "_labels", None, raising=False)


def test_crowd_noise_label_indices_finds_expected_labels():
    indices = _crowd_noise_label_indices(FAKE_LABELS)
    found = {FAKE_LABELS[i] for i in indices}
    assert found == {"Applause", "Cheering", "Crowd"}


def test_crowd_noise_label_indices_raises_if_none_found():
    with pytest.raises(RuntimeError, match="crowd-noise"):
        _crowd_noise_label_indices(["Speech", "Music", "Silence"])


def test_extract_liveness_averages_only_crowd_classes(monkeypatch, tone_wav):
    fake_model = _FakeModel(
        {"Applause": 0.2, "Cheering": 0.4, "Crowd": 0.6, "Speech": 0.99}
    )
    monkeypatch.setattr(
        liveness_panns, "_load_model", lambda checkpoint_path=None: (fake_model, FAKE_LABELS)
    )

    y, sr = load_audio(tone_wav)
    liveness = extract_liveness(y, sr)

    assert liveness == pytest.approx((0.2 + 0.4 + 0.6) / 3)


def test_extract_liveness_resamples_to_panns_rate(monkeypatch, tone_wav):
    fake_model = _FakeModel({"Applause": 0.1, "Cheering": 0.1, "Crowd": 0.1})
    monkeypatch.setattr(
        liveness_panns, "_load_model", lambda checkpoint_path=None: (fake_model, FAKE_LABELS)
    )

    y, sr = load_audio(tone_wav)
    assert sr != PANNS_SAMPLE_RATE
    extract_liveness(y, sr)

    expected_len = round(len(y) * PANNS_SAMPLE_RATE / sr)
    assert abs(fake_model.last_input_length - expected_len) <= 1


def test_missing_panns_inference_package_raises_clear_error(monkeypatch, tone_wav):
    monkeypatch.setitem(sys.modules, "panns_inference", None)

    y, sr = load_audio(tone_wav)
    with pytest.raises(RuntimeError, match="panns-inference is not usable"):
        extract_liveness(y, sr)


def test_panns_inference_init_failure_other_than_importerror_is_also_caught(monkeypatch, tone_wav):
    # Regression test: on Windows, panns_inference raises FileNotFoundError
    # (not ImportError) because its first-import label download shells out
    # to `wget`, which doesn't exist there. Any failure importing/initializing
    # the package should map to the same clear RuntimeError.
    class _FakeBrokenModule:
        def __getattr__(self, name):
            raise FileNotFoundError("class_labels_indices.csv")

    monkeypatch.setitem(sys.modules, "panns_inference", _FakeBrokenModule())

    y, sr = load_audio(tone_wav)
    with pytest.raises(RuntimeError, match="panns-inference is not usable"):
        extract_liveness(y, sr)
