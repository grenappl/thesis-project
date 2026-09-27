import sys

import numpy as np
import pytest

import api.pipelines.demo.panns_embedding as pe
from api.pipelines.demo.audio_io import load_audio


class _FakeModel:
    def __init__(self):
        self.last_len = None

    def inference(self, batch):
        self.last_len = batch.shape[1]
        return np.zeros((1, 527)), np.arange(2048, dtype=np.float32)[None, :]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(pe, "_model", None)


def test_returns_2048_embedding_and_resamples(monkeypatch, tone_wav):
    fake = _FakeModel()
    monkeypatch.setattr(pe, "_load_model", lambda checkpoint_path=None: fake)
    y, sr = load_audio(tone_wav)
    emb = pe.extract_panns_embedding(y, sr)
    assert emb.shape == (2048,)
    assert abs(fake.last_len - round(len(y) * pe.PANNS_SAMPLE_RATE / sr)) <= 1


def test_unusable_panns_inference_raises_clear_error(monkeypatch, tone_wav):
    # on Windows panns_inference fails with FileNotFoundError (wget), not ImportError
    class _Broken:
        def __getattr__(self, name):
            raise FileNotFoundError("class_labels_indices.csv")

    monkeypatch.setitem(sys.modules, "panns_inference", _Broken())
    y, sr = load_audio(tone_wav)
    with pytest.raises(RuntimeError, match="panns-inference is not usable"):
        pe.extract_panns_embedding(y, sr)
