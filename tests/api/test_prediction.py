from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.core.config import get_settings
from api.main import app
from api.schemas.prediction import PredictionRequest
from api.services.popularity_prediction_service import build_feature_vector

SONG = dict(
    tempo=120.0, loudness=-6.0, key=9, mode=0, energy=0.7, danceability=0.6,
    speechiness=0.05, acousticness=0.2, instrumentalness=0.0, liveness=0.1,
    duration_ms=210000.0, valence=0.25, lyric_sentiment=0.3,
)

model_missing = not Path(get_settings().popularity_model_path).is_file()
needs_model = pytest.mark.skipif(model_missing, reason="run scripts/export_popularity_model.py first")


@pytest.fixture
def client():
    return TestClient(app)


def test_feature_vector_one_hot_key_and_gap():
    values = build_feature_vector(PredictionRequest(**SONG), alignment_gap=0.8)
    assert values["key_9"] == 1.0
    assert sum(values[f"key_{i}"] for i in range(12)) == 1.0
    assert values["alignment_gap"] == 0.8


def test_missing_lyrics_is_422_not_a_crash(client):
    song = {k: v for k, v in SONG.items() if k != "lyric_sentiment"}
    response = client.post("/predict", json=song)
    assert response.status_code == 422
    assert "lyrics" in response.json()["detail"]


@needs_model
def test_prediction_matches_its_own_explanation(client):
    body = client.post("/predict", json=SONG).json()
    # alignment_gap = lyric_sentiment - (2 * valence - 1) = 0.3 - (-0.5)
    assert body["alignment_gap"] == pytest.approx(0.8)
    assert 0 <= body["predicted_popularity"] <= 100
    # TreeSHAP contributions plus the base value add back up to the prediction
    total = body["base_value"] + sum(f["contribution"] for f in body["features"])
    assert total == pytest.approx(body["predicted_popularity"], abs=1e-3)
    assert {f["feature"] for f in body["features"]} >= {"key", "alignment_gap", "valence"}


@needs_model
def test_demo_feature_record_is_accepted_as_is(client):
    # /predict takes the extract-features output directly, extra fields included
    record = {**SONG, "spectral_centroid": 2500.0, "key_confidence": 0.05,
              "key_alternative": 0, "mode_alternative": 1, "alignment_gap": 99.0}
    body = client.post("/predict", json=record).json()
    assert body["alignment_gap"] == pytest.approx(0.8)  # recomputed, client value ignored
