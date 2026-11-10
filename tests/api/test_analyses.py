from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.core.analyses_db import get_analyses_session
from api.core.config import get_settings
from api.core.dependencies import get_popularity_prediction_service
from api.main import app
from api.models.analysis import Analysis
from api.pipelines.demo.alignment import compute_alignment_gap, normalize_valence
from api.schemas.prediction import (
    ContributionGroups,
    FeatureContribution,
    ModelInfo,
    PredictionRead,
    PredictionRequest,
)
from api.services.analysis_service import popularity_bucket
from api.services.popularity_prediction_service import PredictionError

SONG = dict(
    tempo=120.0, loudness=-6.0, key=9, mode=0, energy=0.7, danceability=0.6,
    speechiness=0.05, acousticness=0.2, instrumentalness=0.0, liveness=0.1,
    duration_ms=210000.0, valence=0.25, lyric_sentiment=0.3,
)


class FakePredictor:
    """Stands in for the XGBoost model so these tests don't need the exported
    model file. Popularity is the song's tempo / 2, so tests pick any score."""

    def predict(self, request: PredictionRequest) -> PredictionRead:
        if request.lyric_sentiment is None:
            raise PredictionError("A popularity prediction needs lyrics.")
        gap = compute_alignment_gap(request.lyric_sentiment, request.valence)
        return PredictionRead(
            predicted_popularity=request.tempo / 2,
            base_value=24.5,
            lyric_sentiment=request.lyric_sentiment,
            valence_normalized=normalize_valence(request.valence),
            alignment_gap=gap,
            contributions=ContributionGroups(audio=1.0, lyric_sentiment=0.5, alignment_gap=-0.5),
            features=[
                FeatureContribution(feature="danceability", value=request.danceability, contribution=-3.0),
                FeatureContribution(feature="alignment_gap", value=gap, contribution=1.0),
            ],
            model=ModelInfo(name="fake", test_r2=0.27, test_rmse=15.0, trained_rows=1,
                            exported_at="2026-01-01", note="test"),
        )


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'analyses.db').as_posix()}")
    Analysis.__table__.create(engine)
    return engine


@pytest.fixture
def client(engine):
    def session():
        with Session(engine, expire_on_commit=False) as s:
            yield s

    app.dependency_overrides[get_analyses_session] = session
    app.dependency_overrides[get_popularity_prediction_service] = lambda: FakePredictor()
    yield TestClient(app)
    app.dependency_overrides.clear()


def save(client, title="Song", artist=None, **overrides):
    body = {"song_title": title, "features": {**SONG, **overrides}}
    if artist is not None:
        body["artist"] = artist
    response = client.post("/analyses", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_returns_prediction_and_saves_it(client):
    created = save(client, "Neon Static", artist="Rue Haven")
    assert created["song_title"] == "Neon Static"
    assert created["artist"] == "Rue Haven"
    assert created["predicted_popularity"] == 60.0
    assert created["prediction"]["alignment_gap"] == pytest.approx(0.8)
    assert created["created_at"].endswith("+00:00")

    fetched = client.get(f"/analyses/{created['id']}").json()
    assert fetched["prediction"] == created["prediction"]
    assert fetched["features"]["tempo"] == 120.0


def test_gap_is_recomputed_not_trusted(client):
    created = save(client, alignment_gap=-1.5)
    assert created["alignment_gap"] == pytest.approx(0.8)


def test_missing_lyrics_is_422_and_saves_nothing(client):
    features = {k: v for k, v in SONG.items() if k != "lyric_sentiment"}
    response = client.post("/analyses", json={"song_title": "No Words", "features": features})
    assert response.status_code == 422
    assert client.get("/analyses").json()["total"] == 0


def test_blank_title_rejected(client):
    response = client.post("/analyses", json={"song_title": "   ", "features": SONG})
    assert response.status_code == 422


def test_unknown_analysis_is_404(client):
    assert client.get("/analyses/999").status_code == 404
    assert client.delete("/analyses/999").status_code == 404


def test_delete_removes_it(client):
    created = save(client)
    assert client.delete(f"/analyses/{created['id']}").status_code == 204
    assert client.get(f"/analyses/{created['id']}").status_code == 404


def test_list_is_newest_first_and_paginated(client):
    for title in ("first", "second", "third"):
        save(client, title)
    page = client.get("/analyses", params={"limit": 2}).json()
    assert page["total"] == 3
    assert [i["song_title"] for i in page["items"]] == ["third", "second"]
    rest = client.get("/analyses", params={"limit": 2, "offset": 2}).json()
    assert [i["song_title"] for i in rest["items"]] == ["first"]


def test_search_matches_title_or_artist_and_treats_wildcards_literally(client):
    save(client, "Neon Static", artist="Rue Haven")
    save(client, "Glass Halo", artist="Marin Faye")
    save(client, "100% Pure")

    titles = lambda **p: sorted(i["song_title"] for i in client.get("/analyses", params=p).json()["items"])
    assert titles(search="neon") == ["Neon Static"]
    assert titles(search="MARIN") == ["Glass Halo"]
    assert titles(search="%") == ["100% Pure"]
    assert titles(search="_") == []


def test_days_filter(client, engine):
    old = save(client, "old")
    save(client, "recent")
    with Session(engine) as s:
        row = s.get(Analysis, old["id"])
        row.created_at = row.created_at - timedelta(days=40)
        s.commit()
    assert [i["song_title"] for i in client.get("/analyses", params={"days": 30}).json()["items"]] == ["recent"]
    assert client.get("/analyses").json()["total"] == 2


def test_stats_empty(client):
    stats = client.get("/analyses/stats").json()
    assert stats["total"] == 0
    assert stats["avg_popularity"] is None
    assert stats["avg_alignment_gap"] is None
    assert stats["over_time"] == []
    assert [b["count"] for b in stats["distribution"]] == [0, 0, 0, 0, 0]
    assert stats["feature_influence"] == []


def test_stats_aggregates(client):
    save(client, tempo=40.0)    # popularity 20, bucket 0-20
    save(client, tempo=130.0)   # popularity 65, bucket 61-80
    stats = client.get("/analyses/stats").json()
    assert stats["total"] == 2
    assert stats["avg_popularity"] == pytest.approx(42.5)
    assert stats["avg_alignment_gap"] == pytest.approx(0.8)
    assert [b["count"] for b in stats["distribution"]] == [1, 0, 0, 1, 0]
    assert sum(d["predictions"] for d in stats["over_time"]) == 2
    top = stats["feature_influence"][0]
    assert top["feature"] == "danceability"
    assert top["mean_abs_contribution"] == pytest.approx(3.0)


@pytest.mark.parametrize("popularity,bucket", [
    (0, 0), (20, 0), (20.4, 0), (20.6, 1), (40, 1), (41, 2), (80, 3), (81, 4), (100, 4),
])
def test_popularity_bucket_matches_displayed_score(popularity, bucket):
    assert popularity_bucket(popularity) == bucket


@pytest.mark.skipif(not Path(get_settings().popularity_model_path).is_file(),
                    reason="run scripts/export_popularity_model.py first")
def test_real_model_round_trip(engine):
    def session():
        with Session(engine, expire_on_commit=False) as s:
            yield s

    app.dependency_overrides[get_analyses_session] = session
    try:
        client = TestClient(app)
        created = client.post("/analyses", json={"song_title": "Real", "features": SONG}).json()
        direct = client.post("/predict", json=SONG).json()
        assert created["prediction"]["predicted_popularity"] == pytest.approx(direct["predicted_popularity"])
        assert client.get(f"/analyses/{created['id']}").json()["prediction"]["features"] == direct["features"]
    finally:
        app.dependency_overrides.clear()
