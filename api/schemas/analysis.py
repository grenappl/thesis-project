from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_serializer

from api.schemas.prediction import PredictionRead, PredictionRequest

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
Artist = Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]


class AnalysisCreate(BaseModel):
    """What the demo posts after feature extraction: the feature record plus
    the labels the History page shows."""

    song_title: Title
    artist: Artist | None = None
    features: PredictionRequest


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    song_title: str
    artist: str | None
    created_at: datetime
    predicted_popularity: float
    alignment_gap: float

    @field_serializer("created_at")
    def _as_utc(self, value: datetime) -> str:
        # Stored as naive UTC; the trailing offset lets the browser localise it.
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()


class AnalysisRead(AnalysisSummary):
    prediction: PredictionRead
    features: PredictionRequest


class AnalysisList(BaseModel):
    items: list[AnalysisSummary]
    total: int


class DailyCount(BaseModel):
    date: str
    """UTC calendar day, YYYY-MM-DD."""
    predictions: int


class PopularityBucket(BaseModel):
    bucket: str
    count: int


class FeatureInfluence(BaseModel):
    feature: str
    mean_abs_contribution: float
    """Average popularity points this feature moved a prediction, in either direction."""


class AnalyticsModel(BaseModel):
    name: str
    test_r2: float
    test_rmse: float


class AnalyticsRead(BaseModel):
    total: int
    avg_popularity: float | None
    avg_alignment_gap: float | None = Field(
        description="Signed mean of lyric_sentiment - valence_normalized. Positive: "
                    "the analysed songs' lyrics read brighter than their music sounds.",
    )
    over_time: list[DailyCount]
    distribution: list[PopularityBucket]
    feature_influence: list[FeatureInfluence]
    model: AnalyticsModel | None
