from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """One song's feature record: the twelve audio descriptors plus the VADER
    lyric sentiment.

    Deliberately the same field names as DemoFeaturesRead, so the result of
    POST /demo/extract-features can be posted here unchanged. Unknown fields
    (e.g. `spectral_centroid`) are ignored, and so is any client-sent
    `alignment_gap`: the gap is always recomputed server-side from `valence`
    and `lyric_sentiment`, so a stale or hand-edited value can never reach the
    model.
    """

    tempo: float = Field(ge=0, le=300, description="beats per minute")
    loudness: float = Field(ge=-80, le=10, description="dB, Spotify's scale")
    key: int = Field(ge=0, le=11, description="pitch class, 0=C ... 11=B")
    mode: int = Field(ge=0, le=1, description="1=major, 0=minor")
    energy: float = Field(ge=0, le=1)
    danceability: float = Field(ge=0, le=1)
    speechiness: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)
    liveness: float = Field(ge=0, le=1)
    duration_ms: float = Field(gt=0)
    valence: float = Field(ge=0, le=1)
    lyric_sentiment: float | None = Field(
        default=None, ge=-1, le=1,
        description="VADER compound score. Required for a prediction: the model "
                    "was trained with lyrics on every track (Section 4.2).",
    )


class ContributionGroups(BaseModel):
    """Exact SHAP contributions (TreeSHAP), summed per feature group, in
    popularity points. base_value + audio + lyric_sentiment + alignment_gap
    equals the prediction before clipping."""

    audio: float
    lyric_sentiment: float
    alignment_gap: float


class FeatureContribution(BaseModel):
    feature: str
    value: float | int
    contribution: float
    """Popularity points this feature moved the prediction, relative to base_value."""


class ModelInfo(BaseModel):
    name: str
    test_r2: float
    test_rmse: float
    trained_rows: int
    exported_at: str
    note: str


class PredictionRead(BaseModel):
    predicted_popularity: float
    """Spotify-scale popularity, 0-100."""
    base_value: float
    """The model's average prediction -- what it predicts before seeing any feature."""
    lyric_sentiment: float
    valence_normalized: float
    alignment_gap: float
    """lyric_sentiment - valence_normalized. Positive: lyrics read brighter than
    the music sounds; negative: the music sounds brighter than the lyrics read."""
    contributions: ContributionGroups
    features: list[FeatureContribution]
    """Every input feature, largest absolute contribution first. The twelve
    key one-hot columns are reported as a single `key` entry."""
    model: ModelInfo
