from pydantic import BaseModel


class DemoFeaturesRead(BaseModel):
    energy: float
    """Google TF-Hub VGGish embedding + Ridge regression, fit on real Spotify energy values."""
    tempo: float
    spectral_centroid: float
    speechiness: float
    """Google TF-Hub VGGish embedding + Ridge regression, fit on real Spotify speechiness values."""
    liveness: float
    danceability: float
    """Google TF-Hub VGGish embedding + Ridge regression, fit on real Spotify danceability values."""
    valence: float
    """Google TF-Hub VGGish embedding + Ridge regression, fit on real Spotify valence values."""
    loudness: float
    """dB, via Essentia's ReplayGain algorithm — same units as Spotify's
    loudness column."""
    acousticness: float
    """Google TF-Hub VGGish embedding + Ridge regression, fit on real Spotify acousticness values."""
    instrumentalness: float
    """Google TF-Hub VGGish embedding + Ridge regression, fit on real Spotify instrumentalness values."""
    duration_ms: float
    key: int
    """0=C, 1=C#/Db, ..., 11=B — Spotify's key encoding, via chroma-profile
    correlation (Krumhansl-Schmuckler)."""
    mode: int
    """1=major, 0=minor, from the same key-detection step as `key`."""

    lyric_sentiment: float | None = None
    """VADER compound score on the submitted lyrics. None when no lyrics
    were submitted (e.g. an instrumental track)."""

    alignment_gap: float | None = None
    """lyric_sentiment - valence_normalized. None when no lyrics were
    submitted, since it can't be computed without lyric_sentiment."""
