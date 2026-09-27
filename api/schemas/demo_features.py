from pydantic import BaseModel


class DemoFeaturesRead(BaseModel):
    energy: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify energy values."""
    tempo: float
    spectral_centroid: float
    speechiness: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify speechiness values."""
    liveness: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify liveness values."""
    danceability: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify danceability values."""
    valence: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify valence values."""
    loudness: float
    """dB, same units as Spotify's loudness column — VGGish + PANNs-CNN14
    embeddings -> gradient-boosted regression head, fit on real Spotify loudness values."""
    acousticness: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify acousticness values."""
    instrumentalness: float
    """VGGish + PANNs-CNN14 embeddings -> gradient-boosted regression head, fit on real Spotify instrumentalness values."""
    duration_ms: float
    key: int
    """0=C, 1=C#/Db, ..., 11=B — Spotify's key encoding, via chroma-profile
    correlation (Krumhansl-Schmuckler method, Temperley profiles)."""
    mode: int
    """1=major, 0=minor, from the same key-detection step as `key`."""
    key_alternative: int | None = None
    mode_alternative: int | None = None
    """Runner-up key/mode from the same step — often the relative
    major/minor (same notes, different home note)."""
    key_confidence: float | None = None
    """Correlation margin between the winning and runner-up key profiles.
    Below ~0.10 the key/mode call is noticeably less reliable (mode 0.61 vs
    0.73 correct on 66 real songs)."""

    lyric_sentiment: float | None = None
    """VADER compound score on the submitted lyrics. None when no lyrics
    were submitted (e.g. an instrumental track)."""

    alignment_gap: float | None = None
    """lyric_sentiment - valence_normalized. None when no lyrics were
    submitted, since it can't be computed without lyric_sentiment."""
