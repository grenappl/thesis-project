from pydantic import BaseModel, ConfigDict

from api.schemas.artist import ArtistRead


class TrackBase(BaseModel):
    name: str
    lyrics: str | None = None

    danceability: float | None = None
    energy: float | None = None
    loudness: float | None = None
    speechiness: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    liveness: float | None = None
    valence: float | None = None
    tempo: float | None = None
    mode: int | None = None

    lyric_sentiment: float | None = None
    audio_sentiment: float | None = None
    emotional_alignment: float | None = None

    popularity: int | None = None


class TrackCreate(TrackBase):
    artist_ids: list[int] = []


class TrackRead(TrackBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    artists: list[ArtistRead] = []
