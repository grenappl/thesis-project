from sqlalchemy import Column, Float, ForeignKey, Integer, String, Table, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.core.database import Base

track_artists = Table(
    "track_artists",
    Base.metadata,
    Column("track_id", ForeignKey("tracks.id"), primary_key=True),
    Column("artist_id", ForeignKey("artists.id"), primary_key=True),
)


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    lyrics: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Audio features (Spotify audio feature ranges)
    danceability: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    loudness: Mapped[float | None] = mapped_column(Float, nullable=True)
    speechiness: Mapped[float | None] = mapped_column(Float, nullable=True)
    acousticness: Mapped[float | None] = mapped_column(Float, nullable=True)
    instrumentalness: Mapped[float | None] = mapped_column(Float, nullable=True)
    liveness: Mapped[float | None] = mapped_column(Float, nullable=True)
    valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Emotional alignment features (lyric/audio sentiment)
    lyric_sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    emotional_alignment: Mapped[float | None] = mapped_column(Float, nullable=True)

    popularity: Mapped[int | None] = mapped_column(Integer, nullable=True)

    artists: Mapped[list["Artist"]] = relationship(
        secondary=track_artists, back_populates="tracks"
    )
