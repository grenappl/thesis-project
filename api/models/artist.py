from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.core.database import Base


class Artist(Base):
    __tablename__ = "artists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    genre: Mapped[str | None] = mapped_column(String(255), nullable=True)

    tracks: Mapped[list["Track"]] = relationship(
        secondary="track_artists", back_populates="artists"
    )
