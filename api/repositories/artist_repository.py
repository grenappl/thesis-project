from sqlalchemy import select

from api.models.artist import Artist
from api.repositories.base import BaseRepository


class ArtistRepository(BaseRepository[Artist]):
    model = Artist

    async def get_by_name(self, name: str) -> Artist | None:
        result = await self.session.execute(select(Artist).where(Artist.name == name))
        return result.scalar_one_or_none()
