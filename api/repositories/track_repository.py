from sqlalchemy import select
from sqlalchemy.orm import selectinload

from api.models.track import Track
from api.repositories.base import BaseRepository


class TrackRepository(BaseRepository[Track]):
    model = Track

    async def list(self, offset: int = 0, limit: int = 100) -> list[Track]:
        result = await self.session.execute(
            select(Track).options(selectinload(Track.artists)).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def get(self, id_: int) -> Track | None:
        result = await self.session.execute(
            select(Track).options(selectinload(Track.artists)).where(Track.id == id_)
        )
        return result.scalar_one_or_none()
