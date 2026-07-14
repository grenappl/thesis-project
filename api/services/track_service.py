from sqlalchemy.ext.asyncio import AsyncSession

from api.models.artist import Artist
from api.models.track import Track
from api.repositories.artist_repository import ArtistRepository
from api.repositories.track_repository import TrackRepository
from api.schemas.track import TrackCreate


class TrackNotFoundError(Exception):
    pass


class TrackService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.tracks = TrackRepository(session)
        self.artists = ArtistRepository(session)

    async def list_tracks(self, offset: int = 0, limit: int = 100) -> list[Track]:
        return await self.tracks.list(offset=offset, limit=limit)

    async def get_track(self, track_id: int) -> Track:
        track = await self.tracks.get(track_id)
        if track is None:
            raise TrackNotFoundError(f"Track {track_id} not found")
        return track

    async def create_track(self, payload: TrackCreate) -> Track:
        artists: list[Artist] = []
        for artist_id in payload.artist_ids:
            artist = await self.artists.get(artist_id)
            if artist is not None:
                artists.append(artist)

        track = Track(
            **payload.model_dump(exclude={"artist_ids"}),
            artists=artists,
        )
        track = await self.tracks.add(track)
        await self.session.commit()
        return await self.get_track(track.id)

    async def delete_track(self, track_id: int) -> None:
        track = await self.get_track(track_id)
        await self.tracks.delete(track)
        await self.session.commit()
