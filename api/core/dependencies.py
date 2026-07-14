from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db_session
from api.services.track_service import TrackService

DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_track_service(session: DbSession) -> TrackService:
    return TrackService(session)


TrackServiceDep = Annotated[TrackService, Depends(get_track_service)]
