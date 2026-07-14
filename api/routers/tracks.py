from fastapi import APIRouter, HTTPException, status

from api.core.dependencies import TrackServiceDep
from api.schemas.track import TrackCreate, TrackRead
from api.services.track_service import TrackNotFoundError

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("/", response_model=list[TrackRead])
async def list_tracks(service: TrackServiceDep, offset: int = 0, limit: int = 100):
    return await service.list_tracks(offset=offset, limit=limit)


@router.get("/{track_id}", response_model=TrackRead)
async def get_track(track_id: int, service: TrackServiceDep):
    try:
        return await service.get_track(track_id)
    except TrackNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/", response_model=TrackRead, status_code=status.HTTP_201_CREATED)
async def create_track(payload: TrackCreate, service: TrackServiceDep):
    return await service.create_track(payload)


@router.delete("/{track_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_track(track_id: int, service: TrackServiceDep):
    try:
        await service.delete_track(track_id)
    except TrackNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
