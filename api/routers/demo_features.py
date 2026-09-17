import json
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.core.dependencies import DemoFeatureExtractionServiceDep
from api.schemas.demo_features import DemoFeaturesRead
from api.services.demo_feature_extraction_service import DemoExtractionError

router = APIRouter(prefix="/demo", tags=["demo"])

# Optional lyrics text alongside the audio upload — a new song has both
# inputs (audio + lyrics), same as a dataset track does, just neither is
# known to Spotify ahead of time. Omit for instrumental tracks.
LyricsForm = Annotated[str | None, Form()]


@router.post("/extract-features", response_model=DemoFeaturesRead)
async def extract_features(
    audio: UploadFile, service: DemoFeatureExtractionServiceDep, lyrics: LyricsForm = None
):
    try:
        return await service.extract_features(audio, lyrics)
    except DemoExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/extract-features/stream")
async def extract_features_stream(
    audio: UploadFile, service: DemoFeatureExtractionServiceDep, lyrics: LyricsForm = None
):
    """Newline-delimited JSON stream of {"type": "log"|"result"|"error", ...}
    events, so the caller can show live progress instead of a single opaque
    wait. See DemoFeatureExtractionService.stream_extract_features — it
    never raises, so every outcome (including failures) arrives as an event
    rather than an HTTP error status, since headers/status are already sent
    once streaming begins.
    """

    async def event_stream():
        async for event in service.stream_extract_features(audio, lyrics):
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
