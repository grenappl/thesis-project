import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.core.config import Settings
from api.core.dependencies import DemoFeatureExtractionServiceDep, SettingsDep
from api.schemas.demo_features import DemoFeaturesRead
from api.services.demo_feature_extraction_service import DemoExtractionError

router = APIRouter(prefix="/demo", tags=["demo"])

# Optional lyrics text alongside the audio upload — a new song has both
# inputs (audio + lyrics), same as a dataset track does, just neither is
# known to Spotify ahead of time. Omit for instrumental tracks.
LyricsForm = Annotated[str | None, Form()]


def validate_upload(audio: UploadFile, settings: Settings) -> None:
    """SRS REQ-5.3-1: reject unsupported or oversized uploads before the
    pipeline runs. Done here, ahead of the StreamingResponse, because once a
    stream starts its status code is already sent and a failure can only
    arrive as an error event."""
    allowed = settings.demo_allowed_audio_extensions
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio file type '{suffix or '(none)'}'. "
                   f"Upload one of: {', '.join(allowed)}.",
        )
    limit_mb = settings.demo_max_upload_mb
    if audio.size is not None and audio.size > limit_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Audio file is {audio.size / (1024 * 1024):.1f} MB; "
                   f"the limit is {limit_mb} MB.",
        )


@router.post("/extract-features", response_model=DemoFeaturesRead)
async def extract_features(
    audio: UploadFile,
    service: DemoFeatureExtractionServiceDep,
    settings: SettingsDep,
    lyrics: LyricsForm = None,
):
    validate_upload(audio, settings)
    try:
        return await service.extract_features(audio, lyrics)
    except DemoExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/extract-features/stream")
async def extract_features_stream(
    audio: UploadFile,
    service: DemoFeatureExtractionServiceDep,
    settings: SettingsDep,
    lyrics: LyricsForm = None,
):
    """Newline-delimited JSON stream of {"type": "log"|"result"|"error", ...}
    events, so the caller can show live progress instead of a single opaque
    wait. See DemoFeatureExtractionService.stream_extract_features — it
    never raises, so every outcome (including failures) arrives as an event
    rather than an HTTP error status, since headers/status are already sent
    once streaming begins. The one exception is upload validation, which runs
    before streaming starts and so can still return a normal 413/415.
    """
    validate_upload(audio, settings)

    async def event_stream():
        async for event in service.stream_extract_features(audio, lyrics):
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
