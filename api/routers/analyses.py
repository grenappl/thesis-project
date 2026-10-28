from fastapi import APIRouter, HTTPException, Query, Response, status

from api.core.dependencies import AnalysisServiceDep
from api.schemas.analysis import AnalysisCreate, AnalysisList, AnalysisRead, AnalyticsRead
from api.services.analysis_service import AnalysisNotFoundError
from api.services.popularity_prediction_service import ModelUnavailableError, PredictionError

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisRead, status_code=status.HTTP_201_CREATED)
def create_analysis(payload: AnalysisCreate, service: AnalysisServiceDep):
    """Score a song (same model and response as POST /predict) and save the
    result so it appears in History and Analytics."""
    try:
        return service.create(payload)
    except PredictionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("", response_model=AnalysisList)
def list_analyses(
    service: AnalysisServiceDep,
    search: str | None = Query(default=None, max_length=255, description="Matches title or artist."),
    days: int | None = Query(default=None, ge=1, le=3650, description="Only the last N days."),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return service.list(search, days, limit, offset)


# Declared before /{analysis_id}, which would otherwise swallow "stats".
@router.get("/stats", response_model=AnalyticsRead)
def analytics(service: AnalysisServiceDep):
    return service.stats()


@router.get("/{analysis_id}", response_model=AnalysisRead)
def get_analysis(analysis_id: int, service: AnalysisServiceDep):
    try:
        return service.get(analysis_id)
    except AnalysisNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: int, service: AnalysisServiceDep):
    try:
        service.delete(analysis_id)
    except AnalysisNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
