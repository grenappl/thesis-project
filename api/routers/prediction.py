from fastapi import APIRouter, HTTPException, status

from api.core.dependencies import PopularityPredictionServiceDep
from api.schemas.prediction import PredictionRead, PredictionRequest
from api.services.popularity_prediction_service import ModelUnavailableError, PredictionError

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionRead)
def predict(request: PredictionRequest, service: PopularityPredictionServiceDep):
    """Predicted Spotify popularity for one song, with its alignment gap and a
    per-feature breakdown of what drove the prediction.

    Takes the feature record POST /demo/extract-features returns, so the demo
    flow is: extract features from the uploaded audio + lyrics, then post the
    result here. Plain `def`, not `async def`: inference is a short CPU-bound
    call, which FastAPI runs in its threadpool without blocking the event loop.
    """
    try:
        return service.predict(request)
    except PredictionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
