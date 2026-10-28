from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from api.core.analyses_db import get_analyses_session
from api.core.config import Settings, get_settings
from api.core.database import get_db_session
from api.services.analysis_service import AnalysisService
from api.services.demo_feature_extraction_service import DemoFeatureExtractionService
from api.services.popularity_prediction_service import PopularityPredictionService
from api.services.track_service import TrackService

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_track_service(session: DbSession) -> TrackService:
    return TrackService(session)


TrackServiceDep = Annotated[TrackService, Depends(get_track_service)]


def get_demo_feature_extraction_service(settings: SettingsDep) -> DemoFeatureExtractionService:
    return DemoFeatureExtractionService(settings)


DemoFeatureExtractionServiceDep = Annotated[
    DemoFeatureExtractionService, Depends(get_demo_feature_extraction_service)
]


def get_popularity_prediction_service(settings: SettingsDep) -> PopularityPredictionService:
    return PopularityPredictionService(settings)


PopularityPredictionServiceDep = Annotated[
    PopularityPredictionService, Depends(get_popularity_prediction_service)
]

AnalysesSession = Annotated[Session, Depends(get_analyses_session)]


def get_analysis_service(
    session: AnalysesSession,
    predictor: PopularityPredictionServiceDep,
    settings: SettingsDep,
) -> AnalysisService:
    return AnalysisService(session, predictor, settings)


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
