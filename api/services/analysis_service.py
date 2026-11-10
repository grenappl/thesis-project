"""Saved analyses: score a song, keep the result, and aggregate what's kept.

Creating an analysis runs the same PopularityPredictionService that backs
POST /predict -- the stored result is exactly what /predict would have
returned -- and saves it alongside the features that produced it, so the Song
page can redraw the full result screen later without re-running anything.
"""

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from api.core.config import Settings
from api.models.analysis import Analysis
from api.repositories.analysis_repository import AnalysisRepository
from api.schemas.analysis import (
    AnalysisCreate,
    AnalysisList,
    AnalysisRead,
    AnalysisSummary,
    AnalyticsModel,
    AnalyticsRead,
    DailyCount,
    FeatureInfluence,
    PopularityBucket,
)
from api.schemas.prediction import PredictionRead, PredictionRequest
from api.services.popularity_prediction_service import PopularityPredictionService

BUCKET_LABELS = ("0-20", "21-40", "41-60", "61-80", "81-100")
DAYS_SHOWN = 30


class AnalysisNotFoundError(Exception):
    pass


def popularity_bucket(popularity: float) -> int:
    """Index into BUCKET_LABELS, using the whole-number score the UI displays."""
    rounded = round(popularity)
    return 0 if rounded <= 20 else min(4, (rounded - 1) // 20)


def _to_read(analysis: Analysis) -> AnalysisRead:
    return AnalysisRead(
        **AnalysisSummary.model_validate(analysis).model_dump(),
        prediction=PredictionRead.model_validate(analysis.result),
        features=PredictionRequest.model_validate(analysis.inputs),
    )


class AnalysisService:
    def __init__(self, session: Session, predictor: PopularityPredictionService, settings: Settings) -> None:
        self.session = session
        self.analyses = AnalysisRepository(session)
        self.predictor = predictor
        self.settings = settings

    def create(self, payload: AnalysisCreate) -> AnalysisRead:
        # Scored before anything is written, so a song that can't be scored
        # (e.g. no lyrics) leaves no half-saved row behind.
        prediction = self.predictor.predict(payload.features)
        analysis = self.analyses.add(Analysis(
            song_title=payload.song_title,
            artist=payload.artist or None,
            predicted_popularity=prediction.predicted_popularity,
            alignment_gap=prediction.alignment_gap,
            inputs=payload.features.model_dump(mode="json"),
            result=prediction.model_dump(mode="json"),
        ))
        self.session.commit()
        return _to_read(analysis)

    def get(self, analysis_id: int) -> AnalysisRead:
        return _to_read(self._get_row(analysis_id))

    def delete(self, analysis_id: int) -> None:
        self.analyses.delete(self._get_row(analysis_id))
        self.session.commit()

    def list(self, text: str | None, days: int | None, limit: int, offset: int) -> AnalysisList:
        since = None
        if days is not None:
            since = (datetime.now(UTC) - timedelta(days=days)).replace(tzinfo=None)
        rows, total = self.analyses.search(text.strip() if text else None, since, limit, offset)
        return AnalysisList(items=[AnalysisSummary.model_validate(r) for r in rows], total=total)

    def stats(self) -> AnalyticsRead:
        rows = self.analyses.all()

        per_day: Counter[str] = Counter(r.created_at.date().isoformat() for r in rows)
        over_time = [DailyCount(date=day, predictions=n) for day, n in sorted(per_day.items())[-DAYS_SHOWN:]]

        buckets = Counter(popularity_bucket(r.predicted_popularity) for r in rows)
        distribution = [PopularityBucket(bucket=label, count=buckets[i]) for i, label in enumerate(BUCKET_LABELS)]

        influence: defaultdict[str, float] = defaultdict(float)
        for row in rows:
            for item in row.result.get("features", []):
                influence[item["feature"]] += abs(item["contribution"])
        feature_influence = sorted(
            (FeatureInfluence(feature=name, mean_abs_contribution=total / len(rows))
             for name, total in influence.items()),
            key=lambda f: f.mean_abs_contribution, reverse=True,
        )

        return AnalyticsRead(
            total=len(rows),
            avg_popularity=sum(r.predicted_popularity for r in rows) / len(rows) if rows else None,
            avg_alignment_gap=sum(r.alignment_gap for r in rows) / len(rows) if rows else None,
            over_time=over_time,
            distribution=distribution,
            feature_influence=feature_influence,
            model=self._model_info(),
        )

    def _get_row(self, analysis_id: int) -> Analysis:
        analysis = self.analyses.get(analysis_id)
        if analysis is None:
            raise AnalysisNotFoundError(f"Analysis {analysis_id} not found")
        return analysis

    def _model_info(self) -> AnalyticsModel | None:
        """Headline numbers from the exported model's metadata; None if the
        model hasn't been exported, so the page still loads without it."""
        try:
            meta = json.loads(Path(self.settings.popularity_model_metadata_path).read_text(encoding="utf-8"))
            return AnalyticsModel(
                name=meta["model"],
                test_r2=meta["test_metrics"]["r2"],
                test_rmse=meta["test_metrics"]["rmse"],
            )
        except (OSError, ValueError, KeyError):
            return None
