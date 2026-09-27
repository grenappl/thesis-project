"""Popularity prediction for one song -- the demo app's POST /predict.

Serves the XGBoost Alignment-Augmented model, the one Section 4.2.2 names as
"the primary model ... for the deployed demo". The model file is produced by
scripts/export_popularity_model.py, which retrains it from notebook 6's frozen
configuration and refuses to write it unless it reproduces notebook 7's saved
test predictions -- so what this service loads is the model the thesis reports
on, not an approximation of it.

Unlike Pipeline 2, this runs natively on Windows: XGBoost has a Windows wheel
and inference is a single tree-ensemble evaluation, no WSL2 round-trip.

The feature vector is rebuilt here to match training exactly (key one-hot over
all twelve pitch classes, alignment gap recomputed with the demo pipeline's own
formula). Column order is never assumed: it is read from the exported
metadata and checked against the names stored inside the model file at load
time, so any drift between training and serving fails loudly on the first
request instead of silently scrambling inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import xgboost as xgb

from api.core.config import Settings
from api.pipelines.demo.alignment import compute_alignment_gap, normalize_valence
from api.schemas.prediction import (
    ContributionGroups,
    FeatureContribution,
    ModelInfo,
    PredictionRead,
    PredictionRequest,
)

# The eleven non-key audio descriptors, named exactly as in training
# (notebooks/modeling_config.py AUDIO_12 minus `key`).
_AUDIO_CONTINUOUS = (
    "tempo", "loudness", "mode", "energy", "danceability", "speechiness",
    "acousticness", "instrumentalness", "liveness", "duration_ms", "valence",
)
_KEY_COLUMNS = tuple(f"key_{i}" for i in range(12))
_EXPORT_HINT = "Run `uv run python scripts/export_popularity_model.py` from the project root."


class PredictionError(Exception):
    """The request can't be scored; the message is safe to show the caller."""


class ModelUnavailableError(Exception):
    """The model file is missing or doesn't match its metadata."""


@dataclass(frozen=True)
class _LoadedModel:
    booster: xgb.Booster
    feature_names: tuple[str, ...]
    metadata: dict


@lru_cache(maxsize=4)
def _load_model(model_path: str, metadata_path: str) -> _LoadedModel:
    model_file, meta_file = Path(model_path), Path(metadata_path)
    if not model_file.is_file() or not meta_file.is_file():
        raise ModelUnavailableError(f"Popularity model not found at {model_file}. {_EXPORT_HINT}")

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(model_file)

    expected = tuple(metadata["feature_names"])
    if tuple(booster.feature_names or ()) != expected:
        raise ModelUnavailableError(
            "Model file and metadata disagree on feature order; they were not "
            f"exported together. {_EXPORT_HINT}"
        )
    unknown = set(expected) - set(_AUDIO_CONTINUOUS) - set(_KEY_COLUMNS) - {"lyric_sentiment", "alignment_gap"}
    if unknown:
        raise ModelUnavailableError(f"Model expects features this service can't build: {sorted(unknown)}")
    return _LoadedModel(booster=booster, feature_names=expected, metadata=metadata)


def build_feature_vector(request: PredictionRequest, alignment_gap: float) -> dict[str, float]:
    """Every model input by name, constructed exactly as in training."""
    if request.lyric_sentiment is None:
        raise PredictionError("lyric_sentiment is required to build the feature vector.")
    values = {name: float(getattr(request, name)) for name in _AUDIO_CONTINUOUS}
    values.update({col: 0.0 for col in _KEY_COLUMNS})
    values[f"key_{request.key}"] = 1.0
    values["lyric_sentiment"] = float(request.lyric_sentiment)
    values["alignment_gap"] = float(alignment_gap)
    return values


class PopularityPredictionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def predict(self, request: PredictionRequest) -> PredictionRead:
        # SRS REQ-4.5-3: missing lyrics must not crash the pipeline, and the
        # user must be told why no prediction was made. Checked before the model
        # loads, so this answer doesn't depend on the model being present.
        if request.lyric_sentiment is None:
            raise PredictionError(
                "A popularity prediction needs lyrics. The model was trained on songs "
                "that all had lyrics, and the alignment gap is undefined without a "
                "lyric sentiment score. Submit the song's lyrics to get a prediction."
            )

        loaded = _load_model(self.settings.popularity_model_path,
                             self.settings.popularity_model_metadata_path)

        gap = compute_alignment_gap(request.lyric_sentiment, request.valence)
        values = build_feature_vector(request, gap)
        row = np.array([[values[name] for name in loaded.feature_names]], dtype=float)
        matrix = xgb.DMatrix(row, feature_names=list(loaded.feature_names))

        # pred_contribs gives exact TreeSHAP values (the same algorithm as
        # shap.TreeExplainer in notebook 8), one per feature plus a final bias
        # column; they sum to the raw prediction.
        contribs = loaded.booster.predict(matrix, pred_contribs=True)[0]
        per_feature = dict(zip(loaded.feature_names, (float(c) for c in contribs[:-1])))
        base_value = float(contribs[-1])
        raw_prediction = base_value + sum(per_feature.values())

        key_total = sum(per_feature[col] for col in _KEY_COLUMNS)
        features = [
            FeatureContribution(feature=name, value=values[name], contribution=per_feature[name])
            for name in loaded.feature_names if name not in _KEY_COLUMNS
        ]
        features.append(FeatureContribution(feature="key", value=request.key, contribution=key_total))
        features.sort(key=lambda f: abs(f.contribution), reverse=True)

        meta = loaded.metadata
        return PredictionRead(
            predicted_popularity=float(np.clip(raw_prediction, 0.0, 100.0)),
            base_value=base_value,
            lyric_sentiment=float(request.lyric_sentiment),
            valence_normalized=normalize_valence(request.valence),
            alignment_gap=gap,
            contributions=ContributionGroups(
                audio=sum(v for k, v in per_feature.items()
                          if k not in ("lyric_sentiment", "alignment_gap")),
                lyric_sentiment=per_feature["lyric_sentiment"],
                alignment_gap=per_feature["alignment_gap"],
            ),
            features=features,
            model=ModelInfo(
                name=meta["model"],
                test_r2=meta["test_metrics"]["r2"],
                test_rmse=meta["test_metrics"]["rmse"],
                trained_rows=meta["trained_rows"],
                exported_at=meta["exported_at"],
                note=meta["purpose"],
            ),
        )
