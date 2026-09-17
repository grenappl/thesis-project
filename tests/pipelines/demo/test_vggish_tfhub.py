"""compute_vggish_embedding needs a real TF-Hub download + TensorFlow, so it
isn't exercised here (see feature_extraction.md for the manual validation
that produced the 0.97 correlation against the Kaggle dataset). What's
tested is the glue around it: predict_vggish_features combines an embedding
with the fitted Ridge models correctly, and load_ridge_models fails clearly
when a model is missing.
"""

import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge

import api.pipelines.demo.vggish_tfhub as vggish_tfhub
from api.pipelines.demo.vggish_tfhub import load_ridge_models, predict_vggish_features


def _fit_constant_ridge(value: float) -> Ridge:
    """A Ridge model that (approximately) always predicts `value`, regardless
    of input — enough to check predict_vggish_features wires things up right
    without needing a real trained model.
    """
    X = np.zeros((2, 128))
    y = np.array([value, value])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


@pytest.fixture
def models_dir(tmp_path):
    for i, target in enumerate(vggish_tfhub._TARGETS):
        joblib.dump(_fit_constant_ridge(0.1 * (i + 1)), tmp_path / f"{target}.joblib")
    return tmp_path


def test_load_ridge_models_reads_all_targets(models_dir):
    models = load_ridge_models(models_dir)
    assert set(models) == set(vggish_tfhub._TARGETS)


def test_load_ridge_models_raises_clear_error_if_missing(tmp_path):
    with pytest.raises(RuntimeError, match="train_vggish_ridge"):
        load_ridge_models(tmp_path)


def test_predict_vggish_features_uses_computed_embedding(monkeypatch, models_dir):
    monkeypatch.setattr(vggish_tfhub, "compute_vggish_embedding", lambda path: np.zeros(128))

    result = predict_vggish_features("irrelevant.wav", models_dir)

    for i, target in enumerate(vggish_tfhub._TARGETS):
        assert result[target] == pytest.approx(0.1 * (i + 1), abs=1e-6)
