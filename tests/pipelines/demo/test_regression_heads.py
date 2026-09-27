"""The real heads need trained .joblib files; these tests cover the glue:
feature construction, loading, clipping, and the missing-model error."""

import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor

from api.pipelines.demo import regression_heads as rh


@pytest.fixture
def heads_dir(tmp_path):
    rng = np.random.default_rng(0)
    pca = PCA(n_components=256).fit(rng.normal(size=(300, 2048)))
    joblib.dump(pca, tmp_path / "pca.joblib")
    x = np.zeros((2, 128 + 256))
    values = {t: 0.5 for t in rh.TARGETS} | {"loudness": -7.0, "energy": 1.4, "speechiness": -0.2}
    for t, v in values.items():
        joblib.dump(DummyRegressor(strategy="constant", constant=v).fit(x, [v, v]), tmp_path / f"{t}.joblib")
    return tmp_path


def test_build_features_shape(heads_dir):
    pca = joblib.load(heads_dir / "pca.joblib")
    x = rh.build_features(np.zeros(128), np.zeros(2048), pca)
    assert x.shape == (1, 384)


def test_predict_features_clips_bounded_targets_only(heads_dir):
    out = rh.predict_features(np.zeros(128), np.zeros(2048), heads_dir, heads_dir / "pca.joblib")
    assert set(out) == set(rh.TARGETS)
    assert out["energy"] == 1.0
    assert out["speechiness"] == 0.0
    assert out["loudness"] == -7.0  # dB, never clipped
    assert out["valence"] == 0.5


def test_missing_models_raise_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="train_vggish_ridge"):
        rh.load_heads(tmp_path, tmp_path / "pca.joblib")
