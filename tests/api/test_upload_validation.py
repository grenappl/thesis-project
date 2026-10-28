import pytest
from fastapi.testclient import TestClient

from api.core.config import Settings, get_settings
from api.core.dependencies import get_demo_feature_extraction_service
from api.main import app


class _NeverCalledService:
    """Stands in for the WSL2 pipeline: validation must reject the upload
    before the service is ever reached."""

    async def extract_features(self, *args, **kwargs):
        raise AssertionError("pipeline should not run for a rejected upload")

    async def stream_extract_features(self, *args, **kwargs):
        raise AssertionError("pipeline should not run for a rejected upload")
        yield  # pragma: no cover  (makes this an async generator)


@pytest.fixture
def client():
    app.dependency_overrides[get_demo_feature_extraction_service] = lambda: _NeverCalledService()
    app.dependency_overrides[get_settings] = lambda: Settings(demo_max_upload_mb=1)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("path", ["/demo/extract-features", "/demo/extract-features/stream"])
def test_unsupported_file_type_is_415(client, path):
    response = client.post(path, files={"audio": ("notes.txt", b"not audio", "text/plain")})
    assert response.status_code == 415
    assert ".mp3" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/demo/extract-features", "/demo/extract-features/stream"])
def test_oversized_upload_is_413(client, path):
    two_mb = b"\0" * (2 * 1024 * 1024)
    response = client.post(path, files={"audio": ("song.wav", two_mb, "audio/wav")})
    assert response.status_code == 413
    assert "1 MB" in response.json()["detail"]


def test_extension_check_ignores_case(client):
    # An accepted type gets past validation and reaches the (stubbed) service.
    with pytest.raises(AssertionError, match="pipeline should not run"):
        client.post("/demo/extract-features", files={"audio": ("SONG.MP3", b"x", "audio/mpeg")})
