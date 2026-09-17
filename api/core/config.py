from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Thesis API"
    debug: bool = False

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/thesis"

    # Pinned to 3005, not the Next.js default 3000 — 3000/3001/3002 kept
    # silently conflicting with other projects' dev servers on this machine
    # (Next.js falls back to the next free port with no warning, which
    # desyncs from whatever's hardcoded here and manifests as a browser
    # NetworkError, not a CORS error message). package.json's dev/start
    # scripts pass `next dev -p 3005` / `next start -p 3005` to force it —
    # with an explicit -p, Next.js errors instead of silently picking a
    # different port, so a mismatch here would fail loudly, not silently.
    cors_origins: list[str] = ["http://localhost:3005"]

    # Demo feature-extraction pipeline (api/pipelines/demo/) — only runs under
    # WSL2 (Essentia has no Windows wheel), so the API shells out to it. See
    # PIPELINE_SETUP.md for why the whole pipeline lives in WSL2.
    wsl_distro: str = "Ubuntu-24.04"
    wsl_project_path: str = "/mnt/d/Projects/thesis-project"
    # 180s, not 120s: valence/acousticness/instrumentalness now run as their
    # own subprocess (a second TensorFlow runtime load, plus the one-time
    # VGGish module download) on top of everything else — see vggish_tfhub.py.
    demo_pipeline_timeout_seconds: int = 180

    demo_panns_checkpoint: str = "models/panns/Cnn14_mAP=0.431.pth"
    demo_vggish_ridge_models_dir: str = "models/vggish_ridge"


@lru_cache
def get_settings() -> Settings:
    return Settings()
