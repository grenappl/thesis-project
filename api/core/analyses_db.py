"""SQLite storage for saved analyses (History, Song and Analytics pages).

Separate from api/core/database.py on purpose: that is the async Postgres
engine behind the `tracks` stack, which needs a database server. Saved
analyses are a few KB each and the demo is single-user, so a SQLite file
keeps the app runnable with nothing installed. The engine is synchronous --
the endpoints that use it are plain `def`, which FastAPI runs in its
threadpool.
"""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from api.core.config import get_settings
from api.models.analysis import Analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_db_path(configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_engine() -> Engine:
    db_path = resolve_db_path(get_settings().analyses_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    Analysis.__table__.create(engine, checkfirst=True)
    return engine


def get_analyses_session() -> Iterator[Session]:
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
