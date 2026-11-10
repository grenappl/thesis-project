from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from api.models.analysis import Analysis


def _like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class AnalysisRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, id_: int) -> Analysis | None:
        return self.session.get(Analysis, id_)

    def add(self, analysis: Analysis) -> Analysis:
        self.session.add(analysis)
        self.session.flush()
        return analysis

    def delete(self, analysis: Analysis) -> None:
        self.session.delete(analysis)
        self.session.flush()

    def search(
        self,
        text: str | None,
        since: datetime | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Analysis], int]:
        """Newest first. `text` matches title or artist, case-insensitively."""
        conditions = []
        if text:
            pattern = _like_pattern(text)
            conditions.append(or_(
                Analysis.song_title.ilike(pattern, escape="\\"),
                Analysis.artist.ilike(pattern, escape="\\"),
            ))
        if since is not None:
            conditions.append(Analysis.created_at >= since)

        total = self.session.scalar(select(func.count()).select_from(Analysis).where(*conditions)) or 0
        rows = self.session.scalars(
            select(Analysis).where(*conditions)
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(limit).offset(offset)
        )
        return list(rows), total

    def all(self) -> list[Analysis]:
        return list(self.session.scalars(select(Analysis).order_by(Analysis.created_at)))
