# app/application/use_cases/get_last_update_time.py
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.config.config import settings
from app.infrastructure.db.repositories.program_repository import ProgramRepository

# Источник времени на сайте — СПб, поэтому считаем исходную зону Московской
_SOURCE_TZ = ZoneInfo("Europe/Moscow")


class GetLastUpdateTimeUseCase:
    """
    Достаёт максимальную generated_at из submission_stats и
    возвращает tz-aware datetime в таймзоне, заданной в settings.timezone.
    """

    def __init__(self, repo: ProgramRepository):
        self._repo = repo

    def execute(self) -> datetime | None:
        dt = self._repo.get_latest_submission_generated_at()
        if dt is None:
            return None

        # В БД дата может быть наивной (без tz). Локализуем как московскую.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_SOURCE_TZ)

        # Переводим в целевую зону (по умолчанию — settings.timezone_name)
        return dt.astimezone(settings.timezone)
