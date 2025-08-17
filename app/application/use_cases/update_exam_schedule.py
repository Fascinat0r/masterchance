from __future__ import annotations

import re
from datetime import datetime
from typing import List, Set, Tuple

from app.config.logger import logger
from app.domain.models import ExamSession, Program
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.infrastructure.parser.parse_exam_schedule import get_master_exam_schedule

# из "38.04.05_02" берём "38.04.05"
_DEPT_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}")


class UpdateExamScheduleUseCase:
    """
    Парсит страницу расписания и заполняет exam_sessions.

    ⚙️ Матчинг:
      • Берём столбец «Код» вида '38.04.05_02'
      • Из него извлекаем кафедру '38.04.05'
      • Находим ВСЕ наши программы с department_code == '38.04.05'
      • Для каждой даты из строки создаём запись для КАЖДОЙ программы этой кафедры

    🔒 Дедупликация:
      • Не создаём дубль, если (program_code, dt) уже добавляли в текущем запуске
    """

    def __init__(self, repo: ProgramRepository):
        self._repo = repo

    @staticmethod
    def _dept_from_exam_code(exam_code: str) -> str | None:
        m = _DEPT_RE.search(exam_code or "")
        return m.group(0) if m else None

    @staticmethod
    def _parse_dt(s: str) -> datetime | None:
        try:
            # Время МСК, tz-naive (консистентно с остальной БД)
            return datetime.strptime(s.strip(), "%d.%m.%Y %H:%M")
        except Exception:
            return None

    def execute(self) -> int:
        logger.info("=== Синхронизация дат вступительных экзаменов ===")
        data = get_master_exam_schedule()
        logger.info("Спарсили %d строк расписания", len(data))

        sessions: List[ExamSession] = []
        seen: Set[Tuple[str, datetime]] = set()  # (program_code, dt)
        unmatched_depts = 0
        total_rows = 0

        for row in data:
            exam_code = (row.get("code") or "").strip()
            dept = self._dept_from_exam_code(exam_code) or exam_code.split("_")[0].strip()
            if not dept:
                logger.warning("Строка без распознаваемого кода кафедры: %s", exam_code)
                continue

            # Все наши программы этой кафедры
            progs: List[Program] = self._repo.get_programs_by_department(dept)
            if not progs:
                unmatched_depts += 1
                logger.warning("Не нашли ни одной программы для кафедры dept=%s (exam_code=%s)", dept, exam_code)
                continue

            # Каждую дату размножаем на все программы кафедры
            dates = [d for d in (row.get("dates") or []) if d and d.strip()]
            for ds in dates:
                dt = self._parse_dt(ds)
                if not dt:
                    logger.warning("Пропускаем битую дату '%s' (dept=%s, exam_code=%s)", ds, dept, exam_code)
                    continue
                for prog in progs:
                    key = (prog.code, dt)
                    if key in seen:
                        continue
                    seen.add(key)
                    sessions.append(ExamSession(
                        program_code=prog.code,
                        exam_code=exam_code,
                        dt=dt,
                        institute=row.get("institute"),
                        education_form=row.get("education_form"),
                        contract=row.get("contract"),
                        program_name=row.get("program"),
                        program_pdf_url=row.get("program_pdf_url"),
                    ))
                    total_rows += 1

        # Обновляем таблицу целиком
        self._repo.clear_exam_sessions()
        self._repo.add_exam_sessions_bulk(sessions)
        self._repo.commit()

        logger.info("✅ Записей добавлено: %d (не сопоставлено кафедр: %d)", len(sessions), unmatched_depts)
        return len(sessions)
