from __future__ import annotations

import re
from datetime import datetime
from typing import List, Set, Tuple, Optional
from difflib import SequenceMatcher

from app.config.logger import logger
from app.domain.models import ExamSession, Program
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.infrastructure.parser.parse_exam_schedule import get_master_exam_schedule

# из "38.04.05_02" берём "38.04.05"
_DEPT_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}")

# служебные приписки, которые часто мусорят название ОП на сайте
_INTERNATIONAL_TAGS = (
    "международная образовательная программа",
    "international educational program",
)


class UpdateExamScheduleUseCase:
    """
    Парсит страницу расписания и заполняет exam_sessions.

    Главное изменение:
    — Больше НЕ размножаем даты на все программы направления.
      Ищем и привязываем даты только к одной, конкретной ОП из строки расписания.

    ⚙️ Матчинг:
      1) Берём «Код» вида '38.04.05_02' и выделяем направление '38.04.05'.
      2) Получаем все наши программы этого направления.
      3) Пытаемся однозначно сопоставить строку с конкретной ОП:
         • точное совпадение по нормализованному имени;
         • иначе — лучший кандидат по близости имени (SequenceMatcher) с порогом.
      4) Если однозначного матча нет — пропускаем строку и логируем предупреждение.

    🔒 Дедупликация (в рамках запуска):
      • Не создаём дубль, если (program_code, dt) уже добавляли.

    ⏱️ Даты:
      • Парсим как tz-naive МСК в формате '%d.%m.%Y %H:%M' (единый стиль с БД).
    """

    def __init__(self, repo: ProgramRepository):
        self._repo = repo

    @staticmethod
    def _dept_from_exam_code(exam_code: str) -> Optional[str]:
        m = _DEPT_RE.search(exam_code or "")
        return m.group(0) if m else None

    @staticmethod
    def _parse_dt(s: str) -> Optional[datetime]:
        try:
            # Время МСК, tz-naive (консистентно с остальной БД)
            return datetime.strptime(s.strip(), "%d.%m.%Y %H:%M")
        except Exception:
            return None

    @staticmethod
    def _norm_name(s: Optional[str]) -> str:
        """
        Нормализация названия ОП:
          - нижний регистр, 'ё' → 'е'
          - схлопываем пробелы
          - отбрасываем всё после ' / ' (часто дублируется англ. версия)
          - вырезаем служебные пометки про МОП
        """
        s = (s or "").strip().lower().replace("ё", "е")
        # отрезаем англ. хвост после слеша
        if " / " in s:
            s = s.split(" / ", 1)[0]
        # схлопываем пробелы
        s = re.sub(r"\s+", " ", s)

        # убираем тэги в скобках (и без) про МОП
        for tag in _INTERNATIONAL_TAGS:
            s = re.sub(rf"\s*\({tag}\)\s*", " ", s)
            s = s.replace(tag, " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _pick_program(self, progs: List[Program], row_program_name: str) -> Optional[Program]:
        """
        Возвращает единственную программу, однозначно соответствующую названию из строки расписания.
        Сначала ищем точное совпадение нормализованных названий.
        Если нет — выбираем лучший fuzzy-кандидат по порогу и отрыву от второго места.
        """
        norm_target = self._norm_name(row_program_name)

        # 1) точные совпадения по нормализованному имени
        exact = [p for p in progs if self._norm_name(getattr(p, "name", "")) == norm_target]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            # это странно, но лучше явно сообщить и не гадать
            names = ", ".join(p.name for p in exact)
            logger.warning("Неоднозначное точное сопоставление по имени: %r → {%s}", row_program_name, names)
            return None

        # 2) fuzzy-матчинг (безопасный): высокий порог + заметный отрыв
        if not progs:
            return None
        scored = []
        for p in progs:
            pn = self._norm_name(getattr(p, "name", ""))
            scored.append((self._similar(norm_target, pn), p))
        scored.sort(key=lambda t: t[0], reverse=True)

        best_score, best_prog = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        # Порог можно подстроить; сейчас консервативно
        if best_score >= 0.90 and (best_score - second_score) >= 0.05:
            return best_prog

        # иначе считаем сопоставление рискованным
        logger.warning(
            "Не удалось однозначно сопоставить программу по названию: %r (лучший=%.3f, второй=%.3f, кандидат=%r)",
            row_program_name, best_score, second_score, getattr(best_prog, "name", ""),
        )
        return None

    def execute(self) -> int:
        logger.info("=== Синхронизация дат вступительных экзаменов ===")
        data = get_master_exam_schedule()
        logger.info("Спарсили %d строк расписания", len(data))

        sessions: List[ExamSession] = []
        seen: Set[Tuple[str, datetime]] = set()  # (program_code, dt)

        unmatched_depts = 0
        unmatched_programs = 0
        invalid_dates = 0
        matched_rows = 0

        for row in data:
            exam_code_full = (row.get("code") or "").strip()  # например: '38.04.05_02'
            dept = self._dept_from_exam_code(exam_code_full) or exam_code_full.split("_")[0].strip()
            if not dept:
                logger.warning("Строка без распознаваемого кода направления: %r", exam_code_full)
                unmatched_depts += 1
                continue

            # все наши программы этого направления
            progs_in_dept: List[Program] = self._repo.get_programs_by_department(dept)
            if not progs_in_dept:
                unmatched_depts += 1
                logger.warning("Не нашли ни одной нашей программы для направления dept=%s (exam_code=%s)", dept, exam_code_full)
                continue

            # подбираем одну конкретную программу по названию (5-й столбец таблицы)
            target_prog = self._pick_program(progs_in_dept, row.get("program") or "")
            if not target_prog:
                unmatched_programs += 1
                logger.warning(
                    "Пропускаем строку: не смогли сопоставить программу (dept=%s, exam_code=%s, program=%r)",
                    dept, exam_code_full, row.get("program"),
                )
                continue

            # добавляем все валидные даты только к ЭТОЙ программе
            raw_dates = [d for d in (row.get("dates") or []) if d and d.strip()]
            for ds in raw_dates:
                dt = self._parse_dt(ds)
                if not dt:
                    invalid_dates += 1
                    logger.warning(
                        "Пропускаем битую дату '%s' (dept=%s, exam_code=%s, program_code=%s)",
                        ds, dept, exam_code_full, target_prog.code
                    )
                    continue

                key = (target_prog.code, dt)
                if key in seen:
                    continue
                seen.add(key)

                sessions.append(ExamSession(
                    program_code=target_prog.code,
                    exam_code=exam_code_full,
                    dt=dt,
                    institute=row.get("institute"),
                    education_form=row.get("education_form"),
                    contract=row.get("contract"),
                    program_name=row.get("program"),
                    program_pdf_url=row.get("program_pdf_url"),
                ))
                matched_rows += 1

        # Обновляем таблицу целиком
        self._repo.clear_exam_sessions()
        if sessions:
            self._repo.add_exam_sessions_bulk(sessions)
        self._repo.commit()

        logger.info(
            "✅ Записей добавлено: %d | ⚠️ не нашли направление: %d | ⚠️ не сопоставили программу: %d | ⚠️ битых дат: %d",
            len(sessions), unmatched_depts, unmatched_programs, invalid_dates
        )
        return len(sessions)
