from __future__ import annotations

from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from app.config.logger import logger
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.infrastructure.parser.master_applications_parser import MasterApplicationsParser
from app.infrastructure.parser.parallel_master_parser import parse_programs_in_parallel


class UpdateApplicationListsUseCase:
    """
    Полная синхронизация заявок из приёмной комиссии с БД.
    Для каждого направления:
        1. скачиваем свежие данные
        2. удаляем «старые» заявки этого направления
        3. bulk-добавляем абитуриентов и заявки
        4. обновляем submission_stats

    Коммит происходит один раз в конце прохода.
    Если нужно «жёстко» всё-или-ничего — не ловите исключения per-program.
    """

    def __init__(self, repo: ProgramRepository, parser: Optional[MasterApplicationsParser] = None):
        self._repo = repo
        self._parser = parser  # опционально (для последовательного режима)

    # сохранён оригинальный последовательный режим
    def execute(self) -> None:
        if self._parser is None:
            raise RuntimeError("Для последовательного режима требуется parser")
        logger.info("=== Синхронизация списков заявок начинается (последовательно) ===")
        try:
            programs = [
                p
                for inst in self._repo.get_all_institutes()
                for dept in self._repo.get_departments_by_institute(inst.code)
                for p in self._repo.get_programs_by_department(dept.code)
            ]

            for prog in programs:
                logger.info("→ Обработка направления %s …", prog.code)
                try:
                    stats, applications = self._parser.parse(prog.code)
                except Exception as e:
                    logger.warning("✕ Пропускаем %s: %s", prog.code, e)
                    continue

                self._repo.delete_applications_by_program(prog.code)
                applicant_ids = [a.applicant_id for a in applications]
                self._repo.add_applicants_bulk(applicant_ids)
                self._repo.add_applications_bulk(applications)
                self._repo.add_submission_stats(stats)

            self._repo.commit()
            logger.info("✅ Синхронизация списков завершена без ошибок")
        except SQLAlchemyError as db_err:
            logger.exception("Ошибка транзакции, выполняем rollback: %s", db_err)
            self._repo._session.rollback()
            raise

    # новый параллельный режим
    def execute_parallel(self, parallelism: int = 8, headless: bool = True) -> None:
        logger.info("=== Синхронизация списков заявок начинается (параллельно, N=%d) ===", parallelism)
        try:
            programs = [
                p
                for inst in self._repo.get_all_institutes()
                for dept in self._repo.get_departments_by_institute(inst.code)
                for p in self._repo.get_programs_by_department(dept.code)
            ]
            codes = [p.code for p in programs]
            if not codes:
                logger.info("Нет направлений для обработки — выходим")
                return

            # парсим во множестве процессов, один ChromeDriver на процесс
            results = parse_programs_in_parallel(
                program_codes=codes,
                parallelism=parallelism,
                headless=headless,
            )

            # сохраняем в исходном порядке (на случай зависимостей логики)
            ok, skipped = 0, 0
            for prog in programs:
                code = prog.code
                if code not in results:
                    logger.warning("✕ Пропускаем %s: результата нет (ошибка парсинга в воркере?)", code)
                    skipped += 1
                    continue

                stats, applications = results[code]

                # 1) удаляем старые заявки
                self._repo.delete_applications_by_program(code)

                # 2) добавляем (bulk)
                applicant_ids = [a.applicant_id for a in applications]
                self._repo.add_applicants_bulk(applicant_ids)
                self._repo.add_applications_bulk(applications)

                # 3) статистика
                self._repo.add_submission_stats(stats)
                ok += 1

            # один общийCommit
            self._repo.commit()
            logger.info("✅ Параллельная синхронизация завершена. Успешно: %d, пропущено: %d", ok, skipped)

        except SQLAlchemyError as db_err:
            logger.exception("Ошибка транзакции, выполняем rollback: %s", db_err)
            self._repo._session.rollback()
            raise
