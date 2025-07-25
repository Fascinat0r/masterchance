from sqlalchemy.exc import SQLAlchemyError

from app.config.logger import logger
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.infrastructure.parser.master_applications_parser import MasterApplicationsParser


class UpdateApplicationListsUseCase:
    """
    Полная синхронизация заявок из приёмной комиссии с БД.
    Для каждого направления:
        1. скачиваем свежие данные
        2. удаляем «старые» заявки этого направления
        3. bulk‑добавляем абитуриентов и заявки
        4. обновляем submission_stats
    Все операции в одной транзакции: если что‑то упало ⇒ ничего не меняем.
    """

    def __init__(self, repo: ProgramRepository, parser: MasterApplicationsParser):
        self._repo = repo
        self._parser = parser

    def execute(self) -> None:
        logger.info("=== Синхронизация списков заявок начинается ===")
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

                # 1) удаляем старые заявки
                self._repo.delete_applications_by_program(prog.code)

                # 2) добавляем (bulk)
                applicant_ids = [a.applicant_id for a in applications]
                self._repo.add_applicants_bulk(applicant_ids)
                self._repo.add_applications_bulk(applications)

                # 3) статистика
                self._repo.add_submission_stats(stats)

            # commit ТОЛЬКО после успешного прохода всех направлений
            self._repo.commit()
            logger.info("✅ Синхронизация списков завершена без ошибок")

        except SQLAlchemyError as db_err:
            logger.exception("Ошибка транзакции, выполняем rollback: %s", db_err)
            self._repo._session.rollback()
            raise
