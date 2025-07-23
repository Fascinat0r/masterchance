from app.config.logger import logger
from app.infrastructure.db.repositories.program_repository import ProgramRepository
from app.infrastructure.parser.master_applications_parser import MasterApplicationsParser


class UpdateApplicationListsUseCase:
    """
    Для каждого направления:
      1) берёт Program.code из БД
      2) вызывает парсер, который возвращает SubmissionStats и список Application
      3) сохраняет всё через репозиторий
    """

    def __init__(
            self,
            repo: ProgramRepository,
            parser: MasterApplicationsParser
    ):
        self._repo = repo
        self._parser = parser

    def execute(self) -> None:
        # 1) получаем все направления
        programs = []
        for inst in self._repo.get_all_institutes():
            for dept in self._repo.get_departments_by_institute(inst.code):
                programs.extend(self._repo.get_programs_by_department(dept.code))

        # 2) для каждого направления парсим и сохраняем
        for prog in programs:
            try:
                stats, applications = self._parser.parse(prog.code)
            except Exception as e:
                logger.warning("Пропускаем направление %s из‑за ошибки: %s", prog.code, e)
                continue

            # сохраняем статистику
            self._repo.add_submission_stats(stats)

            # сохраняем каждую заявку
            for app in applications:
                self._repo.add_application(app)

        # 3) финальный коммит
        self._repo.commit()
        logger.info("Все данные сохранены в БД")
