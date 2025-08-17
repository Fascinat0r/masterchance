#!/usr/bin/env python3
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.application.use_cases.update_exam_schedule import UpdateExamScheduleUseCase
from app.config.config import settings
from app.config.logger import logger
from app.infrastructure.db.models import Base
from app.infrastructure.db.repositories.program_repository import ProgramRepository


def main():
    logger.info("=== update_exam_schedule старт ===")
    engine = create_engine(settings.database_url, echo=settings.db_echo, future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, future=True)()
    try:
        repo = ProgramRepository(session)
        uc = UpdateExamScheduleUseCase(repo)
        n = uc.execute()
        print(f"✅ Экзаменационных сессий сохранено: {n}")
    except Exception as e:
        logger.exception("Ошибка при обновлении расписания")
        print("❌ Ошибка при обновлении расписания:", e, file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()
        logger.info("=== update_exam_schedule завершён ===")


if __name__ == "__main__":
    main()
