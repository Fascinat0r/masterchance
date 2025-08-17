#!/usr/bin/env python3
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.application.use_cases.update_lists import UpdateApplicationListsUseCase
from app.config.config import settings
from app.config.logger import logger
from app.infrastructure.db.models import Base
from app.infrastructure.db.repositories.program_repository import ProgramRepository

def main():
    logger.info("=== masterchance старт ===")
    # 1) Настройка БД
    engine = create_engine(
        settings.database_url,
        echo=settings.db_echo,
        future=True,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)

    # 2) Инициализация
    session = Session()
    repo = ProgramRepository(session)
    updater = UpdateApplicationListsUseCase(repo=repo)  # parser не нужен для параллельного режима

    # 3) Запуск
    try:
        parallelism = settings.parser_parallelism
        updater.execute_parallel(parallelism=parallelism, headless=True)
        logger.info("Данные по подаче заявлений успешно обновлены.")
        print("✅ Данные по подаче заявлений успешно обновлены.")
    except Exception as e:
        logger.exception("Ошибка при обновлении данных")
        print("❌ Ошибка при обновлении:", e, file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()
        logger.info("=== masterchance завершён ===")


if __name__ == "__main__":
    main()
