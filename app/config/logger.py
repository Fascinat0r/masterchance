# app/config/logger.py
import logging
import sys

from app.config.config import settings

LOG_LEVEL = logging.DEBUG if settings.env == "dev" else logging.INFO

# создаём корневой логгер
logger = logging.getLogger("masterchance")
logger.setLevel(LOG_LEVEL)

# хендлер для вывода в stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(LOG_LEVEL)

# форматтер: время, уровень, [имя логгера], сообщение
fmt = logging.Formatter(
    "%(asctime)s %(levelname)-5s [masterchance] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(fmt)
logger.addHandler(handler)
