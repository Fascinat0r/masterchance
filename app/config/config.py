# app/config/config.py
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        populate_by_name=True,
        extra="ignore",
    )

    # окружение
    env: str = Field("dev", alias="ENV")
    data_dir: Path = Field(_DEFAULT_DATA_DIR, alias="DATA_DIR")

    # таймзона
    timezone_name: str = Field("UTC", alias="TIMEZONE")

    bot_token: str = Field("BOT_TOKEN", alias="BOT_TOKEN")

    chrome_bin: str = Field("CHROME_BIN", alias="CHROME_BIN")

    parser_parallelism: int = Field(8, alias="PARSER_PARALLELISM")

    # БД
    db_url: str | None = Field(None, alias="DATABASE_URL")
    db_filename: str = Field("master.db", alias="DB_FILENAME")
    db_echo: bool = Field(False, alias="DB_ECHO")

    # ───────────────── Monte-Carlo: «отток» сильных ───────────────────
    # Включение механики оттока (True — включено, False — legacy-поведение).
    opt_out_enabled: bool = Field(False, alias="MC_OPTOUT_ENABLED")
    # Доля выбывающих среди тех, у кого НИГДЕ нет согласия (consent=False по всем его заявлениям).
    # Значение 0.2 означает «ровно 20% из пула E будем исключать в каждой симуляции».
    opt_out_ratio: float = Field(0.20, alias="MC_OPTOUT_RATIO")
    # Крутизна зависимости шанса «уйти» от перцентиля способности (vi).
    # Вероятности пропорциональны p^alpha, где p — перцентиль, alpha>=1 (3 — агрессивнее).
    opt_out_alpha: float = Field(1.0, alias="MC_OPTOUT_STRENGTH")
    # Режим выбора набора «ушедших»:
    #  - "per_simulation": в КАЖДОЙ симуляции выбираем заново (по текущим, уже импутированным vi);
    #  - "fixed": один раз при инициализации (по детерминированной «базовой способности»).
    opt_out_mode: Literal["per_simulation", "fixed"] = Field("per_simulation", alias="MC_OPTOUT_MODE")

    bot_show_anchored: bool = Field(True, alias="BOT_SHOW_ANCHORED")

    # ───────────────── Экзамены: freeze после дедлайна ────────────────
    # Включение механики заморозки нулей по истёкшим экзаменам
    exam_freeze_enabled: bool = Field(True, alias="MC_EXAM_FREEZE_ENABLED")
    # Грация (в часах) после последней даты экзамена
    exam_grace_hours: int = Field(24, alias="MC_EXAM_GRACE_HOURS")

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, Any]) -> dict[str, Any]:
        raw = values.get("data_dir", _DEFAULT_DATA_DIR)
        values["data_dir"] = Path(raw).expanduser().resolve()
        return values

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def database_url(self) -> str:
        return self.db_url or f"sqlite:///{self.data_dir / self.db_filename}"


settings = Settings()
