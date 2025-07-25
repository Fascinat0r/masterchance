# app/config/config.py
from pathlib import Path
from typing import Any
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
    # директория для данных
    data_dir: Path = Field(_DEFAULT_DATA_DIR, alias="DATA_DIR")

    # таймзона
    timezone_name: str = Field("UTC", alias="TIMEZONE")

    bot_token: str = Field("BOT_TOKEN", alias="BOT_TOKEN")

    chrome_bin: str = Field("CHROME_BIN", alias="CHROME_BIN")

    # БД
    db_url: str | None = Field(None, alias="DATABASE_URL")
    db_filename: str = Field("master.db", alias="DB_FILENAME")
    db_echo: bool = Field(False, alias="DB_ECHO")

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, Any]) -> dict[str, Any]:
        raw = values.get("data_dir", _DEFAULT_DATA_DIR)
        values["data_dir"] = Path(raw).expanduser().resolve()
        return values

    @property
    def timezone(self) -> ZoneInfo:
        # возвращаем объект зоны из строки
        return ZoneInfo(self.timezone_name)

    @property
    def database_url(self) -> str:
        return self.db_url or f"sqlite:///{self.data_dir / self.db_filename}"


settings = Settings()
