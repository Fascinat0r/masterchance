# app/infrastructure/parser/parallel_master_parser.py
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Tuple

from app.config.logger import logger
from app.domain.models import SubmissionStats, Application
from app.infrastructure.parser.master_applications_parser import MasterApplicationsParser


# --- utils ------------------------------------------------------------------

def _to_dict(obj):
    """Безопасная сериализация для передачи между процессами."""
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    if is_dataclass(obj):
        return asdict(obj)
    # простой объект
    return obj.__dict__


def _from_stats(d: dict) -> SubmissionStats:
    return SubmissionStats(**d)


def _from_app(d: dict) -> Application:
    return Application(**d)


def _chunkify(seq: List[str], n_chunks: int) -> List[List[str]]:
    n = max(1, n_chunks)
    size = max(1, math.ceil(len(seq) / n))
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# --- worker -----------------------------------------------------------------

def _worker_parse_chunk(codes: List[str], headless: bool = True) -> Dict[str, Tuple[dict, List[dict]]]:
    """
    Процесс-воркер: один ChromeDriver на чанк.
    Возвращает сериализованные dict'и, чтобы не споткнуться о pickle.
    """
    # необязательно, но помогает диагностировать
    os.environ.setdefault("WDM_LOG", "0")

    parser = MasterApplicationsParser(headless=headless)
    out: Dict[str, Tuple[dict, List[dict]]] = {}

    try:
        for code in codes:
            try:
                logger.info("[PID %s] → парсим %s", os.getpid(), code)
                stats, apps = parser.parse(code)
                out[code] = (_to_dict(stats), [_to_dict(a) for a in apps])
            except Exception as e:
                logger.exception("[PID %s] Ошибка парсинга %s: %s", os.getpid(), code, e)
                # возвращаем пустые, чтобы основной процесс мог решить судьбу записи
                out[code] = (None, [])
        return out
    finally:
        try:
            parser.close()
        except Exception:
            pass


# --- public API --------------------------------------------------------------

def parse_programs_in_parallel(
    program_codes: Iterable[str],
    parallelism: int = 4,
    headless: bool = True,
) -> Dict[str, Tuple[SubmissionStats, List[Application]]]:
    """
    Запускает N независимых ChromeDriver в отдельных процессах.
    Каждый процесс переиспользует драйвер для своего чанка направлений.
    Возвращает {code: (stats, applications)}.

    Рекомендации соблюдены:
      • один WebDriver — один процесс;
      • никакого шеринга драйвера между потоками/процессами;
      • корректное завершение драйвера в finally;
      • if __name__ == '__main__' должен быть в вызывающем скрипте (Windows).
    """
    codes = list(program_codes)
    if not codes:
        return {}

    n = max(1, int(parallelism))
    if n == 1 or len(codes) == 1:
        # однопоточный бэкап — без накладных расходов на процессы
        parser = MasterApplicationsParser(headless=headless)
        try:
            result: Dict[str, Tuple[SubmissionStats, List[Application]]] = {}
            for code in codes:
                stats, apps = parser.parse(code)
                result[code] = (stats, apps)
            return result
        finally:
            parser.close()

    chunks = _chunkify(codes, n)
    logger.info("Стартуем параллельный парсинг: %d процессов, %d направлений", len(chunks), len(codes))

    aggregated: Dict[str, Tuple[SubmissionStats, List[Application]]] = {}
    started = time.perf_counter()

    # ВАЖНО: на Windows обязательна защита if __name__ == '__main__' в вызывающем коде!
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_worker_parse_chunk, chunk, headless) for chunk in chunks]
        for fut in as_completed(futures):
            data = fut.result()  # dict[code] -> (stats_dict|None, apps_dicts)
            for code, (s_dict, a_dicts) in data.items():
                if s_dict is None:
                    # передаём как исключение вверх? здесь помечаем как пропущенное
                    logger.warning("Направление %s пропущено из-за ошибки парсинга в воркере.", code)
                    continue
                stats = _from_stats(s_dict)
                apps = [_from_app(d) for d in a_dicts]
                aggregated[code] = (stats, apps)

    elapsed = time.perf_counter() - started
    logger.info("Параллельный парсинг завершён за %.2f сек. Успешно: %d/%d", elapsed, len(aggregated), len(codes))
    return aggregated
