import time
from datetime import datetime
from typing import List, Optional, Dict
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    JavascriptException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from app.config.logger import logger
from app.domain.models import SubmissionStats, Application


def _safe_int(text: str, default: int = 0) -> int:
    try:
        return int((text or "").strip().replace("\u00A0", ""))
    except Exception:
        return default


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


class MasterApplicationsParser:
    """
    Через Selenium переходит на страницу списка абитуриентов,
    выставляет фильтры (select2), дожидается AJAX, парсит статистику и таблицу.
    Сделано устойчивым к:
      - select2 (обязательный trigger('change'))
      - отложенной отрисовке tbody
      - возможному отсутствию данных ('.no-data')
      - локальным фильтрам (tr.d-none)
      - изменению порядка столбцов (картирование по заголовкам)
    """

    BASE_URL = "https://my.spbstu.ru/home/abit/list-applicants/master"

    # Значения для "Форма обучения" и "Условия поступления"
    FORM_VALUE = "2"  # Очная
    CONDITION_VALUE = "1"  # Бюджетная основа

    # Тайминги/ретраи
    PAGE_TIMEOUT = 30
    AJAX_TIMEOUT = 40
    OPTION_TIMEOUT = 25
    RETRIES_PER_PROGRAM = 2
    RETRY_SLEEP_SEC = 2.0

    def __init__(self, headless: bool = True):
        logger.info("Запускаем ChromeDriver (headless=%s)", headless)
        options = webdriver.ChromeOptions()

        # Если нужен конкретный бинарник
        # if settings.chrome_bin:
        #     options.binary_location = settings.chrome_bin

        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1200")
        options.add_argument("--lang=ru-RU")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")

        # Чуть агрессивнее ждём появление DOM, а сети — через явные ожидания
        options.page_load_strategy = "normal"

        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=options)
        self._driver.set_page_load_timeout(self.PAGE_TIMEOUT)
        self._wait = WebDriverWait(self._driver, self.PAGE_TIMEOUT)

    # --------------- helpers: select2 / ожидания -----------------

    def _s2_set_and_change(self, select_css: str, value: str) -> None:
        """Ставит value в скрытый <select> и принудительно кидает change
        (и нативный, и jQuery/select2), иначе ajax не запустится.
        """
        el = self._wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, select_css)))
        logger.debug("Выставляем %s = %s", select_css, value)
        try:
            self._driver.execute_script(
                """
                const sel = arguments[0], val = arguments[1];
                // нативный select
                sel.value = val;
                sel.dispatchEvent(new Event('change', { bubbles: true }));
                // jQuery + select2 (если есть)
                if (window.jQuery) {
                    const $sel = jQuery(sel);
                    try { $sel.val(val).trigger('change'); } catch (e) {}
                }
                """,
                el,
                value,
            )
        except JavascriptException as e:
            logger.warning("JS ошибка при установке %s: %s", select_css, e)
            raise

    def _wait_option_present(self, select_css: str, value: str, timeout: Optional[int] = None) -> None:
        """Ждём, пока в селекте появится нужная опция (после ajaxCode())."""
        t = timeout or self.OPTION_TIMEOUT
        logger.debug("Ждём появления option[value='%s'] в %s (до %ss)", value, select_css, t)
        WebDriverWait(self._driver, t).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, f"{select_css} option[value='{value}']")
        )

    def _wait_ajax_table_ready(self) -> None:
        """Ждём окончания ajaxLoadFullTable:
           - спиннер исчез (или не появился — тоже ок),
           - появился tbody с tr ИЛИ блок .no-data.
        """
        logger.debug("Ждём загрузки таблицы (ajaxLoadFullTable)")
        # 1) дождаться, пока спиннер спрячется (или не будет найден)
        try:
            spinner = self._driver.find_element(By.ID, "loading-spinner")
            WebDriverWait(self._driver, self.AJAX_TIMEOUT).until(
                lambda d: "d-none" in spinner.get_attribute("class")
            )
        except NoSuchElementException:
            # на некоторых загрузках спиннер может не попадаться — ок
            logger.debug("Спиннер не найден — продолжаем ожидание по строкам.")

        # 2) дождаться появления строк или .no-data
        WebDriverWait(self._driver, self.AJAX_TIMEOUT).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, "#ajaxTable tbody tr")
                      or d.find_elements(By.CSS_SELECTOR, ".no-data")
        )

        # 3) таблица полосится скриптом — даём UI немного стабилизироваться
        time.sleep(0.3)

    # --------------- helpers: парсинг -----------------

    def _parse_stats(self, program_code: str) -> SubmissionStats:
        """Парсит карточку статистики над таблицей."""

        def _text_by_id(el_id: str) -> str:
            try:
                return self._driver.find_element(By.ID, el_id).text.strip()
            except NoSuchElementException:
                return ""

        raw_places = _text_by_id("numPlaces")
        raw_apps = _text_by_id("numApplications")
        raw_date = _text_by_id("date_info")

        num_places = _safe_int(raw_places, default=0)
        num_apps = _safe_int(raw_apps, default=0)

        # дата может быть пустой или странной — страхуемся
        gen_time: datetime
        if raw_date:
            try:
                gen_time = datetime.strptime(raw_date, "%d.%m.%Y %H:%M")
                # страница явно московская
                gen_time = gen_time.replace(tzinfo=ZoneInfo("Europe/Moscow"))
            except ValueError:
                logger.warning("Неподходящий формат даты '%s' — используем текущее (МСК).", raw_date)
                gen_time = datetime.now(ZoneInfo("Europe/Moscow"))
        else:
            logger.warning("Пустой #date_info — используем текущее (МСК).")
            gen_time = datetime.now(ZoneInfo("Europe/Moscow"))

        stats = SubmissionStats(
            program_code=program_code,
            num_places=num_places,
            num_applications=num_apps,
            generated_at=gen_time,
        )
        logger.info(
            "Статистика %s: места=%d, заявок=%d, время=%s",
            program_code, num_places, num_apps, gen_time.isoformat()
        )
        return stats

    def _header_map(self) -> Dict[str, int]:
        """
        Возвращает карту "нормализованный заголовок -> индекс столбца".
        Нормализация упрощает поиск по русским заголовкам.
        """
        try:
            ths = self._driver.find_elements(By.CSS_SELECTOR, "#ajaxTable thead th")
            header_map: Dict[str, int] = {}
            for idx, th in enumerate(ths):
                key = _normalize(th.text)
                if key:
                    header_map[key] = idx
            if not header_map:
                logger.warning("Не удалось прочитать заголовок таблицы — будет использован жёсткий порядок колонок.")
            return header_map
        except Exception as e:
            logger.warning("Ошибка чтения заголовка таблицы: %s", e)
            return {}

    def _col_index(self, header_map: Dict[str, int], fallback: int, *candidates: str) -> int:
        """Находит индекс колонки по нескольким вариантам заголовка; иначе возвращает fallback."""
        for c in candidates:
            norm = _normalize(c)
            if norm in header_map:
                return header_map[norm]
        return fallback

    def _parse_table(self, program_code: str) -> List[Application]:
        """
        Парсит видимые строки последнего tbody.
        Учитывает, что страница может прятать строки (.d-none) локальными фильтрами.
        """
        # tbody динамически добавляется заново — берём последний
        try:
            rows = self._driver.find_elements(By.CSS_SELECTOR, "#ajaxTable tbody:last-of-type tr:not(.d-none)")
        except NoSuchElementException:
            rows = []

        if not rows:
            # Может быть реальная пустота ('.no-data') — не считаем это за ошибку
            if self._driver.find_elements(By.CSS_SELECTOR, ".no-data"):
                logger.info("Данные не найдены: .no-data присутствует.")
                return []
            logger.warning("Тело таблицы пустое без .no-data — возможно, сбой отрисовки.")
            return []

        header_map = self._header_map()

        # Жёсткие fallback-индексы по текущей верстке (см. HTML)
        idx_applicant = self._col_index(header_map, 1, "уникальный код поступающего")
        idx_total = self._col_index(header_map, 2, "сумма конкурсных баллов")
        idx_vi = self._col_index(header_map, 3, "сумма баллов за ви")
        idx_subj1 = self._col_index(header_map, 4, "предмет 1")
        idx_subj2 = self._col_index(header_map, 5, "предмет 2")
        idx_id = self._col_index(header_map, 6, "кол-во баллов за общие ид")
        idx_target_id = self._col_index(header_map, 7, "кол-во баллов за целевые ид")
        idx_priority = self._col_index(header_map, 8, "приоритет")
        # заголовок колонки про согласие может слегка меняться — fallback = 9
        idx_consent = self._col_index(
            header_map,
            9,
            "наличие согласия на зачисление/ оригинала договора об оказании платных образовательных услуг",
            "согласие на зачисление",
            "информирование о необходимости зачисления",
        )
        idx_status = self._col_index(
            header_map,
            10,
            "информация о рассмотрении заявления о приеме, о допуске к участию в конкурсе",
        )

        applications: List[Application] = []
        for i, tr in enumerate(rows, start=1):
            try:
                tds = tr.find_elements(By.TAG_NAME, "td")
                if not tds:
                    logger.debug("Пустой <tr> #%d — пропускаем.", i)
                    continue

                applicant_id = (tds[idx_applicant].text or "").strip()

                total_score = _safe_int(tds[idx_total].text)
                vi_score = _safe_int(tds[idx_vi].text)
                subject1_score = _safe_int(tds[idx_subj1].text)
                subject2_score = _safe_int(tds[idx_subj2].text)
                id_achievements = _safe_int(tds[idx_id].text)
                target_id_achievements = _safe_int(tds[idx_target_id].text)
                priority = _safe_int(tds[idx_priority].text)

                consent_text = (tds[idx_consent].text or "").strip().lower()
                consent = consent_text in {"+", "получено", "да", "yes"}  # на всякий случай поддержим варианты

                review_status = (tds[idx_status].text or "").strip()

                app = Application(
                    program_code=program_code,
                    applicant_id=applicant_id,
                    total_score=total_score,
                    vi_score=vi_score,
                    subject1_score=subject1_score,
                    subject2_score=subject2_score,
                    id_achievements=id_achievements,
                    target_id_achievements=target_id_achievements,
                    priority=priority,
                    consent=consent,
                    review_status=review_status,
                )
                applications.append(app)
            except Exception as e:
                logger.exception("Не удалось распарсить строку #%d (пропускаем): %s", i, e)

        logger.info("Найдено %d заявок для %s", len(applications), program_code)
        return applications

    # --------------- основной сценарий -----------------

    def parse(self, program_code: str) -> tuple[SubmissionStats, List[Application]]:
        """
        Загружает страницу, выставляет фильтры и парсит данные.
        Делает несколько попыток в случае временных ошибок DOM/AJAX.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.RETRIES_PER_PROGRAM + 1):
            try:
                logger.info("Начинаем парсинг направления %s (попытка %d/%d)",
                            program_code, attempt, self.RETRIES_PER_PROGRAM)

                # Переход на страницу
                self._driver.get(self.BASE_URL)
                self._wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#educationOfForm")))

                # 1) Фильтры "Очная" + "Бюджет"
                self._s2_set_and_change("#educationOfForm", self.FORM_VALUE)
                self._s2_set_and_change("#conditions", self.CONDITION_VALUE)

                # 2) Ждём, пока в #code появится нужная опция
                self._wait_option_present("#code", program_code, timeout=self.OPTION_TIMEOUT)

                # 3) Выбираем код направления → триггерим ajaxDataSummary/ajaxLoadFullTable
                self._s2_set_and_change("#code", program_code)

                # 4) Ждём заполнения таблицы
                self._wait_ajax_table_ready()

                # 5) Парсим статистику и таблицу
                stats = self._parse_stats(program_code)
                applications = self._parse_table(program_code)

                return stats, applications

            except (TimeoutException, StaleElementReferenceException, JavascriptException) as e:
                last_err = e
                logger.warning("Временная ошибка парсинга %s: %s", program_code, e, exc_info=True)
                time.sleep(self.RETRY_SLEEP_SEC)
            except (NoSuchElementException, WebDriverException) as e:
                last_err = e
                logger.error("Критическая ошибка парсинга %s: %s", program_code, e, exc_info=True)
                time.sleep(self.RETRY_SLEEP_SEC)

        # Если все попытки исчерпаны — пробрасываем последнюю ошибку
        assert last_err is not None
        logger.error("Не удалось распарсить %s после %d попыток.", program_code, self.RETRIES_PER_PROGRAM)
        raise last_err

    # --------------- lifecycle -----------------

    def close(self) -> None:
        try:
            self._driver.quit()
            logger.info("ChromeDriver остановлен")
        except Exception:
            pass

    def __del__(self):
        self.close()
