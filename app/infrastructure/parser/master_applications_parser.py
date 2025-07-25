import time
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from webdriver_manager.chrome import ChromeDriverManager

from app.config.config import settings
from app.config.logger import logger
from app.domain.models import SubmissionStats, Application


class MasterApplicationsParser:
    """
    Через Selenium переходит на страницу списка абитуриентов,
    ждёт загрузки нужного направления, парсит статистику и таблицу.
    """
    BASE_URL = "https://my.spbstu.ru/home/abit/list-applicants/master"

    def __init__(self, headless: bool = True):
        logger.info("Запускаем ChromeDriver (headless=%s)", headless)
        options = webdriver.ChromeOptions()
        # options.binary_location = settings.chrome_bin
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=options)
        self._wait = WebDriverWait(self._driver, 15)

    def parse(self, program_code: str) -> tuple[SubmissionStats, List[Application]]:
        logger.info("Начинаем парсинг направления %s", program_code)
        try:
            self._driver.get(self.BASE_URL)

            # 1) выбираем «Очная» и «Бюджет»
            edu = self._wait.until(EC.presence_of_element_located((By.ID, "educationOfForm")))
            Select(edu).select_by_value("2")
            cond = self._wait.until(EC.presence_of_element_located((By.ID, "conditions")))
            Select(cond).select_by_value("1")

            # 2) ждём, пока селектор «code» наполнится нужным option
            code_sel = self._wait.until(EC.presence_of_element_located((By.ID, "code")))
            try:
                self._wait.until(lambda d: any(
                    opt.get_attribute("value") == program_code
                    for opt in code_sel.find_elements(By.TAG_NAME, "option")
                ))
            except TimeoutException:
                logger.warning("Нет опции %s в списке направлений — пропускаем", program_code)
                raise

            Select(code_sel).select_by_value(program_code)
            time.sleep(1)  # даём время на подгрузку

            # 3) парсим статистику
            num_places = int(self._driver.find_element(By.ID, "numPlaces").text or 0)
            num_apps = int(self._driver.find_element(By.ID, "numApplications").text or 0)

            raw_date = self._driver.find_element(By.ID, "date_info").text.strip()
            if raw_date:
                try:
                    gen_time = datetime.strptime(raw_date, "%d.%m.%Y %H:%M")
                except ValueError:
                    logger.warning(
                        "Неподходящий формат даты '%s', используем текущее время", raw_date
                    )
                    gen_time = datetime.now(ZoneInfo("Europe/Moscow"))
            else:
                logger.warning("Пустая дата в #date_info, используем текущее время")
                gen_time = datetime.now()

            stats = SubmissionStats(
                program_code=program_code,
                num_places=num_places,
                num_applications=num_apps,
                generated_at=gen_time
            )
            logger.info(
                "Статистика %s: места=%d, заявок=%d, время=%s",
                program_code, num_places, num_apps, gen_time.isoformat()
            )

            # 4) парсим таблицу заявок
            tbody = self._driver.find_element(By.ID, "ajaxTable").find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            applications: List[Application] = []
            for tr in rows:
                cols = tr.find_elements(By.TAG_NAME, "td")
                app = Application(
                    program_code=program_code,
                    applicant_id=cols[1].text.strip(),
                    total_score=int(cols[2].text),
                    vi_score=int(cols[3].text),
                    subject1_score=int(cols[4].text),
                    subject2_score=int(cols[5].text),
                    id_achievements=int(cols[6].text),
                    target_id_achievements=int(cols[7].text),
                    priority=int(cols[8].text),
                    consent=(cols[9].text.strip() == "+"),
                    review_status=cols[10].text.strip(),
                )
                applications.append(app)

            logger.info("Найдено %d заявок для %s", len(applications), program_code)
            return stats, applications

        except (NoSuchElementException, TimeoutException) as err:
            logger.error("Ошибка парсинга %s: %s", program_code, err)
            raise


def __del__(self):
    try:
        self._driver.quit()
        logger.info("ChromeDriver остановлен")
    except Exception:
        pass
