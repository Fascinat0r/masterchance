#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер дат и времени вступительных испытаний магистратуры СПбПУ.

Источник:
https://www.spbstu.ru/abit/master/pass-the-entrance-tests/the-list-of-entrance-examinations/

Таблица с классами: "abit-table responsive tablefilter tablesaw TF"
Колонки:
  0: Институт
  1: Форма обучения
  2: Контракт
  3: Код
  4: Образовательная программа (внутри <a> с pdf)
  5: Дата и время (несколько дат через <br> → в Selenium это .text с переносами строк)

Вывод: JSON-массив объектов:
{
  "institute": "...",
  "education_form": "Очная|Заочная",
  "contract": "Бюджет|Контракт",
  "code": "01.04.02_01",
  "program": "Название",
  "program_pdf_url": "https://www.spbstu.ru/upload/....pdf" | null,
  "dates": ["DD.MM.YYYY HH:MM", ...]  # строки как на сайте, порядок сохранён
}

Запуск:
  python scripts/parse_exam_schedule.py [output.json]

Если указан путь — запишет туда. Иначе напечатает в stdout.
"""
from __future__ import annotations

import json
import sys
from typing import List, Dict, Any, Optional

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://www.spbstu.ru/abit/master/pass-the-entrance-tests/the-list-of-entrance-examinations/"


def _make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def _find_schedule_table(driver: webdriver.Chrome):
    """
    Ищем нужную таблицу. На странице она одна с классом 'abit-table'.
    Дополнительно валидируем, что в thead есть ожидаемые заголовки.
    """
    tables = driver.find_elements(By.CSS_SELECTOR, "table.abit-table")
    if not tables:
        raise NoSuchElementException("Не найдена таблица с классом 'abit-table'.")

    # Берём первую, где встречаются нужные th
    expected_headers = ["Институт", "Форма обучения", "Контракт", "Код", "Образовательная программа", "Дата и время"]
    for t in tables:
        try:
            ths = [th.text.strip() for th in t.find_elements(By.CSS_SELECTOR, "thead th")]
            if all(any(eh in th for th in ths) for eh in expected_headers):
                return t
        except Exception:
            continue

    # fallback — первая таблица
    return tables[0]


def get_master_exam_schedule() -> List[Dict[str, Any]]:
    """
    Парсит расписание вступительных испытаний и возвращает список словарей.
    """
    driver = _make_driver(headless=True)
    wait = WebDriverWait(driver, 15)

    try:
        driver.get(URL)
        # Ждём появления таблицы
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.abit-table")))
        table = _find_schedule_table(driver)

        # tbody → строки (без заголовка и строки фильтров)
        tbody = table.find_element(By.TAG_NAME, "tbody")
        rows = tbody.find_elements(By.TAG_NAME, "tr")

        results: List[Dict[str, Any]] = []
        for tr in rows:
            tds = tr.find_elements(By.TAG_NAME, "td")
            if len(tds) < 6:
                # пропускаем служебные/битые строки
                continue

            institute = tds[0].text.strip()
            education_form = tds[1].text.strip()
            contract = tds[2].text.strip()
            code = tds[3].text.strip()

            # 5-й столбец — программа (обычно <a>)
            program_name = tds[4].text.strip()
            program_pdf_url: Optional[str] = None
            try:
                a = tds[4].find_element(By.TAG_NAME, "a")
                if a:
                    program_name = (a.text or program_name).strip()
                    program_pdf_url = a.get_attribute("href") or None
            except NoSuchElementException:
                pass

            # 6-й столбец — даты (текст с переносами)
            raw_dates = tds[5].text.splitlines()
            dates = [d.strip() for d in raw_dates if d.strip()]

            results.append({
                "institute": institute,
                "education_form": education_form,
                "contract": contract,
                "code": code,
                "program": program_name,
                "program_pdf_url": program_pdf_url,
                "dates": dates,
            })

        return results

    except TimeoutException as e:
        print(f"Ошибка: не дождались таблицы. {e}", file=sys.stderr)
        return []
    finally:
        driver.quit()


def main() -> None:
    data = get_master_exam_schedule()
    out = json.dumps(data, ensure_ascii=False, indent=2)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Сохранил {len(data)} записей в {path}")
    else:
        print(out)


if __name__ == "__main__":
    main()
