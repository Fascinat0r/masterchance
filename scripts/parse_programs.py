#!/usr/bin/env python3
"""
Скрипт для парсинга направлений магистратуры с сайта SPbSTU
Необходимые библиотеки:
    pip install selenium webdriver-manager
"""
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from webdriver_manager.chrome import ChromeDriverManager


def get_master_programs():
    # Инициализируем драйвер Chrome через Service
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Запуск в фоновом режиме
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Открываем страницу
        url = 'https://my.spbstu.ru/home/abit/list-applicants/master'
        driver.get(url)

        wait = WebDriverWait(driver, 10)

        # Ждем и выбираем 'Очная' в форме обучения
        education_select = wait.until(
            EC.presence_of_element_located((By.ID, 'educationOfForm'))
        )
        Select(education_select).select_by_value('2')

        # Ждем и выбираем 'Бюджетная основа'
        conditions_select = wait.until(
            EC.presence_of_element_located((By.ID, 'conditions'))
        )
        Select(conditions_select).select_by_value('1')

        # Ждем появления селектора направлений и его опций
        code_select = wait.until(
            EC.presence_of_element_located((By.ID, 'code'))
        )
        wait.until(lambda d: len(code_select.find_elements(By.TAG_NAME, 'option')) > 0)

        # Извлекаем все опции из селектора направлений
        programs = []
        for option in code_select.find_elements(By.TAG_NAME, 'option'):
            value = option.get_attribute('value')
            text = option.text.strip()
            if value:
                programs.append({'code': value, 'name': text})

        return programs

    except TimeoutException:
        print('Ошибка: элемент не найден или истекло время ожидания.')
        return []

    finally:
        driver.quit()


if __name__ == '__main__':
    programs = get_master_programs()
    # Выводим в формате JSON
    import json

    print(json.dumps(programs, ensure_ascii=False, indent=2))

