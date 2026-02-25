# MasterChance

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Aiogram](https://img.shields.io/badge/Aiogram-3.x-orange.svg)](https://docs.aiogram.dev/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)

Telegram-бот для абитуриентов магистратуры СПбПУ. Он парсит конкурсные списки с сайта университета, считает симуляции методом Монте-Карло и выдаёт примерную вероятность зачисления — с учётом приоритетов и возможных отказов других людей в очереди.

Честно говоря, идея не нова: многие руками смотрят на своё место в списке и прикидывают шансы. Бот просто делает это быстрее и более системно.

---

## Что умеет бот

- Собирает данные о заявлениях через Selenium — без ручного обновления.
- Запускает симуляции Монте-Карло поверх актуальных данных и показывает вероятность поступления.
- Следит за расписанием вступительных испытаний.
- Строит отчёты по проходным баллам и числу поданных заявлений на направление.

---

## Технологии

| Цель          | Инструмент                      |
| ------------- | ------------------------------- |
| Бот           | `aiogram 3.x`                   |
| БД и миграции | `SQLAlchemy`, `Alembic`, SQLite |
| Парсинг       | `Selenium`, `webdriver-manager` |
| Расчёты       | `numpy`, `numba`, `pandas`      |
| Графики       | `matplotlib`                    |
| Сборка        | Docker                          |

---

## Запуск

### Требования

- Python 3.11+
- Токен Telegram-бота от [@BotFather](https://t.me/BotFather)
- Docker (если запускаете в контейнере) или Chromium (если локально)

### Через Docker

```bash
git clone <repository_url>
cd masterchance
```

Создайте `.env`:

```env
BOT_TOKEN=your_token_here
ENV=dev
```

Запустите:

```bash
make run
```

### Локально

```bash
pip install -r requirements.txt
python bot.py
```

Парсер использует Selenium — нужен установленный Chromium или Chrome.

---

## Обновление данных

Скрипты запускаются вручную или по расписанию:

- `update_lists.py` — обновляет списки абитуриентов.
- `update_exam_schedule.py` — обновляет расписание испытаний.
- `run_monte_carlo.py` — пересчитывает вероятности.

---

## Структура проекта

Код разбит по слоям Clean Architecture:

```
app/
  domain/         # модели данных (dataclasses)
  application/    # сценарии использования
  infrastructure/ # парсеры и работа с БД
  presentation/   # Telegram-бот (aiogram)
migrations/       # миграции Alembic
```

---

## Разработка

### Makefile

```bash
make build        # собрать Docker-образ
make run          # собрать и запустить
make push         # отправить образ в реестр
make bump-version # обновить версию (VERSION=x.y.z)
```

### Миграции

```bash
alembic upgrade head
```

---

## Лицензия

Данные со страниц университета принадлежат СПбПУ. Бот использует их только для расчётов и не хранит в открытом доступе.
