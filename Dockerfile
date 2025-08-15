FROM python:3.11-slim

# Устанавливаем Chromium (без chromium-driver) + все зависимости для headless
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    chromium \
    ca-certificates \
    fonts-liberation \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxshmfence1 \
    libxkbcommon0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
COPY ./data /app/data

# ваш requirements.txt должен включать selenium и webdriver-manager
RUN pip install --no-cache-dir -r requirements.txt

# Подсказываем selenium, где лежит chromium
ENV CHROME_BIN=/usr/bin/chromium

# Оставляем остальные ENV:
ENV ENV=dev \
    DATA_DIR=./data \
    TIMEZONE=Europe/Moscow \
    DB_FILENAME=master.db \
    DB_ECHO=false \

ENTRYPOINT ["python", "bot.py"]
