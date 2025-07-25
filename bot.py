from app.config.config import settings
from app.presentation.bot import start_bot


if __name__ == "__main__":
    BOT_TOKEN = settings.bot_token
    start_bot(BOT_TOKEN)
