"""
Конфигурация бота
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Токен Telegram-бота
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Настройки OpenAI через проксиapi
PROXYAPI_BASE_URL = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Настройки памяти
MAX_HISTORY_MESSAGES = 5  # Количество последних сообщений для контекста

# Путь к файлу с промптами
PROMPTS_FILE = "prompts.json"

# Путь к файлу памяти
MEMORY_FILE = "memory.json"

# Настройки логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Модели для генерации
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1-mini")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "sora-2")
