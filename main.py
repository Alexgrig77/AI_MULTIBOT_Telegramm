"""
Основной файл Telegram-бота с интеграцией OpenAI через проксиapi
"""
import asyncio
import json
import logging
from typing import Optional, Tuple

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from openai import AsyncOpenAI

from config import (
    BOT_TOKEN, PROXYAPI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL,
    MAX_HISTORY_MESSAGES, PROMPTS_FILE, MEMORY_FILE, LOG_LEVEL,
    IMAGE_MODEL, VIDEO_MODEL
)
from memory import MemoryManager

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация компонентов
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
memory_manager = MemoryManager(MEMORY_FILE)

# Загрузка промптов
prompts_data = {}
try:
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    logger.info(f"Загружено {len(prompts_data.get('prompts', {}))} промптов")
except Exception as e:
    logger.error(f"Ошибка загрузки промптов: {e}")

# Инициализация клиента OpenAI через проксиapi
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=PROXYAPI_BASE_URL
)

# Кэш курса валют
usd_rate_cache = None
usd_rate_cache_time = None


async def get_usd_rate() -> Optional[float]:
    """Получает курс USD к RUB с API ЦБ РФ"""
    global usd_rate_cache, usd_rate_cache_time
    import time
    
    # Кэшируем курс на 1 час
    if usd_rate_cache and usd_rate_cache_time:
        if time.time() - usd_rate_cache_time < 3600:
            return usd_rate_cache
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://www.cbr-xml-daily.ru/daily_json.js') as response:
                if response.status == 200:
                    data = await response.json()
                    rate = data['Valute']['USD']['Value']
                    usd_rate_cache = rate
                    usd_rate_cache_time = time.time()
                    logger.info(f"Получен курс USD: {rate} RUB")
                    return rate
    except Exception as e:
        logger.error(f"Ошибка получения курса валют: {e}")
    
    # Если не удалось получить, используем запасной курс
    return 100.0  # Примерный курс


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> Tuple[float, float]:
    """
    Рассчитывает стоимость запроса в USD
    Возвращает (стоимость в USD, стоимость в RUB)
    """
    # Примерные цены для gpt-4o-mini (могут отличаться)
    # Цены обычно: input $0.15/1M tokens, output $0.6/1M tokens
    input_price_per_1m = 0.15
    output_price_per_1m = 0.6
    
    cost_usd = (input_tokens / 1_000_000 * input_price_per_1m) + (output_tokens / 1_000_000 * output_price_per_1m)
    return cost_usd


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """
    Разбивает длинное сообщение на части, не превышающие max_length символов
    """
    if len(text) <= max_length:
        return [text]
    
    parts = []
    current_part = ""
    
    # Пытаемся разбить по абзацам
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # Если текущая часть + новый абзац помещается
        if len(current_part) + len(paragraph) + 2 <= max_length:
            if current_part:
                current_part += '\n\n' + paragraph
            else:
                current_part = paragraph
        else:
            # Если текущая часть не пуста, сохраняем её
            if current_part:
                parts.append(current_part)
                current_part = ""
            
            # Если абзац сам по себе длиннее лимита, разбиваем по предложениям
            if len(paragraph) > max_length:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_part) + len(sentence) + 2 <= max_length:
                        if current_part:
                            current_part += '. ' + sentence
                        else:
                            current_part = sentence
                    else:
                        if current_part:
                            parts.append(current_part)
                        current_part = sentence
            else:
                current_part = paragraph
    
    # Добавляем последнюю часть
    if current_part:
        parts.append(current_part)
    
    return parts


async def get_ai_response(user_id: str, user_message: str) -> tuple[str, dict]:
    """
    Получает ответ от OpenAI через проксиapi
    Возвращает (ответ, метаданные с токенами)
    """
    try:
        # Получаем режим пользователя
        mode = memory_manager.get_mode(user_id)
        prompts = prompts_data.get('prompts', {})
        prompt_info = prompts.get(mode, prompts.get('assistant', {}))
        system_prompt = prompt_info.get('system_prompt', 'Ты — полезный помощник.')
        
        # Получаем историю
        history = memory_manager.get_history(user_id, MAX_HISTORY_MESSAGES)
        
        # Формируем сообщения для OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        
        logger.info(f"Отправка запроса к OpenAI для пользователя {user_id}, режим: {mode}")
        
        # Отправляем запрос
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        
        assistant_message = response.choices[0].message.content
        
        # Сохраняем сообщения в память
        memory_manager.add_message(user_id, "user", user_message)
        memory_manager.add_message(user_id, "assistant", assistant_message)
        
        # Получаем метаданные
        usage = response.usage
        metadata = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
        
        logger.info(f"Получен ответ от OpenAI. Токены: {metadata['input_tokens']} входных, {metadata['output_tokens']} выходных")
        
        return assistant_message, metadata
        
    except Exception as e:
        logger.error(f"Ошибка при запросе к OpenAI: {e}", exc_info=True)
        raise


async def translate_prompt_to_english(prompt: str) -> str:
    """Переводит промпт на английский язык для генерации видео"""
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Ты — переводчик. Переведи промпт на английский язык, сохраняя технические термины и стиль описания. Верни только перевод без дополнительных комментариев."
                },
                {
                    "role": "user",
                    "content": f"Переведи на английский: {prompt}"
                }
            ]
        )
        translated = response.choices[0].message.content.strip()
        logger.info(f"Промпт переведен: {prompt[:50]}... -> {translated[:50]}...")
        return translated
    except Exception as e:
        logger.error(f"Ошибка перевода промпта: {e}")
        # Если перевод не удался, возвращаем оригинал
        return prompt


async def generate_image(prompt: str) -> tuple[str, dict]:
    """
    Генерирует изображение через проксиapi
    Возвращает (URL изображения, метаданные)
    """
    try:
        logger.info(f"Генерация изображения с промптом: {prompt[:100]}...")
        
        # Пробуем с параметром model, если не сработает - без него
        try:
            response = await openai_client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
        except Exception as model_error:
            logger.warning(f"Ошибка с параметром model, пробуем без него: {model_error}")
            # Пробуем без параметра model
            response = await openai_client.images.generate(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
        
        # Логируем структуру ответа для отладки
        logger.debug(f"Структура ответа: {type(response)}")
        logger.debug(f"Response data: {response.data if hasattr(response, 'data') else 'No data attr'}")
        
        # Пробуем разные варианты получения URL
        image_url = None
        
        if hasattr(response, 'data') and response.data and len(response.data) > 0:
            first_item = response.data[0]
            logger.debug(f"First item type: {type(first_item)}")
            logger.debug(f"First item attrs: {dir(first_item)}")
            
            # Пробуем разные атрибуты
            if hasattr(first_item, 'url') and first_item.url:
                image_url = first_item.url
            elif hasattr(first_item, 'image_url') and first_item.image_url:
                image_url = first_item.image_url
            elif hasattr(first_item, 'b64_json') and first_item.b64_json:
                # Если вернулся base64, нужно обработать отдельно
                logger.warning("Получен base64 вместо URL")
                image_url = None
        elif hasattr(response, 'url') and response.url:
            image_url = response.url
        
        if not image_url:
            # Логируем полный ответ для отладки
            logger.error(f"Не удалось получить URL изображения. Полный ответ: {response}")
            raise ValueError("API не вернул URL изображения. Проверьте логи для деталей.")
        
        metadata = {
            "model": IMAGE_MODEL,
            "prompt": prompt
        }
        
        logger.info(f"Изображение сгенерировано: {image_url}")
        return image_url, metadata
        
    except Exception as e:
        logger.error(f"Ошибка генерации изображения: {e}", exc_info=True)
        raise


async def generate_video(prompt: str) -> tuple[str, dict]:
    """
    Генерирует видео через проксиapi (sora-2)
    Возвращает (URL видео или ID задачи, метаданные)
    """
    try:
        # Переводим промпт на английский
        english_prompt = await translate_prompt_to_english(prompt)
        
        logger.info(f"Генерация видео с промптом (EN): {english_prompt[:100]}...")
        
        # Генерируем видео
        response = await openai_client.videos.create(
            model=VIDEO_MODEL,
            prompt=english_prompt
        )
        
        # Для sora-2 может быть асинхронная генерация
        # Проверяем, есть ли URL или нужно ждать
        video_url = None
        video_id = None
        
        if hasattr(response, 'url') and response.url:
            video_url = response.url
        elif hasattr(response, 'id'):
            video_id = response.id
        elif hasattr(response, 'data') and len(response.data) > 0:
            if hasattr(response.data[0], 'url'):
                video_url = response.data[0].url
            elif hasattr(response.data[0], 'id'):
                video_id = response.data[0].id
        
        metadata = {
            "model": VIDEO_MODEL,
            "prompt": prompt,
            "english_prompt": english_prompt,
            "video_id": video_id
        }
        
        logger.info(f"Видео сгенерировано. URL: {video_url}, ID: {video_id}")
        return video_url or f"video_id:{video_id}", metadata
        
    except Exception as e:
        logger.error(f"Ошибка генерации видео: {e}", exc_info=True)
        raise


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    user_id = str(message.from_user.id)
    mode = memory_manager.get_mode(user_id)
    prompt_info = prompts_data.get('prompts', {}).get(mode, {})
    mode_name = prompt_info.get('name', 'Неизвестный режим')
    
    welcome_text = (
        f"Привет! Я AI-ассистент, готовый помочь.\n\n"
        f"Текущий режим: {mode_name}\n\n"
        f"Доступные команды:\n"
        f"/mode - выбрать режим работы\n"
        f"/reset - очистить историю диалога\n\n"
        f"Просто напиши мне что-нибудь, и я отвечу!"
    )
    
    # Создаем клавиатуру с кнопками
    keyboard = [
        [
            InlineKeyboardButton(text="🖼️ Сгенерировать изображение", callback_data="generate_image"),
            InlineKeyboardButton(text="🎬 Сгенерировать видео", callback_data="generate_video")
        ],
        [
            InlineKeyboardButton(text="⚙️ Выбрать режим", callback_data="show_modes")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    await message.answer(welcome_text, reply_markup=reply_markup)
    logger.info(f"Пользователь {user_id} запустил бота")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    """Обработчик команды /reset"""
    user_id = str(message.from_user.id)
    memory_manager.clear_history(user_id)
    await message.answer("История диалога очищена! 🗑️")
    logger.info(f"Пользователь {user_id} очистил историю")


@dp.message(Command("mode"))
async def cmd_mode(message: Message):
    """Обработчик команды /mode - показывает список режимов"""
    prompts = prompts_data.get('prompts', {})
    
    if not prompts:
        await message.answer("Режимы не загружены. Проверьте файл prompts.json")
        return
    
    keyboard = []
    for mode_id, mode_info in prompts.items():
        name = mode_info.get('name', mode_id)
        description = mode_info.get('description', '')
        button_text = f"{name}"
        if description:
            button_text += f" - {description}"
        
        keyboard.append([InlineKeyboardButton(
            text=button_text,
            callback_data=f"mode_{mode_id}"
        )])
    
    reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    await message.answer("Выберите режим работы:", reply_markup=reply_markup)
    logger.info(f"Пользователь {message.from_user.id} запросил список режимов")


@dp.callback_query(F.data == "generate_image")
async def callback_generate_image(callback: CallbackQuery):
    """Обработчик кнопки генерации изображения"""
    user_id = str(callback.from_user.id)
    memory_manager.set_mode(user_id, "image_generator")
    await callback.answer("Режим генерации изображений активирован")
    await callback.message.answer(
        "🖼️ **Режим генерации изображений активирован**\n\n"
        "Теперь просто напиши текстом, что хочешь увидеть, и я сгенерирую изображение!\n\n"
        "Пример: `красивый закат над морем с пальмами`\n"
        "Или: `крыло самолета`",
        parse_mode="Markdown"
    )
    logger.info(f"Пользователь {user_id} переключился в режим генерации изображений")


@dp.callback_query(F.data == "generate_video")
async def callback_generate_video(callback: CallbackQuery):
    """Обработчик кнопки генерации видео"""
    user_id = str(callback.from_user.id)
    memory_manager.set_mode(user_id, "video_generator")
    await callback.answer("Режим генерации видео активирован")
    await callback.message.answer(
        "🎬 **Режим генерации видео активирован**\n\n"
        "Теперь просто напиши текстом, что хочешь увидеть, и я сгенерирую видео!\n\n"
        "Пример: `молодой разработчик кодит за ноутбуком в уютной комнате`\n\n"
        "💡 Промпт автоматически переведется на английский для лучшего результата.",
        parse_mode="Markdown"
    )
    logger.info(f"Пользователь {user_id} переключился в режим генерации видео")


@dp.callback_query(F.data == "show_modes")
async def callback_show_modes(callback: CallbackQuery):
    """Обработчик кнопки показа режимов"""
    await callback.answer()
    prompts = prompts_data.get('prompts', {})
    
    if not prompts:
        await callback.message.answer("Режимы не загружены. Проверьте файл prompts.json")
        return
    
    keyboard = []
    for mode_id, mode_info in prompts.items():
        name = mode_info.get('name', mode_id)
        description = mode_info.get('description', '')
        button_text = f"{name}"
        if description:
            button_text += f" - {description}"
        
        keyboard.append([InlineKeyboardButton(
            text=button_text,
            callback_data=f"mode_{mode_id}"
        )])
    
    reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    await callback.message.answer("Выберите режим работы:", reply_markup=reply_markup)
    logger.info(f"Пользователь {callback.from_user.id} запросил список режимов через кнопку")


@dp.callback_query(F.data.startswith("mode_"))
async def process_mode_callback(callback: CallbackQuery):
    """Обработчик выбора режима"""
    mode_id = callback.data.replace("mode_", "")
    user_id = str(callback.from_user.id)
    
    prompts = prompts_data.get('prompts', {})
    if mode_id not in prompts:
        await callback.answer("Режим не найден", show_alert=True)
        return
    
    memory_manager.set_mode(user_id, mode_id)
    mode_info = prompts[mode_id]
    mode_name = mode_info.get('name', mode_id)
    
    await callback.answer(f"Режим изменен на: {mode_name}")
    await callback.message.edit_text(f"✅ Режим изменен на: **{mode_name}**\n\n{mode_info.get('description', '')}", parse_mode="Markdown")
    logger.info(f"Пользователь {user_id} изменил режим на {mode_id}")


@dp.message(Command("image"))
async def cmd_image(message: Message):
    """Обработчик команды /image - генерация изображения"""
    user_id = str(message.from_user.id)
    
    # Получаем промпт из сообщения
    command_text = message.text or ""
    prompt = command_text.replace("/image", "").strip()
    
    if not prompt:
        await message.answer(
            "Использование: /image <описание>\n\n"
            "Пример: /image красивый закат над морем с пальмами"
        )
        return
    
    logger.info(f"Пользователь {user_id} запросил генерацию изображения: {prompt[:50]}...")
    
    # Отправляем индикатор
    await bot.send_chat_action(message.chat.id, "upload_photo")
    status_msg = await message.answer("🎨 Генерирую изображение...")
    
    try:
        # Генерируем изображение
        image_url, metadata = await generate_image(prompt)
        
        if not image_url:
            raise ValueError("Не удалось получить URL изображения от API")
        
        # Отправляем изображение
        await message.answer_photo(
            photo=image_url,
            caption=f"🖼️ Изображение сгенерировано\n\nПромпт: {prompt}"
        )
        
        # Удаляем сообщение о статусе
        await status_msg.delete()
        
        logger.info(f"Изображение отправлено пользователю {user_id}")
        
    except Exception as e:
        await status_msg.delete()
        error_msg = f"❌ Ошибка генерации изображения: {str(e)}"
        await message.answer(error_msg)
        logger.error(f"Ошибка генерации изображения для {user_id}: {e}", exc_info=True)


@dp.message(Command("video"))
async def cmd_video(message: Message):
    """Обработчик команды /video - генерация видео"""
    user_id = str(message.from_user.id)
    
    # Получаем промпт из сообщения
    command_text = message.text or ""
    prompt = command_text.replace("/video", "").strip()
    
    if not prompt:
        await message.answer(
            "Использование: /video <описание>\n\n"
            "Пример: /video молодой разработчик кодит за ноутбуком в уютной комнате"
        )
        return
    
    logger.info(f"Пользователь {user_id} запросил генерацию видео: {prompt[:50]}...")
    
    # Отправляем индикатор
    await bot.send_chat_action(message.chat.id, "upload_video")
    status_msg = await message.answer("🎬 Генерирую видео... Это может занять некоторое время...")
    
    try:
        # Генерируем видео
        video_result, metadata = await generate_video(prompt)
        
        # Проверяем, это URL или ID
        if video_result.startswith("video_id:"):
            video_id = video_result.replace("video_id:", "")
            await status_msg.edit_text(
                f"⏳ Видео генерируется...\n"
                f"ID задачи: {video_id}\n\n"
                f"Промпт: {prompt}\n"
                f"Переведенный промпт (EN): {metadata.get('english_prompt', 'N/A')}"
            )
            logger.info(f"Видео в процессе генерации для пользователя {user_id}, ID: {video_id}")
        else:
            # Если есть URL, отправляем видео
            await message.answer_video(
                video=video_result,
                caption=f"🎬 Видео сгенерировано\n\nПромпт: {prompt}\nПереведенный промпт (EN): {metadata.get('english_prompt', 'N/A')}"
            )
            await status_msg.delete()
            logger.info(f"Видео отправлено пользователю {user_id}")
        
    except Exception as e:
        await status_msg.delete()
        error_msg = f"❌ Ошибка генерации видео: {str(e)}"
        await message.answer(error_msg)
        logger.error(f"Ошибка генерации видео для {user_id}: {e}", exc_info=True)


@dp.message(F.text)
async def handle_message(message: Message):
    """Обработчик обычных текстовых сообщений"""
    user_id = str(message.from_user.id)
    user_message = message.text
    
    # Проверяем, не является ли это командой (начинается с /)
    if user_message.startswith('/'):
        return  # Команды обрабатываются отдельными обработчиками
    
    logger.info(f"Получено сообщение от пользователя {user_id}: {user_message[:50]}...")
    
    # Проверяем режим пользователя
    mode = memory_manager.get_mode(user_id)
    prompts = prompts_data.get('prompts', {})
    mode_info = prompts.get(mode, {})
    
    # Если режим - генератор изображений
    if mode_info.get('type') == 'image':
        await bot.send_chat_action(message.chat.id, "upload_photo")
        status_msg = await message.answer("🎨 Генерирую изображение...")
        
        try:
            image_url, metadata = await generate_image(user_message)
            
            if not image_url:
                raise ValueError("Не удалось получить URL изображения от API")
            
            await message.answer_photo(
                photo=image_url,
                caption=f"🖼️ Изображение сгенерировано\n\nПромпт: {user_message}"
            )
            await status_msg.delete()
            logger.info(f"Изображение отправлено пользователю {user_id}")
        except Exception as e:
            await status_msg.delete()
            error_msg = f"❌ Ошибка генерации изображения: {str(e)}"
            await message.answer(error_msg)
            logger.error(f"Ошибка генерации изображения для {user_id}: {e}", exc_info=True)
        return
    
    # Если режим - генератор видео
    if mode_info.get('type') == 'video':
        await bot.send_chat_action(message.chat.id, "upload_video")
        status_msg = await message.answer("🎬 Генерирую видео... Это может занять некоторое время...")
        
        try:
            video_result, metadata = await generate_video(user_message)
            
            if video_result.startswith("video_id:"):
                video_id = video_result.replace("video_id:", "")
                await status_msg.edit_text(
                    f"⏳ Видео генерируется...\n"
                    f"ID задачи: {video_id}\n\n"
                    f"Промпт: {user_message}\n"
                    f"Переведенный промпт (EN): {metadata.get('english_prompt', 'N/A')}"
                )
                logger.info(f"Видео в процессе генерации для пользователя {user_id}, ID: {video_id}")
            else:
                await message.answer_video(
                    video=video_result,
                    caption=f"🎬 Видео сгенерировано\n\nПромпт: {user_message}\nПереведенный промпт (EN): {metadata.get('english_prompt', 'N/A')}"
                )
                await status_msg.delete()
                logger.info(f"Видео отправлено пользователю {user_id}")
        except Exception as e:
            await status_msg.delete()
            error_msg = f"❌ Ошибка генерации видео: {str(e)}"
            await message.answer(error_msg)
            logger.error(f"Ошибка генерации видео для {user_id}: {e}", exc_info=True)
        return
    
    # Обычный режим - текстовый ответ
    await bot.send_chat_action(message.chat.id, "typing")
    
    try:
        # Получаем ответ от AI
        response_text, metadata = await get_ai_response(user_id, user_message)
        
        # Рассчитываем стоимость
        cost_usd = calculate_cost(metadata['input_tokens'], metadata['output_tokens'], OPENAI_MODEL)
        usd_rate = await get_usd_rate()
        cost_rub = cost_usd * usd_rate
        
        # Формируем сообщение со стоимостью
        cost_info = (
            f"\n\n"
            f"💵 Стоимость запроса:\n"
            f"• Входные токены: {metadata['input_tokens']}\n"
            f"• Выходные токены: {metadata['output_tokens']}\n"
            f"• Всего токенов: {metadata['total_tokens']}\n"
            f"• Стоимость: ${cost_usd:.6f} ({cost_rub:.4f} ₽)"
        )
        
        full_response = response_text + cost_info
        
        # Разбиваем сообщение на части, если оно слишком длинное
        message_parts = split_message(full_response, max_length=4096)
        
        # Отправляем все части
        for i, part in enumerate(message_parts):
            if i == 0:
                await message.answer(part)
            else:
                await message.answer(part)
                # Небольшая задержка между сообщениями
                await asyncio.sleep(0.1)
        
        logger.info(
            f"Ответ отправлен пользователю {user_id}. "
            f"Токены: {metadata['input_tokens']}/{metadata['output_tokens']}, "
            f"Стоимость: ${cost_usd:.6f} ({cost_rub:.4f} ₽)"
        )
        
    except Exception as e:
        error_msg = f"Произошла ошибка при обработке запроса: {str(e)}"
        await message.answer(error_msg)
        logger.error(f"Ошибка обработки сообщения от {user_id}: {e}", exc_info=True)


async def main():
    """Главная функция запуска бота"""
    logger.info("Запуск бота...")
    logger.info(f"Используется проксиapi: {PROXYAPI_BASE_URL}")
    logger.info(f"Модель: {OPENAI_MODEL}")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
