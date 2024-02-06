import asyncio
import logging
import sys
from os import getenv
from io import BytesIO

from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from aiogram import F

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf


TOKEN = getenv("TOKEN")
# All handlers should be attached to the Router (or Dispatcher)
router = Router()

model = tf.keras.models.load_model('model.h5')
class_names = ['cat', 'dog', 'elephant', 'horse', 'lion']


@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {hbold(message.from_user.full_name)}, I am Animal Classifier! I can classify some animals on the images. I can classify such animals as cat, dog, elephant, horse and lion")


@router.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    await message.answer("/start - start bot \n/help - help \n/classify - classify an image")


@router.message(Command("classify"))
async def command_classify_handler(message: Message) -> None:
    await message.answer("Send me an image!")


@router.message(F.photo)
async def classify_photo(message: Message) -> None:
    await message.answer("Classifying...")
    photo = await message.bot.get_file(message.photo[-1].file_id)
    photo_bytes = await message.bot.download_file(photo.file_path)

    photo_bytes = np.asarray(bytearray(photo_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
    img = np.array(img, dtype=np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    print(predictions)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]
    await message.answer(f"The animal on the picture is - {predicted_class_name}!")


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    dp = Dispatcher()
    dp.include_router(router)
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
