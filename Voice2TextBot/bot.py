import os
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from voice_processor import voice_processor


# Token را از متغیر محیطی TELEGRAM_BOT_TOKEN می‌خوانیم؛ در صورت نبود، از مقدار زیر استفاده می‌شود
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "8165964545:AAFCuFec-tXvoN0QCxQ4g0AtrFiOYuq3Vqk"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_message = """🎤 سلام! بات تبدیل صوت به متن فارسی

✅ فرمت‌های پشتیبانی شده:
• ویس تلگرام (OGG)
• فایل‌های صوتی: MP3, WAV, M4A, FLAC, AAC
• ارسال به عنوان سند

🔧 مدل: wav2vec2 Persian v3 (بهترین کیفیت فارسی)

📝 فقط ویس یا فایل صوتی بفرستید!"""
    await update.message.reply_text(start_message)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📝 پیام متنی دریافت شد! برای تبدیل صوت به متن، لطفاً ویس بفرستید.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """پردازش پیام‌های صوتی (ویس تلگرام)"""
    try:
        # ارسال پیام در حال پردازش
        processing_msg = await update.message.reply_text("🔄 در حال پردازش ویس...")
        
        # دریافت فایل صوتی
        voice_file = await update.message.voice.get_file()
        
        # ایجاد فایل موقت با فرمت OGG
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_file:
            temp_path = temp_file.name
        
        # دانلود فایل صوتی
        await voice_file.download_to_drive(temp_path)
        
        # تبدیل صوت به متن
        text = voice_processor.transcribe_audio(temp_path)
        
        # حذف فایل موقت
        os.unlink(temp_path)
        
        # ارسال نتیجه
        if text and text.strip():
            await processing_msg.edit_text(f"📝 متن تشخیص داده شده:\n\n{text}")
        else:
            await processing_msg.edit_text("❌ متأسفانه نتوانستم متن را تشخیص دهم. لطفاً دوباره تلاش کنید.")
            
    except Exception as e:
        await update.message.reply_text(f"❌ خطا در پردازش ویس: {str(e)}")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """پردازش فایل‌های صوتی (MP3, WAV, M4A و غیره)"""
    try:
        # ارسال پیام در حال پردازش
        processing_msg = await update.message.reply_text("🔄 در حال پردازش فایل صوتی...")
        
        # دریافت فایل صوتی
        audio_file = await update.message.audio.get_file()
        
        # تشخیص فرمت فایل
        file_name = audio_file.file_path.split('/')[-1] if audio_file.file_path else "audio"
        file_extension = os.path.splitext(file_name)[1] or ".mp3"
        
        # ایجاد فایل موقت با فرمت مناسب
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_path = temp_file.name
        
        # دانلود فایل صوتی
        await audio_file.download_to_drive(temp_path)
        
        # تبدیل صوت به متن
        text = voice_processor.transcribe_audio(temp_path)
        
        # حذف فایل موقت
        os.unlink(temp_path)
        
        # ارسال نتیجه
        if text and text.strip():
            await processing_msg.edit_text(f"📝 متن تشخیص داده شده:\n\n{text}")
        else:
            await processing_msg.edit_text("❌ متأسفانه نتوانستم متن را تشخیص دهم. لطفاً دوباره تلاش کنید.")
            
    except Exception as e:
        await update.message.reply_text(f"❌ خطا در پردازش فایل صوتی: {str(e)}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """پردازش فایل‌های صوتی ارسالی به عنوان سند"""
    try:
        document = update.message.document
        
        # بررسی فرمت فایل
        if not document.mime_type.startswith('audio/'):
            await update.message.reply_text("❌ لطفاً فایل صوتی ارسال کنید (MP3, WAV, OGG و غیره)")
            return
        
        # ارسال پیام در حال پردازش
        processing_msg = await update.message.reply_text("🔄 در حال پردازش فایل صوتی...")
        
        # دریافت فایل
        file = await document.get_file()
        
        # تشخیص فرمت فایل
        file_name = document.file_name or "audio"
        file_extension = os.path.splitext(file_name)[1] or ".mp3"
        
        # ایجاد فایل موقت
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_path = temp_file.name
        
        # دانلود فایل
        await file.download_to_drive(temp_path)
        
        # تبدیل صوت به متن
        text = voice_processor.transcribe_audio(temp_path)
        
        # حذف فایل موقت
        os.unlink(temp_path)
        
        # ارسال نتیجه
        if text and text.strip():
            await processing_msg.edit_text(f"📝 متن تشخیص داده شده:\n\n{text}")
        else:
            await processing_msg.edit_text("❌ متأسفانه نتوانستم متن را تشخیص دهم. لطفاً دوباره تلاش کنید.")
            
    except Exception as e:
        await update.message.reply_text(f"❌ خطا در پردازش فایل: {str(e)}")


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.AUDIO, handle_document))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

