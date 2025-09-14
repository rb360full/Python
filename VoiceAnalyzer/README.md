# 🎤 VoiceAnalyzer - تحلیلگر صوتی پیشرفته

یک برنامه پیشرفته برای تبدیل صوت به متن با پشتیبانی از مدل‌های مختلف Speech-to-Text.

## ✨ ویژگی‌ها

- **35+ مدل مختلف** برای تشخیص گفتار
- **پشتیبانی از فارسی و انگلیسی**
- **مدل‌های آفلاین و آنلاین**
- **نوار پیشرفت زیبا** برای دانلود مدل‌ها
- **ذخیره تنظیمات** کاربر
- **رابط کاربری ساده** و زیبا

## 🚀 نصب سریع

### روش 1: نصب خودکار
```bash
python install_dependencies.py
```

### روش 2: نصب دستی
```bash
pip install -r requirements.txt
```

## 🎯 مدل‌های پشتیبانی شده

### 📱 مدل‌های آفلاین (کاملاً رایگان)

#### Vosk Models
- ✅ **Vosk Persian** - مخصوص فارسی (1.13 GB)
- ⚠️ **Vosk Small** - فقط انگلیسی (40 MB)
- ⚠️ **Vosk Large** - فقط انگلیسی (1.8 GB)

#### Whisper Models
- ⚠️ **Whisper Tiny** - ضعیف برای فارسی (75 MB)
- ⚠️ **Whisper Base** - ضعیف برای فارسی (142 MB)
- ✅ **Whisper Small** - تعادل خوب (466 MB)
- ✅ **Whisper Medium** - دقت بالا (1.5 GB)
- ✅ **Whisper Large** - بالاترین دقت (2.9 GB)
- ✅ **Whisper Large V2** - جدیدترین نسخه (2.9 GB)
- ✅ **Whisper Large V3** - جدیدترین نسخه (2.9 GB)

#### Hugging Face Models
- ✅ **Wav2Vec2 Persian** - مخصوص فارسی (1.2 GB)
- ✅ **Whisper Persian** - مخصوص فارسی (1.5 GB)
- ⚠️ **Whisper Tiny HF** - ضعیف برای فارسی (75 MB)
- ⚠️ **Whisper Base HF** - ضعیف برای فارسی (142 MB)
- ✅ **Whisper Small HF** - تعادل خوب (466 MB)
- ✅ **Whisper Medium HF** - دقت بالا (1.5 GB)
- ✅ **Whisper Large HF** - بالاترین دقت (2.9 GB)

#### Silero STT
- ⚠️ **Silero STT English** - فقط انگلیسی (50 MB)
- ✅ **Silero STT Multilingual** - پشتیبانی از فارسی (200 MB)

#### Kaldi
- ✅ **Kaldi Persian** - مخصوص فارسی (500 MB)
- ⚠️ **Kaldi English** - فقط انگلیسی (300 MB)

### 🌐 مدل‌های آنلاین

#### SpeechRecognition
- 🌐 **Google Speech** - رایگان 60دقیقه/ماه
- ⚠️ **CMU Sphinx** - فقط انگلیسی (100 MB)
- 🌐 **Wit.ai** - رایگان تا حدی
- 🌐 **Azure Speech** - رایگان 5ساعت/ماه
- 🌐 **Bing Speech** - رایگان تا حدی
- 🌐 **Houndify** - رایگان تا حدی
- 🌐 **IBM Speech** - رایگان تا حدی

#### سرویس‌های بومی ایرانی
- 🇮🇷 **Arvan Cloud Speech** - سرویس ایرانی
- 🇮🇷 **Fanap Speech API** - سرویس ایرانی
- 🇮🇷 **Parsijoo Speech** - سرویس ایرانی

## 🔧 تنظیمات API Keys

### Google Speech
```bash
set GOOGLE_APPLICATION_CREDENTIALS=path/to/your/key.json
```

### Azure Speech
```bash
set AZURE_SPEECH_KEY=your_key_here
set AZURE_SPEECH_REGION=your_region_here
```

### AssemblyAI
```bash
set ASSEMBLYAI_API_KEY=your_key_here
```

## 🎮 نحوه استفاده

1. **اجرای برنامه:**
   ```bash
   python app.py
   ```

2. **انتخاب فایل صوتی:**
   - روی "Choose File" کلیک کنید
   - فایل صوتی خود را انتخاب کنید

3. **انتخاب مدل:**
   - روی "Start / Pause" کلیک کنید
   - مدل مورد نظر را انتخاب کنید

4. **تغییر مدل:**
   - روی "Change Model" کلیک کنید
   - مدل جدید را انتخاب کنید

5. **دانلود مدل‌ها:**
   - روی "Download Models" کلیک کنید
   - مدل‌های مورد نظر را دانلود کنید

## 📁 ساختار فایل‌ها

```
VoiceAnalyzer/
├── app.py                    # فایل اصلی برنامه
├── custom_dict_manager.py    # مدیریت دیکشنری کاستوم
├── install_dependencies.py   # اسکریپت نصب وابستگی‌ها
├── requirements.txt          # لیست وابستگی‌ها
├── README.md                 # این فایل
├── voice_analyzer_config.json # تنظیمات برنامه
├── custom_dict.json          # دیکشنری کاستوم
└── relying_dict.json         # تکیه کلام‌ها
```

## 🛠️ عیب‌یابی

### خطای "Module not found"
```bash
pip install -r requirements.txt
```

### خطای "FFmpeg not found"
- FFmpeg را نصب کنید و به PATH اضافه کنید

### خطای "Model not found"
- مدل مورد نظر را از "Download Models" دانلود کنید

## 📝 نکات مهم

- **مدل‌های فارسی**: Vosk Persian، Wav2Vec2 Persian، Whisper Persian
- **مدل‌های انگلیسی**: Vosk Small/Large، Silero STT English
- **مدل‌های چندزبانه**: Whisper، Silero STT Multilingual
- **مدل‌های آنلاین**: نیاز به API Key دارند
- **مدل‌های آفلاین**: کاملاً رایگان و بدون نیاز به اینترنت

## 🤝 مشارکت

برای گزارش باگ یا پیشنهاد ویژگی جدید، لطفاً issue ایجاد کنید.

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

---

**🎉 از VoiceAnalyzer لذت ببرید!**
