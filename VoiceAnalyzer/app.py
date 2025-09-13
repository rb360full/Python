import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QTextEdit, QScrollArea, QProgressBar, QMessageBox, QDialog, QComboBox,
    QDialogButtonBox, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal
import subprocess
import json
import collections
import webbrowser
import tempfile
import time
import requests
from tqdm import tqdm
import zipfile
import shutil

# Whisper و تبدیل صوت به متن
import whisper

# تصحیح خودکار فارسی
try:
    from persiantools import digits
    from persiantools import characters
    PERSIAN_TOOLS_AVAILABLE = True
except ImportError:
    PERSIAN_TOOLS_AVAILABLE = False
    print("Warning: persiantools not installed. Install with: pip install persiantools")

# Google Speech-to-Text
try:
    from google.cloud import speech
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    print("Warning: google-cloud-speech not installed. Install with: pip install google-cloud-speech")

# Vosk Speech Recognition
try:
    import vosk
    import json
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Warning: vosk not installed. Install with: pip install vosk")

# Microsoft Azure Speech
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False
    print("Warning: azure-cognitiveservices-speech not installed. Install with: pip install azure-cognitiveservices-speech")

# AssemblyAI
try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    print("Warning: assemblyai not installed. Install with: pip install assemblyai")

# بررسی ffmpeg در PATH
try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except FileNotFoundError:
    raise RuntimeError("FFmpeg not found in PATH! Install it or add to PATH.")

# مسیر دیکشنری کاستوم
CUSTOM_DICT_FILE = Path("custom_dict.json")
if not CUSTOM_DICT_FILE.exists():
    with open(CUSTOM_DICT_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

# مسیر تکیه کلام‌ها
RELYING_DICT_FILE = Path("relying_dict.json")
if not RELYING_DICT_FILE.exists():
    with open(RELYING_DICT_FILE, "w", encoding="utf-8") as f:
        json.dump({"تکیه_کلام": ["خب", "آخه", "یعنی", "حالا"]}, f, ensure_ascii=False, indent=2)

# Import پنجره مدیریت دیکشنری
from custom_dict_manager import DictManager

class ModelDownloader:
    """کلاس دانلود خودکار مدل‌ها"""
    
    # URL های مدل‌های Vosk
    VOSK_MODELS = {
        "vosk_small": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "name": "vosk-model-small-en-us-0.15",
            "size": "40 MB",
            "language": "انگلیسی",
            "warning": "⚠️ فقط انگلیسی"
        },
        "vosk_large": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip", 
            "name": "vosk-model-en-us-0.22",
            "size": "1.8 GB",
            "language": "انگلیسی",
            "warning": "⚠️ فقط انگلیسی"
        },
        "vosk_persian": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-fa-0.5.zip",
            "name": "vosk-model-fa-0.5", 
            "size": "1.13 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی"
        }
    }
    
    @staticmethod
    def download_model(model_id, progress_callback=None, progress_bar_callback=None):
        """دانلود مدل Vosk"""
        if model_id not in ModelDownloader.VOSK_MODELS:
            return False, f"مدل {model_id} پشتیبانی نمی‌شود"
        
        model_info = ModelDownloader.VOSK_MODELS[model_id]
        models_dir = Path.home() / ".vosk" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / model_info["name"]
        
        # اگر مدل قبلاً دانلود شده
        if model_path.exists():
            return True, str(model_path)
        
        try:
            # دانلود فایل
            if progress_callback:
                progress_callback(f"در حال دانلود {model_info['name']} ({model_info['size']})...")
            
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            # محاسبه اندازه فایل
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            # ذخیره فایل
            zip_path = models_dir / f"{model_info['name']}.zip"
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # به‌روزرسانی progress bar
                        if progress_bar_callback and total_size > 0:
                            progress_percent = int((downloaded_size / total_size) * 100)
                            progress_bar_callback(progress_percent)
            
            # استخراج فایل
            if progress_callback:
                progress_callback("در حال استخراج مدل...")
            
            if progress_bar_callback:
                progress_bar_callback(100)  # دانلود کامل
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            
            # حذف فایل zip
            zip_path.unlink()
            
            return True, str(model_path)
            
        except Exception as e:
            return False, f"خطا در دانلود: {str(e)}"
    
    @staticmethod
    def is_model_downloaded(model_id):
        """بررسی وجود مدل"""
        if model_id not in ModelDownloader.VOSK_MODELS:
            return False
        
        model_info = ModelDownloader.VOSK_MODELS[model_id]
        models_dir = Path.home() / ".vosk" / "models"
        model_path = models_dir / model_info["name"]
        
        return model_path.exists()

class ModelSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("انتخاب مدل Whisper")
        self.setModal(True)
        self.resize(400, 200)
        
        layout = QFormLayout(self)
        
        # لیست مدل‌های Whisper
        self.model_combo = QComboBox()
        models = [
            # Whisper Models
            ("whisper_tiny", "⚠️ Whisper Tiny - خیلی ضعیف برای فارسی (75 MB)"),
            ("whisper_base", "⚠️ Whisper Base - ضعیف برای فارسی (142 MB)"),
            ("whisper_small", "✅ Whisper Small - تعادل خوب (466 MB)"),
            ("whisper_medium", "✅ Whisper Medium - دقت بالا (1.5 GB)"),
            ("whisper_large", "✅ Whisper Large - بالاترین دقت (2.9 GB)"),
            # Google Models
            ("google_standard", "✅ Google Standard - رایگان 60دقیقه/ماه (آنلاین)"),
            ("google_enhanced", "💳 Google Enhanced - پولی، دقت بالاتر (آنلاین)"),
            ("google_phone_call", "💳 Google Phone Call - پولی، مخصوص تماس‌ها (آنلاین)"),
            ("google_medical", "💳 Google Medical - پولی، اصطلاحات پزشکی (آنلاین)"),
            ("google_video", "💳 Google Video - پولی، مخصوص ویدیوها (آنلاین)"),
            # Vosk Models (کاملاً رایگان)
            ("vosk_small", "⚠️ Vosk Small - فقط انگلیسی (40 MB)"),
            ("vosk_large", "⚠️ Vosk Large - فقط انگلیسی (1.8 GB)"),
            ("vosk_persian", "✅ Vosk Persian - مخصوص فارسی (1.13 GB)"),
            # Microsoft Azure Speech (رایگان تا حدی)
            ("azure_standard", "✅ Azure Standard - رایگان 5ساعت/ماه (آنلاین)"),
            ("azure_enhanced", "💳 Azure Enhanced - پولی، دقت بالاتر (آنلاین)"),
            # AssemblyAI (رایگان تا حدی)
            ("assemblyai_standard", "✅ AssemblyAI - رایگان 3ساعت/ماه (آنلاین)"),
            ("assemblyai_enhanced", "💳 AssemblyAI Enhanced - پولی، دقت بالاتر (آنلاین)")
        ]
        
        for model_id, description in models:
            self.model_combo.addItem(f"{model_id} - {description}", model_id)
        
        # انتخاب پیش‌فرض Vosk Persian (بهترین برای فارسی - کاملاً رایگان)
        self.model_combo.setCurrentIndex(10)  # vosk_persian
        
        layout.addRow("مدل:", self.model_combo)
        
        # دکمه‌ها
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_selected_model(self):
        return self.model_combo.currentData()

def improve_persian_text(text):
    """بهبود متن فارسی با تصحیح خودکار"""
    if not PERSIAN_TOOLS_AVAILABLE:
        return text
    
    # حذف تکرارهای اضافی کلمات
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word == prev_word:
            repeat_count += 1
            if repeat_count < 3:  # حداکثر 3 تکرار مجاز
                cleaned_words.append(word)
        else:
            repeat_count = 0
            cleaned_words.append(word)
        prev_word = word
    
    text = " ".join(cleaned_words)
    
    # تصحیح کاراکترهای فارسی
    text = characters.ar_to_fa(text)  # تبدیل عربی به فارسی
    
    # تصحیح اعداد
    text = digits.en_to_fa(text)  # تبدیل اعداد انگلیسی به فارسی
    
    return text

class TranscribeThread(QThread):
    progress = Signal(int)
    finished = Signal(str)
    download_progress = Signal(int)
    download_status = Signal(str)

    def __init__(self, audio_path, model_name="vosk_persian"):
        super().__init__()
        self.audio_path = audio_path
        self.model_name = model_name
        self.model = None
        self._pause = False
        self._stop = False

    def run(self):
        try:
            self.progress.emit(5)
            
            # بارگذاری مدل انتخاب شده
            if self.model_name.startswith("whisper_"):
                whisper_model = self.model_name.replace("whisper_", "")
                self.model = whisper.load_model(whisper_model)
            elif self.model_name.startswith("google_"):
                if not GOOGLE_SPEECH_AVAILABLE:
                    self.finished.emit("Error: Google Speech-to-Text not installed. Install with: pip install google-cloud-speech")
                    return
                self.model = "google"  # نشانگر استفاده از Google
            elif self.model_name.startswith("vosk_"):
                if not VOSK_AVAILABLE:
                    self.finished.emit("Error: Vosk not installed. Install with: pip install vosk")
                    return
                self.model = "vosk"  # نشانگر استفاده از Vosk
            elif self.model_name.startswith("azure_"):
                if not AZURE_SPEECH_AVAILABLE:
                    self.finished.emit("Error: Azure Speech not installed. Install with: pip install azure-cognitiveservices-speech")
                    return
                self.model = "azure"  # نشانگر استفاده از Azure
            elif self.model_name.startswith("assemblyai_"):
                if not ASSEMBLYAI_AVAILABLE:
                    self.finished.emit("Error: AssemblyAI not installed. Install with: pip install assemblyai")
                    return
                self.model = "assemblyai"  # نشانگر استفاده از AssemblyAI
            else:
                self.finished.emit(f"Error: Unknown model {self.model_name}")
                return
                
            self.progress.emit(10)
            
            # تبدیل به WAV 16kHz Mono در فایل موقت
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name

            subprocess.run([
                "ffmpeg", "-y", "-i", str(self.audio_path),
                "-ar", "16000", "-ac", "1", 
                "-af", "highpass=f=80,lowpass=f=8000",  # فیلتر صدا برای بهبود کیفیت
                temp_wav
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.progress.emit(35)

            # تبدیل صوت به متن
            if self.model == "google":
                # استفاده از Google Speech-to-Text
                text = self.transcribe_with_google(temp_wav)
            elif self.model == "vosk":
                # استفاده از Vosk
                text = self.transcribe_with_vosk(temp_wav)
            elif self.model == "azure":
                # استفاده از Azure Speech
                text = self.transcribe_with_azure(temp_wav)
            elif self.model == "assemblyai":
                # استفاده از AssemblyAI
                text = self.transcribe_with_assemblyai(temp_wav)
            else:
                # استفاده از Whisper
                result = self.model.transcribe(
                    temp_wav,
                    language="fa",  # زبان فارسی
                    initial_prompt="این یک فایل صوتی فارسی است. لطفاً متن را با املای صحیح فارسی بنویسید.",
                    temperature=0.0,  # کاهش خلاقیت برای دقت بیشتر
                    beam_size=5,  # جستجوی بهتر برای کلمات
                    best_of=3,  # تست چندین بار و انتخاب بهترین نتیجه
                    patience=1.0,  # صبر بیشتر برای تشخیص دقیق‌تر
                    length_penalty=1.0,  # تنظیم طول متن
                    suppress_tokens=[-1],  # حذف توکن‌های اضافی
                    word_timestamps=True,  # تشخیص بهتر کلمات
                    no_speech_threshold=0.6,  # آستانه تشخیص سکوت
                    logprob_threshold=-1.0,  # آستانه احتمال کلمات
                    compression_ratio_threshold=2.4  # آستانه فشرده‌سازی
                )
                text = result["text"]
            
            # بهبود متن فارسی
            text = improve_persian_text(text)
            
            self.progress.emit(75)

            # لغات کاستوم
            with open(CUSTOM_DICT_FILE, encoding="utf-8") as f:
                custom_dict = json.load(f)
            for wrong, correct in custom_dict.items():
                text = text.replace(wrong, correct)

            # آمار
            words = text.split()
            word_count = len(words)

            # تکیه کلام‌ها
            with open(RELYING_DICT_FILE, encoding="utf-8") as f:
                relying_data = json.load(f)
            relying_words = relying_data.get("تکیه_کلام", [])
            detected_relying = [w for w in words if w in relying_words]

            counter = collections.Counter(words)
            most_common = counter.most_common(10)

            stats = "\n\n--- Statistics ---\n"
            stats += f"Word Count: {word_count}\n"
            stats += f"Detected Pauses/Tic Words: {', '.join(detected_relying)}\n"
            stats += "Top 10 Frequent Words:\n"
            for w, c in most_common:
                stats += f"{w}: {c}\n"

            text += stats
            self.progress.emit(100)
            self.finished.emit(text)

            # حذف فایل موقت
            os.remove(temp_wav)

        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")
    
    def transcribe_with_google(self, audio_file):
        """تبدیل صوت به متن با Google Speech-to-Text"""
        try:
            client = speech.SpeechClient()
            
            # خواندن فایل صوتی
            with open(audio_file, "rb") as audio_file_content:
                content = audio_file_content.read()
            
            # تنظیمات تشخیص
            audio = speech.RecognitionAudio(content=content)
            
            # انتخاب مدل Google
            if self.model_name == "google_enhanced":
                model = "phone_call"  # مدل Enhanced
            elif self.model_name == "google_phone_call":
                model = "phone_call"
            elif self.model_name == "google_medical":
                model = "medical_dictation"
            elif self.model_name == "google_video":
                model = "video"
            else:
                model = "default"  # مدل Standard
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="fa-IR",  # فارسی ایران
                model=model,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True
            )
            
            # تشخیص
            response = client.recognize(config=config, audio=audio)
            
            # ترکیب نتایج
            text = ""
            for result in response.results:
                text += result.alternatives[0].transcript + " "
            
            return text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "credentials" in error_msg.lower():
                return f"""Google Speech Error: نیاز به تنظیم credentials

برای استفاده از Google Speech-to-Text:

1. به https://console.cloud.google.com بروید
2. پروژه جدید بسازید یا انتخاب کنید
3. API Speech-to-Text را فعال کنید
4. Service Account بسازید و JSON key دانلود کنید
5. متغیر محیطی تنظیم کنید:
   set GOOGLE_APPLICATION_CREDENTIALS=path/to/your/key.json

یا از Whisper استفاده کنید (آفلاین و رایگان)
"""
            else:
                return f"Google Speech Error: {error_msg}"
    
    def transcribe_with_vosk(self, audio_file):
        """تبدیل صوت به متن با Vosk"""
        try:
            # بررسی و دانلود مدل Vosk
            if not ModelDownloader.is_model_downloaded(self.model_name):
                # دانلود مدل با progress bar
                def progress_callback(message):
                    self.download_status.emit(message)
                
                def progress_bar_callback(percent):
                    self.download_progress.emit(percent)
                
                success, result = ModelDownloader.download_model(
                    self.model_name, 
                    progress_callback=progress_callback,
                    progress_bar_callback=progress_bar_callback
                )
                if not success:
                    return f"Vosk Error: {result}"
            
            # دریافت مسیر مدل
            model_path = self.get_vosk_model_path()
            if not model_path:
                return "Vosk Error: مدل Vosk یافت نشد."
            
            # بارگذاری مدل
            model = vosk.Model(model_path)
            rec = vosk.KaldiRecognizer(model, 16000)
            
            # خواندن فایل صوتی
            with open(audio_file, "rb") as f:
                data = f.read()
            
            # تشخیص
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
            else:
                result = json.loads(rec.FinalResult())
                text = result.get("text", "")
            
            return text.strip()
            
        except Exception as e:
            return f"Vosk Error: {str(e)}"
    
    def get_vosk_model_path(self):
        """دریافت مسیر مدل Vosk"""
        if self.model_name not in ModelDownloader.VOSK_MODELS:
            return None
        
        model_info = ModelDownloader.VOSK_MODELS[self.model_name]
        models_dir = Path.home() / ".vosk" / "models"
        model_path = models_dir / model_info["name"]
        
        if model_path.exists():
            return str(model_path)
        
        return None
    
    def transcribe_with_azure(self, audio_file):
        """تبدیل صوت به متن با Azure Speech"""
        try:
            # تنظیمات Azure (نیاز به API Key)
            speech_key = os.getenv("AZURE_SPEECH_KEY")
            service_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
            
            if not speech_key:
                return """Azure Speech Error: نیاز به تنظیم API Key

برای استفاده از Azure Speech:

1. به https://portal.azure.com بروید
2. Cognitive Services > Speech را ایجاد کنید
3. API Key و Region را کپی کنید
4. متغیرهای محیطی تنظیم کنید:
   set AZURE_SPEECH_KEY=your_key_here
   set AZURE_SPEECH_REGION=your_region_here

یا از Vosk استفاده کنید (کاملاً رایگان)
"""
            
            # تنظیم speech config
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            speech_config.speech_recognition_language = "fa-IR"  # فارسی ایران
            
            # تنظیم audio config
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
            
            # ایجاد speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            # تشخیص
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return "Azure Speech: تشخیص انجام نشد"
            else:
                return f"Azure Speech Error: {result.reason}"
                
        except Exception as e:
            return f"Azure Speech Error: {str(e)}"
    
    def transcribe_with_assemblyai(self, audio_file):
        """تبدیل صوت به متن با AssemblyAI"""
        try:
            # تنظیمات AssemblyAI (نیاز به API Key)
            api_key = os.getenv("ASSEMBLYAI_API_KEY")
            
            if not api_key:
                return """AssemblyAI Error: نیاز به تنظیم API Key

برای استفاده از AssemblyAI:

1. به https://www.assemblyai.com بروید
2. حساب کاربری بسازید
3. API Key را کپی کنید
4. متغیر محیطی تنظیم کنید:
   set ASSEMBLYAI_API_KEY=your_key_here

یا از Vosk استفاده کنید (کاملاً رایگان)
"""
            
            # تنظیم API key
            aai.settings.api_key = api_key
            
            # ایجاد transcriber
            transcriber = aai.Transcriber()
            
            # آپلود فایل
            transcript = transcriber.transcribe(audio_file)
            
            if transcript.status == aai.TranscriptStatus.completed:
                return transcript.text
            else:
                return f"AssemblyAI Error: {transcript.status}"
                
        except Exception as e:
            return f"AssemblyAI Error: {str(e)}"


class VoiceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Analyzer")
        self.resize(900, 600)  # بزرگ‌تر کردن سایز اپ
        self.layout = QVBoxLayout(self)

        self.label = QLabel("Select an audio file:")
        self.layout.addWidget(self.label)
        
        self.model_label = QLabel("Selected Model: vosk_persian (default)")
        self.model_label.setStyleSheet("color: blue; font-weight: bold;")
        self.layout.addWidget(self.model_label)

        # دکمه ها
        self.btn_select = QPushButton("Choose File")
        self.btn_select.setMinimumHeight(40)
        self.btn_select.clicked.connect(self.select_file)
        self.layout.addWidget(self.btn_select)

        self.btn_start_pause = QPushButton("Start / Pause")
        self.btn_start_pause.setMinimumHeight(40)
        self.btn_start_pause.clicked.connect(self.toggle_start_pause)
        self.layout.addWidget(self.btn_start_pause)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.layout.addWidget(self.btn_stop)

        self.btn_open_text = QPushButton("Open Text")
        self.btn_open_text.setMinimumHeight(40)
        self.btn_open_text.clicked.connect(self.open_text_file)
        self.layout.addWidget(self.btn_open_text)

        self.btn_manage_dict = QPushButton("Manage Dictionary")
        self.btn_manage_dict.setMinimumHeight(40)
        self.btn_manage_dict.clicked.connect(self.open_dict_manager)
        self.layout.addWidget(self.btn_manage_dict)

        self.btn_google_setup = QPushButton("Google Setup Guide")
        self.btn_google_setup.setMinimumHeight(40)
        self.btn_google_setup.setStyleSheet("background-color: #4285f4; color: white;")
        self.btn_google_setup.clicked.connect(self.show_google_setup_guide)
        self.layout.addWidget(self.btn_google_setup)

        self.btn_download_models = QPushButton("Download Models")
        self.btn_download_models.setMinimumHeight(40)
        self.btn_download_models.setStyleSheet("background-color: #ff6b35; color: white;")
        self.btn_download_models.clicked.connect(self.show_download_models_dialog)
        self.layout.addWidget(self.btn_download_models)

        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        
        # Progress bar برای دانلود
        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        self.layout.addWidget(self.download_progress)
        
        # Status label برای دانلود
        self.download_status = QLabel("")
        self.download_status.setVisible(False)
        self.download_status.setStyleSheet("color: blue; font-weight: bold;")
        self.layout.addWidget(self.download_status)

        self.scroll = QScrollArea()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFontPointSize(14)  # بزرگ‌تر شدن متن نمایشی
        self.scroll.setWidget(self.text_edit)
        self.scroll.setWidgetResizable(True)
        self.layout.addWidget(self.scroll)

        self.audio_path = None
        self.output_file = None
        self.thread = None
        self.dict_manager_window = None
        self.selected_model = "vosk_persian"  # مدل پیش‌فرض

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.mp3 *.wav *.m4a *.ogg *.flac)"
        )
        if file_path:
            self.audio_path = file_path
            self.label.setText(f"Selected: {file_path}")
            self.text_edit.clear()
            self.output_file = None
            self.thread = None  # ریست کردن thread برای فایل جدید

    def start_transcription(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Warning", "No audio file selected!")
            return

        # نمایش dialog انتخاب مدل
        dialog = ModelSelectionDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.selected_model = dialog.get_selected_model()
            self.model_label.setText(f"Selected Model: {self.selected_model}")
            
            # هشدار برای مدل‌های ضعیف
            if self.selected_model in ["whisper_tiny", "whisper_base"]:
                reply = QMessageBox.question(
                    self, "هشدار", 
                    f"مدل {self.selected_model} برای فارسی ضعیف است و ممکن است نتایج نامناسبی بدهد.\nآیا مطمئن هستید؟",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return  # کاربر انصراف داد
            
            # هشدار برای مدل‌های انگلیسی Vosk
            if self.selected_model in ["vosk_small", "vosk_large"]:
                reply = QMessageBox.question(
                    self, "هشدار زبان", 
                    f"مدل {self.selected_model} فقط برای انگلیسی طراحی شده و فارسی را به انگلیسی تبدیل می‌کند!\n\nبرای فارسی از 'Vosk Persian' استفاده کنید.\nآیا مطمئن هستید؟",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return  # کاربر انصراف داد
            
            # هشدار برای مدل‌های پولی
            paid_models = [
                "google_enhanced", "google_phone_call", "google_medical", "google_video",
                "azure_enhanced", "assemblyai_enhanced"
            ]
            if self.selected_model in paid_models:
                reply = QMessageBox.question(
                    self, "هشدار هزینه", 
                    f"مدل {self.selected_model} پولی است و هزینه بر اساس استفاده محاسبه می‌شود.\nآیا مطمئن هستید؟",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return  # کاربر انصراف داد
        else:
            return  # کاربر cancel کرد

        self.text_edit.clear()
        self.output_file = None
        self.thread = TranscribeThread(self.audio_path, self.selected_model)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.display_result)
        self.thread.download_progress.connect(self.update_download_progress)
        self.thread.download_status.connect(self.update_download_status)
        self.thread.start()

    def update_download_progress(self, percent):
        """به‌روزرسانی progress bar دانلود"""
        self.download_progress.setValue(percent)
        if percent == 100:
            self.download_progress.setVisible(False)
            self.download_status.setVisible(False)
    
    def update_download_status(self, message):
        """به‌روزرسانی status دانلود"""
        self.download_status.setText(message)
        self.download_status.setVisible(True)
        self.download_progress.setVisible(True)

    def display_result(self, text):
        self.text_edit.setPlainText(text)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Text", "", "Text Files (*.txt)"
        )
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            self.output_file = save_path

    def open_text_file(self):
        if self.output_file and os.path.exists(self.output_file):
            webbrowser.open(self.output_file)
        else:
            QMessageBox.warning(self, "Warning", "No saved text file to open!")

    def toggle_start_pause(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Warning", "No audio file selected!")
            return

        if not self.thread:
            self.start_transcription()
        else:
            QMessageBox.information(self, "Info", "Pause / Resume toggle clicked (future implementation).")

    def stop_processing(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "Info", "Stop clicked (future implementation).")

    def open_dict_manager(self):
        if not self.dict_manager_window:
            self.dict_manager_window = DictManager()
        self.dict_manager_window.show()
        self.dict_manager_window.raise_()
        self.dict_manager_window.activateWindow()

    def show_google_setup_guide(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QTextCursor
        
        dialog = QDialog(self)
        dialog.setWindowTitle("راهنمای تنظیم Google Speech")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # متن راهنما
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h2>راهنمای تنظیم Google Speech-to-Text</h2>
        
        <h3>🔧 مراحل تنظیم:</h3>
        
        <h4>1️⃣ ایجاد پروژه Google Cloud:</h4>
        <p>• به <a href="https://console.cloud.google.com">Google Cloud Console</a> بروید</p>
        <p>• روی "New Project" کلیک کنید</p>
        <p>• نام پروژه وارد کنید و "Create" بزنید</p>
        
        <h4>2️⃣ فعال‌سازی API:</h4>
        <p>• در منو، "APIs & Services" > "Library" را انتخاب کنید</p>
        <p>• "Speech-to-Text API" را جستجو کنید</p>
        <p>• روی آن کلیک کرده و "Enable" بزنید</p>
        
        <h4>3️⃣ ایجاد Service Account:</h4>
        <p>• "APIs & Services" > "Credentials" بروید</p>
        <p>• "Create Credentials" > "Service Account" را انتخاب کنید</p>
        <p>• نام و توضیحات وارد کنید</p>
        <p>• Role: "Cloud Speech Client" انتخاب کنید</p>
        
        <h4>4️⃣ دانلود کلید JSON:</h4>
        <p>• روی Service Account کلیک کنید</p>
        <p>• تب "Keys" > "Add Key" > "Create new key"</p>
        <p>• JSON را انتخاب کرده و دانلود کنید</p>
        
        <h4>5️⃣ تنظیم متغیر محیطی:</h4>
        <p>• فایل JSON را در پوشه امن قرار دهید</p>
        <p>• در Command Prompt اجرا کنید:</p>
        <p><code>set GOOGLE_APPLICATION_CREDENTIALS=C:\\path\\to\\your\\key.json</code></p>
        
        <h3>💡 نکات مهم:</h3>
        <p>• Google Speech 60 دقیقه در ماه رایگان است</p>
        <p>• برای استفاده بیشتر، هزینه بر اساس استفاده محاسبه می‌شود</p>
        <p>• Whisper کاملاً رایگان و آفلاین است</p>
        
        <h3>🔗 لینک‌های مفید:</h3>
        <p>• <a href="https://console.cloud.google.com">Google Cloud Console</a></p>
        <p>• <a href="https://cloud.google.com/speech-to-text/docs">مستندات Speech-to-Text</a></p>
        <p>• <a href="https://cloud.google.com/docs/authentication/external/set-up-adc">راهنمای Authentication</a></p>
        """)
        
        layout.addWidget(text_edit)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        open_console_btn = QPushButton("باز کردن Google Cloud Console")
        open_console_btn.setStyleSheet("background-color: #4285f4; color: white; padding: 8px;")
        open_console_btn.clicked.connect(lambda: webbrowser.open("https://console.cloud.google.com"))
        
        open_docs_btn = QPushButton("مستندات Speech-to-Text")
        open_docs_btn.setStyleSheet("background-color: #34a853; color: white; padding: 8px;")
        open_docs_btn.clicked.connect(lambda: webbrowser.open("https://cloud.google.com/speech-to-text/docs"))
        
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(open_console_btn)
        button_layout.addWidget(open_docs_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def show_download_models_dialog(self):
        """نمایش dialog دانلود مدل‌ها"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QLabel
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("دانلود مدل‌های Vosk")
        dialog.setModal(True)
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # توضیحات
        info_label = QLabel("مدل‌های Vosk برای تشخیص گفتار (کاملاً رایگان و آفلاین):")
        layout.addWidget(info_label)
        
        # لیست مدل‌ها
        model_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.VOSK_MODELS.items():
            status = "✅ دانلود شده" if ModelDownloader.is_model_downloaded(model_id) else "❌ دانلود نشده"
            item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
            model_list.addItem(item_text)
        
        layout.addWidget(model_list)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        download_btn = QPushButton("دانلود مدل انتخاب شده")
        download_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        download_btn.clicked.connect(lambda: self.download_selected_model(dialog, model_list))
        
        download_all_btn = QPushButton("دانلود همه مدل‌ها")
        download_all_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        download_all_btn.clicked.connect(lambda: self.download_all_models(dialog))
        
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(download_btn)
        button_layout.addWidget(download_all_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def download_selected_model(self, dialog, model_list):
        """دانلود مدل انتخاب شده"""
        current_row = model_list.currentRow()
        if current_row == -1:
            QMessageBox.warning(dialog, "هشدار", "لطفاً یک مدل انتخاب کنید.")
            return
        
        model_ids = list(ModelDownloader.VOSK_MODELS.keys())
        model_id = model_ids[current_row]
        
        if ModelDownloader.is_model_downloaded(model_id):
            QMessageBox.information(dialog, "اطلاعات", "این مدل قبلاً دانلود شده است.")
            return
        
        # دانلود مدل
        QMessageBox.information(dialog, "شروع دانلود", f"دانلود {model_id} شروع شد. لطفاً صبر کنید...")
        
        success, result = ModelDownloader.download_model(model_id)
        
        if success:
            QMessageBox.information(dialog, "موفق", f"مدل {model_id} با موفقیت دانلود شد!")
            dialog.accept()
        else:
            QMessageBox.critical(dialog, "خطا", f"خطا در دانلود: {result}")
    
    def download_all_models(self, dialog):
        """دانلود همه مدل‌ها"""
        reply = QMessageBox.question(
            dialog, "تأیید", 
            "آیا می‌خواهید همه مدل‌های Vosk را دانلود کنید؟\nاین کار ممکن است زمان زیادی ببرد.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(dialog, "شروع دانلود", "دانلود همه مدل‌ها شروع شد. لطفاً صبر کنید...")
            
            for model_id in ModelDownloader.VOSK_MODELS.keys():
                if not ModelDownloader.is_model_downloaded(model_id):
                    success, result = ModelDownloader.download_model(model_id)
                    if not success:
                        QMessageBox.critical(dialog, "خطا", f"خطا در دانلود {model_id}: {result}")
                        return
            
            QMessageBox.information(dialog, "موفق", "همه مدل‌ها با موفقیت دانلود شدند!")
            dialog.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec())
