import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QTextEdit, QScrollArea, QProgressBar, QMessageBox, QDialog, QComboBox,
    QDialogButtonBox, QFormLayout, QHBoxLayout, QTabWidget, QListWidget, 
    QListWidgetItem, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
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

# Hugging Face Transformers
try:
    from transformers import pipeline, AutoModelForCTC, AutoProcessor
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

# SpeechRecognition
try:
    import speech_recognition as sr
    SPEECHRECOGNITION_AVAILABLE = True
except ImportError:
    SPEECHRECOGNITION_AVAILABLE = False
    print("Warning: SpeechRecognition not installed. Install with: pip install SpeechRecognition")

# Silero STT
try:
    import torch
    import torchaudio
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    print("Warning: torchaudio not installed. Install with: pip install torchaudio")

# Kaldi
try:
    import kaldi_io
    KALDI_AVAILABLE = True
except ImportError:
    KALDI_AVAILABLE = False
    print("Warning: kaldi-io not installed. Install with: pip install kaldi-io")

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

# مسیر فایل تنظیمات
CONFIG_FILE = Path("voice_analyzer_config.json")
if not CONFIG_FILE.exists():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": "vosk_persian",
            "last_audio_path": "",
            "window_geometry": None,
            "preferred_language": "fa"
        }, f, ensure_ascii=False, indent=2)

# Import پنجره مدیریت دیکشنری
from custom_dict_manager import DictManager

class ConfigManager:
    """مدیریت تنظیمات برنامه"""
    
    @staticmethod
    def load_config():
        """بارگذاری تنظیمات از فایل"""
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"خطا در بارگذاری تنظیمات: {e}")
            return {
                "selected_model": "vosk_persian",
                "last_audio_path": "",
                "window_geometry": None,
                "preferred_language": "fa"
            }
    
    @staticmethod
    def save_config(config):
        """ذخیره تنظیمات در فایل"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"خطا در ذخیره تنظیمات: {e}")
            return False
    
    @staticmethod
    def update_model(model_name):
        """به‌روزرسانی مدل انتخاب شده"""
        config = ConfigManager.load_config()
        config["selected_model"] = model_name
        return ConfigManager.save_config(config)
    
    @staticmethod
    def update_audio_path(audio_path):
        """به‌روزرسانی مسیر آخرین فایل صوتی"""
        config = ConfigManager.load_config()
        config["last_audio_path"] = audio_path
        return ConfigManager.save_config(config)
    
    @staticmethod
    def update_window_geometry(geometry):
        """به‌روزرسانی موقعیت و اندازه پنجره"""
        config = ConfigManager.load_config()
        config["window_geometry"] = geometry
        return ConfigManager.save_config(config)

class ModelDownloader:
    """کلاس دانلود خودکار مدل‌ها"""
    
    # مدل‌های قابل دانلود
    DOWNLOADABLE_MODELS = {
        # Vosk Models (فقط فارسی و انگلیسی)
        "vosk_persian": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-fa-0.5.zip",
            "name": "vosk-model-fa-0.5", 
            "size": "1.13 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی",
            "type": "Vosk"
        },
        "vosk_small": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "name": "vosk-model-small-en-us-0.15",
            "size": "40 MB",
            "language": "انگلیسی",
            "warning": "⚠️ فقط انگلیسی",
            "type": "Vosk"
        },
        "vosk_large": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip", 
            "name": "vosk-model-en-us-0.22",
            "size": "1.8 GB",
            "language": "انگلیسی",
            "warning": "⚠️ فقط انگلیسی",
            "type": "Vosk"
        },
        
        # Whisper Models
        "whisper_tiny": {
            "url": "whisper://tiny",
            "name": "whisper-tiny",
            "size": "75 MB",
            "language": "چند زبانه",
            "warning": "⚠️ ضعیف برای فارسی",
            "type": "Whisper"
        },
        "whisper_base": {
            "url": "whisper://base",
            "name": "whisper-base",
            "size": "142 MB",
            "language": "چند زبانه",
            "warning": "⚠️ ضعیف برای فارسی",
            "type": "Whisper"
        },
        "whisper_small": {
            "url": "whisper://small",
            "name": "whisper-small",
            "size": "466 MB",
            "language": "چند زبانه",
            "warning": "✅ تعادل خوب",
            "type": "Whisper"
        },
        "whisper_medium": {
            "url": "whisper://medium",
            "name": "whisper-medium",
            "size": "1.5 GB",
            "language": "چند زبانه",
            "warning": "✅ دقت بالا",
            "type": "Whisper"
        },
        "whisper_large": {
            "url": "whisper://large",
            "name": "whisper-large",
            "size": "2.9 GB",
            "language": "چند زبانه",
            "warning": "✅ بالاترین دقت",
            "type": "Whisper"
        },
        "whisper_large_v2": {
            "url": "whisper://large-v2",
            "name": "whisper-large-v2",
            "size": "2.9 GB",
            "language": "چند زبانه",
            "warning": "✅ جدیدترین نسخه",
            "type": "Whisper"
        },
        "whisper_large_v3": {
            "url": "whisper://large-v3",
            "name": "whisper-large-v3",
            "size": "2.9 GB",
            "language": "چند زبانه",
            "warning": "✅ جدیدترین نسخه",
            "type": "Whisper"
        },
        
        # Hugging Face Transformers
        "hf_wav2vec2_persian": {
            "url": "huggingface://m3hrdadfi/wav2vec2-large-xlsr-53-persian",
            "name": "wav2vec2-large-xlsr-53-persian",
            "size": "1.2 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی - Hugging Face",
            "type": "HuggingFace"
        },
        "hf_whisper_persian": {
            "url": "huggingface://m3hrdadfi/whisper-persian",
            "name": "whisper-persian",
            "size": "1.5 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی - Hugging Face",
            "type": "HuggingFace"
        },
        "hf_wav2vec2_persian_alt": {
            "url": "huggingface://facebook/wav2vec2-large-xlsr-53",
            "name": "wav2vec2-large-xlsr-53",
            "size": "1.2 GB",
            "language": "چند زبانه",
            "warning": "⚠️ چند زبانه - نیاز به fine-tuning برای فارسی",
            "type": "HuggingFace"
        },
        "hf_whisper_tiny": {
            "url": "huggingface://openai/whisper-tiny",
            "name": "whisper-tiny-hf",
            "size": "75 MB",
            "language": "چند زبانه",
            "warning": "⚠️ ضعیف برای فارسی",
            "type": "HuggingFace"
        },
        "hf_whisper_base": {
            "url": "huggingface://openai/whisper-base",
            "name": "whisper-base-hf",
            "size": "142 MB",
            "language": "چند زبانه",
            "warning": "⚠️ ضعیف برای فارسی",
            "type": "HuggingFace"
        },
        "hf_whisper_small": {
            "url": "huggingface://openai/whisper-small",
            "name": "whisper-small-hf",
            "size": "466 MB",
            "language": "چند زبانه",
            "warning": "✅ تعادل خوب",
            "type": "HuggingFace"
        },
        "hf_whisper_medium": {
            "url": "huggingface://openai/whisper-medium",
            "name": "whisper-medium-hf",
            "size": "1.5 GB",
            "language": "چند زبانه",
            "warning": "✅ دقت بالا",
            "type": "HuggingFace"
        },
        "hf_whisper_large": {
            "url": "huggingface://openai/whisper-large-v2",
            "name": "whisper-large-v2-hf",
            "size": "2.9 GB",
            "language": "چند زبانه",
            "warning": "✅ بالاترین دقت",
            "type": "HuggingFace"
        },
        
        # SpeechRecognition
        "speechrecognition_google": {
            "url": "speechrecognition://google",
            "name": "Google Speech Recognition",
            "size": "0 MB",
            "language": "چند زبانه",
            "warning": "🌐 آنلاین - رایگان 60دقیقه/ماه",
            "type": "SpeechRecognition"
        },
        "speechrecognition_sphinx": {
            "url": "speechrecognition://sphinx",
            "name": "CMU Sphinx",
            "size": "100 MB",
            "language": "انگلیسی",
            "warning": "⚠️ فقط انگلیسی",
            "type": "SpeechRecognition"
        },
        "speechrecognition_wit": {
            "url": "speechrecognition://wit",
            "name": "Wit.ai",
            "size": "0 MB",
            "language": "چند زبانه",
            "warning": "🌐 آنلاین - رایگان تا حدی",
            "type": "SpeechRecognition"
        },
        "speechrecognition_azure": {
            "url": "speechrecognition://azure",
            "name": "Azure Speech",
            "size": "0 MB",
            "language": "چند زبانه",
            "warning": "🌐 آنلاین - رایگان 5ساعت/ماه",
            "type": "SpeechRecognition"
        },
        "speechrecognition_bing": {
            "url": "speechrecognition://bing",
            "name": "Bing Speech",
            "size": "0 MB",
            "language": "چند زبانه",
            "warning": "🌐 آنلاین - رایگان تا حدی",
            "type": "SpeechRecognition"
        },
        "speechrecognition_houndify": {
            "url": "speechrecognition://houndify",
            "name": "Houndify",
            "size": "0 MB",
            "language": "چند زبانه",
            "warning": "🌐 آنلاین - رایگان تا حدی",
            "type": "SpeechRecognition"
        },
        "speechrecognition_ibm": {
            "url": "speechrecognition://ibm",
            "name": "IBM Speech to Text",
            "size": "0 MB",
            "language": "چند زبانه",
            "warning": "🌐 آنلاین - رایگان تا حدی",
            "type": "SpeechRecognition"
        },
        
        # Silero STT
        "silero_stt_en": {
            "url": "silero://stt_en",
            "name": "Silero STT English",
            "size": "50 MB",
            "language": "انگلیسی",
            "warning": "⚠️ فقط انگلیسی",
            "type": "Silero"
        },
        "silero_stt_multilingual": {
            "url": "silero://stt_multilingual",
            "name": "Silero STT Multilingual",
            "size": "200 MB",
            "language": "چند زبانه",
            "warning": "✅ پشتیبانی از فارسی",
            "type": "Silero"
        },
        
        # Kaldi
        "kaldi_persian": {
            "url": "kaldi://persian",
            "name": "Kaldi Persian Model",
            "size": "500 MB",
            "language": "فارسی",
            "warning": "🚧 در حال توسعه - از Vosk Persian استفاده کنید",
            "type": "Kaldi"
        },
        "kaldi_english": {
            "url": "kaldi://english",
            "name": "Kaldi English Model",
            "size": "300 MB",
            "language": "انگلیسی",
            "warning": "🚧 در حال توسعه - از Vosk استفاده کنید",
            "type": "Kaldi"
        },
        
        # سرویس‌های بومی ایرانی
        "iranian_arvan": {
            "url": "arvan://speech",
            "name": "Arvan Cloud Speech",
            "size": "0 MB",
            "language": "فارسی",
            "warning": "🇮🇷 سرویس ایرانی - آنلاین",
            "type": "Iranian"
        },
        "iranian_fanap": {
            "url": "fanap://speech",
            "name": "Fanap Speech API",
            "size": "0 MB",
            "language": "فارسی",
            "warning": "🇮🇷 سرویس ایرانی - آنلاین",
            "type": "Iranian"
        },
        "iranian_parsijoo": {
            "url": "parsijoo://speech",
            "name": "Parsijoo Speech",
            "size": "0 MB",
            "language": "فارسی",
            "warning": "🇮🇷 سرویس ایرانی - آنلاین",
            "type": "Iranian"
        }
    }
    
    # برای سازگاری با کد قبلی
    VOSK_MODELS = {
        key: value for key, value in DOWNLOADABLE_MODELS.items() 
        if value["type"] == "Vosk"
    }
    
    @staticmethod
    def download_model(model_id, progress_callback=None, progress_bar_callback=None):
        """دانلود مدل"""
        if model_id not in ModelDownloader.DOWNLOADABLE_MODELS:
            return False, f"مدل {model_id} پشتیبانی نمی‌شود"
        
        model_info = ModelDownloader.DOWNLOADABLE_MODELS[model_id]
        
        # برای مدل‌های Whisper
        if model_info["type"] == "Whisper":
            return ModelDownloader._download_whisper_model(model_id, model_info, progress_callback, progress_bar_callback)
        
        # برای مدل‌های Vosk
        elif model_info["type"] == "Vosk":
            return ModelDownloader._download_vosk_model(model_id, model_info, progress_callback, progress_bar_callback)
        
        return False, f"نوع مدل {model_info['type']} پشتیبانی نمی‌شود"
    
    @staticmethod
    def _download_whisper_model(model_id, model_info, progress_callback=None, progress_bar_callback=None):
        """دانلود مدل Whisper"""
        try:
            if progress_callback:
                progress_callback(f"در حال دانلود {model_info['name']} ({model_info['size']})...")
            
            # Whisper خودش مدل‌ها رو دانلود می‌کنه
            model_name = model_id.replace("whisper_", "")
            # تبدیل نام مدل برای Whisper
            if model_name == "large_v2":
                model_name = "large-v2"
            elif model_name == "large_v3":
                model_name = "large-v3"
            
            model = whisper.load_model(model_name)
            
            if progress_bar_callback:
                progress_bar_callback(100)
            
            return True, f"مدل {model_name} با موفقیت دانلود شد"
            
        except Exception as e:
            return False, f"خطا در دانلود Whisper: {str(e)}"
    
    @staticmethod
    def _download_vosk_model(model_id, model_info, progress_callback=None, progress_bar_callback=None):
        """دانلود مدل Vosk"""
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
        if model_id not in ModelDownloader.DOWNLOADABLE_MODELS:
            return False
        
        model_info = ModelDownloader.DOWNLOADABLE_MODELS[model_id]
        
        # برای مدل‌های Whisper
        if model_info["type"] == "Whisper":
            try:
                model_name = model_id.replace("whisper_", "")
                # تبدیل نام مدل برای Whisper
                if model_name == "large_v2":
                    model_name = "large-v2"
                elif model_name == "large_v3":
                    model_name = "large-v3"
                
                # بررسی وجود مدل در cache Whisper
                import whisper
                model_path = whisper._MODELS.get(model_name)
                return model_path is not None
            except:
                return False
        
        # برای مدل‌های Vosk
        elif model_info["type"] == "Vosk":
            models_dir = Path.home() / ".vosk" / "models"
            model_path = models_dir / model_info["name"]
            return model_path.exists()
        
        return False

class ModelSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("انتخاب مدل Speech-to-Text")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # فیلترهای مدل
        filter_layout = QHBoxLayout()
        
        # چک باکس زبان
        self.checkbox_persian = QCheckBox("فارسی")
        self.checkbox_persian.setChecked(True)
        self.checkbox_persian.setStyleSheet("QCheckBox { font-weight: bold; color: #2e7d32; }")
        self.checkbox_persian.stateChanged.connect(self.filter_models)
        
        self.checkbox_english = QCheckBox("انگلیسی")
        self.checkbox_english.setChecked(True)
        self.checkbox_english.setStyleSheet("QCheckBox { font-weight: bold; color: #1976d2; }")
        self.checkbox_english.stateChanged.connect(self.filter_models)
        
        # چک باکس نوع اتصال
        self.checkbox_online = QCheckBox("آنلاین")
        self.checkbox_online.setChecked(True)
        self.checkbox_online.setStyleSheet("QCheckBox { font-weight: bold; color: #ff6b35; }")
        self.checkbox_online.stateChanged.connect(self.filter_models)
        
        self.checkbox_offline = QCheckBox("آفلاین")
        self.checkbox_offline.setChecked(True)
        self.checkbox_offline.setStyleSheet("QCheckBox { font-weight: bold; color: #9c27b0; }")
        self.checkbox_offline.stateChanged.connect(self.filter_models)
        
        filter_layout.addWidget(QLabel("زبان:"))
        filter_layout.addWidget(self.checkbox_persian)
        filter_layout.addWidget(self.checkbox_english)
        filter_layout.addStretch()
        filter_layout.addWidget(QLabel("نوع:"))
        filter_layout.addWidget(self.checkbox_online)
        filter_layout.addWidget(self.checkbox_offline)
        
        layout.addLayout(filter_layout)
        
        # لیست مدل‌های Speech-to-Text (فقط فارسی و انگلیسی)
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(300)
        self.model_list.setStyleSheet("""
            QListWidget {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
        """)
        layout.addWidget(self.model_list)
        
        # ذخیره تمام مدل‌ها برای فیلتر کردن
        self.all_models = [
            # Vosk Models (آفلاین)
            ("vosk_persian", "✅ Vosk Persian - مخصوص فارسی (1.13 GB)", "persian", "offline"),
            ("vosk_small", "⚠️ Vosk Small - فقط انگلیسی (40 MB)", "english", "offline"),
            ("vosk_large", "⚠️ Vosk Large - فقط انگلیسی (1.8 GB)", "english", "offline"),
            
            # Whisper Models (آفلاین - چند زبانه)
            ("whisper_tiny", "⚠️ Whisper Tiny - خیلی ضعیف برای فارسی (75 MB)", "both", "offline"),
            ("whisper_base", "⚠️ Whisper Base - ضعیف برای فارسی (142 MB)", "both", "offline"),
            ("whisper_small", "✅ Whisper Small - تعادل خوب (466 MB)", "both", "offline"),
            ("whisper_medium", "✅ Whisper Medium - دقت بالا (1.5 GB)", "both", "offline"),
            ("whisper_large", "✅ Whisper Large - بالاترین دقت (2.9 GB)", "both", "offline"),
            ("whisper_large_v2", "✅ Whisper Large V2 - جدیدترین نسخه (2.9 GB)", "both", "offline"),
            ("whisper_large_v3", "✅ Whisper Large V3 - جدیدترین نسخه (2.9 GB)", "both", "offline"),
            
            # Hugging Face Transformers (آفلاین)
            ("hf_wav2vec2_persian", "✅ Wav2Vec2 Persian - مخصوص فارسی (1.2 GB)", "persian", "offline"),
            ("hf_whisper_persian", "✅ Whisper Persian - مخصوص فارسی (1.5 GB)", "persian", "offline"),
            ("hf_wav2vec2_persian_alt", "⚠️ Wav2Vec2 Multilingual - چند زبانه (1.2 GB)", "both", "offline"),
            ("hf_whisper_tiny", "⚠️ Whisper Tiny HF - ضعیف برای فارسی (75 MB)", "both", "offline"),
            ("hf_whisper_base", "⚠️ Whisper Base HF - ضعیف برای فارسی (142 MB)", "both", "offline"),
            ("hf_whisper_small", "✅ Whisper Small HF - تعادل خوب (466 MB)", "both", "offline"),
            ("hf_whisper_medium", "✅ Whisper Medium HF - دقت بالا (1.5 GB)", "both", "offline"),
            ("hf_whisper_large", "✅ Whisper Large HF - بالاترین دقت (2.9 GB)", "both", "offline"),
            
            # SpeechRecognition (آنلاین)
            ("speechrecognition_google", "🌐 Google Speech - رایگان 60دقیقه/ماه (آنلاین)", "both", "online"),
            ("speechrecognition_sphinx", "⚠️ CMU Sphinx - فقط انگلیسی (100 MB)", "english", "offline"),
            ("speechrecognition_wit", "🌐 Wit.ai - رایگان تا حدی (آنلاین)", "both", "online"),
            ("speechrecognition_azure", "🌐 Azure Speech - رایگان 5ساعت/ماه (آنلاین)", "both", "online"),
            ("speechrecognition_bing", "🌐 Bing Speech - رایگان تا حدی (آنلاین)", "both", "online"),
            ("speechrecognition_houndify", "🌐 Houndify - رایگان تا حدی (آنلاین)", "both", "online"),
            ("speechrecognition_ibm", "🌐 IBM Speech - رایگان تا حدی (آنلاین)", "both", "online"),
            
            # Silero STT (آفلاین)
            ("silero_stt_en", "⚠️ Silero STT English - فقط انگلیسی (50 MB)", "english", "offline"),
            ("silero_stt_multilingual", "✅ Silero STT Multilingual - پشتیبانی از فارسی (200 MB)", "both", "offline"),
            
            # Kaldi (آفلاین)
            ("kaldi_persian", "🚧 Kaldi Persian - در حال توسعه (500 MB)", "persian", "offline"),
            ("kaldi_english", "🚧 Kaldi English - در حال توسعه (300 MB)", "english", "offline"),
            
            # سرویس‌های بومی ایرانی (آنلاین)
            ("iranian_arvan", "🇮🇷 Arvan Cloud Speech - سرویس ایرانی (آنلاین)", "persian", "online"),
            ("iranian_fanap", "🇮🇷 Fanap Speech API - سرویس ایرانی (آنلاین)", "persian", "online"),
            ("iranian_parsijoo", "🇮🇷 Parsijoo Speech - سرویس ایرانی (آنلاین)", "persian", "online"),
            
            # Google Models (آنلاین)
            ("google_standard", "🌐 Google Standard - رایگان 60دقیقه/ماه (آنلاین)", "both", "online"),
            ("google_enhanced", "💳 Google Enhanced - پولی، دقت بالاتر (آنلاین)", "both", "online"),
            ("google_phone_call", "💳 Google Phone Call - پولی، مخصوص تماس‌ها (آنلاین)", "both", "online"),
            ("google_medical", "💳 Google Medical - پولی، اصطلاحات پزشکی (آنلاین)", "both", "online"),
            ("google_video", "💳 Google Video - پولی، مخصوص ویدیوها (آنلاین)", "both", "online"),
            
            # Microsoft Azure Speech (آنلاین)
            ("azure_standard", "🌐 Azure Standard - رایگان 5ساعت/ماه (آنلاین)", "both", "online"),
            ("azure_enhanced", "💳 Azure Enhanced - پولی، دقت بالاتر (آنلاین)", "both", "online"),
            
            # AssemblyAI (آنلاین)
            ("assemblyai_standard", "🌐 AssemblyAI - رایگان 3ساعت/ماه (آنلاین)", "both", "online"),
            ("assemblyai_enhanced", "💳 AssemblyAI Enhanced - پولی، دقت بالاتر (آنلاین)", "both", "online")
        ]
        
        # بارگذاری مدل‌ها در لیست
        self.populate_model_list()
        
        # بارگذاری آخرین مدل انتخاب شده
        config = ConfigManager.load_config()
        last_model = config.get("selected_model", "vosk_persian")
        
        # پیدا کردن و انتخاب مدل آخر
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.data(Qt.UserRole) == last_model:
                self.model_list.setCurrentItem(item)
                break
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("تأیید")
        self.ok_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("لغو")
        self.cancel_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px 16px;")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def populate_model_list(self):
        """بارگذاری مدل‌ها در لیست بر اساس فیلترها"""
        self.model_list.clear()
        
        for model_id, description, language, connection_type in self.all_models:
            # بررسی فیلتر زبان
            language_match = False
            if language == "both":
                language_match = True
            elif language == "persian" and self.checkbox_persian.isChecked():
                language_match = True
            elif language == "english" and self.checkbox_english.isChecked():
                language_match = True
            
            # بررسی فیلتر نوع اتصال
            connection_match = False
            if connection_type == "online" and self.checkbox_online.isChecked():
                connection_match = True
            elif connection_type == "offline" and self.checkbox_offline.isChecked():
                connection_match = True
            
            # اگر هر دو فیلتر مطابقت داشت، مدل را اضافه کن
            if language_match and connection_match:
                item = QListWidgetItem(f"{model_id} - {description}")
                item.setData(Qt.UserRole, model_id)
                
                # رنگ‌بندی بر اساس نوع
                if connection_type == "online":
                    item.setBackground(QColor("#fff3e0"))  # نارنجی روشن
                else:
                    item.setBackground(QColor("#f3e5f5"))  # بنفش روشن
                
                self.model_list.addItem(item)
    
    def filter_models(self):
        """فیلتر کردن مدل‌ها بر اساس چک باکس‌ها"""
        self.populate_model_list()
    
    def get_selected_model(self):
        """دریافت مدل انتخاب شده"""
        current_item = self.model_list.currentItem()
        if current_item:
            return current_item.data(Qt.UserRole)
        return None

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
                # تبدیل نام مدل برای Whisper
                if whisper_model == "large_v2":
                    whisper_model = "large-v2"
                elif whisper_model == "large_v3":
                    whisper_model = "large-v3"
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
            elif self.model_name.startswith("hf_"):
                if not HUGGINGFACE_AVAILABLE:
                    self.finished.emit("Error: Hugging Face Transformers not installed. Install with: pip install transformers torch")
                    return
                self.model = "huggingface"  # نشانگر استفاده از Hugging Face
            elif self.model_name.startswith("speechrecognition_"):
                if not SPEECHRECOGNITION_AVAILABLE:
                    self.finished.emit("Error: SpeechRecognition not installed. Install with: pip install SpeechRecognition")
                    return
                self.model = "speechrecognition"  # نشانگر استفاده از SpeechRecognition
            elif self.model_name.startswith("silero_"):
                if not SILERO_AVAILABLE:
                    self.finished.emit("Error: Silero STT not installed. Install with: pip install torchaudio")
                    return
                self.model = "silero"  # نشانگر استفاده از Silero
            elif self.model_name.startswith("kaldi_"):
                if not KALDI_AVAILABLE:
                    self.finished.emit("Error: Kaldi not installed. Install with: pip install kaldi-io")
                    return
                self.model = "kaldi"  # نشانگر استفاده از Kaldi
            elif self.model_name.startswith("iranian_"):
                self.model = "iranian"  # نشانگر استفاده از سرویس‌های ایرانی
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
            elif self.model == "huggingface":
                # استفاده از Hugging Face
                text = self.transcribe_with_huggingface(temp_wav)
            elif self.model == "speechrecognition":
                # استفاده از SpeechRecognition
                text = self.transcribe_with_speechrecognition(temp_wav)
            elif self.model == "silero":
                # استفاده از Silero STT
                text = self.transcribe_with_silero(temp_wav)
            elif self.model == "kaldi":
                # استفاده از Kaldi
                text = self.transcribe_with_kaldi(temp_wav)
            elif self.model == "iranian":
                # استفاده از سرویس‌های ایرانی
                text = self.transcribe_with_iranian(temp_wav)
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
            
            # تنظیم زبان بر اساس مدل انتخاب شده
            if self.model_name == "vosk_persian":
                rec = vosk.KaldiRecognizer(model, 16000)
            elif self.model_name in ["vosk_arabic", "vosk_spanish", "vosk_french", "vosk_german", 
                                   "vosk_italian", "vosk_portuguese", "vosk_russian", 
                                   "vosk_chinese", "vosk_japanese", "vosk_korean"]:
                rec = vosk.KaldiRecognizer(model, 16000)
            else:
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
    
    def transcribe_with_huggingface(self, audio_file):
        """تبدیل صوت به متن با Hugging Face Transformers"""
        try:
            # بارگذاری مدل بر اساس نوع
            if self.model_name == "hf_wav2vec2_persian":
                # تلاش برای بارگذاری مدل فارسی
                try:
                    model_name = "m3hrdadfi/wav2vec2-large-xlsr-53-persian"
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    # اگر مدل فارسی در دسترس نباشد، از مدل عمومی استفاده کن
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل فارسی در دسترس نیست

مشکل: مدل m3hrdadfi/wav2vec2-large-xlsr-53-persian یافت نشد

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. از مدل‌های جایگزین استفاده کنید:
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)
   • Wav2Vec2 Multilingual (Hugging Face)

برای استفاده از Hugging Face:
1. به https://huggingface.co بروید
2. حساب کاربری بسازید
3. از دستور زیر استفاده کنید:
   hf auth login
"""
                    else:
                        return f"Hugging Face Error: {error_msg}. لطفاً از Vosk Persian یا Whisper استفاده کنید."
                    
            elif self.model_name == "hf_whisper_persian":
                # تلاش برای بارگذاری مدل Whisper فارسی
                try:
                    model_name = "m3hrdadfi/whisper-persian"
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل Whisper فارسی در دسترس نیست

مشکل: مدل m3hrdadfi/whisper-persian یافت نشد

راه‌حل‌ها:
1. از مدل‌های جایگزین استفاده کنید:
   • Whisper Medium/Large (چند زبانه)
   • Vosk Persian (بهترین برای فارسی)
   • Whisper عادی (Hugging Face)

برای استفاده از Hugging Face:
1. به https://huggingface.co بروید
2. حساب کاربری بسازید
3. از دستور زیر استفاده کنید:
   hf auth login
"""
                    else:
                        return f"Hugging Face Error: {error_msg}. لطفاً از Whisper عادی استفاده کنید."
                    
            elif self.model_name == "hf_wav2vec2_persian_alt":
                # استفاده از مدل عمومی wav2vec2
                try:
                    model_name = "facebook/wav2vec2-large-xlsr-53"
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    return f"Hugging Face Error: مدل wav2vec2 در دسترس نیست ({str(e)}). لطفاً از مدل‌های دیگر استفاده کنید."
            else:
                # مدل‌های عمومی Whisper
                model_name = self.model_name.replace("hf_", "").replace("_hf", "")
                if model_name == "whisper_large":
                    model_name = "openai/whisper-large-v2"
                else:
                    model_name = f"openai/whisper-{model_name}"
                
                try:
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    return f"Hugging Face Error: مدل {model_name} در دسترس نیست ({str(e)}). لطفاً از مدل‌های دیگر استفاده کنید."
            
            # بارگذاری فایل صوتی
            try:
                import librosa
                audio, sr = librosa.load(audio_file, sr=16000)
            except ImportError:
                try:
                    # استفاده از soundfile به عنوان جایگزین
                    import soundfile as sf
                    audio, sr = sf.read(audio_file)
                    if sr != 16000:
                        # تبدیل sample rate به 16000
                        import numpy as np
                        from scipy import signal
                        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
                        sr = 16000
                except ImportError:
                    return "Hugging Face Error: Neither librosa nor soundfile is available. Install with: pip install librosa soundfile"
            
            # پردازش
            try:
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                
                # تشخیص
                with torch.no_grad():
                    logits = model(inputs.input_values).logits
                
                # تبدیل به متن
                predicted_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(predicted_ids)[0]
                
                return text.strip()
            except Exception as e:
                return f"Hugging Face Error: Model processing failed - {str(e)}"
            
        except Exception as e:
            return f"Hugging Face Error: {str(e)}"
    
    def transcribe_with_speechrecognition(self, audio_file):
        """تبدیل صوت به متن با SpeechRecognition"""
        try:
            r = sr.Recognizer()
            
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
            
            # انتخاب سرویس بر اساس مدل
            if self.model_name == "speechrecognition_google":
                text = r.recognize_google(audio, language="fa-IR")
            elif self.model_name == "speechrecognition_sphinx":
                try:
                    text = r.recognize_sphinx(audio)
                except Exception as e:
                    if "missing PocketSphinx module" in str(e):
                        return """SpeechRecognition Error: PocketSphinx not installed

برای استفاده از CMU Sphinx:
1. PocketSphinx را نصب کنید:
   pip install PocketSphinx
2. یا از Google Speech استفاده کنید (رایگان)

CMU Sphinx فقط انگلیسی را پشتیبانی می‌کند.
"""
                    else:
                        return f"SpeechRecognition Error: {str(e)}"
            elif self.model_name == "speechrecognition_wit":
                api_key = os.getenv("WIT_AI_KEY")
                if not api_key:
                    return """SpeechRecognition Error: Wit.ai API Key not found

برای استفاده از Wit.ai:
1. به https://wit.ai بروید
2. حساب کاربری بسازید
3. API Key دریافت کنید
4. متغیر محیطی تنظیم کنید:
   set WIT_AI_KEY=your_key_here

یا از Google Speech استفاده کنید (رایگان)
"""
                text = r.recognize_wit(audio, key=api_key)
            elif self.model_name == "speechrecognition_azure":
                api_key = os.getenv("AZURE_SPEECH_KEY")
                region = os.getenv("AZURE_SPEECH_REGION")
                if not api_key or not region:
                    return """SpeechRecognition Error: Azure Speech API Key not found

برای استفاده از Azure Speech:
1. به https://portal.azure.com بروید
2. Cognitive Services > Speech ایجاد کنید
3. API Key و Region دریافت کنید
4. متغیرهای محیطی تنظیم کنید:
   set AZURE_SPEECH_KEY=your_key_here
   set AZURE_SPEECH_REGION=your_region_here

یا از Google Speech استفاده کنید (رایگان)
"""
                text = r.recognize_azure(audio, key=api_key, location=region)
            elif self.model_name == "speechrecognition_bing":
                api_key = os.getenv("BING_KEY")
                if not api_key:
                    return """SpeechRecognition Error: Bing Speech API Key not found

برای استفاده از Bing Speech:
1. به https://azure.microsoft.com بروید
2. Bing Speech API فعال کنید
3. API Key دریافت کنید
4. متغیر محیطی تنظیم کنید:
   set BING_KEY=your_key_here

یا از Google Speech استفاده کنید (رایگان)
"""
                text = r.recognize_bing(audio, key=api_key)
            elif self.model_name == "speechrecognition_houndify":
                client_id = os.getenv("HOUNDIFY_CLIENT_ID")
                client_key = os.getenv("HOUNDIFY_CLIENT_KEY")
                if not client_id or not client_key:
                    return """SpeechRecognition Error: Houndify API Keys not found

برای استفاده از Houndify:
1. به https://www.houndify.com بروید
2. حساب کاربری بسازید
3. Client ID و Client Key دریافت کنید
4. متغیرهای محیطی تنظیم کنید:
   set HOUNDIFY_CLIENT_ID=your_client_id
   set HOUNDIFY_CLIENT_KEY=your_client_key

یا از Google Speech استفاده کنید (رایگان)
"""
                text = r.recognize_houndify(audio, client_id=client_id, client_key=client_key)
            elif self.model_name == "speechrecognition_ibm":
                username = os.getenv("IBM_USERNAME")
                password = os.getenv("IBM_PASSWORD")
                if not username or not password:
                    return """SpeechRecognition Error: IBM Speech API Credentials not found

برای استفاده از IBM Speech:
1. به https://www.ibm.com/cloud/watson-speech-to-text بروید
2. حساب کاربری بسازید
3. Username و Password دریافت کنید
4. متغیرهای محیطی تنظیم کنید:
   set IBM_USERNAME=your_username
   set IBM_PASSWORD=your_password

یا از Google Speech استفاده کنید (رایگان)
"""
                text = r.recognize_ibm(audio, username=username, password=password)
            else:
                return "SpeechRecognition Error: Unknown service"
            
            return text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "key must be a string" in error_msg:
                return f"""SpeechRecognition Error: API Key مشکل دارد

مشکل: کلید API به درستی تنظیم نشده است

راه‌حل‌ها:
1. متغیرهای محیطی را بررسی کنید
2. API Key را دوباره تنظیم کنید
3. از Google Speech استفاده کنید (رایگان)

برای تنظیم متغیرهای محیطی:
set API_KEY_NAME=your_key_here
"""
            else:
                return f"SpeechRecognition Error: {error_msg}"
    
    def transcribe_with_silero(self, audio_file):
        """تبدیل صوت به متن با Silero STT"""
        try:
            import torch
            import torchaudio
            
            # بارگذاری مدل
            if self.model_name == "silero_stt_en":
                model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language='en')
            else:  # silero_stt_multilingual
                model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language='multilingual')
            
            # بارگذاری فایل صوتی
            audio, sample_rate = torchaudio.load(audio_file)
            
            # تبدیل به mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # تشخیص
            text = decoder(model(audio[0]))
            
            return text.strip()
            
        except ImportError as e:
            return f"Silero STT Error: Required dependencies not installed. Install with: pip install torch torchaudio"
        except Exception as e:
            return f"Silero STT Error: {str(e)}"
    
    def transcribe_with_kaldi(self, audio_file):
        """تبدیل صوت به متن با Kaldi"""
        try:
            # بررسی وجود Kaldi
            if not KALDI_AVAILABLE:
                return "Kaldi Error: kaldi-io not installed. Install with: pip install kaldi-io"
            
            # پیاده‌سازی ساده Kaldi با استفاده از مدل‌های موجود
            if self.model_name == "kaldi_persian":
                # برای Kaldi Persian، از یک مدل ساده استفاده می‌کنیم
                # این یک پیاده‌سازی نمونه است که می‌تواند با مدل‌های واقعی Kaldi جایگزین شود
                
                # بارگذاری فایل صوتی
                import soundfile as sf
                audio, sample_rate = sf.read(audio_file)
                
                # تبدیل به mono اگر stereo باشد
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # تبدیل sample rate به 16000
                if sample_rate != 16000:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                    sample_rate = 16000
                
                # پیاده‌سازی ساده Kaldi
                # این یک پیاده‌سازی نمونه است که می‌تواند با مدل‌های واقعی Kaldi جایگزین شود
                
                # برای حال حاضر، از یک مدل ساده استفاده می‌کنیم
                # در آینده می‌توان مدل‌های Kaldi واقعی را اضافه کرد
                
                # محاسبه طول فایل صوتی
                duration = len(audio) / sample_rate
                
                # پیام نمونه با اطلاعات فایل
                result = f"Kaldi Persian: فایل صوتی پردازش شد (مدت: {duration:.2f} ثانیه)\n"
                result += "مدل Kaldi فارسی در حال توسعه است.\n"
                result += "برای استفاده از مدل‌های فارسی، لطفاً از گزینه‌های زیر استفاده کنید:\n"
                result += "• Vosk Persian (بهترین برای فارسی)\n"
                result += "• Whisper Medium/Large (چند زبانه)\n"
                result += "• Wav2Vec2 Persian (Hugging Face)"
                
                return result
                
            elif self.model_name == "kaldi_english":
                # برای Kaldi English
                import soundfile as sf
                audio, sample_rate = sf.read(audio_file)
                
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                if sample_rate != 16000:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                    sample_rate = 16000
                
                # محاسبه طول فایل صوتی
                duration = len(audio) / sample_rate
                
                # پیام نمونه با اطلاعات فایل
                result = f"Kaldi English: فایل صوتی پردازش شد (مدت: {duration:.2f} ثانیه)\n"
                result += "مدل Kaldi انگلیسی در حال توسعه است.\n"
                result += "برای استفاده از مدل‌های انگلیسی، لطفاً از گزینه‌های زیر استفاده کنید:\n"
                result += "• Vosk Small/Large (بهترین برای انگلیسی)\n"
                result += "• Whisper (چند زبانه)\n"
                result += "• Silero STT English"
                
                return result
            else:
                return "Kaldi Error: Unknown Kaldi model"
            
        except ImportError as e:
            return f"Kaldi Error: Required dependencies not installed. Install with: pip install kaldi-io soundfile scipy"
        except Exception as e:
            return f"Kaldi Error: {str(e)}"
    
    def transcribe_with_iranian(self, audio_file):
        """تبدیل صوت به متن با سرویس‌های بومی ایرانی"""
        try:
            # این یک پیاده‌سازی نمونه است
            # برای استفاده واقعی نیاز به API key های مربوطه است
            
            if self.model_name == "iranian_arvan":
                return "Arvan Cloud Speech Error: نیاز به تنظیم API Key\nبرای استفاده از Arvan Cloud Speech API key خود را تنظیم کنید."
            elif self.model_name == "iranian_fanap":
                return "Fanap Speech Error: نیاز به تنظیم API Key\nبرای استفاده از Fanap Speech API key خود را تنظیم کنید."
            elif self.model_name == "iranian_parsijoo":
                return "Parsijoo Speech Error: نیاز به تنظیم API Key\nبرای استفاده از Parsijoo Speech API key خود را تنظیم کنید."
            else:
                return "Iranian Service Error: Unknown service"
            
        except Exception as e:
            return f"Iranian Service Error: {str(e)}"


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

        self.btn_change_model = QPushButton("Change Model")
        self.btn_change_model.setMinimumHeight(40)
        self.btn_change_model.setStyleSheet("background-color: #9c27b0; color: white;")
        self.btn_change_model.clicked.connect(self.change_model)
        self.layout.addWidget(self.btn_change_model)

        self.btn_huggingface_setup = QPushButton("Hugging Face Setup")
        self.btn_huggingface_setup.setMinimumHeight(40)
        self.btn_huggingface_setup.setStyleSheet("background-color: #ff6b35; color: white;")
        self.btn_huggingface_setup.clicked.connect(self.show_huggingface_setup_guide)
        self.layout.addWidget(self.btn_huggingface_setup)

        self.btn_speechrecognition_setup = QPushButton("SpeechRecognition Setup")
        self.btn_speechrecognition_setup.setMinimumHeight(40)
        self.btn_speechrecognition_setup.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_speechrecognition_setup.clicked.connect(self.show_speechrecognition_setup_guide)
        self.layout.addWidget(self.btn_speechrecognition_setup)

        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        
        # Progress bar برای دانلود
        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        self.download_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        self.layout.addWidget(self.download_progress)
        
        # Status label برای دانلود
        self.download_status = QLabel("")
        self.download_status.setVisible(False)
        self.download_status.setStyleSheet("""
            color: #2196F3; 
            font-weight: bold; 
            font-size: 14px;
            padding: 5px;
            background-color: #E3F2FD;
            border: 1px solid #2196F3;
            border-radius: 5px;
        """)
        self.download_status.setAlignment(Qt.AlignCenter)
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
        
        # بارگذاری تنظیمات و مدل آخر
        config = ConfigManager.load_config()
        self.selected_model = config.get("selected_model", "vosk_persian")
        
        # بارگذاری آخرین فایل صوتی
        last_audio_path = config.get("last_audio_path", "")
        if last_audio_path and os.path.exists(last_audio_path):
            self.audio_path = last_audio_path
            self.label.setText(f"Last File: {last_audio_path}")
        
        # بازگردانی موقعیت و اندازه پنجره
        window_geometry = config.get("window_geometry")
        if window_geometry:
            self.setGeometry(
                window_geometry.get("x", 100),
                window_geometry.get("y", 100),
                window_geometry.get("width", 900),
                window_geometry.get("height", 600)
            )
        
        # به‌روزرسانی نمایش مدل
        self.update_model_display()

    def update_model_display(self):
        """به‌روزرسانی نمایش مدل انتخاب شده"""
        model_info = ModelDownloader.DOWNLOADABLE_MODELS.get(self.selected_model, {})
        if model_info:
            model_name = model_info.get("name", self.selected_model)
            model_type = model_info.get("type", "Unknown")
            model_language = model_info.get("language", "Unknown")
            self.model_label.setText(f"Selected Model: {self.selected_model} ({model_type} - {model_language})")
        else:
            self.model_label.setText(f"Selected Model: {self.selected_model}")

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
            
            # ذخیره مسیر آخرین فایل صوتی
            ConfigManager.update_audio_path(file_path)

    def start_transcription(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Warning", "No audio file selected!")
            return

        # نمایش dialog انتخاب مدل
        dialog = ModelSelectionDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.selected_model = dialog.get_selected_model()
            
            # ذخیره مدل انتخاب شده
            ConfigManager.update_model(self.selected_model)
            
            # به‌روزرسانی نمایش مدل
            self.update_model_display()
            
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
        self.download_progress.setFormat(f"دانلود: {percent}%")
        
        if percent == 100:
            # کمی صبر کنید تا کاربر progress کامل را ببیند
            import time
            time.sleep(1)
            self.download_progress.setVisible(False)
            self.download_status.setVisible(False)
    
    def update_download_status(self, message):
        """به‌روزرسانی status دانلود"""
        self.download_status.setText(f"📥 {message}")
        self.download_status.setVisible(True)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)  # شروع از صفر

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

    def change_model(self):
        """تغییر مدل بدون شروع transcription"""
        dialog = ModelSelectionDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.selected_model = dialog.get_selected_model()
            
            # ذخیره مدل انتخاب شده
            ConfigManager.update_model(self.selected_model)
            
            # به‌روزرسانی نمایش مدل
            self.update_model_display()
            
            QMessageBox.information(self, "Model Changed", f"Model changed to: {self.selected_model}")

    def closeEvent(self, event):
        """ذخیره تنظیمات هنگام بستن برنامه"""
        # ذخیره موقعیت و اندازه پنجره
        geometry = {
            "x": self.x(),
            "y": self.y(),
            "width": self.width(),
            "height": self.height()
        }
        ConfigManager.update_window_geometry(geometry)
        event.accept()

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
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QLabel, QTabWidget, QWidget
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("دانلود مدل‌های Speech-to-Text")
        dialog.setModal(True)
        dialog.resize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # توضیحات
        info_label = QLabel("مدل‌های قابل دانلود برای تشخیص گفتار (کاملاً رایگان و آفلاین):")
        layout.addWidget(info_label)
        
        # Tab widget برای دسته‌بندی مدل‌ها
        tab_widget = QTabWidget()
        
        # Tab Vosk
        vosk_tab = QWidget()
        vosk_layout = QVBoxLayout(vosk_tab)
        vosk_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "Vosk":
                status = "✅ دانلود شده" if ModelDownloader.is_model_downloaded(model_id) else "❌ دانلود نشده"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                vosk_list.addItem(item_text)
        
        vosk_layout.addWidget(vosk_list)
        tab_widget.addTab(vosk_tab, "Vosk Models")
        
        # Tab Whisper
        whisper_tab = QWidget()
        whisper_layout = QVBoxLayout(whisper_tab)
        whisper_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "Whisper":
                status = "✅ دانلود شده" if ModelDownloader.is_model_downloaded(model_id) else "❌ دانلود نشده"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                whisper_list.addItem(item_text)
        
        whisper_layout.addWidget(whisper_list)
        tab_widget.addTab(whisper_tab, "Whisper Models")
        
        # Tab Hugging Face
        hf_tab = QWidget()
        hf_layout = QVBoxLayout(hf_tab)
        hf_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "HuggingFace":
                status = "✅ دانلود شده" if ModelDownloader.is_model_downloaded(model_id) else "❌ دانلود نشده"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                hf_list.addItem(item_text)
        
        hf_layout.addWidget(hf_list)
        tab_widget.addTab(hf_tab, "Hugging Face Models")
        
        # Tab SpeechRecognition
        sr_tab = QWidget()
        sr_layout = QVBoxLayout(sr_tab)
        sr_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "SpeechRecognition":
                status = "✅ آماده" if True else "❌ نیاز به تنظیم"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                sr_list.addItem(item_text)
        
        sr_layout.addWidget(sr_list)
        tab_widget.addTab(sr_tab, "SpeechRecognition")
        
        # Tab Silero
        silero_tab = QWidget()
        silero_layout = QVBoxLayout(silero_tab)
        silero_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "Silero":
                status = "✅ دانلود شده" if ModelDownloader.is_model_downloaded(model_id) else "❌ دانلود نشده"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                silero_list.addItem(item_text)
        
        silero_layout.addWidget(silero_list)
        tab_widget.addTab(silero_tab, "Silero STT")
        
        # Tab Kaldi
        kaldi_tab = QWidget()
        kaldi_layout = QVBoxLayout(kaldi_tab)
        kaldi_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "Kaldi":
                status = "✅ دانلود شده" if ModelDownloader.is_model_downloaded(model_id) else "❌ دانلود نشده"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                kaldi_list.addItem(item_text)
        
        kaldi_layout.addWidget(kaldi_list)
        tab_widget.addTab(kaldi_tab, "Kaldi Models")
        
        # Tab Iranian Services
        iranian_tab = QWidget()
        iranian_layout = QVBoxLayout(iranian_tab)
        iranian_list = QListWidget()
        
        for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if model_info["type"] == "Iranian":
                status = "✅ آماده" if True else "❌ نیاز به تنظیم"
                item_text = f"{model_info['name']} ({model_info['size']}) - {model_info['warning']} - {status}"
                iranian_list.addItem(item_text)
        
        iranian_layout.addWidget(iranian_list)
        tab_widget.addTab(iranian_tab, "سرویس‌های ایرانی")
        
        layout.addWidget(tab_widget)
        
        # ذخیره reference ها
        dialog.vosk_list = vosk_list
        dialog.whisper_list = whisper_list
        dialog.hf_list = hf_list
        dialog.sr_list = sr_list
        dialog.silero_list = silero_list
        dialog.kaldi_list = kaldi_list
        dialog.iranian_list = iranian_list
        dialog.tab_widget = tab_widget
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        download_btn = QPushButton("دانلود مدل انتخاب شده")
        download_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        download_btn.clicked.connect(lambda: self.download_selected_model(dialog))
        
        download_all_btn = QPushButton("دانلود همه مدل‌های Vosk")
        download_all_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        download_all_btn.clicked.connect(lambda: self.download_all_vosk_models(dialog))
        
        download_whisper_btn = QPushButton("دانلود همه مدل‌های Whisper")
        download_whisper_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        download_whisper_btn.clicked.connect(lambda: self.download_all_whisper_models(dialog))
        
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(download_btn)
        button_layout.addWidget(download_all_btn)
        button_layout.addWidget(download_whisper_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def show_huggingface_setup_guide(self):
        """نمایش راهنمای تنظیم Hugging Face"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("راهنمای تنظیم Hugging Face")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # متن راهنما
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h2>راهنمای تنظیم Hugging Face</h2>
        
        <h3>🔧 مراحل تنظیم:</h3>
        
        <h4>1️⃣ ایجاد حساب کاربری:</h4>
        <p>• به <a href="https://huggingface.co">Hugging Face</a> بروید</p>
        <p>• روی "Sign Up" کلیک کنید</p>
        <p>• حساب کاربری خود را بسازید</p>
        
        <h4>2️⃣ نصب Hugging Face CLI:</h4>
        <p>• در Command Prompt اجرا کنید:</p>
        <p><code>pip install huggingface_hub</code></p>
        
        <h4>3️⃣ ورود به حساب کاربری:</h4>
        <p>• در Command Prompt اجرا کنید:</p>
        <p><code>huggingface-cli login</code></p>
        <p>• Token خود را وارد کنید</p>
        
        <h4>4️⃣ دریافت Token:</h4>
        <p>• به <a href="https://huggingface.co/settings/tokens">Settings > Tokens</a> بروید</p>
        <p>• "New token" کلیک کنید</p>
        <p>• نام و دسترسی‌ها را انتخاب کنید</p>
        <p>• Token را کپی کنید</p>
        
        <h3>💡 نکات مهم:</h3>
        <p>• برخی مدل‌ها نیاز به احراز هویت دارند</p>
        <p>• مدل‌های فارسی ممکن است در دسترس نباشند</p>
        <p>• از مدل‌های جایگزین استفاده کنید</p>
        
        <h3>🔗 لینک‌های مفید:</h3>
        <p>• <a href="https://huggingface.co">Hugging Face</a></p>
        <p>• <a href="https://huggingface.co/models">مدل‌های موجود</a></p>
        <p>• <a href="https://huggingface.co/docs/hub/quick-start">راهنمای سریع</a></p>
        """)
        
        layout.addWidget(text_edit)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        open_hf_btn = QPushButton("باز کردن Hugging Face")
        open_hf_btn.setStyleSheet("background-color: #ff6b35; color: white; padding: 8px;")
        open_hf_btn.clicked.connect(lambda: webbrowser.open("https://huggingface.co"))
        
        open_tokens_btn = QPushButton("مدیریت Tokens")
        open_tokens_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        open_tokens_btn.clicked.connect(lambda: webbrowser.open("https://huggingface.co/settings/tokens"))
        
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(open_hf_btn)
        button_layout.addWidget(open_tokens_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def show_speechrecognition_setup_guide(self):
        """نمایش راهنمای تنظیم SpeechRecognition"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("راهنمای تنظیم SpeechRecognition")
        dialog.setModal(True)
        dialog.resize(700, 600)
        
        layout = QVBoxLayout(dialog)
        
        # متن راهنما
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h2>راهنمای تنظیم SpeechRecognition</h2>
        
        <h3>🔧 تنظیم API Keys:</h3>
        
        <h4>1️⃣ Google Speech (رایگان - پیشنهادی):</h4>
        <p>• نیازی به API Key ندارد</p>
        <p>• 60 دقیقه در ماه رایگان</p>
        <p>• بهترین کیفیت برای فارسی</p>
        
        <h4>2️⃣ Wit.ai:</h4>
        <p>• به <a href="https://wit.ai">Wit.ai</a> بروید</p>
        <p>• حساب کاربری بسازید</p>
        <p>• API Key دریافت کنید</p>
        <p>• متغیر محیطی تنظیم کنید:</p>
        <p><code>set WIT_AI_KEY=your_key_here</code></p>
        
        <h4>3️⃣ Azure Speech:</h4>
        <p>• به <a href="https://portal.azure.com">Azure Portal</a> بروید</p>
        <p>• Cognitive Services > Speech ایجاد کنید</p>
        <p>• API Key و Region دریافت کنید</p>
        <p>• متغیرهای محیطی تنظیم کنید:</p>
        <p><code>set AZURE_SPEECH_KEY=your_key_here</code></p>
        <p><code>set AZURE_SPEECH_REGION=your_region_here</code></p>
        
        <h4>4️⃣ Bing Speech:</h4>
        <p>• به <a href="https://azure.microsoft.com">Azure</a> بروید</p>
        <p>• Bing Speech API فعال کنید</p>
        <p>• API Key دریافت کنید</p>
        <p>• متغیر محیطی تنظیم کنید:</p>
        <p><code>set BING_KEY=your_key_here</code></p>
        
        <h4>5️⃣ Houndify:</h4>
        <p>• به <a href="https://www.houndify.com">Houndify</a> بروید</p>
        <p>• حساب کاربری بسازید</p>
        <p>• Client ID و Client Key دریافت کنید</p>
        <p>• متغیرهای محیطی تنظیم کنید:</p>
        <p><code>set HOUNDIFY_CLIENT_ID=your_client_id</code></p>
        <p><code>set HOUNDIFY_CLIENT_KEY=your_client_key</code></p>
        
        <h4>6️⃣ IBM Speech:</h4>
        <p>• به <a href="https://www.ibm.com/cloud/watson-speech-to-text">IBM Watson</a> بروید</p>
        <p>• حساب کاربری بسازید</p>
        <p>• Username و Password دریافت کنید</p>
        <p>• متغیرهای محیطی تنظیم کنید:</p>
        <p><code>set IBM_USERNAME=your_username</code></p>
        <p><code>set IBM_PASSWORD=your_password</code></p>
        
        <h3>💡 نکات مهم:</h3>
        <p>• Google Speech بهترین گزینه برای فارسی است</p>
        <p>• CMU Sphinx کاملاً آفلاین است (فقط انگلیسی)</p>
        <p>• سایر سرویس‌ها نیاز به API Key دارند</p>
        <p>• متغیرهای محیطی را در Command Prompt تنظیم کنید</p>
        
        <h3>🔗 لینک‌های مفید:</h3>
        <p>• <a href="https://wit.ai">Wit.ai</a></p>
        <p>• <a href="https://portal.azure.com">Azure Portal</a></p>
        <p>• <a href="https://www.houndify.com">Houndify</a></p>
        <p>• <a href="https://www.ibm.com/cloud/watson-speech-to-text">IBM Watson</a></p>
        """)
        
        layout.addWidget(text_edit)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        open_wit_btn = QPushButton("Wit.ai")
        open_wit_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        open_wit_btn.clicked.connect(lambda: webbrowser.open("https://wit.ai"))
        
        open_azure_btn = QPushButton("Azure Portal")
        open_azure_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        open_azure_btn.clicked.connect(lambda: webbrowser.open("https://portal.azure.com"))
        
        open_houndify_btn = QPushButton("Houndify")
        open_houndify_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        open_houndify_btn.clicked.connect(lambda: webbrowser.open("https://www.houndify.com"))
        
        open_ibm_btn = QPushButton("IBM Watson")
        open_ibm_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px;")
        open_ibm_btn.clicked.connect(lambda: webbrowser.open("https://www.ibm.com/cloud/watson-speech-to-text"))
        
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(open_wit_btn)
        button_layout.addWidget(open_azure_btn)
        button_layout.addWidget(open_houndify_btn)
        button_layout.addWidget(open_ibm_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def download_selected_model(self, dialog):
        """دانلود مدل انتخاب شده"""
        # تشخیص tab فعال
        current_tab = dialog.tab_widget.currentIndex()
        
        # تعیین model_list و model_type بر اساس tab
        if current_tab == 0:  # Vosk tab
            model_list = dialog.vosk_list
            model_type = "Vosk"
        elif current_tab == 1:  # Whisper tab
            model_list = dialog.whisper_list
            model_type = "Whisper"
        elif current_tab == 2:  # Hugging Face tab
            model_list = dialog.hf_list
            model_type = "HuggingFace"
        elif current_tab == 3:  # SpeechRecognition tab
            model_list = dialog.sr_list
            model_type = "SpeechRecognition"
        elif current_tab == 4:  # Silero tab
            model_list = dialog.silero_list
            model_type = "Silero"
        elif current_tab == 5:  # Kaldi tab
            model_list = dialog.kaldi_list
            model_type = "Kaldi"
        elif current_tab == 6:  # Iranian tab
            model_list = dialog.iranian_list
            model_type = "Iranian"
        else:
            QMessageBox.warning(dialog, "هشدار", "لطفاً یک tab معتبر انتخاب کنید.")
            return
        
        current_row = model_list.currentRow()
        if current_row == -1:
            QMessageBox.warning(dialog, "هشدار", "لطفاً یک مدل انتخاب کنید.")
            return
        
        # دریافت model_id
        model_ids = [key for key, value in ModelDownloader.DOWNLOADABLE_MODELS.items() 
                    if value["type"] == model_type]
        model_id = model_ids[current_row]
        
        # بررسی وضعیت مدل
        if model_type in ["SpeechRecognition", "Iranian"]:
            QMessageBox.information(dialog, "اطلاعات", f"مدل {model_id} آنلاین است و نیازی به دانلود ندارد.")
            return
        
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
    
    def download_all_vosk_models(self, dialog):
        """دانلود همه مدل‌های Vosk"""
        reply = QMessageBox.question(
            dialog, "تأیید", 
            "آیا می‌خواهید همه مدل‌های Vosk را دانلود کنید؟\nاین کار ممکن است زمان زیادی ببرد.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(dialog, "شروع دانلود", "دانلود همه مدل‌های Vosk شروع شد. لطفاً صبر کنید...")
            
            for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
                if model_info["type"] == "Vosk" and not ModelDownloader.is_model_downloaded(model_id):
                    success, result = ModelDownloader.download_model(model_id)
                    if not success:
                        QMessageBox.critical(dialog, "خطا", f"خطا در دانلود {model_id}: {result}")
                        return
            
            QMessageBox.information(dialog, "موفق", "همه مدل‌های Vosk با موفقیت دانلود شدند!")
            dialog.accept()
    
    def download_all_whisper_models(self, dialog):
        """دانلود همه مدل‌های Whisper"""
        reply = QMessageBox.question(
            dialog, "تأیید", 
            "آیا می‌خواهید همه مدل‌های Whisper را دانلود کنید؟\nاین کار ممکن است زمان زیادی ببرد.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(dialog, "شروع دانلود", "دانلود همه مدل‌های Whisper شروع شد. لطفاً صبر کنید...")
            
            for model_id, model_info in ModelDownloader.DOWNLOADABLE_MODELS.items():
                if model_info["type"] == "Whisper" and not ModelDownloader.is_model_downloaded(model_id):
                    success, result = ModelDownloader.download_model(model_id)
                    if not success:
                        QMessageBox.critical(dialog, "خطا", f"خطا در دانلود {model_id}: {result}")
                        return
            
            QMessageBox.information(dialog, "موفق", "همه مدل‌های Whisper با موفقیت دانلود شدند!")
            dialog.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec())
