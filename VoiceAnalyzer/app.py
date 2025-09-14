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
    import torch.hub
    import omegaconf
    SILERO_AVAILABLE = True
except ImportError as e:
    SILERO_AVAILABLE = False
    print(f"Warning: Silero STT dependencies not available: {e}")
    print("To install Silero STT dependencies:")
    print("1. First install omegaconf: pip install omegaconf")
    print("2. Then install PyTorch:")
    print("   For CPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("   For GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")

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
            "size": "3.1 GB",
            "language": "چند زبانه",
            "warning": "✅ بالاترین دقت",
            "type": "Whisper"
        },
        "whisper_large_v2": {
            "url": "whisper://large-v2",
            "name": "whisper-large-v2",
            "size": "3.1 GB",
            "language": "چند زبانه",
            "warning": "✅ جدیدترین نسخه",
            "type": "Whisper"
        },
        "whisper_large_v3": {
            "url": "whisper://large-v3",
            "name": "whisper-large-v3",
            "size": "3.1 GB",
            "language": "چند زبانه",
            "warning": "✅ جدیدترین نسخه",
            "type": "Whisper"
        },
        
        # Hugging Face Transformers
        "hf_wav2vec2_persian": {
            "url": "huggingface://m3hrdadfi/wav2vec2-large-xlsr-persian",
            "name": "Wav2Vec2-Large-XLSR-53-Persian",
            "size": "1.2 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی - Hugging Face",
            "type": "HuggingFace"
        },
        "hf_wav2vec2_persian_v3": {
            "url": "huggingface://m3hrdadfi/wav2vec2-large-xlsr-persian-v3",
            "name": "Wav2Vec2-Large-XLSR-53-Persian-V3",
            "size": "1.2 GB",
            "language": "فارسی",
            "warning": "✅ جدیدترین نسخه - بهترین کیفیت فارسی (WER: 10.36%)",
            "type": "HuggingFace"
        },
        "hf_wav2vec2_persian_jonatas": {
            "url": "huggingface://jonatasgrosman/wav2vec2-large-xlsr-53-persian",
            "name": "Wav2Vec2-Large-XLSR-53-Persian-Jonatas",
            "size": "1.2 GB",
            "language": "فارسی",
            "warning": "✅ مدل بهینه شده - کیفیت بالا (WER: 30.12%)",
            "type": "HuggingFace"
        },
        "hf_whisper_large_v3_persian": {
            "url": "huggingface://nezamisafa/whisper-large-v3-persian",
            "name": "Whisper-Large-V3-Persian",
            "size": "3.1 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی - بهترین کیفیت",
            "type": "HuggingFace"
        },
        "hf_whisper_large_v3_persian_alt": {
            "url": "huggingface://MohammadKhosravi/whisper-large-v3-Persian",
            "name": "Whisper-Large-V3-Persian-Alt",
            "size": "3.1 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی - جایگزین",
            "type": "HuggingFace"
        },
        "hf_whisper_large_persian_steja": {
            "url": "huggingface://steja/whisper-large-persian",
            "name": "Whisper-Large-Persian-Steja",
            "size": "3.1 GB",
            "language": "فارسی",
            "warning": "✅ مخصوص فارسی - Steja (WER: 26.37%)",
            "type": "HuggingFace"
        },
        "hf_wav2vec2_persian_alt": {
            "url": "huggingface://facebook/wav2vec2-large-xlsr-53",
            "name": "Wav2Vec2-Large-XLSR-53-Multilingual",
            "size": "1.2 GB",
            "language": "چند زبانه",
            "warning": "⚠️ چند زبانه - نیاز به fine-tuning برای فارسی",
            "type": "HuggingFace"
        },
        "hf_whisper_tiny": {
            "url": "huggingface://openai/whisper-tiny",
            "name": "Whisper-Tiny-HF",
            "size": "75 MB",
            "language": "چند زبانه",
            "warning": "⚠️ ضعیف برای فارسی",
            "type": "HuggingFace"
        },
        "hf_whisper_base": {
            "url": "huggingface://openai/whisper-base",
            "name": "Whisper-Base-HF",
            "size": "142 MB",
            "language": "چند زبانه",
            "warning": "⚠️ ضعیف برای فارسی",
            "type": "HuggingFace"
        },
        "hf_whisper_small": {
            "url": "huggingface://openai/whisper-small",
            "name": "Whisper-Small-HF",
            "size": "466 MB",
            "language": "چند زبانه",
            "warning": "✅ تعادل خوب",
            "type": "HuggingFace"
        },
        "hf_whisper_medium": {
            "url": "huggingface://openai/whisper-medium",
            "name": "Whisper-Medium-HF",
            "size": "1.5 GB",
            "language": "چند زبانه",
            "warning": "✅ دقت بالا",
            "type": "HuggingFace"
        },
        "hf_whisper_large": {
            "url": "huggingface://openai/whisper-large-v2",
            "name": "Whisper-Large-V2-HF",
            "size": "3.1 GB",
            "language": "چند زبانه",
            "warning": "✅ بالاترین دقت",
            "type": "HuggingFace"
        },
        "hf_whisper_large_v2": {
            "url": "huggingface://openai/whisper-large-v2",
            "name": "Whisper-Large-V2-HF",
            "size": "3.1 GB",
            "language": "چند زبانه",
            "warning": "✅ جدیدترین نسخه",
            "type": "HuggingFace"
        },
        "hf_whisper_large_v3": {
            "url": "huggingface://openai/whisper-large-v3",
            "name": "Whisper-Large-V3-HF",
            "size": "3.1 GB",
            "language": "چند زبانه",
            "warning": "✅ جدیدترین نسخه",
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
        
        
    }
    
    # برای سازگاری با کد قبلی
    VOSK_MODELS = {
        key: value for key, value in DOWNLOADABLE_MODELS.items() 
        if value["type"] == "Vosk"
    }
    
    @staticmethod
    def download_model(model_id, progress_callback=None):
        """دانلود مدل با نمایش نوتیفیکیشن ساده"""
        if model_id not in ModelDownloader.DOWNLOADABLE_MODELS:
            return False, f"مدل {model_id} پشتیبانی نمی‌شود"
        
        model_info = ModelDownloader.DOWNLOADABLE_MODELS[model_id]
        
        # برای مدل‌های Whisper
        if model_info["type"] == "Whisper":
            return ModelDownloader._download_whisper_model(model_id, model_info, progress_callback)
        
        # برای مدل‌های Vosk
        elif model_info["type"] == "Vosk":
            return ModelDownloader._download_vosk_model(model_id, model_info, progress_callback)
        
        return False, f"نوع مدل {model_info['type']} پشتیبانی نمی‌شود"
    
    @staticmethod
    def _download_whisper_model(model_id, model_info, progress_callback=None):
        """دانلود مدل Whisper با نمایش نوتیفیکیشن ساده"""
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
            
            if progress_callback:
                progress_callback(f"در حال بارگذاری مدل Whisper {model_name}...")
            
            # دانلود مدل
            model = whisper.load_model(model_name)
            
            if progress_callback:
                progress_callback(f"مدل {model_name} با موفقیت دانلود شد")
            
            return True, f"مدل {model_name} با موفقیت دانلود شد"
            
        except Exception as e:
            return False, f"خطا در دانلود Whisper: {str(e)}"
    
    @staticmethod
    def _download_vosk_model(model_id, model_info, progress_callback=None):
        """دانلود مدل Vosk با نمایش نوتیفیکیشن ساده"""
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
            
            # ذخیره فایل
            zip_path = models_dir / f"{model_info['name']}.zip"
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # استخراج فایل
            if progress_callback:
                progress_callback("در حال استخراج مدل...")
            
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
                import os
                
                # مسیر cache Whisper
                cache_dir = os.path.expanduser("~/.cache/whisper")
                model_file = f"{model_name}.pt"
                model_path = os.path.join(cache_dir, model_file)
                
                return os.path.exists(model_path)
            except:
                return False
        
        # برای مدل‌های Vosk
        elif model_info["type"] == "Vosk":
            models_dir = Path.home() / ".vosk" / "models"
            model_path = models_dir / model_info["name"]
            return model_path.exists()
        
        # برای مدل‌های Hugging Face - بررسی cache محلی
        elif model_info["type"] == "HuggingFace":
            try:
                import os
                from transformers import AutoModel, AutoTokenizer
                
                # استخراج نام مدل از URL
                model_url = model_info["url"]
                if model_url.startswith("huggingface://"):
                    model_name = model_url.replace("huggingface://", "")
                    
                    # بررسی cache محلی Hugging Face
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    
                    # جستجو در فولدرهای cache
                    if os.path.exists(cache_dir):
                        for item in os.listdir(cache_dir):
                            if model_name.replace("/", "--") in item:
                                return True
                    
                    # بررسی cache transformers
                    try:
                        # تلاش برای بارگذاری بدون دانلود
                        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                        return True
                    except:
                        return False
                
                return False
            except:
                return False
        
        return False

class ModelDownloadDialog(QDialog):
    """دیالوگ دانلود مدل با نوار پیشرفت"""
    
    def __init__(self, model_id, model_info, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.model_info = model_info
        self.download_thread = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle(f"دانلود مدل {self.model_info['name']}")
        self.setModal(True)
        self.resize(400, 200)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 14px;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 1 #45a049);
                border-radius: 6px;
                margin: 1px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # اطلاعات مدل
        info_label = QLabel(f"در حال دانلود: {self.model_info['name']}")
        info_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(info_label)
        
        size_label = QLabel(f"حجم: {self.model_info['size']}")
        size_label.setStyleSheet("font-size: 14px; color: #81c784;")
        layout.addWidget(size_label)
        
        # نوار پیشرفت
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # برچسب وضعیت
        self.status_label = QLabel("آماده برای دانلود...")
        self.status_label.setStyleSheet("font-size: 12px; color: #ffffff;")
        layout.addWidget(self.status_label)
        
        # دکمه لغو
        self.cancel_button = QPushButton("لغو دانلود")
        self.cancel_button.clicked.connect(self.cancel_download)
        layout.addWidget(self.cancel_button)
        
    def start_download(self):
        """شروع دانلود مدل"""
        self.download_thread = ModelDownloadThread(self.model_id, self.model_info)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.status.connect(self.update_status)
        self.download_thread.finished.connect(self.download_finished)
        self.download_thread.start()
        
    def update_progress(self, value):
        """به‌روزرسانی نوار پیشرفت"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """به‌روزرسانی پیام وضعیت"""
        self.status_label.setText(message)
        
    def cancel_download(self):
        """لغو دانلود"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.terminate()
        self.reject()
        
    def download_finished(self, success, message):
        """پایان دانلود"""
        if success:
            self.accept()
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "خطا در دانلود", message)
            self.reject()

class ModelDownloadThread(QThread):
    """Thread برای دانلود مدل با نوار پیشرفت"""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(bool, str)
    
    def __init__(self, model_id, model_info):
        super().__init__()
        self.model_id = model_id
        self.model_info = model_info
        
    def run(self):
        try:
            self.status.emit("شروع دانلود...")
            self.progress.emit(5)
            
            if self.model_info["type"] == "Whisper":
                success, message = self._download_whisper_with_progress()
            elif self.model_info["type"] == "Vosk":
                success, message = self._download_vosk_with_progress()
            elif self.model_info["type"] == "HuggingFace":
                success, message = self._download_huggingface_with_progress()
            else:
                success, message = False, f"نوع مدل {self.model_info['type']} پشتیبانی نمی‌شود"
                
            self.finished.emit(success, message)
            
        except Exception as e:
            self.finished.emit(False, f"خطا در دانلود: {str(e)}")
    
    def _download_whisper_with_progress(self):
        """دانلود مدل Whisper با نوار پیشرفت"""
        try:
            import whisper
            
            model_name = self.model_id.replace("whisper_", "")
            if model_name == "large_v2":
                model_name = "large-v2"
            elif model_name == "large_v3":
                model_name = "large-v3"
            
            self.status.emit(f"در حال دانلود مدل Whisper {model_name}...")
            self.progress.emit(20)
            
            # دانلود مدل
            model = whisper.load_model(model_name)
            
            self.progress.emit(100)
            self.status.emit("دانلود کامل شد!")
            
            return True, f"مدل {model_name} با موفقیت دانلود شد"
            
        except Exception as e:
            return False, f"خطا در دانلود Whisper: {str(e)}"
    
    def _download_vosk_with_progress(self):
        """دانلود مدل Vosk با نوار پیشرفت"""
        try:
            import requests
            import zipfile
            from pathlib import Path
            
            models_dir = Path.home() / ".vosk" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = models_dir / self.model_info["name"]
            
            # اگر مدل قبلاً دانلود شده
            if model_path.exists():
                self.progress.emit(100)
                self.status.emit("مدل قبلاً دانلود شده!")
                return True, f"مدل {self.model_info['name']} قبلاً دانلود شده"
            
            self.status.emit(f"در حال دانلود {self.model_info['name']}...")
            self.progress.emit(10)
            
            # دانلود فایل
            response = requests.get(self.model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            zip_path = models_dir / f"{self.model_info['name']}.zip"
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = int((downloaded_size / total_size) * 70) + 10  # 10-80%
                            self.progress.emit(progress)
            
            self.status.emit("در حال استخراج مدل...")
            self.progress.emit(85)
            
            # استخراج فایل
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            
            # حذف فایل zip
            zip_path.unlink()
            
            self.progress.emit(100)
            self.status.emit("دانلود و استخراج کامل شد!")
            
            return True, f"مدل {self.model_info['name']} با موفقیت دانلود شد"
            
        except Exception as e:
            return False, f"خطا در دانلود Vosk: {str(e)}"
    
    def _download_huggingface_with_progress(self):
        """دانلود مدل Hugging Face با نوار پیشرفت"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import os
            
            # استخراج نام مدل از URL
            model_url = self.model_info["url"]
            if not model_url.startswith("huggingface://"):
                return False, "URL مدل Hugging Face نامعتبر است"
            
            model_name = model_url.replace("huggingface://", "")
            
            self.status.emit(f"در حال دانلود مدل Hugging Face {model_name}...")
            self.progress.emit(20)
            
            # دانلود tokenizer
            self.status.emit("در حال دانلود tokenizer...")
            self.progress.emit(40)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # دانلود مدل
            self.status.emit("در حال دانلود مدل...")
            self.progress.emit(70)
            model = AutoModel.from_pretrained(model_name)
            
            self.progress.emit(100)
            self.status.emit("دانلود کامل شد!")
            
            return True, f"مدل {model_name} با موفقیت دانلود شد"
            
        except Exception as e:
            return False, f"خطا در دانلود Hugging Face: {str(e)}"

class DownloadedModelsManager(QDialog):
    """مدیریت مدل‌های دانلود شده"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("مدیریت مدل‌های دانلود شده")
        self.setModal(True)
        self.resize(700, 500)
        self.setup_ui()
        self.refresh_models()
        
    def setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 5px;
                color: #ffffff;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: #ffffff;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#clear_button {
                background-color: #f44336;
            }
            QPushButton#clear_button:hover {
                background-color: #d32f2f;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # عنوان
        title_label = QLabel("مدل‌های دانلود شده")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # لیست مدل‌ها
        self.models_list = QListWidget()
        layout.addWidget(self.models_list)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        self.open_folder_button = QPushButton("📁 باز کردن فولدر")
        self.open_folder_button.clicked.connect(self.open_models_folder)
        button_layout.addWidget(self.open_folder_button)
        
        self.refresh_button = QPushButton("🔄 به‌روزرسانی")
        self.refresh_button.clicked.connect(self.refresh_models)
        button_layout.addWidget(self.refresh_button)
        
        self.clear_button = QPushButton("🗑️ پاک کردن همه")
        self.clear_button.setObjectName("clear_button")
        self.clear_button.clicked.connect(self.clear_all_models)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # دکمه بستن
        close_button = QPushButton("بستن")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
    def refresh_models(self):
        """به‌روزرسانی لیست مدل‌های دانلود شده"""
        self.models_list.clear()
        
        # بررسی مدل‌های Vosk
        vosk_dir = Path.home() / ".vosk" / "models"
        if vosk_dir.exists():
            for model_dir in vosk_dir.iterdir():
                if model_dir.is_dir():
                    model_info = self.get_model_info("vosk", model_dir.name)
                    if model_info:
                        item_text = f"🎯 {model_info['name']} ({model_info['size']}) - {model_info['language']}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, ("vosk", model_dir.name, str(model_dir)))
                        self.models_list.addItem(item)
        
        # بررسی مدل‌های Whisper
        whisper_dir = Path.home() / ".cache" / "whisper"
        if whisper_dir.exists():
            for model_file in whisper_dir.glob("*.pt"):
                model_name = model_file.stem
                model_info = self.get_model_info("whisper", model_name)
                if model_info:
                    item_text = f"🎤 {model_info['name']} ({model_info['size']}) - {model_info['language']}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, ("whisper", model_name, str(model_file)))
                    self.models_list.addItem(item)
        
        # بررسی مدل‌های Hugging Face
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache_dir.exists():
            for model_dir in hf_cache_dir.iterdir():
                if model_dir.is_dir():
                    # استخراج نام مدل از نام فولدر
                    model_name = model_dir.name.replace("--", "/")
                    model_info = self.get_model_info("huggingface", model_name)
                    if model_info:
                        item_text = f"🤗 {model_info['name']} ({model_info['size']}) - {model_info['language']}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, ("huggingface", model_name, str(model_dir)))
                        self.models_list.addItem(item)
        
        # نمایش تعداد مدل‌ها
        count = self.models_list.count()
        if count == 0:
            self.models_list.addItem("هیچ مدلی دانلود نشده است.")
        else:
            self.models_list.insertItem(0, f"تعداد مدل‌های دانلود شده: {count}")
            
    def get_model_info(self, model_type, model_name):
        """دریافت اطلاعات مدل"""
        for model_id, info in ModelDownloader.DOWNLOADABLE_MODELS.items():
            if info["type"] == model_type.title():
                if model_type == "whisper":
                    whisper_name = model_id.replace("whisper_", "")
                    if whisper_name == "large_v2":
                        whisper_name = "large-v2"
                    elif whisper_name == "large_v3":
                        whisper_name = "large-v3"
                    if whisper_name == model_name:
                        return info
                elif model_type == "vosk":
                    if info["name"] == model_name:
                        return info
                elif model_type == "huggingface":
                    # استخراج نام مدل از URL
                    model_url = info["url"]
                    if model_url.startswith("huggingface://"):
                        hf_model_name = model_url.replace("huggingface://", "")
                        if hf_model_name == model_name:
                            return info
        return None
        
    def open_models_folder(self):
        """باز کردن فولدر مدل‌ها"""
        import subprocess
        import platform
        import os
        
        current_item = self.models_list.currentItem()
        if not current_item or not current_item.data(Qt.UserRole):
            # باز کردن فولدر اصلی - اول Vosk را چک کن
            models_dir = Path.home() / ".vosk" / "models"
            if not models_dir.exists():
                models_dir = Path.home() / ".cache" / "whisper"
                # اگر فولدر whisper هم وجود ندارد، آن را ایجاد کن
                if not models_dir.exists():
                    models_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_type, model_name, model_path = current_item.data(Qt.UserRole)
            models_dir = Path(model_path).parent
        
        # اطمینان از وجود فولدر
        if not models_dir.exists():
            try:
                models_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "خطا", f"نمی‌توان فولدر را ایجاد کرد:\n{str(e)}")
                return
        
        try:
            if platform.system() == "Windows":
                # استفاده از os.startfile که بهتر کار می‌کند
                os.startfile(str(models_dir))
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(models_dir)], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", str(models_dir)], check=True)
        except Exception as e:
            # اگر os.startfile کار نکرد، از subprocess استفاده کن
            try:
                if platform.system() == "Windows":
                    subprocess.run(["explorer", str(models_dir)], check=True)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", str(models_dir)], check=True)
                else:
                    subprocess.run(["xdg-open", str(models_dir)], check=True)
            except Exception as e2:
                QMessageBox.warning(self, "خطا", f"نمی‌توان فولدر را باز کرد:\n{str(e2)}")
            
    def clear_all_models(self):
        """پاک کردن همه مدل‌ها"""
        reply = QMessageBox.question(
            self, "تأیید پاک کردن", 
            "آیا مطمئن هستید که می‌خواهید همه مدل‌های دانلود شده را پاک کنید؟\nاین عمل قابل بازگشت نیست!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # پاک کردن مدل‌های Vosk
                vosk_dir = Path.home() / ".vosk" / "models"
                if vosk_dir.exists():
                    import shutil
                    shutil.rmtree(vosk_dir)
                
                # پاک کردن مدل‌های Whisper
                whisper_dir = Path.home() / ".cache" / "whisper"
                if whisper_dir.exists():
                    import shutil
                    shutil.rmtree(whisper_dir)
                
                # پاک کردن مدل‌های Hugging Face
                hf_cache_dir = Path.home() / ".cache" / "huggingface"
                if hf_cache_dir.exists():
                    import shutil
                    shutil.rmtree(hf_cache_dir)
                
                QMessageBox.information(self, "موفق", "همه مدل‌های دانلود شده پاک شدند.")
                self.refresh_models()
                
            except Exception as e:
                QMessageBox.critical(self, "خطا", f"خطا در پاک کردن مدل‌ها:\n{str(e)}")

class ModelSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("انتخاب مدل Speech-to-Text")
        self.setModal(True)
        self.resize(600, 500)
        
        # تم دارک برای کل دیالوگ
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # فیلترهای مدل
        filter_layout = QHBoxLayout()
        
        # چک باکس زبان
        self.checkbox_persian = QCheckBox("فارسی")
        self.checkbox_persian.setChecked(True)
        self.checkbox_persian.setStyleSheet("""
            QCheckBox {
                font-weight: bold; 
                color: #81c784;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #81c784;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #81c784;
                border: 2px solid #4caf50;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #a5d6a7;
            }
        """)
        self.checkbox_persian.stateChanged.connect(self.filter_models)
        
        self.checkbox_english = QCheckBox("انگلیسی")
        self.checkbox_english.setChecked(True)
        self.checkbox_english.setStyleSheet("""
            QCheckBox {
                font-weight: bold; 
                color: #64b5f6;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #64b5f6;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #64b5f6;
                border: 2px solid #2196f3;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #90caf9;
            }
        """)
        self.checkbox_english.stateChanged.connect(self.filter_models)
        
        self.checkbox_multilingual = QCheckBox("چند زبانه")
        self.checkbox_multilingual.setChecked(True)
        self.checkbox_multilingual.setStyleSheet("""
            QCheckBox {
                font-weight: bold; 
                color: #ffd54f;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #ffd54f;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #ffd54f;
                border: 2px solid #ffc107;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #fff176;
            }
        """)
        self.checkbox_multilingual.stateChanged.connect(self.filter_models)
        
        # چک باکس نوع اتصال
        self.checkbox_online = QCheckBox("آنلاین")
        self.checkbox_online.setChecked(True)
        self.checkbox_online.setStyleSheet("""
            QCheckBox {
                font-weight: bold; 
                color: #ff8a65;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #ff8a65;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #ff8a65;
                border: 2px solid #ff5722;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #ffab91;
            }
        """)
        self.checkbox_online.stateChanged.connect(self.filter_models)
        
        self.checkbox_offline = QCheckBox("آفلاین")
        self.checkbox_offline.setChecked(True)
        self.checkbox_offline.setStyleSheet("""
            QCheckBox {
                font-weight: bold; 
                color: #9575cd;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #9575cd;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #9575cd;
                border: 2px solid #7e57c2;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #b39ddb;
            }
        """)
        self.checkbox_offline.stateChanged.connect(self.filter_models)
        
        # برچسب زبان
        lang_label = QLabel("زبان:")
        lang_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px; padding: 5px;")
        filter_layout.addWidget(lang_label)
        filter_layout.addWidget(self.checkbox_persian)
        filter_layout.addWidget(self.checkbox_english)
        filter_layout.addWidget(self.checkbox_multilingual)
        filter_layout.addStretch()
        
        # برچسب نوع
        type_label = QLabel("نوع:")
        type_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px; padding: 5px;")
        filter_layout.addWidget(type_label)
        filter_layout.addWidget(self.checkbox_online)
        filter_layout.addWidget(self.checkbox_offline)
        
        layout.addLayout(filter_layout)
        
        # لیست مدل‌های Speech-to-Text (فقط فارسی و انگلیسی)
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(300)
        self.model_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 8px;
                color: #ffffff;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QListWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #444;
                border-radius: 4px;
                margin: 2px 0px;
                color: #ffffff;
                background-color: #3a3a3a;
            }
            QListWidget::item:hover {
                background-color: #4a4a4a;
                border: 1px solid #666;
            }
            QListWidget::item:selected {
                background-color: #1976d2;
                color: #ffffff;
                border: 2px solid #42a5f5;
                font-weight: bold;
            }
            QListWidget::item:selected:hover {
                background-color: #1565c0;
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
            ("whisper_large", "✅ Whisper Large - بالاترین دقت (3.1 GB)", "both", "offline"),
            ("whisper_large_v2", "✅ Whisper Large V2 - جدیدترین نسخه (3.1 GB)", "both", "offline"),
            ("whisper_large_v3", "✅ Whisper Large V3 - جدیدترین نسخه (3.1 GB)", "both", "offline"),
            
            # Hugging Face Transformers (آفلاین)
            ("hf_wav2vec2_persian", "✅ Wav2Vec2 Persian - مخصوص فارسی (1.2 GB)", "persian", "offline"),
            ("hf_wav2vec2_persian_v3", "🏆 Wav2Vec2 Persian V3 - بهترین کیفیت فارسی (1.2 GB)", "persian", "offline"),
            ("hf_wav2vec2_persian_jonatas", "⭐ Wav2Vec2 Persian Jonatas - مدل بهینه شده (1.2 GB)", "persian", "offline"),
            ("hf_whisper_large_v3_persian", "✅ Whisper Large V3 Persian - بهترین کیفیت (3.1 GB)", "persian", "offline"),
            ("hf_whisper_large_v3_persian_alt", "✅ Whisper Large V3 Persian Alt - جایگزین (3.1 GB)", "persian", "offline"),
            ("hf_whisper_large_persian_steja", "✅ Whisper Large Persian Steja - مخصوص فارسی (3.1 GB)", "persian", "offline"),
            ("hf_wav2vec2_persian_alt", "⚠️ Wav2Vec2 Multilingual - چند زبانه (1.2 GB)", "both", "offline"),
            ("hf_whisper_tiny", "⚠️ Whisper Tiny HF - ضعیف برای فارسی (75 MB)", "both", "offline"),
            ("hf_whisper_base", "⚠️ Whisper Base HF - ضعیف برای فارسی (142 MB)", "both", "offline"),
            ("hf_whisper_small", "✅ Whisper Small HF - تعادل خوب (466 MB)", "both", "offline"),
            ("hf_whisper_medium", "✅ Whisper Medium HF - دقت بالا (1.5 GB)", "both", "offline"),
            ("hf_whisper_large", "✅ Whisper Large HF - بالاترین دقت (3.1 GB)", "both", "offline"),
            ("hf_whisper_large_v2", "✅ Whisper Large V2 HF - جدیدترین نسخه (3.1 GB)", "both", "offline"),
            ("hf_whisper_large_v3", "✅ Whisper Large V3 HF - جدیدترین نسخه (3.1 GB)", "both", "offline"),
            
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
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("لغو")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c1170a;
            }
        """)
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
            if language == "both" and self.checkbox_multilingual.isChecked():
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
                
                # رنگ‌بندی بر اساس نوع اتصال
                if connection_type == "online":
                    # آنلاین - نارنجی/قرمز تیره
                    item.setBackground(QColor("#4a2c2a"))  # قرمز تیره
                    item.setForeground(QColor("#ff8a65"))  # نارنجی روشن
                else:
                    # آفلاین - بنفش/آبی تیره
                    item.setBackground(QColor("#2a2c4a"))  # بنفش تیره
                    item.setForeground(QColor("#9575cd"))  # بنفش روشن
                
                # رنگ‌بندی بر اساس زبان
                if language == "persian":
                    # فارسی - سبز
                    item.setForeground(QColor("#81c784"))  # سبز روشن
                elif language == "english":
                    # انگلیسی - آبی
                    item.setForeground(QColor("#64b5f6"))  # آبی روشن
                elif language == "both":
                    # چند زبانه - زرد
                    item.setForeground(QColor("#ffd54f"))  # زرد روشن
                
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
    """بهبود متن فارسی با تصحیح خودکار ساده"""
    if not text or not text.strip():
        return text
    
    # تصحیح کاراکترهای فارسی
    if PERSIAN_TOOLS_AVAILABLE:
        text = characters.ar_to_fa(text)  # تبدیل عربی به فارسی
        text = digits.en_to_fa(text)  # تبدیل اعداد انگلیسی به فارسی
    
    # تصحیح فاصله‌گذاری ساده
    text = text.replace("  ", " ")  # حذف فاصله‌های اضافی
    text = text.replace(" .", ".")  # تصحیح نقطه
    text = text.replace(" ,", ",")  # تصحیح ویرگول
    text = text.replace(" :", ":")  # تصحیح دو نقطه
    text = text.replace(" ;", ";")  # تصحیح نقطه‌ویرگول
    text = text.replace(" !", "!")  # تصحیح علامت تعجب
    text = text.replace(" ?", "?")  # تصحیح علامت سوال
    
    return text.strip()


class TranscribeThread(QThread):
    progress = Signal(int)
    finished = Signal(str)
    download_status = Signal(str)

    def __init__(self, audio_path, model_name="vosk_persian"):
        super().__init__()
        self.audio_path = audio_path
        self.model_name = model_name
        self.model = None
        self._pause = False
        self._stop = False
        self.silero_models = {}  # Cache برای مدل‌های Silero

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
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری مدل Whisper {whisper_model}...")
                
                self.model = whisper.load_model(whisper_model)
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"مدل Whisper {whisper_model} بارگذاری شد")
            elif self.model_name.startswith("google_"):
                if not GOOGLE_SPEECH_AVAILABLE:
                    self.finished.emit("Error: Google Speech-to-Text not installed. Install with: pip install google-cloud-speech")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال اتصال به Google Speech {self.model_name}...")
                
                self.model = "google"  # نشانگر استفاده از Google
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"Google Speech {self.model_name} آماده")
            elif self.model_name.startswith("vosk_"):
                if not VOSK_AVAILABLE:
                    self.finished.emit("Error: Vosk not installed. Install with: pip install vosk")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری مدل Vosk {self.model_name}...")
                
                self.model = "vosk"  # نشانگر استفاده از Vosk
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"مدل Vosk {self.model_name} آماده")
            elif self.model_name.startswith("azure_"):
                if not AZURE_SPEECH_AVAILABLE:
                    self.finished.emit("Error: Azure Speech not installed. Install with: pip install azure-cognitiveservices-speech")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال اتصال به Azure Speech {self.model_name}...")
                
                self.model = "azure"  # نشانگر استفاده از Azure
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"Azure Speech {self.model_name} آماده")
            elif self.model_name.startswith("assemblyai_"):
                if not ASSEMBLYAI_AVAILABLE:
                    self.finished.emit("Error: AssemblyAI not installed. Install with: pip install assemblyai")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال اتصال به AssemblyAI {self.model_name}...")
                
                self.model = "assemblyai"  # نشانگر استفاده از AssemblyAI
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"AssemblyAI {self.model_name} آماده")
            elif self.model_name.startswith("hf_"):
                if not HUGGINGFACE_AVAILABLE:
                    self.finished.emit("Error: Hugging Face Transformers not installed. Install with: pip install transformers torch")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری مدل Hugging Face {self.model_name}...")
                
                self.model = "huggingface"  # نشانگر استفاده از Hugging Face
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"مدل Hugging Face {self.model_name} آماده")
            elif self.model_name.startswith("speechrecognition_"):
                if not SPEECHRECOGNITION_AVAILABLE:
                    self.finished.emit("Error: SpeechRecognition not installed. Install with: pip install SpeechRecognition")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری مدل SpeechRecognition {self.model_name}...")
                
                self.model = "speechrecognition"  # نشانگر استفاده از SpeechRecognition
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"مدل SpeechRecognition {self.model_name} آماده")
            elif self.model_name.startswith("silero_"):
                if not SILERO_AVAILABLE:
                    self.finished.emit("Error: Silero STT not installed. Install with: pip install torchaudio")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری مدل Silero STT {self.model_name}...")
                
                self.model = "silero"  # نشانگر استفاده از Silero
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"مدل Silero STT {self.model_name} آماده")
            elif self.model_name.startswith("kaldi_"):
                if not KALDI_AVAILABLE:
                    self.finished.emit("Error: Kaldi not installed. Install with: pip install kaldi-io")
                    return
                
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری مدل Kaldi {self.model_name}...")
                
                self.model = "kaldi"  # نشانگر استفاده از Kaldi
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"مدل Kaldi {self.model_name} آماده")
            elif self.model_name.startswith("iranian_"):
                # نمایش پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"در حال بارگذاری سرویس ایرانی {self.model_name}...")
                
                self.model = "iranian"  # نشانگر استفاده از سرویس‌های ایرانی
                
                # پنهان کردن پیام بارگذاری
                if hasattr(self, 'download_status'):
                    self.download_status.emit(f"سرویس ایرانی {self.model_name} آماده")
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
                # دانلود مدل با نوتیفیکیشن ساده
                def progress_callback(message):
                    self.download_status.emit(message)
                
                success, result = ModelDownloader.download_model(
                    self.model_name, 
                    progress_callback=progress_callback
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
                    model_name = "m3hrdadfi/wav2vec2-large-xlsr-persian"
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل فارسی در دسترس نیست

مشکل: مدل m3hrdadfi/wav2vec2-large-xlsr-persian یافت نشد

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. از مدل جدیدتر استفاده کنید: Wav2Vec2 Persian V3
3. Token Hugging Face خود را تنظیم کنید:
   huggingface-cli login
   یا
   hf auth login
"""
                    else:
                        return f"Hugging Face Error: {error_msg}. لطفاً از Vosk Persian یا Whisper استفاده کنید."
                        
            elif self.model_name == "hf_wav2vec2_persian_v3":
                # تلاش برای بارگذاری مدل فارسی V3 (جدیدترین نسخه)
                try:
                    model_name = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل فارسی V3 در دسترس نیست

مشکل: مدل m3hrdadfi/wav2vec2-large-xlsr-persian-v3 یافت نشد

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. از مدل‌های جایگزین استفاده کنید:
   • Wav2Vec2 Persian (نسخه قبلی)
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)

برای استفاده از Hugging Face:
1. به https://huggingface.co بروید
2. حساب کاربری بسازید
3. از دستور زیر استفاده کنید:
   hf auth login
"""
                    else:
                        return f"Hugging Face Error: {error_msg}. لطفاً از Vosk Persian یا Whisper استفاده کنید."
                        
            elif self.model_name == "hf_wav2vec2_persian_jonatas":
                # تلاش برای بارگذاری مدل فارسی Jonatas (بهینه شده)
                try:
                    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = AutoModelForCTC.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل فارسی Jonatas در دسترس نیست

مشکل: مدل jonatasgrosman/wav2vec2-large-xlsr-53-persian یافت نشد

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. از مدل‌های جایگزین استفاده کنید:
   • Wav2Vec2 Persian V3 (جدیدترین)
   • Wav2Vec2 Persian (نسخه قبلی)
   • Vosk Persian (بهترین برای فارسی)

برای استفاده از Hugging Face:
1. به https://huggingface.co بروید
2. حساب کاربری بسازید
3. از دستور زیر استفاده کنید:
   hf auth login
"""
                    else:
                        return f"Hugging Face Error: {error_msg}. لطفاً از Vosk Persian یا Whisper استفاده کنید."
                        
                    
            elif self.model_name == "hf_whisper_large_v3_persian":
                # تلاش برای بارگذاری مدل Whisper فارسی
                try:
                    from transformers import WhisperForConditionalGeneration, WhisperProcessor
                    model_name = "nezamisafa/whisper-large-v3-persian"
                    processor = WhisperProcessor.from_pretrained(model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل Whisper فارسی در دسترس نیست

مشکل: مدل nezamisafa/whisper-large-v3-persian یافت نشد

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
                        
            elif self.model_name == "hf_whisper_large_v3_persian_alt":
                # تلاش برای بارگذاری مدل Whisper فارسی جایگزین
                try:
                    from transformers import WhisperForConditionalGeneration, WhisperProcessor
                    model_name = "MohammadKhosravi/whisper-large-v3-Persian"
                    processor = WhisperProcessor.from_pretrained(model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل Whisper فارسی در دسترس نیست

مشکل: مدل MohammadKhosravi/whisper-large-v3-Persian یافت نشد

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
                        
            elif self.model_name == "hf_whisper_large_persian_steja":
                # تلاش برای بارگذاری مدل Whisper فارسی Steja
                try:
                    from transformers import WhisperForConditionalGeneration, WhisperProcessor
                    model_name = "steja/whisper-large-persian"
                    processor = WhisperProcessor.from_pretrained(model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
                except Exception as e:
                    error_msg = str(e)
                    if "not a valid model identifier" in error_msg:
                        return f"""Hugging Face Error: مدل Whisper فارسی Steja در دسترس نیست

مشکل: مدل steja/whisper-large-persian یافت نشد

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. از مدل‌های جایگزین استفاده کنید:
   • Wav2Vec2 Persian V3 (بهترین کیفیت)
   • Wav2Vec2 Persian (نسخه قبلی)
   • Vosk Persian (بهترین برای فارسی)

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
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                model_name = self.model_name.replace("hf_", "").replace("_hf", "")
                if model_name == "whisper_large":
                    model_name = "openai/whisper-large-v2"
                elif model_name == "whisper_large_v2":
                    model_name = "openai/whisper-large-v2"
                elif model_name == "whisper_large_v3":
                    model_name = "openai/whisper-large-v3"
                else:
                    model_name = f"openai/whisper-{model_name}"
                
                try:
                    processor = WhisperProcessor.from_pretrained(model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
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
            if not SILERO_AVAILABLE:
                return """Silero STT Error: Dependencies not installed

برای نصب Silero STT:

1. ابتدا omegaconf را نصب کنید:
   pip install omegaconf

2. سپس PyTorch را نصب کنید:
   برای CPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   برای GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

3. سپس برنامه را دوباره اجرا کنید

یا از مدل‌های دیگر استفاده کنید:
• Vosk Persian (بهترین برای فارسی)
• Whisper Medium/Large (چند زبانه)
• Google Speech (آنلاین)
"""
            
            import torch
            import torchaudio
            import omegaconf
            
            # بررسی نسخه PyTorch
            torch_version = torch.__version__
            
            # بررسی cache مدل
            model_key = f"silero_{self.model_name}"
            if model_key in self.silero_models:
                model, decoder, utils = self.silero_models[model_key]
                if hasattr(self, 'download_status'):
                    self.download_status.emit("استفاده از مدل Silero STT موجود")
            else:
                # بارگذاری مدل با تنظیمات بهتر
                try:
                    # تنظیمات torch.hub برای حل مشکل اتصال
                    import os
                    os.environ['TORCH_HOME'] = str(Path.home() / '.cache' / 'torch')
                    
                    # تنظیم timeout برای دانلود
                    import socket
                    socket.setdefaulttimeout(60)
                    
                    # نمایش پیام دانلود
                    if hasattr(self, 'download_status'):
                        self.download_status.emit("در حال بارگذاری مدل Silero STT...")
                    
                    if self.model_name == "silero_stt_en":
                        model, decoder, utils = torch.hub.load(
                            repo_or_dir='snakers4/silero-models', 
                            model='silero_stt', 
                            language='en',
                            force_reload=False,
                            trust_repo=True,
                            verbose=True  # نمایش پیشرفت دانلود
                        )
                    else:  # silero_stt_multilingual
                        model, decoder, utils = torch.hub.load(
                            repo_or_dir='snakers4/silero-models', 
                            model='silero_stt', 
                            language='multilingual',
                            force_reload=False,
                            trust_repo=True,
                            verbose=True  # نمایش پیشرفت دانلود
                        )
                    
                    # ذخیره در cache
                    self.silero_models[model_key] = (model, decoder, utils)
                    
                    # پنهان کردن پیام دانلود
                    if hasattr(self, 'download_status'):
                        self.download_status.emit("مدل Silero STT بارگذاری شد")
                
                except Exception as model_error:
                    error_msg = str(model_error)
                    
                    # اگر خطا خالی است، پیام پیش‌فرض نمایش دهید
                    if not error_msg or error_msg.strip() == "":
                        error_msg = "خطای نامشخص در بارگذاری مدل"
                    
                    # تشخیص نوع خطا
                    if "SSL" in error_msg or "certificate" in error_msg.lower():
                        return f"""Silero STT Error: مشکل SSL Certificate

مشکل: {error_msg}

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. VPN را خاموش کنید (اگر استفاده می‌کنید)
3. فایروال را بررسی کنید
4. از مدل‌های دیگر استفاده کنید:
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)
   • Google Speech (آنلاین)
"""
                    elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                        return f"""Silero STT Error: مشکل اتصال

مشکل: {error_msg}

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. مدل را دوباره دانلود کنید:
   - پوشه ~/.cache/torch/hub را حذف کنید
   - برنامه را دوباره اجرا کنید
3. از مدل‌های دیگر استفاده کنید:
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)
   • Google Speech (آنلاین)
"""
                    else:
                        return f"""Silero STT Error: مدل بارگذاری نشد

مشکل: {error_msg}

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. مدل را دوباره دانلود کنید:
   - پوشه ~/.cache/torch/hub را حذف کنید
   - برنامه را دوباره اجرا کنید
3. از مدل‌های دیگر استفاده کنید:
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)
   • Google Speech (آنلاین)
"""
            
            # بارگذاری فایل صوتی
            try:
                audio, sample_rate = torchaudio.load(audio_file)
                
                # تبدیل به mono
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                
                # تبدیل sample rate به 16000 اگر لازم باشد
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio = resampler(audio)
                
                # تشخیص
                with torch.no_grad():
                    text = decoder(model(audio[0]))
                
                return text.strip()
                
            except Exception as audio_error:
                return f"""Silero STT Error: پردازش فایل صوتی

مشکل: {str(audio_error)}

راه‌حل‌ها:
1. فایل صوتی را بررسی کنید
2. از فرمت‌های پشتیبانی شده استفاده کنید (WAV, MP3, M4A)
3. از مدل‌های دیگر استفاده کنید:
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)
   • Google Speech (آنلاین)
"""
            
        except Exception as e:
            error_msg = str(e)
            if "No module named 'torch'" in error_msg:
                return """Silero STT Error: PyTorch not installed

برای نصب PyTorch:

1. ابتدا omegaconf را نصب کنید:
   pip install omegaconf

2. سپس PyTorch را نصب کنید:
   برای CPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   برای GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

سپس برنامه را دوباره اجرا کنید.
"""
            elif "No module named 'torchaudio'" in error_msg:
                return """Silero STT Error: TorchAudio not installed

برای نصب TorchAudio:

1. ابتدا omegaconf را نصب کنید:
   pip install omegaconf

2. سپس PyTorch را نصب کنید:
   برای CPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   برای GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

سپس برنامه را دوباره اجرا کنید.
"""
            elif "No module named 'omegaconf'" in error_msg:
                return """Silero STT Error: OmegaConf not installed

برای نصب OmegaConf:

1. ابتدا omegaconf را نصب کنید:
   pip install omegaconf

2. سپس PyTorch را نصب کنید:
   برای CPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   برای GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

سپس برنامه را دوباره اجرا کنید.
"""
            else:
                return f"""Silero STT Error: {error_msg}

راه‌حل‌ها:
1. اتصال اینترنت خود را بررسی کنید
2. وابستگی‌ها را دوباره نصب کنید:
   pip install omegaconf
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
3. از مدل‌های دیگر استفاده کنید:
   • Vosk Persian (بهترین برای فارسی)
   • Whisper Medium/Large (چند زبانه)
   • Google Speech (آنلاین)
"""
    
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

        # دکمه مدیریت مدل‌های دانلود شده
        self.btn_manage_models = QPushButton("📦 مدیریت مدل‌های دانلود شده")
        self.btn_manage_models.setMinimumHeight(40)
        self.btn_manage_models.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; 
                color: white; 
                font-weight: bold;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
        """)
        self.btn_manage_models.clicked.connect(self.open_models_manager)
        self.layout.addWidget(self.btn_manage_models)

        # Help menu button
        self.btn_help = QPushButton("📚 راهنما و تنظیمات")
        self.btn_help.setMinimumHeight(40)
        self.btn_help.setStyleSheet("""
            QPushButton {
                background-color: #607D8B; 
                color: white; 
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #455A64;
            }
        """)
        self.btn_help.clicked.connect(self.show_help_menu)
        self.layout.addWidget(self.btn_help)

        self.btn_download_models = QPushButton("Download Models")
        self.btn_download_models.setMinimumHeight(40)
        self.btn_download_models.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
            QPushButton:pressed {
                background-color: #cc4a1f;
            }
        """)
        self.btn_download_models.clicked.connect(self.show_download_models_dialog)
        self.layout.addWidget(self.btn_download_models)

        self.btn_change_model = QPushButton("Change Model")
        self.btn_change_model.setMinimumHeight(40)
        self.btn_change_model.setStyleSheet("""
            QPushButton {
                background-color: #9c27b0; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8e24aa;
            }
            QPushButton:pressed {
                background-color: #7b1fa2;
            }
        """)
        self.btn_change_model.clicked.connect(self.change_model)
        self.layout.addWidget(self.btn_change_model)

        # Model status label - simple text without button styling
        self.model_status = QLabel("مدل بارگذاری نشده")
        self.model_status.setStyleSheet("""
            color: #666666; 
            font-size: 14px;
            padding: 8px;
            background-color: transparent;
            border: none;
            outline: none;
        """)
        self.model_status.setAlignment(Qt.AlignCenter)
        self.model_status.setFocusPolicy(Qt.NoFocus)  # Remove focus ability
        self.layout.addWidget(self.model_status)
        
        # Progress bar for conversion
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 14px;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 1 #45a049);
                border-radius: 6px;
                margin: 1px;
            }
        """)
        self.layout.addWidget(self.progress)
        
        # Temporary notification label
        self.notification_label = QLabel("")
        self.notification_label.setVisible(False)
        self.notification_label.setStyleSheet("""
            color: #ffffff; 
            font-weight: bold; 
            font-size: 14px;
            padding: 12px;
            background-color: #424242;
            border: 2px solid #2196F3;
            border-radius: 8px;
        """)
        self.notification_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.notification_label)

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
            
            # بررسی دانلود مدل
            if not self.check_and_download_model():
                return  # کاربر انصراف داد یا خطا رخ داد
            
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
            
            # شروع کانورت
            self.start_conversion()
        else:
            return  # کاربر cancel کرد

    def check_and_download_model(self):
        """بررسی و دانلود مدل در صورت نیاز"""
        # بررسی اینکه آیا مدل نیاز به دانلود دارد
        if self.selected_model not in ModelDownloader.DOWNLOADABLE_MODELS:
            return True  # مدل آنلاین است یا نیازی به دانلود ندارد
        
        model_info = ModelDownloader.DOWNLOADABLE_MODELS[self.selected_model]
        
        # بررسی اینکه آیا مدل قبلاً دانلود شده
        if ModelDownloader.is_model_downloaded(self.selected_model):
            return True  # مدل قبلاً دانلود شده
        
        # نمایش پیام تأیید دانلود
        reply = QMessageBox.question(
            self, "دانلود مدل", 
            f"مدل {model_info['name']} ({model_info['size']}) دانلود نشده است.\n\nآیا می‌خواهید آن را دانلود کنید؟",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return False  # کاربر انصراف داد
        
        # نمایش دیالوگ دانلود با نوار پیشرفت
        download_dialog = ModelDownloadDialog(self.selected_model, model_info, self)
        download_dialog.start_download()  # شروع دانلود
        if download_dialog.exec() == QDialog.Accepted:
            # بعد از اتمام دانلود، از کاربر بپرس که آیا می‌خواهد کانورت را شروع کند
            reply = QMessageBox.question(
                self, "شروع کانورت", 
                f"مدل {model_info['name']} با موفقیت دانلود شد.\n\nآیا می‌خواهید کانورت را شروع کنید؟",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            return reply == QMessageBox.Yes
        else:
            return False  # دانلود لغو شد

    def start_conversion(self):
        """شروع فرآیند کانورت"""
        self.text_edit.clear()
        self.output_file = None
        self.thread = TranscribeThread(self.audio_path, self.selected_model)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.display_result)
        self.thread.download_status.connect(self.show_notification)
        self.thread.start()

    def show_notification(self, message):
        """نمایش نوتیفیکیشن موقت"""
        from PySide6.QtCore import QTimer
        
        # Update model status when model is ready
        if "آماده" in message or "بارگذاری شد" in message:
            self.model_status.setText("✅ مدل بارگذاری شد")
            self.model_status.setStyleSheet("""
                color: #4CAF50; 
                font-size: 14px;
                padding: 8px;
                background-color: transparent;
                border: none;
                outline: none;
            """)
        
        # Show temporary notification
        self.notification_label.setText(f"📥 {message}")
        self.notification_label.setVisible(True)
        
        # Auto-hide after 3 seconds
        QTimer.singleShot(3000, self.hide_notification)
    
    def hide_notification(self):
        """مخفی کردن نوتیفیکیشن"""
        self.notification_label.setVisible(False)

    def display_result(self, text):
        # Add model information below the converted text
        model_info = f"\n\n---\nکانورت شده با مدل: {self.selected_model}"
        full_text = text + model_info
        
        self.text_edit.setPlainText(full_text)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Text", "", "Text Files (*.txt)"
        )
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            self.output_file = save_path
        
        # Reset thread to allow re-conversion
        self.thread = None

    def open_text_file(self):
        if self.output_file and os.path.exists(self.output_file):
            webbrowser.open(self.output_file)
        else:
            QMessageBox.warning(self, "Warning", "No saved text file to open!")

    def toggle_start_pause(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Warning", "No audio file selected!")
            return

        if not self.thread or not self.thread.isRunning():
            # Allow re-conversion of the same file
            self.start_transcription()
        else:
            QMessageBox.information(self, "Info", "Conversion is already in progress. Please wait for it to complete.")

    def stop_processing(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "Info", "Stop clicked (future implementation).")

    def open_dict_manager(self):
        if not self.dict_manager_window:
            self.dict_manager_window = DictManager()
        self.dict_manager_window.show()
        self.dict_manager_window.raise_()
        self.dict_manager_window.activateWindow()

    def open_models_manager(self):
        """باز کردن پنجره مدیریت مدل‌های دانلود شده"""
        dialog = DownloadedModelsManager(self)
        dialog.exec()

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

    def show_help_menu(self):
        """نمایش منوی راهنما و تنظیمات"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QScrollArea, QWidget
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("📚 راهنما و تنظیمات")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("راهنما و تنظیمات برنامه")
        title_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #1976D2; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Scroll area for buttons
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Setup guides section
        setup_label = QLabel("🔧 راهنمای تنظیم مدل‌ها:")
        setup_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #424242; padding: 5px;")
        scroll_layout.addWidget(setup_label)
        
        # Google Setup
        btn_google_setup = QPushButton("Google Speech Setup Guide")
        btn_google_setup.setMinimumHeight(40)
        btn_google_setup.setStyleSheet("""
            QPushButton {
                background-color: #4285f4; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3367d6;
            }
            QPushButton:pressed {
                background-color: #2c5aa0;
            }
        """)
        btn_google_setup.clicked.connect(lambda: (dialog.accept(), self.show_google_setup_guide()))
        scroll_layout.addWidget(btn_google_setup)
        
        # Hugging Face Setup
        btn_huggingface_setup = QPushButton("Hugging Face Setup Guide")
        btn_huggingface_setup.setMinimumHeight(40)
        btn_huggingface_setup.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
            QPushButton:pressed {
                background-color: #cc4a1f;
            }
        """)
        btn_huggingface_setup.clicked.connect(lambda: (dialog.accept(), self.show_huggingface_setup_guide()))
        scroll_layout.addWidget(btn_huggingface_setup)
        
        # SpeechRecognition Setup
        btn_speechrecognition_setup = QPushButton("SpeechRecognition Setup Guide")
        btn_speechrecognition_setup.setMinimumHeight(40)
        btn_speechrecognition_setup.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        btn_speechrecognition_setup.clicked.connect(lambda: (dialog.accept(), self.show_speechrecognition_setup_guide()))
        scroll_layout.addWidget(btn_speechrecognition_setup)
        
        # PyTorch Install Guide
        btn_install_pytorch = QPushButton("نصب PyTorch (برای Silero)")
        btn_install_pytorch.setMinimumHeight(40)
        btn_install_pytorch.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8E24AA;
            }
            QPushButton:pressed {
                background-color: #7B1FA2;
            }
        """)
        btn_install_pytorch.clicked.connect(lambda: (dialog.accept(), self.show_pytorch_install_guide()))
        scroll_layout.addWidget(btn_install_pytorch)
        
        # Test and maintenance section
        test_label = QLabel("🧪 تست و نگهداری:")
        test_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #424242; padding: 5px; margin-top: 10px;")
        scroll_layout.addWidget(test_label)
        
        # Test Silero
        btn_test_silero = QPushButton("تست Silero STT")
        btn_test_silero.setMinimumHeight(40)
        btn_test_silero.setStyleSheet("""
            QPushButton {
                background-color: #FF5722; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:pressed {
                background-color: #D84315;
            }
        """)
        btn_test_silero.clicked.connect(lambda: (dialog.accept(), self.test_silero_stt()))
        scroll_layout.addWidget(btn_test_silero)
        
        # Clear Silero Cache
        btn_clear_silero_cache = QPushButton("پاک کردن Cache Silero")
        btn_clear_silero_cache.setMinimumHeight(40)
        btn_clear_silero_cache.setStyleSheet("""
            QPushButton {
                background-color: #795548; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #6D4C41;
            }
            QPushButton:pressed {
                background-color: #5D4037;
            }
        """)
        btn_clear_silero_cache.clicked.connect(lambda: (dialog.accept(), self.clear_silero_cache()))
        scroll_layout.addWidget(btn_clear_silero_cache)
        
        # Add stretch to push buttons to top
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("بستن")
        close_btn.setMinimumHeight(40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B; 
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #455A64;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        close_layout.addWidget(close_btn)
        close_layout.addStretch()
        layout.addLayout(close_layout)
        
        dialog.exec()

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
        
        
        layout.addWidget(tab_widget)
        
        # ذخیره reference ها
        dialog.vosk_list = vosk_list
        dialog.whisper_list = whisper_list
        dialog.hf_list = hf_list
        dialog.sr_list = sr_list
        dialog.silero_list = silero_list
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
    
    
    def show_pytorch_install_guide(self):
        """نمایش راهنمای نصب PyTorch"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("راهنمای نصب PyTorch برای Silero STT")
        dialog.setModal(True)
        dialog.resize(700, 600)
        
        layout = QVBoxLayout(dialog)
        
        # متن راهنما
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h2>راهنمای نصب PyTorch برای Silero STT</h2>
        
        <h3>🔧 مراحل نصب:</h3>
        
        <h4>1️⃣ تشخیص نوع سیستم:</h4>
        <p>ابتدا نوع سیستم خود را تشخیص دهید:</p>
        <p>• <strong>CPU:</strong> اگر کارت گرافیک ندارید یا نمی‌خواهید از GPU استفاده کنید</p>
        <p>• <strong>GPU:</strong> اگر کارت گرافیک NVIDIA دارید و می‌خواهید از آن استفاده کنید</p>
        
        <h4>2️⃣ نصب omegaconf:</h4>
        <p>ابتدا omegaconf را نصب کنید:</p>
        <p><code>pip install omegaconf</code></p>
        
        <h4>3️⃣ نصب برای CPU:</h4>
        <p>در Command Prompt یا Terminal اجرا کنید:</p>
        <p><code>pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu</code></p>
        
        <h4>4️⃣ نصب برای GPU (CUDA 11.8):</h4>
        <p>در Command Prompt یا Terminal اجرا کنید:</p>
        <p><code>pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118</code></p>
        
        <h4>5️⃣ نصب برای GPU (CUDA 12.1):</h4>
        <p>در Command Prompt یا Terminal اجرا کنید:</p>
        <p><code>pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121</code></p>
        
        <h4>6️⃣ بررسی نصب:</h4>
        <p>برای بررسی نصب، در Python اجرا کنید:</p>
        <p><code>import torch</code></p>
        <p><code>print(torch.__version__)</code></p>
        <p><code>print(torch.cuda.is_available())  # برای GPU</code></p>
        
        <h3>💡 نکات مهم:</h3>
        <p>• اگر قبلاً PyTorch نصب کرده‌اید، ابتدا آن را حذف کنید:</p>
        <p><code>pip uninstall torch torchaudio</code></p>
        <p>• سپس دستور نصب جدید را اجرا کنید</p>
        <p>• پس از نصب، برنامه را دوباره اجرا کنید</p>
        
        <h3>🚨 مشکلات رایج:</h3>
        <p>• <strong>خطای "No module named 'torch'":</strong> PyTorch نصب نشده است</p>
        <p>• <strong>خطای "No module named 'torchaudio'":</strong> TorchAudio نصب نشده است</p>
        <p>• <strong>خطای CUDA:</strong> نسخه CUDA با PyTorch سازگار نیست</p>
        
        <h3>🔗 لینک‌های مفید:</h3>
        <p>• <a href="https://pytorch.org/get-started/locally/">راهنمای رسمی PyTorch</a></p>
        <p>• <a href="https://pytorch.org/get-started/previous-versions/">نسخه‌های قبلی PyTorch</a></p>
        <p>• <a href="https://github.com/snakers4/silero-models">Silero Models</a></p>
        """)
        
        layout.addWidget(text_edit)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        copy_cpu_btn = QPushButton("کپی دستور CPU")
        copy_cpu_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        copy_omegaconf_btn = QPushButton("کپی دستور omegaconf")
        copy_omegaconf_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        copy_omegaconf_btn.clicked.connect(lambda: self.copy_to_clipboard("pip install omegaconf"))
        
        copy_cpu_btn = QPushButton("کپی دستور CPU")
        copy_cpu_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        copy_cpu_btn.clicked.connect(lambda: self.copy_to_clipboard("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"))
        
        copy_gpu_btn = QPushButton("کپی دستور GPU")
        copy_gpu_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        copy_gpu_btn.clicked.connect(lambda: self.copy_to_clipboard("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"))
        
        open_pytorch_btn = QPushButton("باز کردن PyTorch")
        open_pytorch_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        open_pytorch_btn.clicked.connect(lambda: webbrowser.open("https://pytorch.org/get-started/locally/"))
        
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(copy_omegaconf_btn)
        button_layout.addWidget(copy_cpu_btn)
        button_layout.addWidget(copy_gpu_btn)
        button_layout.addWidget(open_pytorch_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def copy_to_clipboard(self, text):
        """کپی متن به کلیپ‌بورد"""
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "کپی شد", f"دستور کپی شد:\n{text}")
    
    def test_silero_stt(self):
        """تست Silero STT"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLabel
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("تست Silero STT")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # متن راهنما
        info_label = QLabel("تست وابستگی‌های Silero STT")
        info_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #1976D2; padding: 10px;")
        layout.addWidget(info_label)
        
        # نتایج تست
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        results_text.setMaximumHeight(200)
        layout.addWidget(results_text)
        
        def run_test():
            results = []
            
            # تست omegaconf
            try:
                import omegaconf
                results.append("✅ omegaconf: نصب شده")
            except ImportError:
                results.append("❌ omegaconf: نصب نشده")
            
            # تست torch
            try:
                import torch
                results.append(f"✅ torch: نصب شده (نسخه {torch.__version__})")
            except ImportError:
                results.append("❌ torch: نصب نشده")
            
            # تست torchaudio
            try:
                import torchaudio
                results.append(f"✅ torchaudio: نصب شده (نسخه {torchaudio.__version__})")
            except ImportError:
                results.append("❌ torchaudio: نصب نشده")
            
            # تست cache
            cache_path = Path.home() / '.cache' / 'torch' / 'hub'
            if cache_path.exists():
                results.append(f"✅ Cache موجود: {cache_path}")
                # لیست فایل‌های cache
                try:
                    cache_files = list(cache_path.rglob("*"))
                    results.append(f"📁 تعداد فایل‌های cache: {len(cache_files)}")
                except:
                    results.append("⚠️ خطا در خواندن cache")
            else:
                results.append(f"❌ Cache موجود نیست: {cache_path}")
            
            # تست بارگذاری مدل
            try:
                import torch
                import omegaconf
                
                results.append("\n🔄 تست بارگذاری مدل...")
                
                # تنظیمات برای حل مشکل اتصال
                import os
                os.environ['TORCH_HOME'] = str(Path.home() / '.cache' / 'torch')
                import socket
                socket.setdefaulttimeout(30)
                
                # تست مدل انگلیسی
                try:
                    model, decoder, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-models', 
                        model='silero_stt', 
                        language='en',
                        force_reload=False,
                        trust_repo=True,
                        verbose=False
                    )
                    results.append("✅ مدل انگلیسی: بارگذاری موفق")
                except Exception as e:
                    results.append(f"❌ مدل انگلیسی: خطا - {str(e)}")
                
                # تست مدل چند زبانه
                try:
                    model, decoder, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-models', 
                        model='silero_stt', 
                        language='multilingual',
                        force_reload=False,
                        trust_repo=True,
                        verbose=False
                    )
                    results.append("✅ مدل چند زبانه: بارگذاری موفق")
                except Exception as e:
                    results.append(f"❌ مدل چند زبانه: خطا - {str(e)}")
                
            except Exception as e:
                results.append(f"❌ خطا در تست مدل: {str(e)}")
            
            # نمایش نتایج
            results_text.setPlainText("\n".join(results))
        
        # دکمه تست
        test_btn = QPushButton("اجرای تست")
        test_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        test_btn.clicked.connect(run_test)
        layout.addWidget(test_btn)
        
        # دکمه بستن
        close_btn = QPushButton("بستن")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        # اجرای خودکار تست
        run_test()
        
        dialog.exec()
    
    def clear_silero_cache(self):
        """پاک کردن cache Silero STT"""
        from PySide6.QtWidgets import QMessageBox
        import shutil
        
        try:
            # مسیر cache
            cache_path = Path.home() / '.cache' / 'torch' / 'hub'
            
            if cache_path.exists():
                # تأیید از کاربر
                reply = QMessageBox.question(
                    self, "تأیید پاک کردن Cache", 
                    f"آیا می‌خواهید cache Silero STT را پاک کنید؟\n\nمسیر: {cache_path}\n\nاین کار مدل‌های دانلود شده را حذف می‌کند.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # پاک کردن cache
                    shutil.rmtree(cache_path)
                    QMessageBox.information(
                        self, "موفق", 
                        "Cache Silero STT با موفقیت پاک شد!\n\nحالا می‌توانید Silero STT را دوباره تست کنید."
                    )
                else:
                    QMessageBox.information(self, "لغو شد", "عملیات لغو شد.")
            else:
                QMessageBox.information(
                    self, "اطلاعات", 
                    "Cache Silero STT یافت نشد.\n\nمسیر: " + str(cache_path)
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, "خطا", 
                f"خطا در پاک کردن cache:\n{str(e)}\n\nلطفاً دستی پاک کنید:\n{Path.home() / '.cache' / 'torch' / 'hub'}"
            )
    
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
        if model_type in ["SpeechRecognition"]:
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
