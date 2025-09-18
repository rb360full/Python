import os
import tempfile
import torch
import librosa
import soundfile as sf
from transformers import AutoModelForCTC, AutoProcessor
from scipy import signal
import numpy as np
from pathlib import Path


class VoiceProcessor:
    """کلاس پردازش صوت برای تبدیل ویس به متن با مدل wav2vec2 Persian v3"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
        self._model_loaded = False
    
    def load_model(self):
        """بارگذاری مدل wav2vec2 Persian v3"""
        if self._model_loaded:
            return True
            
        try:
            print("🔄 در حال بارگذاری مدل wav2vec2 Persian v3...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCTC.from_pretrained(self.model_name)
            self._model_loaded = True
            print("✅ مدل wav2vec2 Persian v3 بارگذاری شد")
            return True
        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل: {e}")
            return False
    
    def transcribe_audio(self, audio_file_path):
        """تبدیل فایل صوتی به متن - پشتیبانی از فرمت‌های مختلف"""
        if not self._model_loaded:
            if not self.load_model():
                return "خطا: مدل بارگذاری نشد"
        
        try:
            # تشخیص فرمت فایل
            file_extension = Path(audio_file_path).suffix.lower()
            print(f"🔍 پردازش فایل صوتی: {file_extension}")
            
            # بارگذاری فایل صوتی با فرمت‌های مختلف
            audio, sr = self._load_audio_file(audio_file_path, file_extension)
            
            if audio is None:
                return f"خطا: فرمت {file_extension} پشتیبانی نمی‌شود"
            
            # پردازش صوت
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            
            # تشخیص گفتار
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # تبدیل به متن
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(predicted_ids)[0]
            
            # پاکسازی متن
            text = self._clean_persian_text(text)
            
            return text if text.strip() else "متن تشخیص داده نشد"
            
        except Exception as e:
            return f"خطا در پردازش صوت: {str(e)}"
    
    def _load_audio_file(self, file_path, extension):
        """بارگذاری فایل صوتی با پشتیبانی از فرمت‌های مختلف"""
        try:
            # استفاده از librosa برای فرمت‌های مختلف
            if extension in ['.ogg', '.mp3', '.wav', '.m4a', '.flac', '.aac']:
                audio, sr = librosa.load(file_path, sr=16000)
                return audio, sr
            else:
                # تلاش با soundfile برای فرمت‌های دیگر
                try:
                    audio, sr = sf.read(file_path)
                    if sr != 16000:
                        # تبدیل sample rate به 16000
                        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
                        sr = 16000
                    return audio, sr
                except Exception:
                    return None, None
        except Exception as e:
            print(f"خطا در بارگذاری فایل {extension}: {e}")
            return None, None
    
    def _clean_persian_text(self, text):
        """پاکسازی و بهبود متن فارسی"""
        if not text:
            return ""
        
        # حذف فاصله‌های اضافی
        text = " ".join(text.split())
        
        # تصحیح کاراکترهای فارسی
        text = text.replace("ي", "ی")
        text = text.replace("ك", "ک")
        text = text.replace("ة", "ه")
        
        # حذف کاراکترهای غیرضروری
        import re
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]', '', text)
        
        return text.strip()


# نمونه سراسری برای استفاده در بات
voice_processor = VoiceProcessor()
