#!/usr/bin/env python3
"""
تست ساده برای بررسی عملکرد Hugging Face
"""

import sys
import os
import tempfile
import numpy as np

# اضافه کردن مسیر فعلی به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import TranscribeThread

def test_huggingface():
    """تست عملکرد Hugging Face"""
    print("🤗 تست عملکرد Hugging Face...")
    
    # ایجاد یک فایل صوتی نمونه (1 ثانیه سکوت)
    sample_rate = 16000
    duration = 1.0  # 1 ثانیه
    samples = int(sample_rate * duration)
    
    # ایجاد فایل صوتی موقت
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_audio = tmp.name
    
    try:
        # ایجاد فایل صوتی نمونه
        import soundfile as sf
        audio_data = np.zeros(samples, dtype=np.float32)
        sf.write(temp_audio, audio_data, sample_rate)
        
        print(f"✅ فایل صوتی نمونه ایجاد شد: {temp_audio}")
        
        # تست Hugging Face Persian
        print("\n🔍 تست Hugging Face Persian...")
        thread = TranscribeThread(temp_audio, "hf_wav2vec2_persian")
        result = thread.transcribe_with_huggingface(temp_audio)
        print(f"نتیجه Hugging Face Persian: {result}")
        
        # تست Hugging Face Whisper Persian
        print("\n🔍 تست Hugging Face Whisper Persian...")
        thread = TranscribeThread(temp_audio, "hf_whisper_persian")
        result = thread.transcribe_with_huggingface(temp_audio)
        print(f"نتیجه Hugging Face Whisper Persian: {result}")
        
        # تست Hugging Face Multilingual
        print("\n🔍 تست Hugging Face Multilingual...")
        thread = TranscribeThread(temp_audio, "hf_wav2vec2_persian_alt")
        result = thread.transcribe_with_huggingface(temp_audio)
        print(f"نتیجه Hugging Face Multilingual: {result}")
        
        print("\n✅ تست Hugging Face با موفقیت انجام شد!")
        
    except Exception as e:
        print(f"❌ خطا در تست Hugging Face: {e}")
    finally:
        # حذف فایل موقت
        if os.path.exists(temp_audio):
            os.unlink(temp_audio)
            print(f"🗑️ فایل موقت حذف شد: {temp_audio}")

if __name__ == "__main__":
    test_huggingface()
