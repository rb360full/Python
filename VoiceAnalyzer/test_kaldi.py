#!/usr/bin/env python3
"""
تست ساده برای بررسی عملکرد Kaldi
"""

import sys
import os
import tempfile
import numpy as np

# اضافه کردن مسیر فعلی به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import TranscribeThread

def test_kaldi():
    """تست عملکرد Kaldi"""
    print("🧪 تست عملکرد Kaldi...")
    
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
        
        # تست Kaldi Persian
        print("\n🔍 تست Kaldi Persian...")
        thread = TranscribeThread(temp_audio, "kaldi_persian")
        result = thread.transcribe_with_kaldi(temp_audio)
        print(f"نتیجه Kaldi Persian: {result}")
        
        # تست Kaldi English
        print("\n🔍 تست Kaldi English...")
        thread = TranscribeThread(temp_audio, "kaldi_english")
        result = thread.transcribe_with_kaldi(temp_audio)
        print(f"نتیجه Kaldi English: {result}")
        
        print("\n✅ تست Kaldi با موفقیت انجام شد!")
        
    except Exception as e:
        print(f"❌ خطا در تست Kaldi: {e}")
    finally:
        # حذف فایل موقت
        if os.path.exists(temp_audio):
            os.unlink(temp_audio)
            print(f"🗑️ فایل موقت حذف شد: {temp_audio}")

if __name__ == "__main__":
    test_kaldi()
