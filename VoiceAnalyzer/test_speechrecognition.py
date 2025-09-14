#!/usr/bin/env python3
"""
تست ساده برای بررسی عملکرد SpeechRecognition
"""

import sys
import os
import tempfile
import numpy as np

# اضافه کردن مسیر فعلی به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import TranscribeThread

def test_speechrecognition():
    """تست عملکرد SpeechRecognition"""
    print("🎤 تست عملکرد SpeechRecognition...")
    
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
        
        # تست Google Speech (بدون API Key)
        print("\n🔍 تست Google Speech...")
        thread = TranscribeThread(temp_audio, "speechrecognition_google")
        result = thread.transcribe_with_speechrecognition(temp_audio)
        print(f"نتیجه Google Speech: {result}")
        
        # تست Wit.ai (بدون API Key)
        print("\n🔍 تست Wit.ai...")
        thread = TranscribeThread(temp_audio, "speechrecognition_wit")
        result = thread.transcribe_with_speechrecognition(temp_audio)
        print(f"نتیجه Wit.ai: {result}")
        
        # تست Azure Speech (بدون API Key)
        print("\n🔍 تست Azure Speech...")
        thread = TranscribeThread(temp_audio, "speechrecognition_azure")
        result = thread.transcribe_with_speechrecognition(temp_audio)
        print(f"نتیجه Azure Speech: {result}")
        
        # تست CMU Sphinx (آفلاین)
        print("\n🔍 تست CMU Sphinx...")
        thread = TranscribeThread(temp_audio, "speechrecognition_sphinx")
        result = thread.transcribe_with_speechrecognition(temp_audio)
        print(f"نتیجه CMU Sphinx: {result}")
        
        print("\n✅ تست SpeechRecognition با موفقیت انجام شد!")
        
    except Exception as e:
        print(f"❌ خطا در تست SpeechRecognition: {e}")
    finally:
        # حذف فایل موقت
        if os.path.exists(temp_audio):
            os.unlink(temp_audio)
            print(f"🗑️ فایل موقت حذف شد: {temp_audio}")

if __name__ == "__main__":
    test_speechrecognition()
