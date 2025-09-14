#!/usr/bin/env python3
"""
اسکریپت نصب وابستگی‌های VoiceAnalyzer
این اسکریپت تمام وابستگی‌های مورد نیاز را نصب می‌کند
"""

import subprocess
import sys
import os

def install_package(package):
    """نصب یک پکیج"""
    try:
        print(f"در حال نصب {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} با موفقیت نصب شد")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ خطا در نصب {package}: {e}")
        return False

def main():
    """تابع اصلی"""
    print("🎤 VoiceAnalyzer - نصب وابستگی‌ها")
    print("=" * 50)
    
    # لیست وابستگی‌های اصلی
    core_packages = [
        "openai-whisper",
        "PySide6", 
        "arabic-reshaper",
        "python-bidi",
        "numpy",
        "regex",
        "tqdm",
        "persian-tools",
        "soundfile",
        "requests"
    ]
    
    # لیست وابستگی‌های مدل‌های Speech-to-Text
    speech_packages = [
        "google-cloud-speech",
        "vosk",
        "azure-cognitiveservices-speech",
        "assemblyai"
    ]
    
    # لیست وابستگی‌های مدل‌های جدید
    new_model_packages = [
        "transformers",
        "torch",
        "SpeechRecognition",
        "torchaudio",
        "kaldi-io",
        "librosa",
        "scipy"
    ]
    
    print("📦 نصب وابستگی‌های اصلی...")
    core_success = 0
    for package in core_packages:
        if install_package(package):
            core_success += 1
    
    print(f"\n📦 نصب وابستگی‌های Speech-to-Text...")
    speech_success = 0
    for package in speech_packages:
        if install_package(package):
            speech_success += 1
    
    print(f"\n📦 نصب وابستگی‌های مدل‌های جدید...")
    new_model_success = 0
    for package in new_model_packages:
        if install_package(package):
            new_model_success += 1
    
    # خلاصه نتایج
    print("\n" + "=" * 50)
    print("📊 خلاصه نصب:")
    print(f"✅ وابستگی‌های اصلی: {core_success}/{len(core_packages)}")
    print(f"✅ وابستگی‌های Speech-to-Text: {speech_success}/{len(speech_packages)}")
    print(f"✅ وابستگی‌های مدل‌های جدید: {new_model_success}/{len(new_model_packages)}")
    
    total_success = core_success + speech_success + new_model_success
    total_packages = len(core_packages) + len(speech_packages) + len(new_model_packages)
    
    print(f"\n🎯 کل: {total_success}/{total_packages} پکیج با موفقیت نصب شد")
    
    if total_success == total_packages:
        print("🎉 تمام وابستگی‌ها با موفقیت نصب شدند!")
        print("حالا می‌توانید VoiceAnalyzer را اجرا کنید.")
    else:
        print("⚠️ برخی وابستگی‌ها نصب نشدند. لطفاً دستی نصب کنید.")
        print("برای نصب دستی از فایل requirements.txt استفاده کنید:")
        print("pip install -r requirements.txt")
    
    print("\n💡 نکات مهم:")
    print("- برای استفاده از Google Speech، API key تنظیم کنید")
    print("- برای استفاده از Azure Speech، API key تنظیم کنید")
    print("- برای استفاده از AssemblyAI، API key تنظیم کنید")
    print("- مدل‌های Vosk و Whisper کاملاً آفلاین هستند")

if __name__ == "__main__":
    main()
