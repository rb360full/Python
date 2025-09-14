@echo off
echo 🎤 VoiceAnalyzer - نصب وابستگی‌ها
echo ====================================
echo.

echo در حال نصب وابستگی‌های اصلی...
pip install openai-whisper PySide6 arabic-reshaper python-bidi
pip install numpy regex tqdm persian-tools soundfile requests

echo.
echo در حال نصب وابستگی‌های Speech-to-Text...
pip install google-cloud-speech vosk azure-cognitiveservices-speech assemblyai

echo.
echo در حال نصب وابستگی‌های مدل‌های جدید...
pip install transformers torch SpeechRecognition torchaudio kaldi-io librosa scipy

echo.
echo ✅ نصب کامل شد!
echo حالا می‌توانید VoiceAnalyzer را اجرا کنید:
echo python app.py
echo.
pause
