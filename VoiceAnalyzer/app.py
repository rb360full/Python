import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QTextEdit, QScrollArea, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
import subprocess
import json
import re
from collections import Counter

# Whisper و تبدیل صوت به متن
import whisper

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

# Worker thread برای پردازش
class TranscribeThread(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path
        self.model = whisper.load_model("medium")

    def run(self):
        self.progress.emit(10)
        try:
            result = self.model.transcribe(str(self.audio_path))
            text = result["text"]

            # اعمال لغات کاستوم
            with open(CUSTOM_DICT_FILE, encoding="utf-8") as f:
                custom_dict = json.load(f)
            for wrong, correct in custom_dict.items():
                text = text.replace(wrong, correct)

            self.progress.emit(100)
            self.finished.emit(text)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

# تولید گزارش آماری
def generate_report(text):
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)

    # مثال تکیه کلام ها
    filler_words = ["مثلا", "اینکه", "خب", "یعنی", "حالا"]
    found_fillers = [w for w in words if w in filler_words]
    filler_counts = Counter(found_fillers)

    word_counts = Counter(words)
    most_common = word_counts.most_common(10)

    report = "\n--- گزارش آماری ---\n"
    report += f"تعداد کل کلمات: {total_words}\n"
    report += "تکیه کلام‌ها:\n"
    for word, count in filler_counts.items():
        report += f"{word}: {count}\n"
    report += "۱۰ کلمه پرتکرار:\n"
    for word, count in most_common:
        report += f"{word}: {count}\n"
    report += "\n"
    return report

class VoiceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Analyzer")
        self.resize(700, 500)

        self.layout = QVBoxLayout(self)

        self.label = QLabel("Select an audio file:")
        self.layout.addWidget(self.label)

        self.btn_select = QPushButton("Choose File")
        self.btn_select.setMinimumHeight(40)
        self.btn_select.clicked.connect(self.select_file)
        self.layout.addWidget(self.btn_select)

        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)

        self.scroll = QScrollArea()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.scroll.setWidget(self.text_edit)
        self.scroll.setWidgetResizable(True)
        self.layout.addWidget(self.scroll)

        self.btn_save = QPushButton("Save Text")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_text)
        self.layout.addWidget(self.btn_save)

        self.audio_path = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.m4a)"
        )
        if file_path:
            self.audio_path = file_path
            self.label.setText(f"Selected: {file_path}")
            self.start_transcription()

    def start_transcription(self):
        self.text_edit.clear()
        self.thread = TranscribeThread(self.audio_path)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.display_result)
        self.thread.start()

    def display_result(self, text):
        self.text_edit.setPlainText(text)

    def save_text(self):
        if self.text_edit.toPlainText():
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Text", "", "Text Files (*.txt)"
            )
            if save_path:
                text = self.text_edit.toPlainText()
                report = generate_report(text)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(report + text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec())
