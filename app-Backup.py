import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QTextEdit, QScrollArea, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
import subprocess
import json
import collections
import webbrowser

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

# مسیر تکیه کلام‌ها
RELYING_DICT_FILE = Path("relying_dict.json")
if not RELYING_DICT_FILE.exists():
    with open(RELYING_DICT_FILE, "w", encoding="utf-8") as f:
        json.dump({"تکیه_کلام": ["خب", "آخه", "یعنی", "حالا"]}, f, ensure_ascii=False, indent=2)

# Import پنجره مدیریت دیکشنری
from custom_dict_manager import DictManager

class TranscribeThread(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path
        self.model = whisper.load_model("medium")
        self._pause = False
        self._stop = False

    def run(self):
        self.progress.emit(10)
        try:
            result = self.model.transcribe(str(self.audio_path))
            text = result["text"]

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
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class VoiceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Analyzer")
        self.resize(700, 500)

        self.layout = QVBoxLayout(self)

        self.label = QLabel("Select an audio file:")
        self.layout.addWidget(self.label)

        # دکمه ها
        self.btn_select = QPushButton("Choose File")
        self.btn_select.clicked.connect(self.select_file)
        self.layout.addWidget(self.btn_select)

        self.btn_start_pause = QPushButton("Start/Pause")
        self.btn_start_pause.clicked.connect(self.toggle_start_pause)
        self.layout.addWidget(self.btn_start_pause)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.layout.addWidget(self.btn_stop)

        self.btn_open_text = QPushButton("Open Text")
        self.btn_open_text.clicked.connect(self.open_text_file)
        self.layout.addWidget(self.btn_open_text)

        self.btn_manage_dict = QPushButton("Manage Dictionary")
        self.btn_manage_dict.clicked.connect(self.open_dict_manager)
        self.layout.addWidget(self.btn_manage_dict)

        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)

        self.scroll = QScrollArea()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.scroll.setWidget(self.text_edit)
        self.scroll.setWidgetResizable(True)
        self.layout.addWidget(self.scroll)

        self.audio_path = None
        self.output_file = None
        self.thread = None
        self.dict_manager_window = None

    # فقط این قسمت اصلاح شد، حذف start_transcription خودکار
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.m4a)"
        )
        if file_path:
            self.audio_path = file_path
            self.label.setText(f"Selected: {file_path}")
            self.text_edit.clear()
            self.output_file = None

    def start_transcription(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Warning", "No audio file selected!")
            return

        self.text_edit.clear()
        self.output_file = None
        self.thread = TranscribeThread(self.audio_path)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.display_result)
        self.thread.start()

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
            # ایجاد و شروع thread
            self.thread = TranscribeThread(self.audio_path)
            self.thread.progress.connect(self.progress.setValue)
            self.thread.finished.connect(self.display_result)
            self.thread.start()
        else:
            # برای آینده: Pause/Resume
            QMessageBox.information(self, "Info", "Pause/Resume toggle clicked (future implementation).")

    def stop_processing(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "Info", "Stop clicked (future implementation).")

    def open_dict_manager(self):
        if not self.dict_manager_window:
            self.dict_manager_window = DictManager()
        self.dict_manager_window.show()
        self.dict_manager_window.raise_()
        self.dict_manager_window.activateWindow()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec())
