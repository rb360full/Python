import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QTextEdit, QScrollArea, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
import json
import whisper

# ست کردن مسیر ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\ffmpeg\bin\ffmpeg.exe"

# مسیر فونت Vazir
FONT_PATH = str(Path(__file__).parent / "Vazir.ttf")

class TranscribeThread(QThread):
    progress = Signal(str)
    finished = Signal(str)

    def __init__(self, file_path, model_name="medium"):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name

    def run(self):
        try:
            self.progress.emit("Loading Whisper model...")
            model = whisper.load_model(self.model_name)

            self.progress.emit("Transcribing audio...")
            result = model.transcribe(self.file_path)
            text = result["text"]

            self.finished.emit(text)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")


class VoiceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Analyzer")
        self.setGeometry(100, 100, 700, 500)

        layout = QVBoxLayout()

        # دکمه انتخاب فایل
        self.btn_select = QPushButton("Choose Audio File")
        self.btn_select.clicked.connect(self.select_file)
        layout.addWidget(self.btn_select)

        # برچسب وضعیت
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        # پروگرس بار
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # TextEdit با اسکرول
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFontPointSize(12)
        layout.addWidget(self.text_edit)

        # دکمه ذخیره JSON
        self.btn_save = QPushButton("Save as JSON")
        self.btn_save.clicked.connect(self.save_json)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        self.setLayout(layout)
        self.audio_path = None
        self.transcribed_text = ""

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.m4a)"
        )
        if file_path:
            self.audio_path = file_path
            self.start_transcription()

    def start_transcription(self):
        self.status_label.setText("Status: Processing...")
        self.progress_bar.setVisible(True)
        self.text_edit.clear()
        self.btn_save.setEnabled(False)

        self.thread = TranscribeThread(self.audio_path)
        self.thread.progress.connect(self.update_status)
        self.thread.finished.connect(self.transcription_finished)
        self.thread.start()

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def transcription_finished(self, text):
        self.progress_bar.setVisible(False)
        self.transcribed_text = text
        self.text_edit.setPlainText(text)
        self.status_label.setText("Status: Finished")
        self.btn_save.setEnabled(True)

    def save_json(self):
        if not self.transcribed_text:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"text": self.transcribed_text}, f, ensure_ascii=False)
            self.status_label.setText(f"Status: Saved to {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec())
