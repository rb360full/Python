import os
import json
import threading
from collections import Counter
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

from bidi.algorithm import get_display
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen

import whisper
import subprocess

# مسیر پیش‌فرض ffmpeg در سیستم
FFMPEG_PATH = r"C:\ffmpeg\bin"

# ثبت فونت فارسی Vazir
LabelBase.register(name="Vazir", fn_regular=os.path.join(os.path.dirname(__file__), "Vazir.ttf"))

KV = """
BoxLayout:
    orientation: "vertical"
    padding: 20
    spacing: 20

    MDLabel:
        id: status_label
        text: "Initializing..."
        halign: "center"
        font_name: "Vazir"
        size_hint_y: None
        height: self.texture_size[1]

    MDRaisedButton:
        text: "Choose Audio File"
        pos_hint: {"center_x":0.5}
        on_release: app.choose_audio_file()

    MDRaisedButton:
        text: "Save JSON Output"
        pos_hint: {"center_x":0.5}
        on_release: app.save_json_file()

    MDLabel:
        id: result_label
        text: ""
        halign: "center"
        font_name: "Vazir"
        size_hint_y: None
        height: self.texture_size[1]
"""

class VoiceAnalyzer(MDScreen):
    pass

class VoiceApp(MDApp):
    def build(self):
        self.title = "VoiceAnalyzer"
        self.theme_cls.theme_style = "Dark"

        self.main_screen = Builder.load_string(KV)
        self.model = None
        self.data = {}

        # اضافه کردن ffmpeg به PATH
        self.add_ffmpeg_to_path()

        # بارگذاری مدل در Thread
        Clock.schedule_once(lambda dt: threading.Thread(target=self.load_model).start(), 0)
        return self.main_screen

    def add_ffmpeg_to_path(self):
        if not self.is_ffmpeg_available():
            os.environ["PATH"] += os.pathsep + FFMPEG_PATH
            if not self.is_ffmpeg_available():
                self.update_status("ffmpeg not found! Please install or adjust FFMPEG_PATH ⚠️")

    def is_ffmpeg_available(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def load_model(self):
        self.update_status("Loading Whisper model... Please wait")
        try:
            self.model = whisper.load_model("small")
            self.update_status("Model ready ✅")
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}")

    def update_status(self, text):
        self.main_screen.ids.status_label.text = text

    def choose_audio_file(self):
        Tk().withdraw()
        filepath = askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.ogg")]
        )
        if filepath:
            self.update_status(f"Processing: {os.path.basename(filepath)}")
            threading.Thread(target=self.process_audio, args=(filepath,)).start()

    def save_json_file(self):
        if not self.data:
            self.update_status("No data to save ❌")
            return
        Tk().withdraw()
        filepath = asksaveasfilename(
            title="Save JSON Output",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=4)
                self.update_status(f"JSON saved: {filepath}")
            except Exception as e:
                self.update_status(f"Error saving JSON: {str(e)}")

    def process_audio(self, filepath):
        try:
            result = self.model.transcribe(filepath)
            text = result['text']
            words = text.split()
            word_count = len(words)
            word_freq = dict(Counter(words).most_common(10))
            self.data = {
                "file": filepath,
                "text": text,
                "word_count": word_count,
                "top_words": word_freq
            }
            display_text = get_display(f"Word count: {word_count}\nTop words: {word_freq}\n\n{text}")
            Clock.schedule_once(lambda dt: self.update_result(display_text), 0)
            Clock.schedule_once(lambda dt: self.update_status("Processing done ✅"), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self.update_result(f"Error: {str(e)}"), 0)
            Clock.schedule_once(lambda dt: self.update_status("Error occurred ❌"), 0)

    def update_result(self, text):
        self.main_screen.ids.result_label.text = text

if __name__ == "__main__":
    VoiceApp().run()
