import whisper
import re
import json
from collections import Counter
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView


class VoiceAnalyzerApp(App):
    def build(self):
        self.title = "VoiceAnalyzer"  # اسم برنامه
        self.model = whisper.load_model("small")  # مدل Whisper
        self.analysis_result = {}  # ذخیره نتایج برای JSON

        main_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)

        # برچسب راهنما
        self.label = Label(text="🎤 یک فایل صوتی انتخاب کنید", font_size=20, color=(1, 1, 1, 1))
        main_layout.add_widget(self.label)

        # انتخاب فایل صوتی
        self.filechooser = FileChooserIconView(filters=["*.mp3", "*.wav", "*.m4a"])
        main_layout.add_widget(self.filechooser)

        # دکمه‌ها
        button_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)

        self.analyze_button = Button(text="تحلیل وویس", background_color=(0.2, 0.2, 0.2, 1))
        self.analyze_button.bind(on_press=self.analyze_voice)
        button_layout.add_widget(self.analyze_button)

        self.save_button = Button(text="ذخیره JSON", background_color=(0.1, 0.5, 0.1, 1))
        self.save_button.bind(on_press=self.ask_save_path)
        button_layout.add_widget(self.save_button)

        main_layout.add_widget(button_layout)

        # نمایش نتایج با ScrollView
        self.result_label = Label(text="", font_size=16, color=(1, 1, 1, 1), halign="left", valign="top")
        self.result_label.bind(size=self.result_label.setter('text_size'))

        scroll = ScrollView(size_hint=(1, 0.5))
        scroll.add_widget(self.result_label)
        main_layout.add_widget(scroll)

        # بک‌گراند دارک
        main_layout.canvas.before.clear()
        with main_layout.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.1, 0.1, 0.1, 1)
            self.rect = Rectangle(size=main_layout.size, pos=main_layout.pos)
            main_layout.bind(size=self._update_rect, pos=self._update_rect)

        return main_layout

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def analyze_voice(self, instance):
        if not self.filechooser.selection:
            self.result_label.text = "⚠️ لطفاً یک فایل انتخاب کنید"
            return

        file_path = self.filechooser.selection[0]

        # استخراج متن
        result = self.model.transcribe(file_path, fp16=False)
        text = result["text"].strip()

        # جدا کردن کلمات
        words = re.findall(r'\w+', text.lower(), re.UNICODE)
        total_words = len(words)
        word_freq = Counter(words)
        top_words = word_freq.most_common(5)

        # ذخیره نتایج برای JSON
        self.analysis_result = {
            "text": text,
            "total_words": total_words,
            "top_words": top_words
        }

        # نمایش در UI
        output = f"📝 متن:\n{text}\n\n"
        output += f"🔢 تعداد کل کلمات: {total_words}\n\n"
        output += "🔥 کلمات پرتکرار:\n"
        for word, count in top_words:
            output += f"{word}: {count}\n"

        self.result_label.text = output

    def ask_save_path(self, instance):
        if not self.analysis_result:
            self.result_label.text = "⚠️ هنوز تحلیلی انجام نشده!"
            return

        # پاپ‌آپ برای انتخاب نام فایل
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        text_input = TextInput(text="analysis_result.json", multiline=False)

        btn_layout = BoxLayout(size_hint=(1, 0.3), spacing=10)
        save_btn = Button(text="ذخیره", background_color=(0.1, 0.6, 0.1, 1))
        cancel_btn = Button(text="لغو", background_color=(0.6, 0.1, 0.1, 1))

        btn_layout.add_widget(save_btn)
        btn_layout.add_widget(cancel_btn)

        content.add_widget(Label(text="📁 نام فایل خروجی را وارد کنید:", color=(1, 1, 1, 1)))
        content.add_widget(text_input)
        content.add_widget(btn_layout)

        popup = Popup(title="ذخیره JSON", content=content, size_hint=(0.8, 0.4))

        save_btn.bind(on_press=lambda x: self.save_json(text_input.text, popup))
        cancel_btn.bind(on_press=popup.dismiss)

        popup.open()

    def save_json(self, filename, popup):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.analysis_result, f, ensure_ascii=False, indent=4)

            self.result_label.text += f"\n\n✅ خروجی در {filename} ذخیره شد."
            popup.dismiss()
        except Exception as e:
            self.result_label.text = f"❌ خطا در ذخیره فایل: {str(e)}"


if __name__ == "__main__":
    VoiceAnalyzerApp().run()
