from kivy.app import App
from kivy.uix.label import Label
from kivy.core.text import LabelBase
import os

LabelBase.register(name="Vazir", fn_regular=os.path.join(os.path.dirname(__file__), "Vazir.ttf"))

class TestApp(App):
    def build(self):
        return Label(text="سلام دنیا!", font_name="Vazir", font_size=40)

TestApp().run()
