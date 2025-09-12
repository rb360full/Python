import sys
import json
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QListWidget, QMessageBox
)

CUSTOM_DICT_FILE = "custom_dict.json"

class DictManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Dictionary Manager")
        self.resize(400, 500)

        self.layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        self.input_wrong = QLineEdit()
        self.input_wrong.setPlaceholderText("Wrong word")
        self.layout.addWidget(self.input_wrong)

        self.input_correct = QLineEdit()
        self.input_correct.setPlaceholderText("Correct replacement")
        self.layout.addWidget(self.input_correct)

        self.btn_add = QPushButton("Add / Update")
        self.btn_add.clicked.connect(self.add_word)
        self.layout.addWidget(self.btn_add)

        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_word)
        self.layout.addWidget(self.btn_delete)

        self.setLayout(self.layout)

        self.load_dict()

    def load_dict(self):
        try:
            with open(CUSTOM_DICT_FILE, "r", encoding="utf-8") as f:
                self.custom_dict = json.load(f)
        except:
            self.custom_dict = {}
        self.refresh_list()

    def refresh_list(self):
        self.list_widget.clear()
        for k, v in self.custom_dict.items():
            self.list_widget.addItem(f"{k} -> {v}")

    def add_word(self):
        wrong = self.input_wrong.text().strip()
        correct = self.input_correct.text().strip()
        if wrong and correct:
            self.custom_dict[wrong] = correct
            with open(CUSTOM_DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.custom_dict, f, ensure_ascii=False, indent=2)
            self.refresh_list()
            self.input_wrong.clear()
            self.input_correct.clear()
        else:
            QMessageBox.warning(self, "Input error", "Both fields are required")

    def delete_word(self):
        selected = self.list_widget.currentRow()
        if selected >= 0:
            key = list(self.custom_dict.keys())[selected]
            self.custom_dict.pop(key)
            with open(CUSTOM_DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.custom_dict, f, ensure_ascii=False, indent=2)
            self.refresh_list()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DictManager()
    window.show()
    sys.exit(app.exec())
