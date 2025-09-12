# custom_dict_manager.py
import sys
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QListWidget, QLineEdit, QHBoxLayout, QMessageBox
)

CUSTOM_DICT_FILE = Path("custom_dict.json")
RELYING_DICT_FILE = Path("relying_dict.json")

class DictManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dictionary Manager")
        self.resize(500, 400)
        self.layout = QVBoxLayout(self)

        # Custom dict
        self.layout.addWidget(QLabel("Custom corrections:"))
        self.list_custom = QListWidget()
        self.layout.addWidget(self.list_custom)

        self.input_custom_key = QLineEdit()
        self.input_custom_key.setPlaceholderText("Wrong word")
        self.input_custom_value = QLineEdit()
        self.input_custom_value.setPlaceholderText("Correct word")
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.input_custom_key)
        h_layout1.addWidget(self.input_custom_value)
        self.layout.addLayout(h_layout1)

        btn_add_custom = QPushButton("Add/Update")
        btn_add_custom.clicked.connect(self.add_custom)
        self.layout.addWidget(btn_add_custom)

        btn_delete_custom = QPushButton("Delete Selected")
        btn_delete_custom.clicked.connect(self.delete_custom)
        self.layout.addWidget(btn_delete_custom)

        # Relying dict (pause words)
        self.layout.addWidget(QLabel("Relying words:"))
        self.list_rely = QListWidget()
        self.layout.addWidget(self.list_rely)

        self.input_rely = QLineEdit()
        self.input_rely.setPlaceholderText("Relying word")
        self.layout.addWidget(self.input_rely)

        btn_add_rely = QPushButton("Add/Update")
        btn_add_rely.clicked.connect(self.add_rely)
        self.layout.addWidget(btn_add_rely)

        btn_delete_rely = QPushButton("Delete Selected")
        btn_delete_rely.clicked.connect(self.delete_rely)
        self.layout.addWidget(btn_delete_rely)

        self.load_dicts()

    def load_dicts(self):
        self.custom_dict = {}
        self.relying_dict = {}

        if CUSTOM_DICT_FILE.exists():
            with open(CUSTOM_DICT_FILE, encoding="utf-8") as f:
                self.custom_dict = json.load(f)
        if RELYING_DICT_FILE.exists():
            with open(RELYING_DICT_FILE, encoding="utf-8") as f:
                self.relying_dict = json.load(f)

        self.refresh_lists()

    def refresh_lists(self):
        self.list_custom.clear()
        for k, v in self.custom_dict.items():
            self.list_custom.addItem(f"{k} -> {v}")

        self.list_rely.clear()
        for k in self.relying_dict.keys():
            self.list_rely.addItem(k)

    def add_custom(self):
        key = self.input_custom_key.text().strip()
        value = self.input_custom_value.text().strip()
        if key and value:
            self.custom_dict[key] = value
            with open(CUSTOM_DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.custom_dict, f, ensure_ascii=False, indent=2)
            self.refresh_lists()
            self.input_custom_key.clear()
            self.input_custom_value.clear()

    def delete_custom(self):
        selected = self.list_custom.selectedItems()
        if selected:
            for item in selected:
                key = item.text().split(" -> ")[0]
                if key in self.custom_dict:
                    del self.custom_dict[key]
            with open(CUSTOM_DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.custom_dict, f, ensure_ascii=False, indent=2)
            self.refresh_lists()

    def add_rely(self):
        key = self.input_rely.text().strip()
        if key:
            self.relying_dict[key] = True
            with open(RELYING_DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.relying_dict, f, ensure_ascii=False, indent=2)
            self.refresh_lists()
            self.input_rely.clear()

    def delete_rely(self):
        selected = self.list_rely.selectedItems()
        if selected:
            for item in selected:
                key = item.text()
                if key in self.relying_dict:
                    del self.relying_dict[key]
            with open(RELYING_DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.relying_dict, f, ensure_ascii=False, indent=2)
            self.refresh_lists()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DictManager()
    window.show()
    sys.exit(app.exec())
