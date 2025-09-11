import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QGridLayout, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut


class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ماشین حساب")
        self.setStyleSheet("background-color: #2d2d2d; color: white; font-size: 18px;")

        # نمایشگر
        self.display = QLineEdit()
        self.display.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 22px; padding: 5px;")
        self.display.setAlignment(Qt.AlignRight)
        self.display.setReadOnly(True)

        # دکمه‌ها
        buttons = [
            ('7', 0, 0), ('8', 0, 1), ('9', 0, 2), ('/', 0, 3),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2), ('*', 1, 3),
            ('1', 2, 0), ('2', 2, 1), ('3', 2, 2), ('-', 2, 3),
            ('0', 3, 0), ('.', 3, 1), ('=', 3, 2), ('+', 3, 3),
            ('C', 4, 0, 1, 4)
        ]

        grid = QGridLayout()
        for text, row, col, rowspan, colspan in [(*b, 1, 1) if len(b) == 3 else b for b in buttons]:
            button = QPushButton(text)
            button.setStyleSheet("background-color: #444; color: white; font-size: 18px; padding: 15px;")
            button.clicked.connect(self.on_button_click)
            grid.addWidget(button, row, col, rowspan, colspan)

        layout = QVBoxLayout()
        layout.addWidget(self.display)
        layout.addLayout(grid)
        self.setLayout(layout)

        # اتصال کیبورد
        self.shortcuts()

    def shortcuts(self):
        for key in "0123456789.+-*/":
            QShortcut(QKeySequence(key), self, activated=lambda k=key: self.display.insert(k))
        QShortcut(QKeySequence("Backspace"), self, activated=lambda: self.display.backspace())
        QShortcut(QKeySequence("Enter"), self, activated=self.calculate_result)
        QShortcut(QKeySequence("Return"), self, activated=self.calculate_result)
        QShortcut(QKeySequence("Escape"), self, activated=lambda: self.display.clear())

    def on_button_click(self):
        text = self.sender().text()
        if text == "=":
            self.calculate_result()
        elif text == "C":
            self.display.clear()
        else:
            self.display.insert(text)

    def calculate_result(self):
        try:
            result = str(eval(self.display.text()))
            self.display.setText(result)
        except:
            self.display.setText("خطا")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    calc = Calculator()
    calc.resize(300, 400)
    calc.show()
    sys.exit(app.exec())
