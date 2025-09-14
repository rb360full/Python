#!/usr/bin/env python3
"""
تست تم دارک برای دیالوگ انتخاب مدل
"""

import sys
import os

# اضافه کردن مسیر فعلی به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from app import ModelSelectionDialog

def test_dark_theme():
    """تست تم دارک"""
    app = QApplication(sys.argv)
    
    # ایجاد دیالوگ انتخاب مدل
    dialog = ModelSelectionDialog()
    
    # نمایش دیالوگ
    result = dialog.exec()
    
    if result == dialog.Accepted:
        selected_model = dialog.get_selected_model()
        print(f"مدل انتخاب شده: {selected_model}")
    else:
        print("دیالوگ لغو شد")
    
    sys.exit(0)

if __name__ == "__main__":
    test_dark_theme()
