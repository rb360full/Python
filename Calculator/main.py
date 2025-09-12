# calculator.py
import tkinter as tk
from tkinter import ttk
import ast
import operator

# --- یک ارزیاب ساده و امن برای عبارات ریاضی (بدون eval ناامن) ---
# از AST استفاده می‌کنیم و فقط گره‌های ریاضی مجاز را پردازش می‌کنیم.
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    # می‌توان در صورت نیاز عملگرهای بیشتری اضافه کرد
}

def safe_eval(expr: str):
    """
    محاسبه‌ی امن یک عبارت ریاضی شامل اعداد، + - * / % ** و پرانتز.
    در صورت وجود هر گره غیرمجاز یا خطا، استثنا پرت می‌شود.
    """
    # به جای نماد × و ÷ کاراکترهای پایتونی معادل را می‌گذاریم
    expr = expr.replace('×', '*').replace('÷', '/').replace('^', '**')
    node = ast.parse(expr, mode='eval')

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"عملگر مجاز نیست: {op_type}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"عملگر یک‌طرفه مجاز نیست: {op_type}")
        if isinstance(node, ast.Num):  # Python <3.8
            return node.n
        if hasattr(ast, "Constant") and isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("مقدار غیرعددی مجاز نیست")
        if isinstance(node, ast.Call):
            raise ValueError("توابع مجاز نیستند")
        if isinstance(node, ast.Name):
            raise ValueError("نام‌ها مجاز نیستند")
        raise ValueError(f"گره ناشناخته: {type(node)}")

    return _eval(node)

# --- رابط کاربری Tkinter ---
class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ماشین‌حساب ساده")
        self.resizable(False, False)
        self.configure(padx=10, pady=10)

        # نمایشگر (Entry)
        self.entry_var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self.entry_var, font=("Helvetica", 18), justify="right", width=18)
        entry.grid(row=0, column=0, columnspan=4, sticky="we", pady=(0,8))
        entry.focus_set()

        # دکمه‌ها
        buttons = [
            ('C', 1, 0, self.clear_all),
            ('⌫', 1, 1, self.backspace),
            ('%', 1, 2, lambda: self.insert_char('%')),
            ('÷', 1, 3, lambda: self.insert_char('÷')),

            ('7', 2, 0, lambda: self.insert_char('7')),
            ('8', 2, 1, lambda: self.insert_char('8')),
            ('9', 2, 2, lambda: self.insert_char('9')),
            ('×', 2, 3, lambda: self.insert_char('×')),

            ('4', 3, 0, lambda: self.insert_char('4')),
            ('5', 3, 1, lambda: self.insert_char('5')),
            ('6', 3, 2, lambda: self.insert_char('6')),
            ('−', 3, 3, lambda: self.insert_char('-')),

            ('1', 4, 0, lambda: self.insert_char('1')),
            ('2', 4, 1, lambda: self.insert_char('2')),
            ('3', 4, 2, lambda: self.insert_char('3')),
            ('+', 4, 3, lambda: self.insert_char('+')),

            ('±', 5, 0, self.negate),
            ('0', 5, 1, lambda: self.insert_char('0')),
            ('.', 5, 2, lambda: self.insert_char('.')),
            ('=', 5, 3, self.calculate),
        ]

        for (text, r, c, cmd) in buttons:
            b = ttk.Button(self, text=text, command=cmd)
            b.grid(row=r, column=c, sticky="nsew", padx=4, pady=4, ipadx=6, ipady=6)

        # ریز تنظیمات شبکه برای اندازه یکنواخت دکمه‌ها
        for i in range(6):
            self.grid_rowconfigure(i, weight=1)
        for j in range(4):
            self.grid_columnconfigure(j, weight=1)

        # کیبورد بایندینگ‌ها
        self.bind_all('<Return>', lambda e: self.calculate())
        self.bind_all('<BackSpace>', lambda e: self.backspace())
        self.bind_all('<Escape>', lambda e: self.clear_all())
        for key in '0123456789.+-*/()%':
            self.bind_all(key, lambda e, k=key: self.insert_char(k))
        # allow ^ for power
        self.bind_all('^', lambda e: self.insert_char('^'))

    def insert_char(self, ch: str):
        cur = self.entry_var.get()
        # جلوگیری از چند نقطه متوالی در یک عدد ساده ساده (به صورت خیلی ساده)
        if ch == '.':
            # اگر آخرین توکن عددی شامل '.' هست، از وارد شدن دوم جلوگیری کن
            # این روش ساده است و برای اکثر استفاده‌ها کفایت می‌کند.
            import re
            parts = re.split(r'[\+\-\*/%\^\(\)]', cur)
            if parts and '.' in parts[-1]:
                return
        self.entry_var.set(cur + ch)

    def clear_all(self):
        self.entry_var.set('')

    def backspace(self):
        cur = self.entry_var.get()
        if cur:
            self.entry_var.set(cur[:-1])

    def negate(self):
        cur = self.entry_var.get()
        if not cur:
            return
        # سعی می‌کنیم آخرین عدد را منفی/مثبت کنیم
        # پیدا کردن محل آخرین عملگر
        import re
        tokens = list(re.finditer(r'[\+\-\*/%\^\(\)]', cur))
        if tokens:
            last = tokens[-1]
            prefix = cur[:last.end()]
            last_num = cur[last.end():]
            if last_num.startswith('-'):
                last_num = last_num[1:]
            else:
                last_num = '-' + last_num
            self.entry_var.set(prefix + last_num)
        else:
            # تمام عبارت یک عدد است
            if cur.startswith('-'):
                self.entry_var.set(cur[1:])
            else:
                self.entry_var.set('-' + cur)

    def calculate(self):
        expr = self.entry_var.get().strip()
        if not expr:
            return
        try:
            result = safe_eval(expr)
            # نمایش نتیجه با حذف .0 اضافی برای اعداد صحیح
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            self.entry_var.set(str(result))
        except Exception as e:
            # پیام خطا را کوتاه نشان می‌دهیم
            self.entry_var.set("خطا")
            # و بعد از نیم ثانیه مقدار را پاک می‌کنیم
            self.after(700, lambda: self.entry_var.set(''))


if __name__ == "__main__":
    app = Calculator()
    app.mainloop()
