import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy import integrate
from scipy.special import j0
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from datetime import datetime

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MicrostripPatchAntennaCalculator:
    def __init__(self):
        self.c = 3e8  # скорость света в м/с

    def calculate_antenna_parameters(self, fr, epsilon_r, h):

        """Расчет параметров микрополосковой патч-антенны по точным формулам"""

        # 1. Расчет ширины патча (W)
        W = self.calculate_patch_width(fr, epsilon_r)

        # 2. Расчет эффективной диэлектрической проницаемости
        epsilon_ref = self.calculate_effective_permittivity(epsilon_r, h, W)

        # 3. Расчет длины патча (L)
        L = self.calculate_patch_length(fr, epsilon_ref, h, W)

        # 4. Расчет проводимостей и входного сопротивления
        G1, G12 = self.calculate_conductances(W, L, fr)
        R_in_edge = self.calculate_input_impedance_edge(G1, G12)

        # 5. Расчет точки питания для 50 Ом
        y0 = self.calculate_feed_position(R_in_edge, L)

        # 6. Расчет направленности
        D = self.calculate_directivity(W, L, fr, epsilon_ref)

        results = {
            'resonant_frequency': fr,
            'substrate_permittivity': epsilon_r,
            'substrate_height': h,
            'patch_width': W,
            'patch_length': L,
            'effective_permittivity': epsilon_ref,
            'single_slot_conductance': G1,
            'mutual_conductance': G12,
            'input_impedance_edge': R_in_edge,
            'feed_position_50ohm': y0,
            'directivity': D,
        }

        return results

    def calculate_patch_width(self, fr, epsilon_r):
        """Ширина патча"""

        W = (self.c / (2 * fr)) * np.sqrt(2 / (epsilon_r + 1))
        return W

    def calculate_effective_permittivity(self, epsilon_r, h, W):
        """Эффективная диэлектрическая проницаемость"""

        term = 1 / np.sqrt(1 + 12 * (h / W))
        epsilon_ref = ((epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * term)
        return epsilon_ref

    def calculate_patch_length(self, fr, epsilon_ref, h, W):
        """Длина патча"""

        term1 = self.c / (2 * fr * np.sqrt(epsilon_ref))
        term2_numerator = (epsilon_ref + 0.3) * (W / h + 0.264)
        term2_denominator = (epsilon_ref - 0.258) * (W / h + 0.8)
        term2 = 0.824 * h * (term2_numerator / term2_denominator)

        L = term1 - term2
        return L


    def calculate_conductances(self, W, L, fr):
        lambda_0 = self.c / fr
        k0 = 2 * np.pi / lambda_0

        def integrand_G1(theta):
            value1 = np.sin((k0 * W / 2) * np.cos(theta)) / np.cos(theta)
            value = (value1 ** 2) * (np.sin(theta) ** 3)
            return value

        def integrand_G12(theta):
            value1 = np.sin((k0 * W / 2) * np.cos(theta)) / np.cos(theta)
            value = (value1 ** 2) * j0(k0 * L * np.sin(theta)) * (np.sin(theta) ** 3)
            return value

        G1, _ = integrate.quad(integrand_G1, 0, np.pi)
        G12, _ = integrate.quad(integrand_G12, 0, np.pi)

        G1 /= (120 * np.pi ** 2)
        G12 /= (120 * np.pi ** 2)

        return max(G1, 1e-12), max(G12, 1e-12)


    def calculate_input_impedance_edge(self, G1, G12):
        """Входное сопротивление на краю"""

        R_in = 1 / (2 * (G1 + G12))
        return R_in

    def calculate_feed_position(self, R_in_edge, L):
        """Положение точки питания"""

        if R_in_edge < 50:
            # Если сопротивление на краю меньше 50 Ом, питаем с края
            y0 = 0
        else:
            y0 = (L / np.pi) * np.acos(np.sqrt(50 / R_in_edge))
        return y0 

    def calculate_directivity(self, W, L, fr, epsilon_ref):
        """Расчет направленности через двойное интегрирование:"""

        lambda_0 = self.c / fr
        k0 = 2 * np.pi / lambda_0

        # Подынтегральная функция для I1
        def integrand_I1(theta, phi):
            value1 = np.sin((k0 * W / 2) * np.cos(theta))/np.cos(theta)
            term1 = (value1 ** 2) * (np.sin(theta) ** 3)

            Leff = self.c / (2 * fr * np.sqrt(epsilon_ref))
            term2 = np.cos((k0 * Leff / 2) * np.sin(theta) * np.sin(phi)) ** 2

            return term1 * term2

        # Двойное численное интегрирование
        I1, error = integrate.dblquad(integrand_I1, 0, np.pi, lambda x: 0, lambda x: np.pi)

        # Расчет направленности
        if I1 > 0:
            D = ((2 * np.pi * W) / lambda_0) ** 2 * (np.pi / I1)
            # Переводим в dBi
            D_dBi = 10 * np.log10(D)
        else:
            D_dBi = 0

        return D_dBi

class MicrostripPatchAntennaCalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Калькулятор микрополосковой патч-антенны")
        self.root.geometry("1300x850")
        self.root.configure(bg='#2c3e50')

        # Инициализация калькулятора
        self.calculator = MicrostripPatchAntennaCalculator()
        self.current_results = None
        self.tk_image = None  # Для хранения ссылки на изображение

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style() # Объект для управления стилями
        style.theme_use('clam')

        # Настройка цветовой схемы для текстовых форматов
        # font - шрифт, размер, жирный; background - цвет фона; foreground - цвет текста
        style.configure('Custom.TFrame', background='#34495e')
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), background='#34495e', foreground='white')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), background='#34495e', foreground='#ecf0f1')
        style.configure('Input.TLabel', font=('Arial', 10), background='#34495e', foreground='#bdc3c7')
        style.configure('ResultTitle.TLabel', font=('Arial', 11, 'bold'), background='#2c3e50', foreground='#3498db')
        style.configure('ResultValue.TLabel', font=('Arial', 11, 'bold'), background='#2c3e50', foreground='#ecf0f1')
        style.configure('Unit.TLabel', font=('Arial', 9), background='#2c3e50', foreground='#95a5a6')

        style.configure('Custom.TButton', font=('Arial', 10, 'bold'), background='#3498db', foreground='white')
        style.map('Custom.TButton', background=[('active', '#2980b9')]) # Состояние виджета при наведении мышки

        style.configure('Input.TEntry', fieldbackground='#ecf0f1', foreground='#2c3e50')
        style.configure('Custom.TLabelframe', background='#34495e', foreground='white')
        style.configure('Custom.TLabelframe.Label', background='#34495e', foreground='white')

    def create_widgets(self):
        """Создание интерфейса"""
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Заголовок
        title_label = ttk.Label(main_frame, text="КАЛЬКУЛЯТОР МИКРОПОЛОСКОВОЙ ПАТЧ-АНТЕННЫ", style='Title.TLabel')
        title_label.pack(pady=(0, 30))

        # Основной контейнер с тремя колонками
        container = ttk.Frame(main_frame, style='Custom.TFrame')
        container.pack(fill='both', expand=True)

        # Левая панель - ввод параметров
        self.setup_input_panel(container)

        # Центральная панель - изображение антенны
        self.setup_image_panel(container)

        # Правая панель - результаты
        self.setup_results_panel(container)

    def setup_input_panel(self, parent):
        """Панель ввода параметров"""
        input_frame = ttk.LabelFrame(parent, text="ВХОДНЫЕ ПАРАМЕТРЫ",
                                     style='Custom.TLabelframe', padding=25)
        input_frame.pack(side='left', fill='both', padx=(0, 15))

        # Параметры ввода
        input_params = [
            ("Резонансная частота", "freq", "ГГц", "2.4"),
            ("Диэлектрическая проницаемость", "epsilon", "", "4.4"),
            ("Толщина подложки", "height", "мм", "1.6"),
        ]

        self.input_vars = {}

        for i, (label, key, unit, default) in enumerate(input_params):
            # Метка параметра
            param_label = ttk.Label(input_frame, text=label, style='Input.TLabel')
            param_label.grid(row=i * 2, column=0, sticky='w', pady=(20, 5))

            # Поле ввода
            var = tk.StringVar(value=default)
            entry = ttk.Entry(input_frame, textvariable=var, width=18, style='Input.TEntry', font=('Arial', 11))
            entry.grid(row=i * 2 + 1, column=0, sticky='w', pady=(0, 15))
            self.input_vars[key] = var

            # Единица измерения
            if unit:
                unit_label = ttk.Label(input_frame, text=unit, style='Input.TLabel')
                unit_label.grid(row=i * 2 + 1, column=1, sticky='w', padx=(10, 0), pady=(0, 15))

        # Кнопка расчета
        calc_btn = ttk.Button(input_frame, text="ВЫЧИСЛИТЬ ПАРАМЕТРЫ", style='Custom.TButton', command=self.calculate_parameters, width=20)
        calc_btn.grid(row=6, column=0, columnspan=2, pady=(20, 10), sticky='we')

        # Информация о допустимых диапазонах
        info_text = """
Допустимые диапазоны:
• Частота: 0.1 - 100 ГГц
• Диэлектрическая проницаемость: 1 - 20
• Толщина подложки: 0.01 - 100 мм

Рекомендуемые значения:
• Wi-Fi 2.4 ГГц: 2.4 ГГц, εr=4.4, h=1.6 мм
• Wi-Fi 5 ГГц: 5.0 ГГц, εr=4.4, h=1.0 мм
• GSM 900 МГц: 0.9 ГГц, εr=2.2, h=3.2 мм
"""
        info_label = tk.Label(input_frame, text=info_text, bg='#34495e', fg='#bdc3c7', font=('Arial', 9), justify=tk.LEFT)
        info_label.grid(row=7, column=0, columnspan=2, sticky='w', pady=(20, 0))

    def setup_image_panel(self, parent):
        """Панель с изображением антенны"""

        image_frame = ttk.LabelFrame(parent, text="СХЕМА АНТЕННЫ", style='Custom.TLabelframe', padding=20)
        image_frame.pack(side='left', fill='both', expand=True, padx=15)

        # Загружаем изображение
        pil_image = Image.open("image.png")

        # Масштабируем
        max_width, max_height = 400, 280
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Конвертируем для tkinter
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Отображаем
        img_label = tk.Label(image_frame, image=self.tk_image, bg='#34495e')
        img_label.pack(pady=10)

        # Подпись
        caption_label = tk.Label(image_frame,text="Схема микрополосковой патч-антенны", bg='#34495e', fg='#3498db', font=('Arial', 10, 'bold'))
        caption_label.pack()


    def setup_results_panel(self, parent):
        """Панель результатов в современном стиле"""

        results_frame = ttk.LabelFrame(parent, text="РЕЗУЛЬТАТЫ РАСЧЕТА", style='Custom.TLabelframe', padding=25)
        results_frame.pack(side='right', fill='both', expand=True)

        # Сетка для результатов
        results_grid = ttk.Frame(results_frame, style='Custom.TFrame')
        results_grid.pack(fill='both', expand=True)

        # Секция 1: Геометрические параметры
        geom_frame = ttk.LabelFrame(results_grid, text="ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ", style='Custom.TLabelframe', padding=20)
        geom_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=(0, 15))

        geom_params = [
            ("Ширина патча (W)", "patch_width", "мм"),
            ("Длина патча (L)", "patch_length", "мм"),
            ("Точка питания (y₀)", "feed_position_50ohm", "мм"),
            ("Эффективная εr", "effective_permittivity", ""),
        ]

        self.result_vars = {}

        for i, (label, key, unit) in enumerate(geom_params):
            self.create_result_row(geom_frame, label, key, unit, i)

        # Секция 2: Электрические параметры
        elec_frame = ttk.LabelFrame(results_grid, text="ЭЛЕКТРИЧЕСКИЕ ПАРАМЕТРЫ", style='Custom.TLabelframe', padding=20)
        elec_frame.grid(row=0, column=1, sticky='nsew', padx=(10, 0), pady=(0, 15))

        elec_params = [
            ("Сопр. на краю", "input_impedance_edge", "Ом"),
            ("Проводимость G₁", "single_slot_conductance", "См"),
            ("Проводимость G₁₂", "mutual_conductance", "См"),
            ("Направленность", "directivity", "dBi"),
        ]

        for i, (label, key, unit) in enumerate(elec_params):
            self.create_result_row(elec_frame, label, key, unit, i)

        # Настройка весов сетки
        results_grid.columnconfigure(0, weight=1)
        results_grid.columnconfigure(1, weight=1)
        results_grid.rowconfigure(0, weight=1)

        # Панель управления результатами
        self.setup_results_control(results_frame)

    def create_result_row(self, parent, label, key, unit, row):
        """Создание строки результата"""

        # Метка параметра
        label_widget = ttk.Label(parent, text=label, style='ResultTitle.TLabel')
        label_widget.grid(row=row, column=0, sticky='w', pady=12)

        # Значение
        value_var = tk.StringVar(value="--")
        value_label = ttk.Label(parent, textvariable=value_var, style='ResultValue.TLabel')
        value_label.grid(row=row, column=1, sticky='w', padx=(15, 5), pady=12)

        # Единица измерения
        if unit:
            unit_label = ttk.Label(parent, text=unit, style='Unit.TLabel')
            unit_label.grid(row=row, column=2, sticky='w', pady=12)

        self.result_vars[key] = value_var

    def setup_results_control(self, parent):
        """Панель управления результатами"""

        control_frame = ttk.Frame(parent, style='Custom.TFrame')
        control_frame.pack(fill='x', pady=(25, 0))

        # Заголовок раздела
        action_label = ttk.Label(control_frame, text="ЭКСПОРТ РЕЗУЛЬТАТОВ", style='Subtitle.TLabel')
        action_label.pack(anchor='w', pady=(0, 10))

        # Кнопки действий
        actions_frame = ttk.Frame(control_frame, style='Custom.TFrame')
        actions_frame.pack(fill='x')

        actions = [
            ("Сохранить в TXT", self.save_results),
        ]

        for i, (text, command) in enumerate(actions):
            btn = ttk.Button(actions_frame, text=text, style='Custom.TButton', command=command, width=18)
            btn.pack(side='left', padx=(0, 10))

    def calculate_parameters(self):
        """Расчет параметров антенны"""
        try:
            self.root.update()

            # Получение и валидация входных данных
            freq = self.input_vars['freq'].get().strip()
            epsilon = self.input_vars['epsilon'].get().strip()
            height = self.input_vars['height'].get().strip()

            # Проверка на пустые значения
            if not freq or not epsilon or not height:
                messagebox.showerror("Ошибка", "Все поля должны быть заполнены")
                return

            # Проверка на числа
            try:
                fr = float(freq) * 1e9
                epsilon_r = float(epsilon)
                h = float(height) * 1e-3
            except ValueError:
                messagebox.showerror("Ошибка", "Все параметры должны быть числами")
                return

            # Выполнение расчета
            results = self.calculator.calculate_antenna_parameters(fr, epsilon_r, h)
            self.current_results = results

            # Обновление интерфейса
            self.update_results_display(results)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка расчета: {str(e)}")
            return

    def update_results_display(self, results):
        """Обновление отображения результатов"""
        # Форматирование значений
        formatted_results = {
            'patch_width': f"{results['patch_width'] * 1000:.2f}",
            'patch_length': f"{results['patch_length'] * 1000:.2f}",
            'feed_position_50ohm': f"{results['feed_position_50ohm'] * 1000:.2f}",
            'effective_permittivity': f"{results['effective_permittivity']:.3f}",
            'input_impedance_edge': f"{results['input_impedance_edge']:.2f}",
            'single_slot_conductance': f"{results['single_slot_conductance']:.2e}",
            'mutual_conductance': f"{results['mutual_conductance']:.2e}",
            'directivity': f"{results['directivity']:.2f}",
        }

        # Обновление переменных
        for key, value in formatted_results.items():
            if key in self.result_vars:
                self.result_vars[key].set(value)

    def save_results(self):
        """Сохранение результатов в файл"""
        if not self.current_results:
            messagebox.showwarning("Нет данных", "Сначала выполните расчет")
            return

        try:
            filename = f"antenna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("РЕЗУЛЬТАТЫ РАСЧЕТА МИКРОПОЛОСКОВОЙ ПАТЧ-АНТЕННЫ\n")
                f.write("=" * 60 + "\n\n")

                f.write("ВХОДНЫЕ ПАРАМЕТРЫ:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Резонансная частота: {self.current_results['resonant_frequency'] / 1e9:.3f} ГГц\n")
                f.write(f"Диэлектрическая проницаемость: {self.current_results['substrate_permittivity']}\n")
                f.write(f"Толщина подложки: {self.current_results['substrate_height'] * 1000:.2f} мм\n\n")

                f.write("ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Ширина патча (W): {self.current_results['patch_width'] * 1000:.2f} мм\n")
                f.write(f"Длина патча (L): {self.current_results['patch_length'] * 1000:.2f} мм\n")
                f.write(f"Точка питания (y₀): {self.current_results['feed_position_50ohm'] * 1000:.2f} мм\n")
                f.write(f"Эффективная εr: {self.current_results['effective_permittivity']:.3f}\n\n")

                f.write("ЭЛЕКТРИЧЕСКИЕ ПАРАМЕТРЫ:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Сопротивление на краю: {self.current_results['input_impedance_edge']:.2f} Ом\n")
                f.write(f"Проводимость G₁: {self.current_results['single_slot_conductance']:.2e} См\n")
                f.write(f"Проводимость G₁₂: {self.current_results['mutual_conductance']:.2e} См\n")
                f.write(f"Направленность: {self.current_results['directivity']:.2f} dBi\n")
                f.write("=" * 60 + "\n")

            messagebox.showinfo("Сохранено", f"Результаты сохранены в файл:\n{filename}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")

if __name__ == "__main__":
    # Проверка зависимостей
    try:
        root = tk.Tk()
        app = MicrostripPatchAntennaCalculatorGUI(root)
        root.mainloop()
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что установлены все зависимости:")
        print("pip install numpy scipy matplotlib pillow")