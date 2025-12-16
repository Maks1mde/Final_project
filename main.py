"""
Обновленный main.py с корректным подключением переводов
"""

import sys
import os
import traceback
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QIcon
from gui.main_window import AugmentationApp
import translations

def exception_hook(exctype, value, traceback_obj):
    """Глобальный обработчик исключений"""
    error_msg = ''.join(traceback.format_exception(exctype, value, traceback_obj))
    print(f"Unhandled exception: {error_msg}")

    # Показать сообщение об ошибке
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle("Error")
    msg_box.setText(f"An error occurred:\n{str(value)}")
    msg_box.setDetailedText(error_msg)
    msg_box.exec_()

    # Вызвать стандартный обработчик
    sys.__excepthook__(exctype, value, traceback_obj)

def main():
    """Точка входа в приложение"""
    # Установить глобальный обработчик исключений
    sys.excepthook = exception_hook

    # Создать приложение
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    # Определить язык системы
    system_locale = QLocale.system().name()
    if system_locale.startswith("ru"):
        translations.set_language("ru")
    else:
        translations.set_language("en")

    # Установить иконку приложения
    try:
        if hasattr(sys, '_MEIPASS'):
            # Для PyInstaller
            icon_path = os.path.join(sys._MEIPASS, 'icon.ico')
        else:
            icon_path = 'icon.ico'

        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except:
        pass

    try:
        # Создать главное окно
        window = AugmentationApp()
        window.show()

        # Запустить главный цикл
        sys.exit(app.exec_())

    except Exception as e:
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()