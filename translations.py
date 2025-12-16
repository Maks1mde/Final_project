"""
translations.py - Модуль с переводами для интерфейса
"""

TRANSLATIONS = {
    "en": {
        "window_title": "Image Augmentation Tool",
        "menu_file": "File",
        "menu_language": "Language",
        "menu_exit": "Exit",
        "menu_english": "English",
        "menu_russian": "Russian",
        "menu_help": "Help",
        "menu_about": "About",

        "dir_group": "Dataset Directory",
        "no_dir_selected": "No directory selected",
        "browse": "Browse...",
        "aug_params": "Augmentation Parameters",
        "num_copies": "Number of augmented copies:",
        "dataset_split": "Dataset Split",
        "train": "Train:",
        "val": "Validation:",
        "test": "Test:",
        "preview": "Preview Augmentation",
        "apply_all": "Apply to All Images",
        "save_results": "Save Results...",
        "original_image": "Original Image",
        "augmented_image": "Augmented Image",
        "log": "Log",

        "noise_tab": "Noise",
        "noise_type": "Noise Type:",
        "intensity": "Intensity:",
        "gaussian": "Gaussian",
        "rayleigh": "Rayleigh",
        "exponential": "Exponential",

        "geom_tab": "Geometric",
        "transform_type": "Transform Type:",
        "rotation_angle": "Rotation Angle:",
        "rotation": "Rotation",
        "scaling": "Scaling",
        "translation": "Translation",
        "shear": "Shear",
        "reflection": "Reflection",
        "affine": "Affine",
        "perspective": "Perspective",

        "color_tab": "Color",
        "color_transform": "Color Transform:",
        "binary_threshold": "Binary Threshold:",
        "rgb_to_grayscale": "RGB to Grayscale",
        "rgb_to_binary": "RGB to Binary",
        "yiq_transform": "YIQ Transform",

        "status_ready": "Ready",
        "status_loading": "Loading...",
        "status_augmenting": "Augmenting...",
        "status_complete": "Complete",

        "warning_no_dir": "Please select a directory first",
        "error": "Error",
        "success": "Success",
        "augmentation_completed": "Augmentation completed successfully!",
        "no_images": "No images found",
        "select_dir": "Select Directory",
        "select_save_dir": "Select Save Directory",
        "starting_aug": "Starting augmentation of {} copies...",
        "created_splits": "Created dataset splits: Train: {}, Val: {}, Test: {}",
        "no_results": "No results to save",
        "select_save_location": "Select Save Location",
        "results_at": "Results available at: {}",
        "confirm_exit": "Confirm Exit",
        "exit_message": "Are you sure you want to exit?",
        "about_title": "About",
        "about_message": "Image Augmentation Tool\nVersion 1.0\n\nDeveloped for course project",

        "scale_factor": "Scale factor:",
        "x_shift": "X shift:",
        "y_shift": "Y shift:",
        "shear_x": "Shear X:",
        "shear_y": "Shear Y:",
        "flip_direction": "Flip direction:",
        "horizontal": "Horizontal",
        "vertical": "Vertical",
        "both": "Both",

        "browse_dir": "Browse Directory...",
        "browse_image": "Browse Image...",
        "no_image_selected": "No image selected",
        "current_image": "Current: {}",
        "warning_no_image": "Please select an image first",
        "select_image": "Select Image",

        "operations": {
            "noise": "Add Noise",
            "denoise": "Remove Noise",
            "histogram": "Histogram Operations",
            "color": "Color Transformations",
            "gradient": "Gradient Operations",
            "mixing": "Image Mixing",
            "geometric": "Geometric Transformations"
        },

        "filter_types": {
            "average": "Average Filter",
            "gaussian": "Gaussian Filter",
            "median": "Median Filter"
        }
    },

    "ru": {
        "window_title": "Инструмент аугментации изображений",
        "menu_file": "Файл",
        "menu_language": "Язык",
        "menu_exit": "Выход",
        "menu_english": "Английский",
        "menu_russian": "Русский",
        "menu_help": "Помощь",
        "menu_about": "О программе",

        "dir_group": "Директория набора данных",
        "no_dir_selected": "Директория не выбрана",
        "browse": "Обзор...",
        "aug_params": "Параметры аугментации",
        "num_copies": "Количество аугментированных копий:",
        "dataset_split": "Разделение набора данных",
        "train": "Обучающая:",
        "val": "Валидационная:",
        "test": "Тестовая:",
        "preview": "Предпросмотр аугментации",
        "apply_all": "Применить ко всем изображениям",
        "save_results": "Сохранить результаты...",
        "original_image": "Исходное изображение",
        "augmented_image": "Аугментированное изображение",
        "log": "Лог",

        "noise_tab": "Шум",
        "noise_type": "Тип шума:",
        "intensity": "Интенсивность:",
        "gaussian": "Гауссов шум",
        "rayleigh": "Шум Релея",
        "exponential": "Экспоненциальный шум",

        "geom_tab": "Геометрия",
        "transform_type": "Тип преобразования:",
        "rotation_angle": "Угол поворота:",
        "rotation": "Поворот",
        "scaling": "Масштабирование",
        "translation": "Сдвиг",
        "shear": "Сдвиг (shear)",
        "reflection": "Отражение",
        "affine": "Аффинное преобразование",
        "perspective": "Перспективное преобразование",

        "color_tab": "Цвет",
        "color_transform": "Преобразование цвета:",
        "binary_threshold": "Порог бинаризации:",
        "rgb_to_grayscale": "RGB в оттенки серого",
        "rgb_to_binary": "RGB в бинарное",
        "yiq_transform": "Преобразование YIQ",

        "status_ready": "Готов",
        "status_loading": "Загрузка...",
        "status_augmenting": "Аугментация...",
        "status_complete": "Завершено",

        "warning_no_dir": "Пожалуйста, сначала выберите директорию",
        "error": "Ошибка",
        "success": "Успех",
        "augmentation_completed": "Аугментация успешно завершена!",
        "no_images": "Изображения не найдены",
        "select_dir": "Выберите директорию",
        "select_save_dir": "Выберите директорию для сохранения",
        "starting_aug": "Начало аугментации {} копий...",
        "created_splits": "Созданы разделы: Обучающая: {}, Валидационная: {}, Тестовая: {}",
        "no_results": "Нет результатов для сохранения",
        "select_save_location": "Выберите место сохранения",
        "results_at": "Результаты доступны в: {}",
        "confirm_exit": "Подтверждение выхода",
        "exit_message": "Вы уверены, что хотите выйти?",
        "about_title": "О программе",
        "about_message": "Инструмент аугментации изображений\nВерсия 1.0\n\nРазработано для курсового проекта",

        "scale_factor": "Коэффициент масштаба:",
        "x_shift": "Сдвиг по X:",
        "y_shift": "Сдвиг по Y:",
        "shear_x": "Сдвиг по X:",
        "shear_y": "Сдвиг по Y:",
        "flip_direction": "Направление отражения:",
        "horizontal": "Горизонтальное",
        "vertical": "Вертикальное",
        "both": "Оба направления",

        "browse_dir": "Обзор директории...",
        "browse_image": "Выбрать изображение...",
        "no_image_selected": "Изображение не выбрано",
        "current_image": "Текущее: {}",
        "warning_no_image": "Пожалуйста, сначала выберите изображение",
        "select_image": "Выберите изображение",

        "operations": {
            "noise": "Добавить шум",
            "denoise": "Удалить шум",
            "histogram": "Операции с гистограммой",
            "color": "Цветовые преобразования",
            "gradient": "Операции с градиентом",
            "mixing": "Смешение изображений",
            "geometric": "Геометрические преобразования"
        },

        "filter_types": {
            "average": "Усредняющий фильтр",
            "gaussian": "Фильтр Гаусса",
            "median": "Медианный фильтр"
        }
    }
}

class TranslationManager:
    """Менеджер переводов"""

    def __init__(self, language="en"):
        self.language = language
        self.translations = TRANSLATIONS.get(language, TRANSLATIONS["en"])

    def set_language(self, language):
        """Установить язык"""
        if language in TRANSLATIONS:
            self.language = language
            self.translations = TRANSLATIONS[language]
            return True
        return False

    def get(self, key, default=None):
        """Получить перевод по ключу"""
        return self.translations.get(key, default)

    def tr(self, key, default=""):
        """Алиас для get"""
        return self.get(key, default)

    def get_current_language(self):
        """Получить текущий язык"""
        return self.language

    def get_available_languages(self):
        """Получить список доступных языков"""
        return list(TRANSLATIONS.keys())

# Глобальный экземпляр менеджера переводов
_translation_manager = TranslationManager()

def get_translation_manager():
    """Получить глобальный менеджер переводов"""
    return _translation_manager

def set_language(language):
    """Установить язык глобально"""
    return _translation_manager.set_language(language)

def tr(key, default=""):
    """Получить перевод по ключу"""
    return _translation_manager.tr(key, default)

def get_current_language():
    """Получить текущий язык"""
    return _translation_manager.get_current_language()