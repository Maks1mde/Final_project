from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget,
                             QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QComboBox, QSlider, QCheckBox, QSplitter,
                             QMessageBox, QProgressBar, QTextEdit, QMenuBar,
                             QMenu, QAction, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
import numpy as np
from pathlib import Path
import traceback
import sys
import os
import time
import hashlib

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модуль переводов
import translations
from augmentations.augmentor import ImageAugmentor

class ImageLabel(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap = None
        self._display_size = QSize(400, 400)

    def set_display_size(self, size):
        """Установить размер для отображения"""
        self._display_size = size
        self.update_display()

    def set_image(self, image: np.ndarray):
        """Установить изображение и создать пиксмап"""
        if image is None or image.size == 0:
            self.clear()
            return

        try:
            # Конвертировать в RGB если нужно
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = image.copy()

            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)

            # Создать QImage
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            if qimage.isNull():
                self.clear()
                return

            # Создать пиксмап
            self._pixmap = QPixmap.fromImage(qimage)
            self.update_display()

        except Exception as e:
            print(f"Error setting image: {e}")
            self.clear()

    def update_display(self):
        """Обновить отображение с фиксированным масштабированием"""
        if self._pixmap is None or self._pixmap.isNull():
            self.clear()
            return

        scaled_pixmap = self._pixmap.scaled(
            self._display_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Обработчик изменения размера"""
        super().resizeEvent(event)
        pass

    def clear(self):
        """Очистить изображение"""
        self._pixmap = None
        super().clear()

class AugmentationThread(QThread):
    """Поток для выполнения аугментации с уникальными именами файлов"""
    progress = pyqtSignal(int)
    progress_text = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, augmentor, input_dir, output_dir, augmentations, num_copies):
        super().__init__()
        self.augmentor = augmentor
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augmentations = augmentations
        self.num_copies = num_copies
        self.processed_count = 0
        self.total_count = 0

    def generate_unique_name(self, original_name, aug_params, copy_index):

        base_name = Path(original_name).stem

        # Создать описательную часть имени на основе параметров аугментации
        desc_parts = []
        for aug_type, params in aug_params:
            if aug_type == 'noise':
                noise_type = params.get('noise_type', 'noise')
                intensity = params.get('intensity', 0.1)
                desc_parts.append(f"{noise_type[:3]}_{intensity:.2f}")
            elif aug_type == 'geometric':
                transform = params.get('transform_type', 'geom')
                if transform == 'rotation':
                    angle = params.get('angle', 0)
                    desc_parts.append(f"rot_{angle}")
                elif transform == 'scaling':
                    scale = params.get('scale', 1.0)
                    desc_parts.append(f"scale_{scale:.1f}")
                elif transform == 'translation':
                    tx = params.get('tx', 0)
                    ty = params.get('ty', 0)
                    desc_parts.append(f"trans_{tx}_{ty}")
                else:
                    desc_parts.append(transform[:5])
            elif aug_type == 'color':
                transform = params.get('transform_type', 'color')
                desc_parts.append(transform[:5])

        # Создать хэш для уникальности
        param_str = str(aug_params) + str(time.time()) + str(copy_index)
        short_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]

        # Собрать имя
        desc_str = "_".join(desc_parts[:3])  # Берем максимум 3 параметра
        timestamp = int(time.time() % 1000000)  # Только последние 6 цифр

        if desc_str:
            new_name = f"{base_name}_aug_{desc_str}_{short_hash}_{timestamp}_{copy_index:03d}{Path(original_name).suffix}"
        else:
            new_name = f"{base_name}_aug_{short_hash}_{timestamp}_{copy_index:03d}{Path(original_name).suffix}"

        return new_name

    def run(self):
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []

            for ext in image_extensions:
                image_files.extend(self.input_dir.glob(f"*{ext}"))
                image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

            self.total_count = len(image_files)

            if self.total_count == 0:
                self.error.emit(translations.tr("no_images"))
                return

            self.processed_count = 0

            for img_path in image_files:
                try:
                    image = self.augmentor.load_image(img_path)
                    if image is None:
                        continue

                    self.progress_text.emit(f"Processing: {img_path.name}")

                    for i in range(self.num_copies):
                        augmented = image.copy()

                        for aug_type, params in self.augmentations:
                            try:
                                augmented = self.augmentor.apply_augmentation(
                                    augmented, aug_type, params)
                            except Exception as e:
                                print(f"Error applying augmentation {aug_type}: {e}")
                                continue

                        new_name = self.generate_unique_name(
                            img_path.name, self.augmentations, i)
                        save_path = self.output_dir / new_name

                        # Проверить, не существует ли уже файл с таким именем
                        counter = 1
                        original_save_path = save_path
                        while save_path.exists():
                            # Добавить суффикс если файл уже существует
                            save_path = original_save_path.parent / f"{original_save_path.stem}_{counter:02d}{original_save_path.suffix}"
                            counter += 1
                            if counter > 100:  # Защита от бесконечного цикла
                                save_path = self.output_dir / f"{Path(img_path).stem}_aug_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}_{i:03d}{Path(img_path).suffix}"
                                break

                        if not self.augmentor.save_image(augmented, save_path):
                            print(f"Failed to save: {save_path}")

                    self.processed_count += 1
                    progress = int((self.processed_count / self.total_count) * 100)
                    self.progress.emit(progress)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()

class AugmentationApp(QMainWindow):
    """Главное окно приложения с выбором изображения для превью"""

    def __init__(self):
        super().__init__()
        self.augmentor = ImageAugmentor()
        self.current_image = None
        self.current_image_path = None  # Добавлено: путь к текущему изображению
        self.input_dir = None
        self.augmentation_thread = None

        self.translation_manager = translations.get_translation_manager()

        self.init_ui()
        self.init_menu()
        self.retranslate_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle(self.translation_manager.tr("window_title"))
        self.setGeometry(100, 100, 1200, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QHBoxLayout(central_widget)

        # Сплиттер для разделения
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Группа выбора данных
        self.data_group = QGroupBox()
        data_layout = QVBoxLayout()

        # Выбор директории
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel()
        self.dir_label.setWordWrap(True)
        dir_layout.addWidget(self.dir_label)

        self.browse_dir_btn = QPushButton()
        self.browse_dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.browse_dir_btn)

        data_layout.addLayout(dir_layout)

        # Выбор конкретного изображения для превью
        image_layout = QHBoxLayout()
        self.image_label = QLabel()
        self.image_label.setWordWrap(True)
        image_layout.addWidget(self.image_label)

        self.browse_image_btn = QPushButton()
        self.browse_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(self.browse_image_btn)

        data_layout.addLayout(image_layout)

        # Список изображений в директории
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(150)
        self.image_list.itemClicked.connect(self.on_image_selected)
        data_layout.addWidget(QLabel("Available images:"))
        data_layout.addWidget(self.image_list)

        self.data_group.setLayout(data_layout)
        left_layout.addWidget(self.data_group)

        # Параметры аугментации
        self.tabs = QTabWidget()

        # Вкладка шума
        self.noise_tab = self.create_noise_tab()
        self.tabs.addTab(self.noise_tab, "")

        # Вкладка геометрических преобразований
        self.geom_tab = self.create_geometric_tab()
        self.tabs.addTab(self.geom_tab, "")

        # Вкладка цветовых преобразований
        self.color_tab = self.create_color_tab()
        self.tabs.addTab(self.color_tab, "")

        left_layout.addWidget(self.tabs)

        # Количество копий
        self.copies_group = QGroupBox()
        copies_layout = QVBoxLayout()

        self.num_copies_label = QLabel()
        copies_layout.addWidget(self.num_copies_label)

        self.num_copies = QSpinBox()
        self.num_copies.setRange(1, 100)
        self.num_copies.setValue(5)
        copies_layout.addWidget(self.num_copies)

        # Процентное соотношение train/val/test
        self.splits_group = QGroupBox()
        splits_layout = QHBoxLayout()

        self.train_label = QLabel()
        splits_layout.addWidget(self.train_label)

        self.train_split = QSpinBox()
        self.train_split.setRange(0, 100)
        self.train_split.setValue(70)
        splits_layout.addWidget(self.train_split)

        self.val_label = QLabel()
        splits_layout.addWidget(self.val_label)

        self.val_split = QSpinBox()
        self.val_split.setRange(0, 100)
        self.val_split.setValue(15)
        splits_layout.addWidget(self.val_split)

        self.test_label = QLabel()
        splits_layout.addWidget(self.test_label)

        self.test_split = QSpinBox()
        self.test_split.setRange(0, 100)
        self.test_split.setValue(15)
        splits_layout.addWidget(self.test_split)

        self.splits_group.setLayout(splits_layout)
        copies_layout.addWidget(self.splits_group)

        self.copies_group.setLayout(copies_layout)
        left_layout.addWidget(self.copies_group)

        # Кнопки управления
        self.preview_btn = QPushButton()
        self.preview_btn.clicked.connect(self.preview_augmentation)
        left_layout.addWidget(self.preview_btn)

        self.apply_btn = QPushButton()
        self.apply_btn.clicked.connect(self.apply_augmentation)
        left_layout.addWidget(self.apply_btn)

        self.save_btn = QPushButton()
        self.save_btn.clicked.connect(self.save_results)
        left_layout.addWidget(self.save_btn)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel()  # Добавлено: метка прогресса
        self.progress_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.progress_bar)

        splitter.addWidget(left_panel)

        # Правая панель - просмотр изображений
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Изображения до/после
        images_splitter = QSplitter(Qt.Horizontal)

        # Исходное изображение
        self.original_label = ImageLabel()
        self.original_label.set_display_size(QSize(400, 400))
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        images_splitter.addWidget(self.original_label)

        # Аугментированное изображение
        self.augmented_label = ImageLabel()
        self.augmented_label.set_display_size(QSize(400, 400))
        self.augmented_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        images_splitter.addWidget(self.augmented_label)

        right_layout.addWidget(images_splitter)

        # Лог
        self.log_group = QGroupBox()
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.log_group.setLayout(log_layout)
        right_layout.addWidget(self.log_group)

        splitter.addWidget(right_panel)

        main_layout.addWidget(splitter)

        # Установить размеры сплиттера
        splitter.setSizes([400, 800])

    def init_menu(self):
        """Инициализация меню"""
        menubar = self.menuBar()

        # Меню File
        self.file_menu = menubar.addMenu("")

        # Exit action
        self.exit_action = QAction("", self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # Меню Language
        self.lang_menu = menubar.addMenu("")

        # English action
        self.english_action = QAction("", self)
        self.english_action.triggered.connect(lambda: self.change_language("en"))
        self.lang_menu.addAction(self.english_action)

        # Russian action
        self.russian_action = QAction("", self)
        self.russian_action.triggered.connect(lambda: self.change_language("ru"))
        self.lang_menu.addAction(self.russian_action)

        # Меню Help
        self.help_menu = menubar.addMenu("")

        # About action
        self.about_action = QAction("", self)
        self.about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(self.about_action)

    def retranslate_ui(self):
        """Перевод интерфейса"""
        # Обновить все тексты через менеджер переводов
        tr = self.translation_manager.tr

        # Заголовок окна
        self.setWindowTitle(tr("window_title"))

        # Меню
        self.file_menu.setTitle(tr("menu_file"))
        self.exit_action.setText(tr("menu_exit"))

        self.lang_menu.setTitle(tr("menu_language"))
        self.english_action.setText(tr("menu_english"))
        self.russian_action.setText(tr("menu_russian"))

        self.help_menu.setTitle(tr("menu_help"))
        self.about_action.setText(tr("menu_about"))

        # Группы
        self.data_group.setTitle(tr("dir_group"))
        self.copies_group.setTitle(tr("aug_params"))
        self.splits_group.setTitle(tr("dataset_split"))
        self.log_group.setTitle(tr("log"))

        # Тексты
        self.dir_label.setText(tr("no_dir_selected"))
        self.browse_dir_btn.setText(tr("browse_dir"))
        self.image_label.setText(tr("no_image_selected"))  # НОВОЕ
        self.browse_image_btn.setText(tr("browse_image"))  # НОВОЕ
        self.num_copies_label.setText(tr("num_copies"))
        self.train_label.setText(tr("train"))
        self.val_label.setText(tr("val"))
        self.test_label.setText(tr("test"))
        self.preview_btn.setText(tr("preview"))
        self.apply_btn.setText(tr("apply_all"))
        self.save_btn.setText(tr("save_results"))

        # Вкладки
        self.tabs.setTabText(0, tr("noise_tab"))
        self.tabs.setTabText(1, tr("geom_tab"))
        self.tabs.setTabText(2, tr("color_tab"))

        # Заголовки изображений
        self.original_label.setText(tr("original_image"))
        self.augmented_label.setText(tr("augmented_image"))

        # Обновить вкладки
        self.update_tab_texts()

    def update_tab_texts(self):
        """Обновить тексты на вкладках"""
        tr = self.translation_manager.tr

        # Вкладка шума
        self.noise_type_label.setText(tr("noise_type"))
        self.intensity_label.setText(tr("intensity"))

        # Обновить комбобоксы с переводами
        self.noise_type.clear()
        self.noise_type.addItems([
            tr("gaussian"),
            tr("rayleigh"),
            tr("exponential")
        ])

        # Вкладка геометрии
        self.transform_type_label.setText(tr("transform_type"))
        self.rotation_label.setText(tr("rotation_angle"))
        self.scaling_label.setText(tr("scale_factor"))  # НОВОЕ
        self.translation_x_label.setText(tr("x_shift"))  # НОВОЕ
        self.translation_y_label.setText(tr("y_shift"))  # НОВОЕ
        self.shear_x_label.setText(tr("shear_x"))  # НОВОЕ
        self.shear_y_label.setText(tr("shear_y"))  # НОВОЕ
        self.reflection_label.setText(tr("flip_direction"))  # НОВОЕ

        self.geom_type.clear()
        self.geom_type.addItems([
            tr("rotation"),
            tr("scaling"),
            tr("translation"),
            tr("shear"),
            tr("reflection"),
            tr("affine"),
            tr("perspective")
        ])

        self.flip_code.clear()  # НОВОЕ
        self.flip_code.addItems([  # НОВОЕ
            tr("horizontal"),
            tr("vertical"),
            tr("both")
        ])

        # Вкладка цвета
        self.color_transform_label.setText(tr("color_transform"))
        self.threshold_label.setText(tr("binary_threshold"))

        self.color_type.clear()
        self.color_type.addItems([
            tr("rgb_to_grayscale"),
            tr("rgb_to_binary"),
            tr("yiq_transform")
        ])

    def change_language(self, language_code):
        """Изменить язык интерфейса"""
        if self.translation_manager.set_language(language_code):
            self.retranslate_ui()
            self.log_text.append(f"Language changed to {language_code}")

    def create_noise_tab(self):
        """Создать вкладку для шумовых операций"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Тип шума
        self.noise_type_label = QLabel()
        layout.addWidget(self.noise_type_label)

        self.noise_type = QComboBox()
        layout.addWidget(self.noise_type)

        # Интенсивность
        self.intensity_label = QLabel()
        layout.addWidget(self.intensity_label)

        self.noise_intensity = QDoubleSpinBox()
        self.noise_intensity.setRange(0.0, 1.0)
        self.noise_intensity.setSingleStep(0.05)
        self.noise_intensity.setValue(0.1)
        layout.addWidget(self.noise_intensity)

        return tab

    def create_geometric_tab(self):
        """Создать вкладку для геометрических преобразований"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Тип преобразования
        self.transform_type_label = QLabel()
        layout.addWidget(self.transform_type_label)

        self.geom_type = QComboBox()
        self.geom_type.currentIndexChanged.connect(self.on_geom_type_changed)  # Изменено
        layout.addWidget(self.geom_type)

        # Стек виджетов для разных параметров
        self.geom_params_stack = QWidget()
        self.geom_params_layout = QVBoxLayout(self.geom_params_stack)

        # Виджеты для поворота
        self.rotation_widget = QWidget()
        rotation_layout = QVBoxLayout(self.rotation_widget)
        self.rotation_label = QLabel()
        rotation_layout.addWidget(self.rotation_label)
        self.rotation_angle = QSpinBox()
        self.rotation_angle.setRange(-180, 180)
        self.rotation_angle.setValue(15)
        rotation_layout.addWidget(self.rotation_angle)
        self.geom_params_layout.addWidget(self.rotation_widget)

        # Виджеты для масштабирования
        self.scaling_widget = QWidget()
        scaling_layout = QVBoxLayout(self.scaling_widget)
        self.scaling_label = QLabel()
        scaling_layout.addWidget(self.scaling_label)
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.1, 3.0)
        self.scale_factor.setSingleStep(0.1)
        self.scale_factor.setValue(0.8)
        scaling_layout.addWidget(self.scale_factor)
        self.geom_params_layout.addWidget(self.scaling_widget)

        # Виджеты для сдвига
        self.translation_widget = QWidget()
        translation_layout = QVBoxLayout(self.translation_widget)

        self.translation_x_label = QLabel()
        translation_layout.addWidget(self.translation_x_label)
        self.translation_x = QSpinBox()
        self.translation_x.setRange(-100, 100)
        self.translation_x.setValue(10)
        translation_layout.addWidget(self.translation_x)

        self.translation_y_label = QLabel()
        translation_layout.addWidget(self.translation_y_label)
        self.translation_y = QSpinBox()
        self.translation_y.setRange(-100, 100)
        self.translation_y.setValue(10)
        translation_layout.addWidget(self.translation_y)

        self.geom_params_layout.addWidget(self.translation_widget)

        # Виджеты для сдвига (shear)
        self.shear_widget = QWidget()
        shear_layout = QVBoxLayout(self.shear_widget)

        self.shear_x_label = QLabel()
        shear_layout.addWidget(self.shear_x_label)
        self.shear_x = QDoubleSpinBox()
        self.shear_x.setRange(-1.0, 1.0)
        self.shear_x.setSingleStep(0.1)
        self.shear_x.setValue(0.1)
        shear_layout.addWidget(self.shear_x)

        self.shear_y_label = QLabel()
        shear_layout.addWidget(self.shear_y_label)
        self.shear_y = QDoubleSpinBox()
        self.shear_y.setRange(-1.0, 1.0)
        self.shear_y.setSingleStep(0.1)
        self.shear_y.setValue(0.1)
        shear_layout.addWidget(self.shear_y)

        self.geom_params_layout.addWidget(self.shear_widget)

        # Виджеты для отражения
        self.reflection_widget = QWidget()
        reflection_layout = QVBoxLayout(self.reflection_widget)
        self.reflection_label = QLabel()
        reflection_layout.addWidget(self.reflection_label)
        self.flip_code = QComboBox()
        reflection_layout.addWidget(self.flip_code)
        self.geom_params_layout.addWidget(self.reflection_widget)

        layout.addWidget(self.geom_params_stack)

        # Скрыть все виджеты кроме первого
        self.scaling_widget.hide()
        self.translation_widget.hide()
        self.shear_widget.hide()
        self.reflection_widget.hide()

        # Подключить сигнал изменения типа преобразования
        self.geom_type.currentIndexChanged.connect(self.on_geom_type_changed)

        return tab

    def on_geom_type_changed(self, index):
        """Обработчик изменения типа геометрического преобразования"""
        # Скрыть все виджеты параметров
        self.rotation_widget.hide()
        self.scaling_widget.hide()
        self.translation_widget.hide()
        self.shear_widget.hide()
        self.reflection_widget.hide()

        # Показать нужный виджет параметров
        if index == 0:  # rotation
            self.rotation_widget.show()
        elif index == 1:  # scaling
            self.scaling_widget.show()
        elif index == 2:  # translation
            self.translation_widget.show()
        elif index == 3:  # shear
            self.shear_widget.show()
        elif index == 4:  # reflection
            self.reflection_widget.show()
        # affine и perspective не требуют дополнительных параметров

    def create_color_tab(self):
        """Создать вкладку для цветовых преобразований"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Тип преобразования
        self.color_transform_label = QLabel()
        layout.addWidget(self.color_transform_label)

        self.color_type = QComboBox()
        layout.addWidget(self.color_type)

        # Порог для бинаризации
        self.threshold_label = QLabel()
        layout.addWidget(self.threshold_label)

        self.binary_threshold = QSpinBox()
        self.binary_threshold.setRange(0, 255)
        self.binary_threshold.setValue(127)
        layout.addWidget(self.binary_threshold)

        return tab

    def show_about(self):
        """Показать окно 'О программе'"""
        QMessageBox.about(
            self,
            self.translation_manager.tr("about_title"),
            self.translation_manager.tr("about_message")
        )

    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        reply = QMessageBox.question(
            self,
            self.translation_manager.tr("confirm_exit"),
            self.translation_manager.tr("exit_message"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Остановить поток если запущен
            if self.augmentation_thread and self.augmentation_thread.isRunning():
                self.augmentation_thread.terminate()
                self.augmentation_thread.wait()
            event.accept()
        else:
            event.ignore()

    def select_directory(self):
        """Выбрать директорию с изображениями"""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                self.translation_manager.tr("select_dir")
            )

            if dir_path:
                self.input_dir = Path(dir_path)
                self.dir_label.setText(str(self.input_dir))
                self.log_text.append(f"Selected directory: {dir_path}")

                # Загрузить список изображений в директории
                self.load_image_list()

                # Загрузить первое изображение для превью
                QTimer.singleShot(100, self.load_sample_image)

        except Exception as e:
            self.log_text.append(f"Error selecting directory: {e}")
            QMessageBox.critical(
                self,
                self.translation_manager.tr("error"),
                f"Failed to select directory: {e}"
            )

    def select_image(self):
        """Выбрать конкретное изображение для превью"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self.translation_manager.tr("select_image"),
                str(self.input_dir) if self.input_dir else "",
                "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
            )

            if file_path:
                self.current_image_path = Path(file_path)
                self.image_label.setText(
                    self.translation_manager.tr("current_image").format(
                        self.current_image_path.name
                    )
                )
                self.load_image(self.current_image_path)

        except Exception as e:
            self.log_text.append(f"Error selecting image: {e}")
            QMessageBox.critical(
                self,
                self.translation_manager.tr("error"),
                f"Failed to select image: {e}"
            )

    def load_image_list(self):
        """Загрузить список изображений в директории"""
        self.image_list.clear()

        if not self.input_dir or not self.input_dir.exists():
            return

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

        for img_file in sorted(image_files):
            self.image_list.addItem(img_file.name)

    def on_image_selected(self, item):
        """Обработчик выбора изображения из списка"""
        if self.input_dir:
            image_path = self.input_dir / item.text()
            if image_path.exists():
                self.current_image_path = image_path
                self.image_label.setText(
                    self.translation_manager.tr("current_image").format(
                        self.current_image_path.name
                    )
                )
                self.load_image(image_path)

    def load_sample_image(self):
        """Загрузить пример изображения для предпросмотра"""
        try:
            if not self.input_dir or not self.input_dir.exists():
                return

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []

            for ext in image_extensions:
                image_files.extend(self.input_dir.glob(f"*{ext}"))
                image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

            if image_files:
                # Взять первое изображение
                first_image = image_files[0]
                self.current_image_path = first_image
                self.current_image = self.augmentor.load_image(first_image)

                if self.current_image is not None:
                    self.original_label.set_image(self.current_image)
                    self.image_label.setText(
                        self.translation_manager.tr("current_image").format(
                            self.current_image_path.name
                        )
                    )
                    self.log_text.append(f"Loaded sample image: {first_image.name}")
                else:
                    self.original_label.setText(self.translation_manager.tr("no_images"))
                    self.log_text.append(f"Failed to load image: {first_image}")
            else:
                self.original_label.setText(self.translation_manager.tr("no_images"))
                self.log_text.append(self.translation_manager.tr("no_images"))

        except Exception as e:
            self.log_text.append(f"Error loading sample image: {e}")
            self.original_label.setText(f"Error: {str(e)}")

    def load_image(self, image_path):
        """Загрузить изображение"""
        try:
            self.current_image = self.augmentor.load_image(image_path)

            if self.current_image is not None:
                self.original_label.set_image(self.current_image)
                self.log_text.append(f"Loaded image: {image_path.name}")
            else:
                self.original_label.setText(self.translation_manager.tr("no_images"))
                self.log_text.append(f"Failed to load image: {image_path}")

        except Exception as e:
            self.log_text.append(f"Error loading image: {e}")
            self.original_label.setText(f"Error: {str(e)}")

    def preview_augmentation(self):
        """Предпросмотр аугментации"""
        try:
            if self.current_image is None:
                QMessageBox.warning(
                    self,
                    self.translation_manager.tr("warning_no_image"),  # Изменено
                    self.translation_manager.tr("warning_no_image")   # Изменено
                )
                return

            augmentations = self.get_current_augmentations()
            augmented = self.current_image.copy()

            for aug_type, params in augmentations:
                try:
                    augmented = self.augmentor.apply_augmentation(augmented, aug_type, params)
                except Exception as e:
                    self.log_text.append(f"Error in preview {aug_type}: {e}")

            self.augmented_label.set_image(augmented)
            self.log_text.append("Preview augmentation applied")

        except Exception as e:
            self.log_text.append(f"Error in preview: {e}")
            QMessageBox.critical(
                self,
                self.translation_manager.tr("error"),
                f"Preview failed: {e}"
            )

    def get_current_augmentations(self):
        """Получить текущие параметры аугментации"""
        augmentations = []

        try:
            # Параметры шума
            if self.tabs.currentIndex() == 0:  # Noise tab
                noise_types = ['gaussian', 'rayleigh', 'exponential']
                current_index = self.noise_type.currentIndex() if hasattr(self, 'noise_type') else 0
                noise_params = {
                    'noise_type': noise_types[current_index],
                    'intensity': self.noise_intensity.value() if hasattr(self, 'noise_intensity') else 0.1,
                    'sigma': 25
                }
                augmentations.append(('noise', noise_params))

            # Параметры геометрических преобразований
            elif self.tabs.currentIndex() == 1:  # Geometric tab
                transform_types = ['rotation', 'scaling', 'translation',
                                   'shear', 'reflection', 'affine', 'perspective']
                current_index = self.geom_type.currentIndex() if hasattr(self, 'geom_type') else 0
                transform_type = transform_types[current_index]

                geom_params = {'transform_type': transform_type}

                if transform_type == 'rotation':
                    geom_params['angle'] = self.rotation_angle.value() if hasattr(self, 'rotation_angle') else 15
                elif transform_type == 'scaling':
                    geom_params['scale'] = self.scale_factor.value() if hasattr(self, 'scale_factor') else 0.8
                elif transform_type == 'translation':
                    geom_params['tx'] = self.translation_x.value() if hasattr(self, 'translation_x') else 10
                    geom_params['ty'] = self.translation_y.value() if hasattr(self, 'translation_y') else 10
                elif transform_type == 'shear':
                    geom_params['shear_x'] = self.shear_x.value() if hasattr(self, 'shear_x') else 0.1
                    geom_params['shear_y'] = self.shear_y.value() if hasattr(self, 'shear_y') else 0.1
                elif transform_type == 'reflection':
                    flip_codes = [0, 1, -1]
                    current_index = self.flip_code.currentIndex() if hasattr(self, 'flip_code') else 0
                    geom_params['flip_code'] = flip_codes[current_index]

                augmentations.append(('geometric', geom_params))

            # Параметры цветовых преобразований
            elif self.tabs.currentIndex() == 2:  # Color tab
                color_types = ['rgb_to_grayscale', 'rgb_to_binary', 'yiq_transform']
                current_index = self.color_type.currentIndex() if hasattr(self, 'color_type') else 0
                color_params = {
                    'transform_type': color_types[current_index],
                    'threshold': self.binary_threshold.value() if hasattr(self, 'binary_threshold') else 127
                }
                augmentations.append(('color', color_params))

        except Exception as e:
            self.log_text.append(f"Error getting augmentation params: {e}")

        return augmentations

    def apply_augmentation(self):
        """Применить аугментацию ко всем изображениями"""
        try:
            if not self.input_dir or not self.input_dir.exists():
                QMessageBox.warning(
                    self,
                    self.translation_manager.tr("warning_no_dir"),
                    self.translation_manager.tr("warning_no_dir")
                )
                return

            save_dir = QFileDialog.getExistingDirectory(
                self,
                self.translation_manager.tr("select_save_dir")
            )

            if not save_dir:
                return

            self.output_dir = Path(save_dir)
            augmentations = self.get_current_augmentations()
            num_copies = self.num_copies.value()

            # Создать поток аугментации
            self.augmentation_thread = AugmentationThread(
                self.augmentor, self.input_dir, self.output_dir,
                augmentations, num_copies
            )

            self.augmentation_thread.progress.connect(self.progress_bar.setValue)
            self.augmentation_thread.progress_text.connect(self.progress_label.setText)  # НОВОЕ
            self.augmentation_thread.finished.connect(self.on_augmentation_finished)
            self.augmentation_thread.error.connect(self.on_augmentation_error)

            self.progress_bar.setValue(0)
            self.progress_label.setText("Starting...")  # НОВОЕ
            self.augmentation_thread.start()

            self.log_text.append(
                self.translation_manager.tr("starting_aug").format(num_copies)
            )

        except Exception as e:
            self.log_text.append(f"Error starting augmentation: {e}")
            QMessageBox.critical(
                self,
                self.translation_manager.tr("error"),
                f"Failed to start augmentation: {e}"
            )

    def on_augmentation_finished(self):
        """Обработка завершения аугментации"""
        try:
            self.progress_bar.setValue(100)
            self.progress_label.setText("Complete")  # НОВОЕ
            self.log_text.append(self.translation_manager.tr("augmentation_completed"))

            self.create_dataset_splits()

            QMessageBox.information(
                self,
                self.translation_manager.tr("success"),
                self.translation_manager.tr("augmentation_completed")
            )

        except Exception as e:
            self.log_text.append(f"Error in completion: {e}")
            QMessageBox.critical(
                self,
                self.translation_manager.tr("error"),
                f"Error in completion: {e}"
            )

    def on_augmentation_error(self, error_msg):
        """Обработка ошибки аугментации"""
        self.log_text.append(f"Error: {error_msg}")
        self.progress_label.setText("Error")
        QMessageBox.critical(
            self,
            self.translation_manager.tr("error"),
            f"Augmentation failed: {error_msg}"
        )

    def create_dataset_splits(self):
        """Создать разделы train/val/test"""
        try:
            if not hasattr(self, 'output_dir') or not self.output_dir.exists():
                return

            aug_files = list(self.output_dir.glob("*_aug_*.*"))

            if not aug_files:
                self.log_text.append("No augmented files found for splitting")
                return

            np.random.shuffle(aug_files)

            total = len(aug_files)
            train_pct = self.train_split.value()
            val_pct = self.val_split.value()

            train_end = int(total * train_pct / 100)
            val_end = train_end + int(total * val_pct / 100)

            train_dir = self.output_dir / "train"
            val_dir = self.output_dir / "val"
            test_dir = self.output_dir / "test"

            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
            test_dir.mkdir(exist_ok=True)

            moved = 0
            for i, file_path in enumerate(aug_files):
                try:
                    if i < train_end:
                        dest = train_dir / file_path.name
                    elif i < val_end:
                        dest = val_dir / file_path.name
                    else:
                        dest = test_dir / file_path.name

                    file_path.rename(dest)
                    moved += 1

                except Exception as e:
                    self.log_text.append(f"Error moving {file_path.name}: {e}")
                    continue

            self.log_text.append(
                self.translation_manager.tr("created_splits").format(
                    min(train_end, moved),
                    min(val_end - train_end, moved - train_end),
                    moved - min(val_end, moved)
                )
            )

        except Exception as e:
            self.log_text.append(f"Error creating splits: {e}")

    def save_results(self):
        """Сохранить результаты"""
        try:
            if not hasattr(self, 'output_dir') or not self.output_dir.exists():
                QMessageBox.warning(
                    self,
                    self.translation_manager.tr("no_results"),
                    self.translation_manager.tr("no_results")
                )
                return

            save_path = QFileDialog.getExistingDirectory(
                self,
                self.translation_manager.tr("select_save_location")
            )

            if save_path:
                self.log_text.append(
                    self.translation_manager.tr("results_at").format(str(self.output_dir))
                )

        except Exception as e:
            self.log_text.append(f"Error saving results: {e}")
            QMessageBox.critical(
                self,
                self.translation_manager.tr("error"),
                f"Failed to save: {e}"

            )
