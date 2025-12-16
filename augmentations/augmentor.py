"""
Исправленный augmentations/augmentor.py
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import traceback
from .noise_filters import NoiseAugmentation, DenoiseAugmentation
from .color_transforms import ColorAugmentation
from .geometric_transforms import GeometricAugmentation
from .gradient_operations import GradientAugmentation
from .image_mixing import MixingAugmentation
from .histogram_operations import HistogramAugmentation

class ImageAugmentor:
    """Основной класс для аугментации изображений"""

    def __init__(self):
        self.augmentations = {
            'noise': NoiseAugmentation(),
            'denoise': DenoiseAugmentation(),
            'histogram': HistogramAugmentation(),
            'color': ColorAugmentation(),
            'gradient': GradientAugmentation(),
            'mixing': MixingAugmentation(),
            'geometric': GeometricAugmentation(),
        }

    def load_image(self, image_path: Path) -> np.ndarray:
        """Безопасная загрузка изображения"""
        try:
            if not image_path.exists():
                print(f"File not found: {image_path}")
                return None

            # Загрузить изображение
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Failed to load image: {image_path}")
                return None

            # Конвертировать в BGR если нужно
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR или RGB
                # Предполагаем BGR (стандарт OpenCV)
                pass
            else:
                print(f"Unsupported image format: {image.shape}")
                return None

            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            traceback.print_exc()
            return None

    def save_image(self, image: np.ndarray, save_path: Path) -> bool:
        """Безопасное сохранение изображения"""
        try:
            if image is None or image.size == 0:
                print(f"Invalid image for saving: {save_path}")
                return False

            # Создать директорию если нужно
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Сохранить изображение
            success = cv2.imwrite(str(save_path), image)

            if not success:
                print(f"Failed to save image: {save_path}")

            return success

        except Exception as e:
            print(f"Error saving image {save_path}: {e}")
            traceback.print_exc()
            return False

    def apply_augmentation(self, image: np.ndarray, aug_type: str,
                          params: Dict[str, Any]) -> np.ndarray:
        """Безопасное применение аугментации"""
        try:
            if image is None or image.size == 0:
                return image

            if aug_type in self.augmentations:
                augmenter = self.augmentations[aug_type]
                augmenter.set_params(params)
                result = augmenter.apply(image)

                if result is None:
                    print(f"Augmentation {aug_type} returned None")
                    return image.copy()

                return result
            else:
                print(f"Unknown augmentation type: {aug_type}")
                return image.copy()

        except Exception as e:
            print(f"Error applying augmentation {aug_type}: {e}")
            traceback.print_exc()
            return image.copy() if image is not None else None

    def batch_augment(self, input_dir: Path, output_dir: Path,
                     augmentations: List[Tuple[str, Dict]], num_copies: int = 1):
        """Пакетная аугментация изображений"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Получить список изображений
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []

            for ext in image_extensions:
                image_files.extend(input_dir.glob(f"*{ext}"))
                image_files.extend(input_dir.glob(f"*{ext.upper()}"))

            for img_path in image_files:
                image = self.load_image(img_path)
                if image is None:
                    continue

                for i in range(num_copies):
                    augmented = image.copy()

                    for aug_type, params in augmentations:
                        augmented = self.apply_augmentation(augmented, aug_type, params)

                    # Сохранить результат
                    new_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                    self.save_image(augmented, output_dir / new_name)

        except Exception as e:
            print(f"Error in batch_augment: {e}")
            traceback.print_exc()