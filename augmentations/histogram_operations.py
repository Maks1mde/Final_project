# Модуль для операций с гистограммой:

import numpy as np
import cv2
from .base_augmentation import BaseAugmentation


class HistogramAugmentation(BaseAugmentation):
    """Класс для операций с гистограммой"""

    def __init__(self):
        super().__init__()
        self.params = {
            'operation': 'equalization',
            'clip_limit': 2.0,
            'tile_size': 8
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Применить операцию к гистограмме"""
        operation = self.params['operation']

        if operation == 'equalization':
            return self._histogram_equalization(image)
        elif operation == 'color_correction':
            return self._statistical_color_correction(image)
        else:
            return image

    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Эквализация гистограммы"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clip_limit'],
            tileGridSize=(self.params['tile_size'], self.params['tile_size'])
        )

        # Применить к каждому каналу
        channels = cv2.split(image)
        equalized_channels = []

        for channel in channels:
            equalized_channels.append(clahe.apply(channel))

        return cv2.merge(equalized_channels)

    def _statistical_color_correction(self, image: np.ndarray) -> np.ndarray:
        """Статистическая цветокоррекция"""
        # Нормализация цвета на основе статистики
        result = image.copy().astype(np.float32)

        for channel in range(3):
            channel_data = result[:, :, channel]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)

            # Нормализация
            if std_val > 0:
                normalized = (channel_data - mean_val) / std_val
                # Масштабирование к [0, 255]
                normalized = normalized * 64 + 128
                result[:, :, channel] = np.clip(normalized, 0, 255)

        return result.astype(np.uint8)