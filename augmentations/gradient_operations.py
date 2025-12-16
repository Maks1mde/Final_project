# Модуль для операций с градиентом

import numpy as np
import cv2
from .base_augmentation import BaseAugmentation


class GradientAugmentation(BaseAugmentation):
    """Класс для операций с градиентом"""

    def __init__(self):
        super().__init__()
        self.params = {
            'operator': 'sobel',
            'alpha': 0.5  # Коэффициент для нерезкого маскирования
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Применить оператор градиента"""
        operator = self.params['operator']

        if operator == 'sobel':
            gradient = self._sobel_operator(image)
        elif operator == 'prewitt':
            gradient = self._prewitt_operator(image)
        else:
            return image

        # Нерезкое маскирование
        alpha = self.params['alpha']
        sharpened = cv2.addWeighted(image, 1 + alpha, gradient, -alpha, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _sobel_operator(self, image: np.ndarray) -> np.ndarray:
        """Оператор Собеля"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))

        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    def _prewitt_operator(self, image: np.ndarray) -> np.ndarray:
        """Оператор Превитта"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

        magnitude = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))

        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)