#Модуль для шумовых операций

import numpy as np
import cv2
from .base_augmentation import BaseAugmentation


class NoiseAugmentation(BaseAugmentation):
    """Класс для добавления шума к изображениям"""

    def __init__(self):
        super().__init__()
        self.params = {
            'noise_type': 'gaussian',
            'intensity': 0.1,
            'mean': 0,
            'sigma': 25
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Добавить шум к изображению"""
        noise_type = self.params['noise_type']
        intensity = self.params['intensity']

        if noise_type == 'gaussian':
            return self._add_gaussian_noise(image)
        elif noise_type == 'rayleigh':
            return self._add_rayleigh_noise(image)
        elif noise_type == 'exponential':
            return self._add_exponential_noise(image)
        else:
            return image

    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Добавить гауссов шум"""
        row, col, ch = image.shape
        mean = self.params['mean']
        sigma = self.params['sigma']
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _add_rayleigh_noise(self, image: np.ndarray) -> np.ndarray:
        """Добавить шум Релея"""
        row, col, ch = image.shape
        scale = self.params['sigma']
        rayleigh = np.random.rayleigh(scale, (row, col, ch))
        noisy = image + rayleigh
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _add_exponential_noise(self, image: np.ndarray) -> np.ndarray:
        """Добавить экспоненциальный шум"""
        row, col, ch = image.shape
        scale = self.params['sigma']
        exponential = np.random.exponential(scale, (row, col, ch))
        noisy = image + exponential
        return np.clip(noisy, 0, 255).astype(np.uint8)


class DenoiseAugmentation(BaseAugmentation):
    """Класс для удаления шума с изображений"""

    def __init__(self):
        super().__init__()
        self.params = {
            'filter_type': 'gaussian',
            'kernel_size': 5,
            'sigma': 1.0
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Применить фильтр удаления шума"""
        filter_type = self.params['filter_type']

        if filter_type == 'average':
            return self._apply_average_filter(image)
        elif filter_type == 'gaussian':
            return self._apply_gaussian_filter(image)
        elif filter_type == 'median':
            return self._apply_median_filter(image)
        else:
            return image

    def _apply_average_filter(self, image: np.ndarray) -> np.ndarray:
        """Применить усредняющий фильтр"""
        ksize = self.params['kernel_size']
        return cv2.blur(image, (ksize, ksize))

    def _apply_gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """Применить фильтр Гаусса"""
        ksize = self.params['kernel_size']
        sigma = self.params['sigma']
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _apply_median_filter(self, image: np.ndarray) -> np.ndarray:
        """Применить медианный фильтр"""
        ksize = self.params['kernel_size']
        return cv2.medianBlur(image, ksize)