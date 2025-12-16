# Модуль для смешения изображений

import numpy as np
import random
from .base_augmentation import BaseAugmentation


class MixingAugmentation(BaseAugmentation):
    """Класс для смешения изображений"""

    def __init__(self):
        super().__init__()
        self.params = {
            'mix_type': 'random',
            'alpha': 0.5,
            'patch_size': 32,
            'chessboard': False
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Смешать два изображения"""
        # Второе изображение случайного шума
        h, w = image.shape[:2]
        second_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        mix_type = self.params['mix_type']

        if mix_type == 'random':
            return self._random_mixing(image, second_image)
        elif mix_type == 'chessboard':
            return self._chessboard_mixing(image, second_image)
        else:
            return image

    def _random_mixing(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Случайное смешение патчей"""
        h, w = img1.shape[:2]
        patch_size = self.params['patch_size']
        result = img1.copy()

        # Определить количество патчей
        num_patches = (h // patch_size) * (w // patch_size)
        patches_to_replace = random.sample(range(num_patches), num_patches // 2)

        for patch_idx in patches_to_replace:
            i = (patch_idx // (w // patch_size)) * patch_size
            j = (patch_idx % (w // patch_size)) * patch_size

            # Линейное смешивание на границах
            alpha = self.params['alpha']
            blend_width = min(patch_size // 4, 10)

            for x in range(patch_size):
                for y in range(patch_size):
                    if i + x < h and j + y < w:
                        if x < blend_width or y < blend_width or \
                                x >= patch_size - blend_width or y >= patch_size - blend_width:
                            # Граничная зона - смешивание
                            w1 = alpha
                            w2 = 1 - alpha
                            result[i + x, j + y] = w1 * img1[i + x, j + y] + w2 * img2[i + x, j + y]
                        else:
                            # Внутренняя часть - полная замена
                            result[i + x, j + y] = img2[i + x, j + y]

        return result.astype(np.uint8)

    def _chessboard_mixing(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Шахматное смешение"""
        h, w = img1.shape[:2]
        patch_size = self.params['patch_size']
        result = img1.copy()

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                # Шахматный порядок
                if ((i // patch_size) + (j // patch_size)) % 2 == 1:
                    # Заменить патч
                    patch_end_i = min(i + patch_size, h)
                    patch_end_j = min(j + patch_size, w)

                    # Смешивание на границах
                    alpha = self.params['alpha']
                    blend_width = min(patch_size // 4, 10)

                    for x in range(i, patch_end_i):
                        for y in range(j, patch_end_j):
                            rel_x = x - i
                            rel_y = y - j

                            if rel_x < blend_width or rel_y < blend_width or \
                                    rel_x >= patch_size - blend_width or rel_y >= patch_size - blend_width:
                                # Граничная зона
                                w1 = alpha
                                w2 = 1 - alpha
                                result[x, y] = w1 * img1[x, y] + w2 * img2[x, y]
                            else:
                                # Внутренняя часть
                                result[x, y] = img2[x, y]

        return result.astype(np.uint8)