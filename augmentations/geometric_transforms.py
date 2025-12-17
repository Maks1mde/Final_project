"""
Модуль геометрических преобразований
"""

import numpy as np
import cv2
from .base_augmentation import BaseAugmentation


class GeometricAugmentation(BaseAugmentation):
    """Класс для геометрических преобразований"""

    def __init__(self):
        super().__init__()
        self.params = {
            'transform_type': 'rotation',
            'angle': 15,
            'scale': 0.8,
            'tx': 10,
            'ty': 10,
            'flip_code': 0,
            'shear_x': 0.1,
            'shear_y': 0.1
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Применить геометрическое преобразование"""
        transform_type = self.params['transform_type']

        if transform_type == 'rotation':
            return self._rotate(image)
        elif transform_type == 'scaling':
            return self._scale(image)
        elif transform_type == 'translation':
            return self._translate(image)
        elif transform_type == 'shear':
            return self._shear(image)
        elif transform_type == 'reflection':
            return self._reflect(image)
        elif transform_type == 'affine':
            return self._affine_transform(image)
        elif transform_type == 'perspective':
            return self._perspective_transform(image)
        else:
            return image

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        """Поворот изображения с сохранением всех пикселей"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        angle = self.params['angle']

        # Получить матрицу поворота
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Рассчитать новые границы
        cos_angle = np.abs(matrix[0, 0])
        sin_angle = np.abs(matrix[0, 1])

        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))

        # Отрегулировать матрицу для нового центра
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Применить аффинное преобразование
        rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_REFLECT)

        # Масштабировать обратно к исходному размеру
        if rotated.shape[0] != h or rotated.shape[1] != w:
            rotated = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_LINEAR)

        return rotated

    def _scale(self, image: np.ndarray) -> np.ndarray:
        """Масштабирование изображения с сохранением размера"""
        scale = self.params['scale']

        if scale <= 0:
            return image

        h, w = image.shape[:2]

        # Рассчитать новый размер
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Масштабировать изображение
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Создать изображение исходного размера
        result = np.zeros_like(image)

        # Вычислить смещения для центрирования
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        # Убедиться, что смещения не отрицательные
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # Вычислить конечные координаты
        x_end = min(x_offset + new_w, w)
        y_end = min(y_offset + new_h, h)

        # Вычислить сколько пикселей можно скопировать
        copy_w = x_end - x_offset
        copy_h = y_end - y_offset

        if copy_w > 0 and copy_h > 0 and copy_w <= scaled.shape[1] and copy_h <= scaled.shape[0]:
            # Скопировать масштабированное изображение в центр
            result[y_offset:y_end, x_offset:x_end] = scaled[:copy_h, :copy_w]

        return result

    def _translate(self, image: np.ndarray) -> np.ndarray:
        """Сдвиг изображения с отражением границ"""
        h, w = image.shape[:2]
        tx = self.params['tx']
        ty = self.params['ty']

        # Создать матрицу трансформации
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # Применить аффинное преобразование с отражением границ
        translated = cv2.warpAffine(image, matrix, (w, h),
                                    borderMode=cv2.BORDER_REFLECT)

        return translated

    def _shear(self, image: np.ndarray) -> np.ndarray:
        """Сдвиг (shear) изображения"""
        h, w = image.shape[:2]
        shear_x = self.params.get('shear_x', 0.1)
        shear_y = self.params.get('shear_y', 0.1)

        # Точки до преобразования
        src_points = np.float32([[0, 0], [w, 0], [0, h]])

        # Точки после преобразования
        dst_points = np.float32([[0, 0],
                                 [w + shear_x * h, shear_y * w],
                                 [shear_x * h, h + shear_y * w]])

        # Получить матрицу аффинного преобразования
        matrix = cv2.getAffineTransform(src_points, dst_points)

        # Рассчитать новый размер для сохранения всего изображения
        new_w = int(w + abs(shear_x * h))
        new_h = int(h + abs(shear_y * w))

        # Применить преобразование
        sheared = cv2.warpAffine(image, matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_REFLECT)

        # Вернуть к исходному размеру
        if sheared.shape[0] != h or sheared.shape[1] != w:
            sheared = cv2.resize(sheared, (w, h), interpolation=cv2.INTER_LINEAR)

        return sheared

    def _reflect(self, image: np.ndarray) -> np.ndarray:
        """Отражение изображения"""
        flip_code = self.params['flip_code']

        # 0 - по вертикали, 1 - по горизонтали, -1 - по обоим осям
        if flip_code in [0, 1, -1]:
            return cv2.flip(image, flip_code)
        else:
            return image

    def _affine_transform(self, image: np.ndarray) -> np.ndarray:
        """Аффинное преобразование"""
        h, w = image.shape[:2]

        # Случайные точки для аффинного преобразования
        src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])

        # Случайное смещение точек (не более 30% от размера)
        max_offset = min(w, h) * 0.3
        dst_points = np.float32([
            [np.random.uniform(0, max_offset), np.random.uniform(0, max_offset)],
            [w - 1 - np.random.uniform(0, max_offset), np.random.uniform(0, max_offset)],
            [np.random.uniform(0, max_offset), h - 1 - np.random.uniform(0, max_offset)]
        ])

        matrix = cv2.getAffineTransform(src_points, dst_points)

        # Применить преобразование
        transformed = cv2.warpAffine(image, matrix, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)

        return transformed

    def _perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Перспективное преобразование"""
        h, w = image.shape[:2]

        # Исходные точки (углы изображения)
        src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])

        # Случайное смещение точек для перспективы (не более 20% от размера)
        max_offset = min(w, h) * 0.2
        dst_points = np.float32([
            [np.random.uniform(0, max_offset), np.random.uniform(0, max_offset)],
            [w - 1 - np.random.uniform(0, max_offset), np.random.uniform(0, max_offset)],
            [np.random.uniform(0, max_offset), h - 1 - np.random.uniform(0, max_offset)],
            [w - 1 - np.random.uniform(0, max_offset), h - 1 - np.random.uniform(0, max_offset)]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Применить перспективное преобразование
        perspective = cv2.warpPerspective(image, matrix, (w, h),
                                          borderMode=cv2.BORDER_REFLECT)


        return perspective
