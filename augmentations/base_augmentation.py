# Базовый класс аугментации

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAugmentation(ABC):
    """Базовый класс для всех аугментаций"""

    def __init__(self):
        self.params = {}

    def set_params(self, params: Dict[str, Any]):
        """Установить параметры аугментации"""
        self.params.update(params)

    @abstractmethod
    def apply(self, image):
        """Применить аугментацию к изображению"""
        pass

    def get_name(self) -> str:
        """Получить имя класса аугментации"""
        return self.__class__.__name__