from abc import ABC, abstractmethod

import numpy as np


class Processor(ABC):
    """Abstract class for Image and Video processors"""

    @abstractmethod
    def transform(self, src: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def change_setting(self, new_setting: str) -> None:
        raise NotImplementedError


