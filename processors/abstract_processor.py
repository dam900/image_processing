from __future__ import annotations
from abc import ABC, abstractmethod
from .processor_settings import ProcessorSettings

import numpy as np


class Processor(ABC):
    """Abstract class for Image and Video processors"""

    @abstractmethod
    def transform(self, transformation: ProcessorSettings) -> Processor:
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Processor:
        raise NotImplementedError
