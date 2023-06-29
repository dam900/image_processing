import numpy as np
from image_processing_algorithms import grey_average, grey_weighted

from .processor_abstract import Processor


class VideoProcessor(Processor):

    def __init__(self, setting: str = 'n'):
        self.setting = setting

    def transform(self, src: np.ndarray) -> np.ndarray:
        if self.setting == 'g':
            return grey_average(src)
        if self.setting == 'w':
            return grey_weighted(src)
        return src

    def change_setting(self, new_setting: str) -> None:
        self.setting = new_setting
