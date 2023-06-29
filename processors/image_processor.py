import numpy as np
from image_processing_algorithms import img_grey_average, img_grey_weighted
from processors import Processor


class ImageProcessor(Processor):

    def __init__(self, setting: str = 'n'):
        self.setting = setting

    def transform(self, src: np.ndarray) -> np.ndarray:
        if self.setting == 'g':
            return img_grey_average(src)
        if self.setting == 'w':
            return img_grey_weighted(src)
        return src

    def change_setting(self, new_setting: str) -> None:
        self.setting = new_setting
