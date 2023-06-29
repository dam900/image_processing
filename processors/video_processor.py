import numpy as np
from image_processing_algorithms import vid_grey_average, vid_grey_weighted
from .abstract_processor import Processor
from .processor_settings import ProcessorSettings


class VideoProcessor(Processor):

    def __init__(self, setting: ProcessorSettings = ProcessorSettings.NO_SETTING):
        self.setting = setting

    def transform(self, src: np.ndarray) -> np.ndarray:
        if self.setting == ProcessorSettings.GREY_AVERAGE:
            return vid_grey_average(src)
        if self.setting == ProcessorSettings.GREY_WEIGHTED:
            return vid_grey_weighted(src)
        if self.setting == ProcessorSettings.NO_SETTING:
            return src

    def change_setting(self, new_setting: ProcessorSettings) -> None:
        self.setting = new_setting
