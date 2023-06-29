import numpy as np
import image_processing_algorithms as ipa
from .abstract_processor import Processor
from .processor_settings import ProcessorSettings


class VideoProcessor(Processor):

    def __init__(self, setting: ProcessorSettings = ProcessorSettings.NO_SETTING):
        self.setting = setting

    def transform(self, src: np.ndarray) -> np.ndarray:
        if self.setting == ProcessorSettings.VERTICAL_EDGES:
            src = ipa.vid_grey_weighted(src)
            return ipa.vid_vertical_edges(src)
        if self.setting == ProcessorSettings.HORIZONTAL_EDGES:
            src = ipa.vid_grey_weighted(src)
            return ipa.vid_horizontal_edges(src)
        if self.setting == ProcessorSettings.GREY_AVERAGE:
            return ipa.vid_grey_average(src)
        if self.setting == ProcessorSettings.GREY_WEIGHTED:
            return ipa.vid_grey_weighted(src)
        if self.setting == ProcessorSettings.NO_SETTING:
            return src

    def change_setting(self, new_setting: ProcessorSettings) -> None:
        self.setting = new_setting
