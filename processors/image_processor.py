import numpy as np
import image_processing_algorithms as ipa
from .abstract_processor import Processor
from .processor_settings import ProcessorSettings


class ImageProcessor(Processor):

    def __init__(self, img: np.ndarray):
        self.original = img
        self.transformed = img.copy()

    def transform(self, transformation: ProcessorSettings) -> Processor:
        if transformation == ProcessorSettings.VERTICAL_EDGES:
            self.transformed = ipa.img_vertical_edges(self.transformed)
        if transformation == ProcessorSettings.GAUSSIAN_FILTER:
            self.transformed = ipa.img_gaussian_filter(self.transformed)
        if transformation == ProcessorSettings.HORIZONTAL_EDGES:
            self.transformed = ipa.img_horizontal_edges(self.transformed)
        if transformation == ProcessorSettings.GREY_AVERAGE:
            self.transformed = ipa.img_grey_average(self.transformed)
        if transformation == ProcessorSettings.GREY_WEIGHTED:
            self.transformed = ipa.img_grey_weighted(self.transformed)
        if transformation == ProcessorSettings.NO_SETTING:
            self.transformed = self.transformed
        return self

    def get_result(self) -> np.ndarray:
        return self.transformed

    def reset(self) -> Processor:
        self.transformed = self.original.copy()
        return self
