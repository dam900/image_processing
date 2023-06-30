import numpy as np

import image_processing_algorithms as ipa


class CannyEdgeDetector:

    def __init__(self):
        pass

    @classmethod
    def canny_edge_detector(cls, src: np.ndarray) -> np.ndarray:
        grey = ipa.img_grey_weighted(src)
        smooth = ipa.img_gaussian_filter(grey)
        hor = ipa.img_horizontal_edges(smooth)
        ver = ipa.img_vertical_edges(smooth)
        dest = (hor + ver)
        angle_space = np.arctan2(ver, hor)
        dest = ipa.img_non_max_suppression(dest, angle_space)
        dest = ipa.img_double_thresholding(dest, Tl=0.5, Th=0.7)
        dest = ipa.img_edge_tracking_hysteresis(dest, Tl=0.5, Th=0.7)
        return dest
