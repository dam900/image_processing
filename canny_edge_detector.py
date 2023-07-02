import numpy as np
import image_processing_algorithms as ipa


class CannyEdgeDetector:

    def __init__(self):
        pass

    @classmethod
    def Canny(cls, src: np.ndarray, Tl: int = 100, Th: int = 150) -> np.ndarray:
        grey = ipa.grey_weighted(src)
        blur = ipa.gaussian_filter(grey)
        hor = ipa.horizontal_edges(blur)
        ver = ipa.vertical_edges(blur)
        dest = (hor + ver)
        angle_space = np.arctan2(hor, ver)
        dest = ipa.non_max_suppression(dest, angle_space)
        dest = ipa.double_edge_thresholding(dest, Tl=Tl, Th=Th)
        dest = ipa.edge_tracking_hysteresis(dest, Tl=Tl, Th=Th)
        return dest

