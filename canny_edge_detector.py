import numpy as np
import cv2 as cv

import image_processing_algorithms as ipa


class CannyEdgeDetector:

    def __init__(self):
        pass

    @classmethod
    def canny_edge_detector(cls, src: np.ndarray, Tl: float = 0.3, Th: float = 0.6) -> np.ndarray:
        grey = ipa.img_grey_weighted(src)
        smooth = ipa.img_gaussian_filter(grey)
        hor = ipa.img_horizontal_edges(smooth)
        ver = ipa.img_vertical_edges(smooth)
        dest = (hor + ver)
        angle_space = np.arctan2(hor, ver)
        dest = ipa.img_non_max_suppression(dest, angle_space)
        dest = ipa.img_double_thresholding(dest, Tl=Tl, Th=Th)
        dest = ipa.img_edge_tracking_hysteresis(dest, Tl=Tl, Th=Th)
        return dest

    @classmethod
    def canny_edge_detector_with_cv(cls, src: np.ndarray, Tl: float = 0.3, Th: float = 0.5) -> np.ndarray:
        grey = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(grey, (5, 5), 1.4)
        dest, angle_space = cls.__sobel_filter(blur)
        dest = ipa.img_non_max_suppression(dest, angle_space)
        dest = cls.__double_edge_thresholding(dest, Th=Th, Tl=Tl)
        dest = ipa.img_edge_tracking_hysteresis(dest)
        return dest

    @classmethod
    def __sobel_filter(cls, src: np.ndarray) -> (np.ndarray, np.ndarray):
        sobel_ver = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
        sobel_hor = np.array([[-1, -2, -1],
                              [0,  0,  0],
                              [1,  2,  1]])
        sobel_ver = cv.filter2D(src, cv.CV_8UC1, sobel_ver)
        sobel_hor = cv.filter2D(src, cv.CV_8UC1, sobel_hor)
        angle_space = np.arctan2(sobel_hor, sobel_ver)
        return sobel_ver + sobel_hor, angle_space

    @classmethod
    def __double_edge_thresholding(cls, src, Tl: float = 0.3, Th: float = 0.6) -> np.ndarray:
        src[src <= int(Tl*255)] = 0
        src[src >= int(Th*255)] = 255
        return src
