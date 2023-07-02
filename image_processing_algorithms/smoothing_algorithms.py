import cv2 as cv
import numpy as np


def gaussian_filter(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, 3, 'edge')
    kernel = np.array([[2, 4,  5,   4, 2],
                       [4, 9,  12,  9, 4],
                       [5, 12, 15, 12, 5],
                       [4, 9,  12,  9, 4],
                       [2, 4,  5,   4, 2]]) / 159
    return cv.filter2D(src, cv.CV_8UC1, kernel)

