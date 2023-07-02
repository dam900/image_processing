import numpy as np
import cv2 as cv


def vertical_edges(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    return cv.filter2D(src, cv.CV_8UC1, kernel)


def horizontal_edges(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    return cv.filter2D(src, cv.CV_8UC1, kernel)


def edge_tracking_hysteresis(src: np.ndarray, Tl: np.uint8 = 100, Th: np.uint8 = 200) -> np.ndarray:
    h, w = src.shape
    src = np.pad(src, 1, 'edge')
    dest = np.zeros((h, w))
    for x in range(h - 1):
        for y in range(w - 1):
            if (src[x, y] > Tl) and (src[x, y] < Th):
                if np.array(src[x:x + 3, y:y + 3] == 1).flatten().any(axis=0):
                    dest[x, y] = 255
                else:
                    dest[x, y] = 0
            else:
                dest[x, y] = src[x, y]
    return dest
