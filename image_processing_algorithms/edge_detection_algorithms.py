import numpy as np
import cv2 as cv


def vertical_edges(src: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    return cv.filter2D(src, cv.CV_8UC1, kernel)


def horizontal_edges(src: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    return cv.filter2D(src, cv.CV_8UC1, kernel)


def edge_tracking_hysteresis(src: np.ndarray, Tl: int = 100, Th: int = 200) -> np.ndarray:
    h, w = src.shape
    src = np.pad(src, 1, 'edge')
    dest = np.zeros((h, w), dtype=np.uint8)
    for x in range(h - 1):
        for y in range(w - 1):
            if (src[x, y] > Tl) and (src[x, y] < Th):
                if np.array(src[x:x + 3, y:y + 3] == 255).flatten().any(axis=0):
                    dest[x, y] = 255
                else:
                    dest[x, y] = 0
            else:
                dest[x, y] = src[x, y]
    return dest
