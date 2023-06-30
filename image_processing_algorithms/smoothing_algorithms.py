import numpy as np


def img_gaussian_filter(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w), dtype=float)
    src = np.pad(src, 3, 'edge')
    kernel = np.array([[2, 4,  5,   4, 2],
                       [4, 9,  12,  9, 4],
                       [5, 12, 15, 12, 5],
                       [4, 9,  12,  9, 4],
                       [2, 4,  5,   4, 2]]) / 159
    for x in range(0, h - 1):
        for y in range(0, w - 1):
            dest[x, y] = (src[x:x + 5, y:y + 5] * kernel).sum()
    return dest


def img_double_thresholding(src: np.ndarray, Tl: float = 0.3, Th: float = 0.5) -> np.ndarray:
    h, w = src.shape
    for x in range(h):
        for y in range(w):
            if src[x, y] < Tl:
                src[x, y] = 0
            elif src[x, y] > Th:
                src[x, y] = 1
    return src
