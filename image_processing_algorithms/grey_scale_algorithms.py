import numpy as np


def img_grey_average(src: np.ndarray) -> np.ndarray:
    h, w, c = src.shape
    dest = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            dest[x][y] = src[x, y, :].sum() / 3 / 255
    return dest


def img_grey_weighted(src: np.ndarray) -> np.ndarray:
    h, w, c = src.shape
    weights = np.array([0.11, 0.59, 0.30])
    dest = np.zeros((h, w), dtype=float)
    for x in range(h):
        for y in range(w):
            dest[x][y] = (src[x, y, :] * weights).sum() / 255
    return dest

