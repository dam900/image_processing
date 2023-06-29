import numpy as np


def grey_average(src: np.ndarray) -> np.ndarray:
    return src.sum(axis=2) / 3 / 255


def grey_weighted(src: np.ndarray) -> np.ndarray:
    return np.dot(src, [0.11, 0.59, 0.30]) / 255


def img_grey_average(src: np.ndarray) -> np.ndarray:
    h, w, c = src.shape
    dest = np.zeros((h, w, 1))
    for x in range(h):
        for y in range(w):
            dest[x][y] = src[x, y, :].sum() / 3 / 255
    return dest


def img_grey_weighted(src: np.ndarray) -> np.ndarray:
    h, w, c = src.shape
    weights = np.array([0.11, 0.59, 0.30])
    dest = np.zeros((h, w, 1), dtype=float)
    for x in range(h):
        for y in range(w):
            dest[x][y] = (src[x, y, :] * weights).sum() / 255
    return dest

