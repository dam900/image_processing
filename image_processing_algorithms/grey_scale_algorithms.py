import numpy as np


def grey_average(src: np.ndarray) -> np.ndarray:
    return src.sum(axis=2) / 3 / 255


def grey_weighted(src: np.ndarray) -> np.ndarray:
    return np.dot(src, [0.11, 0.59, 0.30]) / 255
