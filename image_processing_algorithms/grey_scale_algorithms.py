import numpy as np


def grey_average(src: np.ndarray) -> np.ndarray:
    return np.array(src.sum(axis=2) / 3, dtype=np.uint8)


def grey_weighted(src: np.ndarray) -> np.ndarray:
    weights = np.array([0.11, 0.59, 0.30])
    return np.array(np.dot(src, weights), dtype=np.uint8)

