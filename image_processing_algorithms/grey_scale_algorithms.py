import numpy as np


def grey_average(src: np.ndarray) -> np.ndarray:
    return src.sum(axis=2) / 3 / 255


