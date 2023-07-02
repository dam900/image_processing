import numpy as np


def double_edge_thresholding(src: np.ndarray, Tl: np.uint8 = 100, Th: np.uint8 = 200) -> np.ndarray:
    src[src <= Tl] = 0
    src[src >= Th] = 255
    return src
