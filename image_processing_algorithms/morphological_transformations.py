import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def erosion(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w),  dtype=np.uint8)
    src = np.pad(src, 1, 'constant')
    v = sliding_window_view(src, (3, 3))
    for y, view in enumerate(v):
        for x, window in enumerate(view):
            dest[y, x] = np.min(window)
    return dest


def dilatation(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w), dtype=np.uint8)
    src = np.pad(src, 1, 'constant')
    v = sliding_window_view(src, (3, 3))
    for y, view in enumerate(v):
        for x, window in enumerate(view):
            dest[y, x] = np.max(window)
    return dest
