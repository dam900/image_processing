import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def erosion(src: np.ndarray, kernel) -> np.ndarray:
    h, w = src.shape
    kh, kw = kernel.shape
    src = np.pad(src, max(kh, kw)//2, 'constant')
    dest = np.zeros((h, w), dtype=np.uint8)
    v = sliding_window_view(src, (kh, kw))
    for y, view in enumerate(v):
        for x, window in enumerate(view):
            try:
                dest[y, x] = np.min(window)
            except:
                pass
    return dest


def dilatation(src: np.ndarray, kernel) -> np.ndarray:
    h, w = src.shape
    kh, kw = kernel.shape
    dest = np.zeros((h, w), dtype=np.uint8)
    src = np.pad(src, max(kh, kw) // 2, 'constant')
    v = sliding_window_view(src, (kh, kw))
    for y, view in enumerate(v):
        for x, window in enumerate(view):
            try:
                dest[y, x] = np.max(window)
            except:
                pass
    return dest


def hit_miss(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w), dtype=np.uint8)
    src = np.pad(src, 1, 'constant')
    kernel = np.array([[-1, -1, 0],
                       [-1,  1, 0],
                       [-1,  -1,  0]])
    v = sliding_window_view(src, (3, 3))
    for y, view in enumerate(v):
        for x, window in enumerate(view):
            if check_pattern(window, kernel):
                dest[y, x] = 255
    return dest


def check_pattern(view: np.ndarray, pattern: np.ndarray):
    v = view.flatten()
    p = pattern.flatten()
    for i, j in zip(v, p):
        if j == 0:
            continue
        elif j == -1:
            if i != 0:
                return False
        elif j == 1:
            if i != 255:
                return False
    return True




