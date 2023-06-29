import numpy as np


def vid_vertical_edges(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros(src.shape)
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    indexes = np.array([0, 1, 2])
    for x in range(0, h-2):
        dest[indexes, x] = (src[indexes, 0+x:3+x] * kernel).sum()
    return dest


def vid_horizontal_edges(src: np.ndarray) -> np.ndarray:
    pass


def img_vertical_edges(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w, 1), dtype=float)
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    for x in range(0, h - 1):
        for y in range(0, w - 1):
            dest[x, y] = (src[x:x + 3, y:y + 3] * kernel).sum()
    return dest


def img_horizontal_edges(src: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w, 1), dtype=float)
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    for x in range(0, h - 1):
        for y in range(0, w - 1):
            dest[x, y] = (src[x:x + 3, y:y + 3] * kernel).sum()
    return dest
