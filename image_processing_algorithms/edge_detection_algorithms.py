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
    h, w, = src.shape
    dest = np.zeros((h, w,), dtype=float)
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
    dest = np.zeros((h, w), dtype=float)
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    for x in range(0, h - 1):
        for y in range(0, w - 1):
            dest[x, y] = (src[x:x + 3, y:y + 3] * kernel).sum()
    return dest


def img_non_max_suppression(src: np.ndarray, angle_space: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w), dtype=float)
    src = np.pad(src, 1, 'edge')

    angle_space = angle_space * 180. / np.pi
    angle_space[angle_space < 0] += 180

    for x in range(0, h - 1):
        for y in range(0, w - 1):
            q = 1
            r = 1
            if (0 <= angle_space[x, y] < 22.5) or (157.5 <= angle_space[x, y] <= 180):
                q = src[x, y + 1]
                r = src[x, y - 1]
            elif 22.5 <= angle_space[x, y] < 67.5:
                q = src[x + 1, y - 1]
                r = src[x - 1, y + 1]
            elif 67.5 <= angle_space[x, y] < 112.5:
                q = src[x + 1, y]
                r = src[x - 1, y]
            elif 112.5 <= angle_space[x, y] < 157.5:
                q = src[x - 1, y - 1]
                r = src[x + 1, y + 1]
            if (src[x, y] >= q) and (src[x, y] >= r):
                dest[x, y] = src[x, y]
            else:
                dest[x, y] = 0
    return dest


def img_edge_tracking_hysteresis(src: np.ndarray, Tl: float = 0.3, Th: float = 0.5) -> np.ndarray:
    h, w = src.shape
    src = np.pad(src, 2, 'edge')
    dest = np.zeros((h, w), dtype=float)
    for x in range(h-1):
        for y in range(w-1):
            if (src[x, y] > Tl) and (src[x, y] < Th):
                if np.array(src[x:x+5, y:y+5] == 1).flatten().any(axis=0):
                    dest[x, y] = 1
                else:
                    dest[x, y] = 0
    return dest

