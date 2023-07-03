import numpy as np


def double_edge_thresholding(src: np.ndarray, Tl: int = 100, Th: int = 200) -> np.ndarray:
    src[src <= Tl] = 0
    src[src >= Th] = 255
    return src


def non_max_suppression(src: np.ndarray, angle_space: np.ndarray) -> np.ndarray:
    h, w = src.shape
    src = np.pad(src, 1, 'edge')
    dest = np.zeros((h, w), dtype=np.uint8)
    angle_space = angle_space * 180. / np.pi
    angle_space[angle_space < 0] += 180

    with np.nditer(angle_space, flags=['multi_index'], op_flags=['readonly']) as it:
        for k in it:
            x, y = it.multi_index
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
    return dest


def binarize(src: np.ndarray) -> np.ndarray:
    src[src < np.floor(255 / 2)] = 0
    src[src >= np.floor(255 / 2)] = 255
    return src
