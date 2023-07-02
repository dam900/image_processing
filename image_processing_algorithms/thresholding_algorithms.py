import numpy as np


def double_edge_thresholding(src: np.ndarray, Tl: np.uint8 = 100, Th: np.uint8 = 200) -> np.ndarray:
    src[src <= Tl] = 0
    src[src >= Th] = 255
    return src


def non_max_suppression(src: np.ndarray, angle_space: np.ndarray) -> np.ndarray:
    h, w = src.shape
    dest = np.zeros((h, w))
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
