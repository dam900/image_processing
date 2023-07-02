import numpy as np
import cv2 as cv


def vertical_edges(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    return cv.filter2D(src, cv.CV_8UC1, kernel)


def horizontal_edges(src: np.ndarray) -> np.ndarray:
    src = np.pad(src, 1, 'edge')
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    return cv.filter2D(src, cv.CV_8UC1, kernel)


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


def edge_tracking_hysteresis(src: np.ndarray, Tl: np.uint8 = 100, Th: np.uint8 = 200) -> np.ndarray:
    h, w = src.shape
    src = np.pad(src, 1, 'edge')
    dest = np.zeros((h, w))
    for x in range(h - 1):
        for y in range(w - 1):
            if (src[x, y] > Tl) and (src[x, y] < Th):
                if np.array(src[x:x + 3, y:y + 3] == 1).flatten().any(axis=0):
                    dest[x, y] = 255
                else:
                    dest[x, y] = 0
            else:
                dest[x, y] = src[x, y]
    return dest
