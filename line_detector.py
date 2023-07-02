import numpy as np


class LineDetector:

    @classmethod
    def hough_line(cls, src: np.ndarray):
        y, x = src.shape
        # max dist from origin in polar coordinates
        max_r = int(np.round(np.sqrt(y ** 2) + np.sqrt(x ** 2)))
        # angles in polar coordinates
        thetas = np.deg2rad(np.arange(-90, 90))
        # radius range
        rs = np.linspace(-max_r, max_r, 2 * max_r)
        # array for the accumulation of crossings in polar coordinates
        acc = np.zeros((2 * max_r, len(rs)))

        with np.nditer(src, flags=['multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                if x > 0:
                    with np.nditer(thetas, flags=['f_index']) as thetas_it:
                        for k in thetas_it:
                            i, j = it.multi_index
                            r = i*np.cos(k) + j*np.sin(k)
                            acc[int(r) + max_r, thetas_it.index] += 1
        return acc, thetas, rs


