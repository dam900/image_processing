import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return (f"Point({self.x} ,{self.y})")


class Path:
    def __init__(self, start: Point, stop: Point):
        self.start = start
        self.stop = stop

    def __repr__(self):
        return f"start {self.start} ,stop {self.stop}"


class LineDetector:

    @classmethod
    def hough_line(cls, src: np.ndarray, threshold: int = 100):
        Ny, Nx = src.shape
        # max dist from origin in polar coordinates
        max_r = int(np.round(np.sqrt(Ny ** 2) + np.sqrt(Nx ** 2)))
        # angles in polar coordinates
        thetas = np.deg2rad(np.arange(-90, 90))
        # radius range
        rs = np.linspace(-max_r, max_r, 2 * max_r)
        # array for the accumulation of crossings in polar coordinates
        acc = np.zeros((2 * max_r, len(thetas)))

        with np.nditer(src, flags=['multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                if x > 0:
                    with np.nditer(thetas, flags=['f_index']) as thetas_it:
                        for k in thetas_it:
                            i, j = it.multi_index
                            r = j * np.cos(k) + i * np.sin(k)
                            acc[int(r) + max_r, thetas_it.index] += 1
        lines = np.argwhere(acc > threshold)
        dest_theta, dest_rs = [], []
        for y, x in lines:
            dest_theta.append(np.rad2deg(thetas[x]))
            dest_rs.append(int(rs[y]))
        return acc, dest_theta, dest_rs

    @classmethod
    def get_points(cls, thetas: list, rs: list):
        paths: [Path] = []
        for theta, rho in zip(thetas, rs):
            sin, cos = np.sin(theta), np.cos(theta)
            x0, y0 = cos * rho, sin * rho
            pt1 = Point(int(x0 + 1000 * (-sin)), int(y0 + 1000 * cos))
            pt2 = Point(int(x0 - 1000 * (-sin)), int(y0 - 1000 * cos))
            paths.append(Path(pt1, pt2))
        return paths
