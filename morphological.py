import numpy as np

import image_processing_algorithms as ipa


class Morphological:

    @classmethod
    def close(cls, src: np.ndarray, times: int, kernel):
        for x in range(times):
            src = ipa.dilatation(src, kernel)
        for x in range(times):
            src = ipa.erosion(src, kernel)
        return src

    @classmethod
    def open(cls, src: np.ndarray, times: int, kernel):
        for x in range(times):
            src = ipa.erosion(src, kernel)
        for x in range(times):
            src = ipa.dilatation(src, kernel)
        return src

    @classmethod
    def gradient(cls, src: np.ndarray, kernel):
        err = ipa.erosion(src, kernel)
        dil = ipa.dilatation(src, kernel)
        return dil - err

    @classmethod
    def top_hat(cls, src: np.ndarray, kernel):
        dest = src.copy()
        return dest - cls.open(src, 1, kernel)

    @classmethod
    def black_hat(cls, src: np.ndarray, kernel):
        dest = src.copy()
        return cls.close(src, 1, kernel) - dest
