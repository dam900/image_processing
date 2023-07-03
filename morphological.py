import numpy as np

import image_processing_algorithms as ipa


class Morphological:

    @classmethod
    def close(cls, src: np.ndarray, times: int):
        for x in range(times):
            src = ipa.dilatation(src)
        for x in range(times):
            src = ipa.erosion(src)
        return src

    @classmethod
    def open(cls, src: np.ndarray, times: int):
        for x in range(times):
            src = ipa.erosion(src)
        for x in range(times):
            src = ipa.dilatation(src)
        return src

    @classmethod
    def gradient(cls, src: np.ndarray):
        err = ipa.erosion(src)
        dil = ipa.dilatation(src)
        return dil - err

    @classmethod
    def top_hat(cls, src: np.ndarray):
        pass

    @classmethod
    def black_hat(cls, src: np.ndarray):
        pass