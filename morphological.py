import numpy as np

import image_processing_algorithms as ipa


class Morphological:

    @classmethod
    def open(cls, src: np.ndarray, times: int):
        for x in range(times):
            src = ipa.dilatation(src)
        for x in range(times):
            src = ipa.erosion(src)
        return src

    @classmethod
    def close(cls, src: np.ndarray, times: int):
        for x in range(times):
            src = ipa.erosion(src)
        for x in range(times):
            src = ipa.dilatation(src)
        return src

