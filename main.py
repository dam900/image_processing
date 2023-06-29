import cv2 as cv
import sys

import numpy as np

from processors import ImageProcessor
from processors import ProcessorSettings as Ps


def main():
    img = np.asarray(cv.imread(r'blok.jpg'))
    i = ImageProcessor(img)
    if img is None:
        sys.exit("Could not read the image.")

    img = i.transform(Ps.GREY_AVERAGE).transform(Ps.VERTICAL_EDGES).get_result()
    img2 = i.reset().transform(Ps.GREY_AVERAGE).transform(Ps.HORIZONTAL_EDGES).get_result()
    combined = (img2 + img) / 2

    cv.imshow("Display window", combined)
    key = cv.waitKey(0)
    if key == ord("q"):
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
