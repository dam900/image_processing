import cv2 as cv
import sys

import numpy as np

from canny_edge_detector import CannyEdgeDetector
from processors import ImageProcessor
from processors import ProcessorSettings as Ps


def main():
    img = cv.imread(r'blok.jpg')
    i = ImageProcessor(img)
    if img is None:
        sys.exit("Could not read the image.")

    img = CannyEdgeDetector.canny_edge_detector(img)
    cv.imshow("Display window", img)
    key = cv.waitKey(0)
    if key == ord("q"):
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
