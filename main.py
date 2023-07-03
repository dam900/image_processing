import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
from morphological import Morphological
import image_processing_algorithms as ipa

from canny_edge_detector import CannyEdgeDetector
from line_detector import LineDetector

matplotlib.use('TkAgg')


def main():
    img = cv.imread(r'samples/white-horse-running.jpg')
    if img is None:
        sys.exit("Could not read the image.")

    # Horse_original = plt.imread('samples/white-horse-running.jpg')
    # Horse_gradient = plt.imread('samples/white-horse-running-gradient.jpg')
    # fig, ax = plt.subplots(1, 2, figsize=(4, 1.5), dpi=400)
    # ax[0].imshow(Horse_original)
    # ax[0].set(title='Original', xticks = [], yticks = [])
    #
    # ax[1].imshow(Horse_gradient, cmap='gray')
    # ax[1].set(title='After gradient', xticks=[], yticks=[])
    # plt.savefig('readme_samples/HorseFig.png')
    # plt.show()

    img = ipa.grey_weighted(img)
    img = ipa.binarize(img)
    img = ipa.hit_miss(img)

    cv.imshow("Display window", img)
    key = cv.waitKey(0)
    if key == ord("q"):
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
