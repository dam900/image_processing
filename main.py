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
    # img = cv.imread(r'samples/Finger_print.png')
    # if img is None:
    #     sys.exit("Could not read the image.")

    Notes_original = plt.imread('samples/notes.png')
    Notes_horizontal = plt.imread('samples/notes_horizontal.png')
    Notes_vertical = plt.imread('samples/notes_vertical.png')
    fig, ax = plt.subplots(3, 1, figsize=(6, 6), dpi=200)
    ax[0].imshow(Notes_original)
    ax[0].set(title='Original', xticks = [], yticks = [])

    ax[1].imshow(Notes_horizontal, cmap='gray')
    ax[1].set(title='Horizontal', xticks=[], yticks=[])

    ax[2].imshow(Notes_vertical, cmap='gray')
    ax[2].set(title='Vertical', xticks=[], yticks=[])

    plt.savefig('readme_samples/NotesFig.png')
    plt.show()

    # img = ipa.grey_weighted(img)
    # img = ipa.binarize(img)
    # kernel = np.ones((3, 3))
    # img = Morphological.open(img, 1, kernel)

    # cv.imshow("Display window", img)
    key = cv.waitKey(0)
    if key == ord("q"):
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
