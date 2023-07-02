import cv2 as cv
import numpy as np
import sys

from canny_edge_detector import CannyEdgeDetector
from line_detector import LineDetector
from morphological import Morphological
import image_processing_algorithms as ipa


def main():
    img = cv.imread(r'samples/Finger_print.png')
    if img is None:
        sys.exit("Could not read the image.")

    img = ipa.grey_weighted(img)
    img = ipa.binarize(img)
    img = Morphological.close(img, 1)

    cv.imshow("Display window", img)
    key = cv.waitKey(0)
    if key == ord("q"):
        cv.destroyAllWindows()

    # cap = cv.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     # Our operations on the frame come here
    #     # frame = cv.Canny(frame, threshold1=160, threshold2=150)
    #     # canny = CannyEdgeDetector.canny_edge_detector_with_cv(frame, Tl=0.16, Th=0.92)
    #     # canny = cv.Canny(frame, 50, 150)
    #     # lines = cv.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)
    #
    #     # Display the resulting frame
    #     cv.imshow('frame', frame)
    #     if cv.waitKey(1) == ord('q'):
    #         break
    #


if __name__ == '__main__':
    main()
