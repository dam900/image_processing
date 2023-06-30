import cv2 as cv
import sys

from canny_edge_detector import CannyEdgeDetector


def main():
    img = cv.imread(r'Large_Scaled_Forest_Lizard.jpg')
    if img is None:
        sys.exit("Could not read the image.")

    img = CannyEdgeDetector.canny_edge_detector(img)
    cv.imshow("Display window", img)
    key = cv.waitKey(0)
    if key == ord("q"):
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
