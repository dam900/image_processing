import cv2 as cv
import numpy as np
import sys

from canny_edge_detector import CannyEdgeDetector


def main():
    img = cv.imread(r'blok.jpg')
    if img is None:
        sys.exit("Could not read the image.")

    # canny = CannyEdgeDetector.Canny(img)


    canny = cv.Canny(img, threshold1=150, threshold2=200)
    lines = cv.HoughLines(canny, 1, np.pi / 180, 400,  None, 0, 0)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    #         cv.line(img, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)

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
