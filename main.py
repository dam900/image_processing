import cv2 as cv
import sys

from canny_edge_detector import CannyEdgeDetector


def main():
    img = cv.imread(r'blok.jpg')
    if img is None:
        sys.exit("Could not read the image.")
    # img = CannyEdgeDetector.canny_edge_detector(img, 0.3, 0.6)
    # img = CannyEdgeDetector.canny_edge_detector_with_cv(img, Tl=0.16, Th=0.92)
    img = cv.Canny(img, threshold1=150, threshold2=200)
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
    #     frame = cv.Canny(frame, threshold1=160, threshold2=150)
    #     # frame = CannyEdgeDetector.canny_edge_detector_with_cv(frame, Tl=0.16, Th=0.92)
    #     # Display the resulting frame
    #     cv.imshow('frame', frame)
    #     if cv.waitKey(1) == ord('q'):
    #         break
    #

if __name__ == '__main__':
    main()
