import cv2 as cv
from image_processing_algorithms import grey_average


def main():
    grey_img = False
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if grey_img:
            frame = grey_average(frame)
        cv.imshow('frame', frame)
        setting = cv.waitKey(1)
        match setting:
            case 113:
                break
            case 103:
                grey_img = not grey_img


if __name__ == '__main__':
    main()
