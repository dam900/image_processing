import cv2 as cv
from image_processing_algorithms import grey_average, grey_weighted
from processors import VideoProcessor

v = VideoProcessor('n')


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = v.transform(frame)
        cv.imshow('frame', frame)
        setting = cv.waitKey(1)

        match setting:
            # q character
            case 113:
                break
            # g character
            case 103:
                v.change_setting('g')
            case 119:
                v.change_setting('w')


if __name__ == '__main__':
    main()
