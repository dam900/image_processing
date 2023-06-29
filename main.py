import cv2 as cv
from image_processing_algorithms import grey_average, grey_weighted
from processors import VideoProcessor, ImageProcessor, Processor

v = VideoProcessor()
i = ImageProcessor()


def menu(setting: str, p: Processor):
    match setting:
        # q character
        case 113:
            exit()
        # g character
        case 103:
            p.change_setting('g')
        case 119:
            p.change_setting('w')


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
        frame = i.transform(frame)
        cv.imshow('frame', frame)
        setting = cv.waitKey(1)

        menu(setting, i)


if __name__ == '__main__':
    main()
