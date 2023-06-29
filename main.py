import cv2 as cv
from image_processing_algorithms import vid_grey_average, vid_grey_weighted
from processors import VideoProcessor, ImageProcessor, Processor
from processors import ProcessorSettings as Ps

v = VideoProcessor()
i = ImageProcessor()


def menu(setting: str, p: Processor):
    match setting:
        # q character
        case 113:
            exit()
        # g character
        case 103:
            p.change_setting(Ps.GREY_AVERAGE)
        # w character
        case 119:
            p.change_setting(Ps.GREY_WEIGHTED)
        case 110:
            p.change_setting(Ps.NO_SETTING)


def main(proc: Processor):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = proc.transform(frame)
        cv.imshow('frame', frame)
        setting = cv.waitKey(1)

        menu(setting, proc)


if __name__ == '__main__':
    main(v)
