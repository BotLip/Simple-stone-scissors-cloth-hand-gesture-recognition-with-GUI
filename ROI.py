import cv2 as cv
from Processing import Blur, skinDetc, erode_dilate, drawContour, rotate
import numpy as np
record_times = 100
rock, paper, scissors = 1, 1, 1
rotate_times = 10


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1: y2, x1: x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    return dst


def process(roi):
    roi_blured = Blur(roi)
    roi_skined = skinDetc(roi_blured)
    roi_skined = erode_dilate(roi_skined)
    hand_contour = drawContour(roi_skined)
    # cv.imshow('gray', hand_contour)
    return hand_contour


def get_rotate(times):
    global rock, paper, scissors
    hand_list = ['R/r_', 'P/p_', 'S/s_']
    hands = [rock, paper, scissors]
    flag = 0
    for hand in hand_list:
        path = 'Hand/' + hand
        print('enhancing ' + hand)
        for count in range(1, record_times+1):
            dst = cv.imread(path+str(count)+'.jpg')
            for i in range(times):
                dst_rotated = rotate(dst)
                cv.imwrite(path+str(hands[flag])+'.jpg', dst_rotated)
                hands[flag] += 1
        print("enhance " + hand + " over")
        flag += 1


def cap_save_image():
    global rock, paper, scissors

    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        roi = get_roi(frame, 100, 350, 50, 300)     # 250x250
        cv.imshow('roi', roi)
        # cv.imshow('frame', frame)
        contour = process(roi)
        cv.imshow('contour', contour)

        key = cv.waitKey(50)
        if key == 27:   # Esc to quit
            break

        elif key == ord('r'):   # r to save rocks image
            if rock == record_times+1:
                print("you have saved ", record_times, " images of rock")
                continue

            path_rock = 'Hand/R/r_' + str(rock) + '.jpg'
            # cv.imwrite(path_rock, roi)
            rock_contour = process(roi)
            cv.imwrite(path_rock, rock_contour)

            rock += 1

        elif key == ord('p'):   # p to save paper image
            if paper == record_times+1:
                print("you have saved ", record_times, " images of paper")
                continue

            path_paper = 'Hand/P/p_' + str(paper) + '.jpg'
            # cv.imwrite(path_paper, roi)
            paper_contour = process(roi)
            cv.imwrite(path_paper, paper_contour)

            paper += 1

        elif key == ord('s'):   # s to save scissors image
            if scissors == record_times+1:
                print("you have saved ", record_times, " images of scissors")
                continue

            path_scissors = 'Hand/S/s_' + str(scissors) + '.jpg'
            # cv.imwrite(path_scissors, roi)
            scissors_contour = process(roi)
            cv.imwrite(path_scissors, scissors_contour)

            scissors += 1

    # rock, paper, scissors = record_times+1, record_times+1, record_times+1
    if (rock, paper, scissors) == (record_times+1, record_times+1, record_times+1):
        print('录取完成，开始训练')
    else:
        return 0
    cap.release()
    cv.destroyAllWindows()
    return 1

'''cap_save_image()
get_rotate(rotate_times)'''

