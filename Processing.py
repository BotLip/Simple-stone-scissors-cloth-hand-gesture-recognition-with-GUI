import cv2 as cv
import numpy as np
import random


def Blur(roi):
    roi_blured = cv.GaussianBlur(roi, (3, 3), 0)
    return roi_blured


def skinDetc(roi_blured):
    '''skin_low = np.array([0, 28, 50])
    skin_high = np.array([20, 255, 255])
    hsv = cv.cvtColor(roi_blured, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, skin_low, skin_high)
    roi_masked = cv.bitwise_and(roi_blured, roi_blured, mask=mask)

    k = np.ones((3, 3), np.uint8)
    roi_skined_erode = cv.erode(roi_masked, k)
    roi_skined = cv.dilate(roi_skined_erode, k)'''

    YCrCb = cv.cvtColor(roi_blured, cv.COLOR_BGR2YCR_CB)
    y, cr, cb = cv.split(YCrCb)
    _, skin = cv.threshold(cr, 0, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)
    roi_skined = cv.bitwise_and(roi_blured, roi_blured, mask=skin)

    '''
    YCrCb = cv.cvtColor(roi_blured, cv.COLOR_BGR2YCR_CB)
    y, Cr, Cb = cv.split(YCrCb)
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv.ellipse(skinCrCbHist, (113, 155), (23, 15), 43, 0, 360, (255, 255, 255), -1)
    mask = np.zeros(Cr.shape, dtype=np.uint8)
    for i in range(Cr.shape[0]):
        for j in range(Cr.shape[1]):
            cr, cb = YCrCb[i, j, 1], YCrCb[i, j, 2]
            if skinCrCbHist[cr, cb] > 0:
                mask[i, j] = 255
    roi_skined = cv.bitwise_and(roi_blured, roi_blured, mask=mask)       
    '''

    return roi_skined


def erode_dilate(roi_skined):
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    roi_skined = cv.erode(roi_skined, kernel)
    roi_skined = cv.dilate(roi_skined, kernel)
    return roi_skined


def drawContour(roi_skined):
    gray = cv.cvtColor(roi_skined, cv.COLOR_BGR2GRAY)
    hand_contour = cv.Canny(gray, 50, 200)
    return hand_contour


def rotate(hand_contour, scale=0.9):
    angle = random.randrange(-180, 180)
    width = hand_contour.shape[1]
    height = hand_contour.shape[0]
    center = (width/2, height/2)

    M = cv.getRotationMatrix2D(center, angle, scale)
    hand_contour_rotated = cv.warpAffine(hand_contour, M, (width, height))

    return hand_contour_rotated

