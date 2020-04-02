import cv2 as cv
import numpy as np
import math
import os


def videoCapture():
    video = cv.VideoCapture(os.getcwd() + '/background-subtraction/vids/003_Resize.mp4')

    while video.isOpened():
        ret, frame = video.read()
        houghP = detectLinesHoughP(frame)
        cv.imshow('Frame', houghP)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()


"""
    Background subtraction doesn't really work I find, causes too much noise. I used
    two methods: 
        method 1 = absolute subtraction by taking a still frame
                    then subtracting from that
        method 2 = using the built in function provided by opencv which is a little
                    better but causes too much noise even after reducing through median gradient
"""
def backgroundSubtraction():
    video = cv.VideoCapture(os.getcwd() + '/background-subtraction/vids/003_Resize.mp4')

    _, first_frame = video.read()
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    subtractor = cv.createBackgroundSubtractorMOG2(_, varThreshold=35)
    subtractor2 = cv.createBackgroundSubtractorKNN(_, dist2Threshold=100)
    subtractor3 = cv.createBackgroundSubtractorMOG2(_, varThreshold=35)

    while video.isOpened():
        _, frame = video.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        difference = cv.absdiff(first_gray, gray_frame)
        _, difference = cv.threshold(difference, 50, 255, cv.THRESH_BINARY)
        _, difference = cv.threshold(difference, 50, 255, cv.THRESH_BINARY)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        average = np.mean(v).astype(int)
        _, vThresh = cv.threshold(v, average, 255, cv.THRESH_TRUNC)
        thresh = cv.merge([h, s, vThresh])
        thresh = cv.cvtColor(thresh, cv.COLOR_HSV2BGR)

        mask = subtractor.apply(frame, 0.1)
        mask2 = subtractor2.apply(frame, 0.1)
        maskThresh = subtractor.apply(thresh)
        mask = cv.medianBlur(mask, 5)

        cv.imshow('Frame', frame)
        cv.imshow('Mask', mask)
        # cv.imshow('Mask 2', mask2)
        cv.imshow('Thresh', maskThresh)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()


# Probablistic Hough Detection which is much better
def detectLinesHoughP(frame):
    frameGray = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)

    dst = cv.Canny(frameGray, 100, 140, None, 3)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 5, np.pi / 180, 100, None, 80, 20)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
    cdstP = cv.medianBlur(cdstP, 5)
    return cdstP


# Normal Hough Detection
def detectLinesHough(frame):
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    dst = cv.Canny(frameGray, 180, 200, None, 3)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
    return cdst


# videoCapture()
backgroundSubtraction()
