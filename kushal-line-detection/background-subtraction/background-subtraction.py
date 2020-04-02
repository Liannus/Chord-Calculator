import cv2 as cv
import numpy as np
import math
import os

class Point:
    def __init__(self, pointTuple):
        self.x = pointTuple[0]
        self.y = pointTuple[1]

class Line:
    def __init__(self, lineVector):
        self.p1 = Point((lineVector[0], lineVector[1]))
        self.p2 = Point((lineVector[2], lineVector[3]))

        if (self.p2.x - self.p1.x == 0):
            self.m = float(0)
        else:
            self.m = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
        self.b = self.p1.y - self.m * self.p1.x


def videoCapture():
    video = cv.VideoCapture(os.getcwd() + '/vids/003_Resize.mp4')

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
    video = cv.VideoCapture(os.getcwd() + '/vids/003_Resize.mp4')

    _, first_frame = video.read()
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    subtractor = cv.createBackgroundSubtractorMOG2(_, varThreshold=35)

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

        mask = subtractor.apply(thresh, 0.1)
        # mask = subtractor.apply(frame, 0.1)
        houghP = detectLinesHoughP(thresh)

        cv.imshow('Frame', frame)
        cv.imshow('Thresh', thresh)
        cv.imshow('Mask', mask)
        cv.imshow('hough', houghP)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()


# Probablistic Hough Detection which is much better
def detectLinesHoughP(frame):
    frameGray = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)

    dst = cv.Canny(frameGray, 40, 160, None, 3)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # cdstP = cv.medianBlur(cdstP, 5)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 120, None, 30, 10)
    cluster = filterLinesBySlope(linesP)
    yint = filterLinesByYInt(cluster)
    hough = colorLines(yint, cdstP, (0, 0, 255))
    return hough


def getAllLines(lines):
    allLines = []
    if lines is not None:
        for i in range(len(lines)):
            allLines.append(Line(lines[i][0]))
    return allLines


def filterLinesBySlope(lines):
    allLines = getAllLines(lines)
    allLines.sort(key=lambda x: x.m)

    res = []
    if lines is not None:
        median = allLines[int(len(allLines)/2)]
        for i in range(len(allLines)):
            if abs(allLines[i].m - median.m) <= 0.1:
                res.append(allLines[i])
    return res


def filterLinesByYInt(lines):
    res = []
    if len(lines) is not 0:
        lines.sort(key=lambda x: x.b)
        print(len(lines))
        medianInt = lines[int(len(lines)/2)].b
        for i in range(0, len(lines)):
            diff = abs(lines[i].b - medianInt)
            if 0 <= diff <= 100:
                res.append(lines[i])
    return res


def colorLines(lines, dst, color):
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i]
            cv.line(dst, (l.p1.x, l.p1.y), (l.p2.x, l.p2.y), color, 3, cv.LINE_AA)
    return dst


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
