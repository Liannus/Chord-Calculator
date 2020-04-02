"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
from math import inf
import sys
import math
import statistics as stat
import cv2 as cv
import imutils
import numpy as np

"""
Explanation
    Input:
    Output:
"""


def removeOutliers(imageToCrop):
    thresh1 = cv.cvtColor(imageToCrop, cv.COLOR_BGR2GRAY)
    lines = cv.HoughLinesP(thresh1, 1, np.pi / 180, 150, None, 50, 30)
    y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            y.append([y1, abs(y2-y1/x2-x1), y2, x1, x2])
            cv.line(imageToCrop, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv.imshow("Start", imageToCrop)

    y = sorted(y, key=lambda x: x[0])
    y_med = np.median([x[1] for x in y])

    y_mean = np.mean([x[0] for x in y])
    y_std = np.std([x[0] for x in y])
    y_dis = [abs(x[0]-y_mean) for x in y]
    max_deviations = 2
    not_outlier = y_dis < max_deviations * y_std
    y_final = []
    for x, y in zip(y, not_outlier):
        if y == True:
            y_final.append(x)
    return y_final


"""
Explanation
    Input:
    Output:
"""


def removeVerticalOutliers(imageToCrop):
    lines = cv.HoughLinesP(imageToCrop, 1, np.pi / 180, 40, None, 20, 20)
    x_diff = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            x_diff.append(x1 - x2)

    x_mean = np.mean(x_diff)
    x_med = np.median(x_diff)
    x_std = np.std(x_diff)
    x_dis = abs(x_diff-x_med)
    max_deviations = 0.1
    not_outlier = x_dis < max_deviations * x_std
    not_outliers = lines[not_outlier]

    return not_outliers


"""
Deprecated
This function was deprecated due to the high variance in fret markers per/guitar
somee guitars have:
- markers starting on fret 3 versus fret 5
- stylized markers that can both not be dots or the colour white.

This function should work for most guitars that have white circular fret markers,
it attempts to:
1. Threshold out the white colour of the markers
2. Morphologically erode pixels to remove noise
3. Morphologically dilate to increase size of still existing pixels for easier detection
4. Do the same thing again to increase success rate
5. Find contours in the remaining image
6. Sort the returned values and draw circles around them in an image

"""


def findFretMarkers(image):
    kernel = np.ones((3, 3), np.uint8)
    ret, mask = cv.threshold(image_start, 180, 190, cv.THRESH_BINARY)
    mask = cv.erode(mask, kernel, iterations=2)
    mask = cv.dilate(mask, kernel, iterations=4)
    closing = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda x: cv.boundingRect(x)[0])

    array = []
    ii = 1
    for c in contours:
        (x, y), r = cv.minEnclosingCircle(c)
        center = (int(x), int(y))
        r = int(r)
        if r >= 6 and r <= 8:
            cv.circle(image, center, r, (0, 255, 0), 2)
            array.append(center)


"""
Cropping the picture so we only work on the region of interest (i.e. the neck)
We're looking for a very dense region where we detect horizontal line
Currently, we identify it by looking at parts where there are more than two lines at the same y (height)
:param image: an Image object of the neck (rotated horizontally if necessary)
:return cropped_neck_picture: an Image object cropped around the neck
"""


def crop_neck_picture(image):
    image_to_crop = image

    dst = cv.Canny(image_to_crop, 50, 200, None, 3)
    image_to_crop = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    edges = cv.Sobel(image_to_crop, cv.CV_8U, 0, 1, ksize=5)
    edges, thresh1 = cv.threshold(edges, 127, 255, cv.THRESH_BINARY)

    y_final = removeOutliers(thresh1)

    center = y_final[len(y_final)//2]
    angle = np.rad2deg(np.arctan2(
        center[2] - center[0], center[4] - center[3]))

    image_to_crop = imutils.rotate_bound(image_to_crop, -angle)
    y_final = removeOutliers(image_to_crop)

    yBottom = y_final[0][0]-10
    yTop = y_final[len(y_final)-1][0]+10

    image_start = imutils.rotate_bound(image, -angle)
    image_start = image_start[yBottom:yTop]

    return image_start


def createMatches(firstImage, secondImage):
    # User a feature detector (view README for why not sift/surf)
    detector = cv.AKAZE_create()

    # find the keypoints and descriptors with AKAZE
    kp1, ds1 = detector.detectAndCompute(firstImage, None)
    kp2, ds2 = detector.detectAndCompute(secondImage, None)

    # create a brute force matcher (could use flann here too)
    bfmatcher = cv.DescriptorMatcher_create("BruteForce")

    # Match descriptors.
    matchObjects = bfmatcher.knnMatch(ds1, ds2, 2)
    pointMatches = []

    for m in matchObjects:
        pointMatches.append((m[0].trainIdx, m[0].queryIdx))

    pts1 = np.float32([kp1[i].pt for (_, i) in pointMatches])
    pts2 = np.float32([kp2[i].pt for (i, _) in pointMatches])

    return (pts1, pts2)


def getWarpedImage(img1Pts, img2Pts, img1, img2):
    retVal, mask = cv.findHomography(img1Pts, img2Pts, cv.RANSAC)
    height, width = img2.shape

    return cv.warpPerspective(img1, retVal, (width, height))


def detectFrets(image):
    edges = cv.Sobel(image, cv.CV_8U, 1, 0, ksize=3)
    ret, edges = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, (3, 3), iterations=3)

    # Remove the unneeded outlier lines (lines that are not at the same slope as the frets)
    linesNoOutliers = removeVerticalOutliers(closing)

    # Break and sort the lines by their X values
    onlyCoords = []
    for line in linesNoOutliers:
        for x1, y1, x2, y2, in line:
            onlyCoords.append([x1, y1, x2, y2])
    sortedLinesNoOutliers = sorted(onlyCoords, key=lambda x: x[0])

    # Compile the lines into each other to create singular fret lines
    onlyFretLines = []
    for line in sortedLinesNoOutliers:
        add = True
        for nestedLine in onlyFretLines:
            if line[0] < nestedLine[0] + 5 and line[0] > nestedLine[0] - 5:
                maxVal = max([nestedLine[1], nestedLine[3], line[1], line[3]])
                minVal = min([nestedLine[1], nestedLine[3], line[1], line[3]])
                nestedLine[1] = maxVal
                nestedLine[3] = minVal
                add = False
                break
        if add == True:
            onlyFretLines.append([line[0], line[1], line[2], line[3]])

    # Calculate the median y values and set all the frets to the same "height"
    y1_med = int(np.median([x[1] for x in onlyFretLines]))
    y2_med = int(np.median([x[3] for x in onlyFretLines]))
    for line in onlyFretLines:
        line[1] = y1_med
        line[3] = y2_med

    # Create the string lines
    stringLines = []
    lastFret = onlyFretLines[len(onlyFretLines)-1]
    firstFret = onlyFretLines[0]

    # Create the lines at the top and bottom of the fretboard
    stringLines.append([firstFret[0], firstFret[1], lastFret[0], lastFret[1]])
    stringLines.append([firstFret[2], firstFret[3], lastFret[2], lastFret[3]])

    # Create lines in between the strings
    neckHeight = firstFret[1] - firstFret[3]
    stringDist = np.round(neckHeight / 6)

    for z in range(int(firstFret[3]), int(neckHeight), int(stringDist)):
        currentString = z + int(stringDist)
        stringLines.append([firstFret[0], currentString,
                            lastFret[0], currentString])

    return (onlyFretLines, stringLines)


def drawLines(image, lines, colour):
    for x1, y1, x2, y2 in lines:
        cv.line(image, (x1, y1), (x2, y2), colour, 2)


def main(argv):

    baseFilename = 'inputo.jpg'
    dChordFilename = 'dmaj.jpg'

    baseImage = cv.imread(cv.samples.findFile(baseFilename), 0)

    croppedBaseImage = crop_neck_picture(baseImage)
    fretLines, stringLines = detectFrets(croppedBaseImage)

    # Draw the fret lines
    cv.imshow("beginning", baseImage)
    cv.imshow("crop algorithm", croppedBaseImage)

    dChord = cv.imread(cv.samples.findFile(dChordFilename), 0)

    croppedDChord = crop_neck_picture(dChord)

    # Get the points matching the Image A and Image B
    ptsA, ptsB = createMatches(croppedDChord, croppedBaseImage)
    # Now warp image A onto image B
    warpedDChord = getWarpedImage(ptsA, ptsB, croppedDChord, croppedBaseImage)

    cv.imshow("second-image", croppedDChord)
    cv.imshow("second-image warped", warpedDChord)

    # Reintroduce colour dimension for lines
    warpedDChord = cv.cvtColor(warpedDChord, cv.COLOR_GRAY2BGR)
    drawLines(warpedDChord, stringLines, (0, 255, 0))
    drawLines(warpedDChord, fretLines, (255, 0, 0))

    croppedBaseImage = cv.cvtColor(croppedBaseImage, cv.COLOR_GRAY2BGR)
    drawLines(croppedBaseImage, stringLines, (0, 255, 0))
    drawLines(croppedBaseImage, fretLines, (255, 0, 0))

    cv.imshow("base-image warped with lines", croppedBaseImage)
    cv.imshow("second-image warped with lines", warpedDChord)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
