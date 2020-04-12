"""
    ******  Chord Calculator / computer vision implementation *****
        Authors:
        Harry Ismayilov, Josh Gorman, Kushal Choksi


        Carleton University
        COMP4102
        Rosa Azami

    Main.py
        This code file contains the implementation of full computer vision in chord recognition.

        It contains two separate modes that make use of the same algorithm to determine chords with 
        different inputs.

        Image mode:
            Image mode can be used by selecting "i" when starting the program and shows the result of chord
            calculator on images found in the root directory.

        Live mode:
            Live mode can be used by selecting "w" when starting the program and shows the result of chord
            calculator on the webcam of your current computer (recommended to read function documentation
            for runtime instructions)
        
        The short summation of the main algorithm:
            1. Take an image of the base fretboard (with no fingers one)
            2. Use the horizontal lines (strings) of the image to rotate the image by the slope of the lines
            3. use the top and bottom lines to crop the image to show only the fretboard
            4. Find the fretlines of the image using houghlines and various thresholding operations
            5. Create a virtual grid of the fretboard to represent string locations
            6. take another image of the fretboard with fingers on in a similar position
            7. use an AKAZE matcher to create a homography of the chord image onto the base image
            8. Apply the string grid onto the warped chord image 
            9. Use thresholding operations to grab the hand as white pixels
            10. Determine if white pixels are inside of the frets of a given chord
            11. Return the resulting chord, if fingers are in frets representing one.


"""
from math import inf
import sys
import math
import statistics as stat
import cv2 as cv
import imutils
import numpy as np
import threading

"""
Explanation
    Input: image
    Output: horizontal hough lines without outliers

    Attempt to find all of the horizontal houghlines inside of the image and remove any outliers
    by checking if their slope is too different from the other lines
"""
def removeOutliers(imageToCrop):
    # Conver the image to grayscale and grab the houghlines
    thresh1 = cv.cvtColor(imageToCrop, cv.COLOR_BGR2GRAY)
    lines = cv.HoughLinesP(thresh1, 1, np.pi / 180, 150, None, 70, 30)
    y = []

    # Return no found lines if none are found
    if lines is None:
        return []

    # For every line create an array that has the [slope, y1, y2, x1, x2]
    for line in lines:
        for x1, y1, x2, y2 in line:
            y.append([abs(y2-y1/x2-x1), y1, y2, x1, x2])
            cv.line(imageToCrop, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Sort by the y1 of the lines
    y = sorted(y, key=lambda x: x[1])

    # Grab the median of the slopes
    y_med = np.median([x[0] for x in y])

    # Grab the mean, standard deviation and distance from the mean using y1
    y_mean = np.mean([x[1] for x in y])
    y_std = np.std([x[1] for x in y])
    y_dis = [abs(x[1]-y_mean) for x in y]
    max_deviations = 2

    # Create array of true/false values matching the following condition
    not_outlier = y_dis < max_deviations * y_std
    y_final = []

    # Iterate through true/false array keeping non outliers (true's)
    for x, y in zip(y, not_outlier):
        if y == True:
            y_final.append(x)
    return y_final


"""
Explanation
    Input: image
    Output: vertical hough lines without outliers 

    Attempt to find all of the vertical houghlines inside of the image, the houghlines function is
    optimized to find shorter lengths of lines (matching that of frets) and removes lines that are not
    sufficiently close to being vertical (due to the algorithm making the fretboard horizontal)
"""
def removeVerticalOutliers(imageToCrop):
    # Make the code grayscale if it's not already
    if (not len(imageToCrop.shape) < 3):
        imageToCrop = cv.cvtColor(imageToCrop, cv.COLOR_BGR2GRAY)

    # Grab the houghlines of the image 
    lines = cv.HoughLinesP(imageToCrop, 1, np.pi / 180, 30, None, 20, 20)
    x_diff = []

    # If there are no lines return no lines
    if lines is None:
        return []
    # Otherwise create an array of differences in the x coordinate
    for line in lines:
        for x1, y1, x2, y2 in line:
            x_diff.append(abs(x1 - x2))

    # Remove any that have a high difference (non-vertical lines)
    remove_large = np.array(x_diff) < 15
    lines = lines[remove_large]

    # Rebuild x_diff
    x_diff = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            x_diff.append(abs(x1 - x2))

    # Use same calculation as remove_outliers to remove outliers from our array
    x_mean = np.mean(x_diff)
    x_med = np.median(x_diff)
    x_std = np.std(x_diff)
    x_dis = abs(x_diff-x_med)
    max_deviations = 2

    # Create the condition for whether the diff is acceptable
    not_outlier = x_dis < max_deviations * x_std

    # Use numpy functinality to only keep true's in array "lines"
    not_outliers = lines[not_outlier]

    return not_outliers


"""
Deprecated
    This function was deprecated due to the high variance in fret markers per/guitar although it does work
    some guitars have:
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
Explanation
    Input: image
    Output: image cropped and rotated to find only guitar fretboard

    This algorithm attempts to use edge detection to find a high number of long horizontal lines inside 
    of an image, using these lines it attempts to rotate an image by the median slope of these lines to 
    create a horizontal guitar fretboard.

    Upon creating the fretboard it takes the top and bottom lines pixel values and crops out the rest of the
    image to allow for a more focused region when determining your chords.
"""
def cropNeckPicture(image):
    image_to_crop = image

    # Use canny edge detection to make the image easier to grab lines
    dst = cv.Canny(image_to_crop, 50, 220, None, 3)
    image_to_crop = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    # Use a horizontal sobel filter to allow for only finding horizontal lines
    edges = cv.Sobel(image_to_crop, cv.CV_8U, 0, 1, ksize=5)
    edges, thresh1 = cv.threshold(edges, 127, 255, cv.THRESH_BINARY)

    # Remove the outliers from the image to get only the strings of the guitar
    y_final = removeOutliers(thresh1)

    # If you didn't find at least 10 lines, don't take the output and return the canny result
    if len(y_final) < 10:
        return image_to_crop
    image_start = image_to_crop

    # If you found it, take the median point and using those values calculate an angle to rotate by
    center = y_final[len(y_final)//2]
    angle = np.rad2deg(np.arctan2(
        center[2] - center[1], center[4] - center[3]))

    # Use imutils "rotate_bound" which will allow for keeping what we want and not losing the edges of the imagejj
    image_to_crop = imutils.rotate_bound(image_to_crop, -angle)

    # Re-complete the same steps shown above but this time to get the top and bottom lines of the newly
    # rotated image
    y_final = removeOutliers(image_to_crop)

    if len(y_final) == 0:
        return image_start

    # Now we take the bottom and top pixel values of the lines
    yBottom = y_final[0][1]-10
    yTop = y_final[len(y_final)-1][1]+50

    # Now rotate and crop the image to return
    image_start = imutils.rotate_bound(image, -angle)
    image_start = image_start[yBottom:yTop]

    return image_start


"""
Explanation
    Input: image, image
    Output: AKAZE matched points between the two images

    This algorithm creates an AKAZE matcher and calculates the feature points between the two of them using
    bruteforce matching. It then returns the matched points to be used in a warping function
"""
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


"""
Explanation
    Input: feature_points_image1, feature_points_image2, image1, image2
    Output: AKAZE matched points between the two images

    This algorithm uses the determined AKAZE feature points to calculate and return a homography of the second
    image onto the first image
"""
def getWarpedImage(img1Pts, img2Pts, img1, img2):
    retVal, mask = cv.findHomography(img1Pts, img2Pts, cv.RANSAC)
    height, width = img2.shape

    return cv.warpPerspective(img1, retVal, (width, height))


"""
Explanation
    Input: cropped image of the fretboard
    Output: Fretlines and stringlines

    This algorithm uses the cropped image of a fretboard and attempts to create a grid which can be used to
    determine finger positions of chords.

    It does this by grabbing each of the frets on the fretboard, cutting out any outliers and setting their 
    top y value and bottom y value to be equal to each other line. After doing this it writes a top and bottom
    line along the frets showing the exact bounding box of the fretboard.

    With the bounding box of the fretboard it then draws 5 lines in between to represnt *zones* in which
    the strings are likely to be inside of. If a finger is inside this zone it is likely being used for a chord
    and can be used to determine chord shapes.
"""
def detectFrets(image):
    # If the image is not grayscale return no lines
    if (not len(image.shape) < 3):
        return [], []

    # If it is use some cleaning functions to try to get a better representation of the frets
    # First use a vertical sobel filter, then threshold to remove noise, and finish by using
    # The morphological operator *closing* to further reduce outside noise and connect lines
    edges = cv.Sobel(image, cv.CV_8U, 1, 0, ksize=3)
    ret, edges = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, (3, 3), iterations=3)

    # Remove the unneeded outlier lines (lines that are not at the same slope as the frets)
    linesNoOutliers = removeVerticalOutliers(closing)
    if len(linesNoOutliers) == 0:
        return [], []

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
            # If the lines are not within five pixels of each other then do not add them
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

    # Create the lines at the top of the fretboard
    stringLines.append([firstFret[2], firstFret[3], lastFret[2], lastFret[3]])

    # Create lines in between the strings
    neckHeight = firstFret[1] - firstFret[3]
    stringDist = np.round(neckHeight / 6)

    if stringDist == 0:
        return [], []
    for z in range(int(firstFret[3]), int(neckHeight), int(stringDist)):
        currentString = z + int(stringDist)
        stringLines.append([firstFret[0], currentString,
                            lastFret[0], currentString])

    # Create the line at the bottom of the fretboard
    stringLines.append([firstFret[0], firstFret[1], lastFret[0], lastFret[1]])

    # Reverse the lines so we have a better visual for further code
    stringLines.reverse()
    onlyFretLines.reverse()
    return (onlyFretLines, stringLines)


"""
Explanation
    Input: image, lines, desired colours of lines
    Output: 

    A simple function to draw lines onto an image of the desired colour
"""
def drawLines(image, lines, colour):
    for x1, y1, x2, y2 in lines:
        cv.line(image, (x1, y1), (x2, y2), colour, 2)


"""
Deprecated

Explanation
    Input: video capture object
    Output: 

    A simple function which attempts to create a timer to grab frames from set intervals instead of constant
    video, this could assist in the runtime and demands of the live chord recognition
"""
def grabFrame(cap):
    threading.Timer(20.0, grabFrame).start()
    ret, frame = cap.read()
    return ret, frame

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    croppedFrame = cropNeckPicture(gray)

    # Display the resulting frame
    cv.imshow('frame', croppedFrame)

    if cv.waitKey(1) == ord('q'):
        exit()


"""
Explanation
    Input: stringlines, fretlines, image with hands on chord
    Output: chord found in image

    This code was implemented after testing out both *contours* and *houghcircles* with an attempt at locating
    the fingernails of the image, in both cases there was a low success rate with normal fingers due to a lack
    of differentiation in the fingernails.

    This code attempts to make use of simple thresholding and pixel manipulation to determine where the fingers
    are inside of the image. 
"""
def checkForChords(stringLines, fretLines, image):
    # Define in which frets of the grid a C chord requires fingers
    chords = {
        "CChord": [
            [0, 1],
            [1, 3],
            [2, 4]
        ],
        "EminChord": [
            [1, 3],
            [1, 4]
        ],
        "DChord": [
            [1, 0],
            [2, 1],
            [1, 2]
        ]
    }

    # Cut to only the first few frets (only because we don't need them all for the choreds we're looking for)
    fretLines = fretLines[0:5]

    # use some simple cleaning functions to first threshold to remove darker parts of the image (since in this
    # case skin colour is pale/white) then morphologically close (erode -> dilate) to focus on larger parts in the
    # image since fingers take up a significant portions of the image
    if (not len(image.shape) < 3):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    ret, mask = cv.threshold(
        image, 100, 255, cv.THRESH_BINARY)
    mask = cv.erode(mask, kernel, iterations=4)
    mask = cv.dilate(mask, kernel, iterations=4)
    temp = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # Generate the fretgrid by looking at the pixels of the string and fretlines and generating bounding boxes in 
    # the form of numpy arrays
    fretGrid = []
    for y in range(0, len(fretLines)-1):
        fretGrid.append([])
        for x in range(0, len(stringLines)-1):
            fretGrid[y].append(
                np.array([[[fretLines[y][0], stringLines[x][1]],
                           [fretLines[y+1][0], stringLines[x][1]],
                           [fretLines[y+1][0], stringLines[x+1][1]],
                           [fretLines[y][0], stringLines[x+1][1]]]], dtype=np.int32)
            )

    resultChord = "none"

    # Iterate through the chords
    for chord in chords:
        isChord = True

        for fret in chords[chord]:
            # For a given fret check if there are white pixels inside of the fret

            #Create a mask image that has a white box equal to the fret we are looking at
            theMask = np.zeros((temp.shape), dtype=np.uint8)
            cv.fillPoly(theMask, fretGrid[fret[0]][fret[1]], (255, 255, 255))

            # Now grab the values of each pixel inside our fretboard image corresponding with our frets bounding box
            values = temp[np.where((theMask == (255, 255, 255)).all(axis=2))]

            # if the values inside our bounding box are all equal to 0 then we it can't be the current chord
            # because the finger has not been found inside of one of the frets that are in the chord
            if np.all(values == [0, 0, 0]) == True:
                isChord = False
                break

        if isChord == True:

            print(chord)
            resultChord = chord
            break

    return resultChord


"""
Explanation
    Input: 
    Output: 

    This algorithm runs a loop that captures a frame every 100 milliseconds.

    It starts by running the canny edge detector while looking for a suitable number of horizontal lines to 
    rotate the image and crop it (ideally a guitar freboard), it then, upon finding a suitable fretboard, 
    it attempts to place the fretboard lines visually onto it for the user to see.

    Upon finding a decent position to capture the fretboard the user can continue the algorithm by selecting
    "t" to, move into the chord recognition portion, this attempts to warp the new frames in the video onto
    the base fretboard which allows the same fret and string lines to be used on the new warped image

    the string lines and fretlines are then used along with the warped image to try to find out what chord is
    being shown
"""
def liveChordCheck():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # ret, frame = grabFrame(cap)
    setupStage = True
    croppedFrame = 0

    # Variables used in warping new images onto existing ones
    fretboard = 0
    fretlines = 0,
    stringLines = 0
    imageWithLines = 0
    mask = 0

    while True:
        # Capture frame by frame (only every 100milliseconds)
        key = cv.waitKey(100)

        # Controls for the program:
        # Q: exit the program
        # T: move to the chord recognition stage
        # G: move back to the fret detection stage
        if key == ord("q"):
            break

        elif key == ord("t"):
            # Set the global variable fretboard to the current frame for use in image warping
            fretboard = croppedFrame

            # Change to the chord warping stage
            setupStage = False

            # Create a visual so the user can see what fretboard they have chosen (to potentially re-do)
            linedFretboard = fretboard
            drawLines(linedFretboard, stringLines, (0, 255, 0))
            drawLines(linedFretboard, fretLines, (255, 0, 0))
            cv.imshow('frame', linedFretboard)
            cv.waitKey(0)

        elif key == ord("g"):
            # Go back to the fret detection stage
            setupStage = True

        if setupStage == True:
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                croppedFrame = cropNeckPicture(gray)
                imageWithLines = croppedFrame

                if (len(croppedFrame.shape) < 3):
                    imageWithLines = cv.cvtColor(
                        croppedFrame, cv.COLOR_GRAY2BGR)
                    fretLines, stringLines = detectFrets(croppedFrame)

                    drawLines(imageWithLines, stringLines, (0, 255, 0))
                    drawLines(imageWithLines, fretLines, (255, 0, 0))

                cv.imshow('frame', imageWithLines)

            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        else:
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                croppedFrame = cropNeckPicture(gray)

                if (len(croppedFrame.shape) < 3):
                    ptsA, ptsB = createMatches(croppedFrame, fretboard)
                    # Now warp image A onto image B
                    warpedChord = getWarpedImage(
                        ptsA, ptsB, croppedFrame, fretboard)

                    imageWithLines = cv.cvtColor(
                        croppedFrame, cv.COLOR_GRAY2BGR)
                    checkForChords(
                        stringLines, fretLines, warpedChord)

                cv.imshow('frame', imageWithLines)
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

    cap.release()
    cv.destroyAllWindows()

"""
Explanation
    Input: 
    Output: 

    This algorithm shows a few example chords of which have images that can be found in the root directory
    It works in the same way as the live chord recognition however on a few sample images instead.

    This is by again by:
    1. grabbing and cropping the fretboard with no fingers on
    2. overlaying frets and strings on it
    3. taking an image with the fingers on it in a similar position
    4. warping the image with fingers on, onto the base image
    5. overlaying the frets and strings onto the warped image
    6. thresholding the fingers and looking for pixels inside of the frets of each chor d
"""
def imageChordCheck():
    # Grab some filenames for loading in images
    baseFilename = 'computerVision1/base.jpg'
    dChordFilename = 'computerVision1/dmaj.jpg'
    eChordFilename = 'computerVision1/emin.jpg'
    cChordFilename = 'computerVision1/cmaj.jpg'

    # Load in the base image and find it's fretlines
    baseImage = cv.imread(cv.samples.findFile(baseFilename), 0)
    croppedBaseImage = cropNeckPicture(baseImage)
    fretLines, stringLines = detectFrets(croppedBaseImage)

    # Load in all the chord images
    dChord = cv.imread(cv.samples.findFile(dChordFilename), 0)
    eChord = cv.imread(cv.samples.findFile(eChordFilename), 0)
    cChord = cv.imread(cv.samples.findFile(cChordFilename), 0)

    # Crop them in the same way as the base image
    croppedDChord = cropNeckPicture(dChord)
    croppedEChord = cropNeckPicture(eChord)
    croppedCChord = cropNeckPicture(cChord)

    # Get the points matching the chord images and the base image
    ptsA, ptsB = createMatches(croppedDChord, croppedBaseImage)
    ptsC, ptsD = createMatches(croppedEChord, croppedBaseImage)
    ptsE, ptsF = createMatches(croppedCChord, croppedBaseImage)

    # Warp the chord images onto the base image
    warpedDChord = getWarpedImage(ptsA, ptsB, croppedDChord, croppedBaseImage)
    warpedEChord = getWarpedImage(ptsC, ptsD, croppedEChord, croppedBaseImage)
    warpedCChord = getWarpedImage(ptsE, ptsF, croppedCChord, croppedBaseImage)

    # Run the checkForChords algorithm to find out what chord is in the warped image
    dChordResult = checkForChords(stringLines, fretLines, warpedDChord)
    eChordResult = checkForChords(stringLines, fretLines, warpedEChord)
    cChordResult = checkForChords(stringLines, fretLines, warpedCChord)

    # Draw text showing what chord was found in each image
    warpedDChord = cv.cvtColor(warpedDChord, cv.COLOR_GRAY2BGR)
    warpedEChord = cv.cvtColor(warpedEChord, cv.COLOR_GRAY2BGR)
    warpedCChord = cv.cvtColor(warpedCChord, cv.COLOR_GRAY2BGR)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(warpedDChord, dChordResult, (10, 120),
               font, 1, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(warpedEChord, eChordResult, (10, 120),
               font, 1, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(warpedCChord, cChordResult, (10, 120),
               font, 1, (0, 0, 255), 1, cv.LINE_AA)

    # Draw the fretboards (for visual purposes) and how the resulting images with their chords on the image
    drawLines(warpedDChord, stringLines, (0, 255, 0))
    drawLines(warpedDChord, fretLines, (255, 0, 0))
    drawLines(warpedEChord, stringLines, (0, 255, 0))
    drawLines(warpedEChord, fretLines, (255, 0, 0))
    drawLines(warpedCChord, stringLines, (0, 255, 0))
    drawLines(warpedCChord, fretLines, (255, 0, 0))

    cv.imshow("DChord", warpedDChord)
    cv.imshow("EChord", warpedEChord)
    cv.imshow("CChord", warpedCChord)
    cv.waitKey()


"""
Explanation
    Input: command line args
    Output: 

    A simple main function which allows for entering I or W to decide whether to test the webcam functionality
    or the provided image examples inside of the repo.
"""
def main(argv):
    print("-------------------------------------")
    print("Input I for image based chord recognition")
    print("Input w for webcam based chord recognition")
    print("-------------------------------------")

    key = ""
    while key != "i" and key != "w":
        key = input()
        if key == "i":
            imageChordCheck()
            break
        elif key == "w":
            liveChordCheck()
            break
        print("Please select a valid option ('i', or 'w')")

    return (0)

if __name__ == "__main__":
    main(sys.argv[1:])
