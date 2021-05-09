"""
Script de calibration de la caméra sur échecs
"""
#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import json

from consts import *

# Defining the dimensions of the checkerboard
BOARD_SIZE = (9, 6)

# To scale input frames
SCALE = False

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[1], 0:BOARD_SIZE[0]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(CALIB_FOLDER + '/*.jpg')

print("Processing distortion calibration on {} images from {}...".format(str(len(images)), CALIB_FOLDER))
for fname in images:
    img = cv2.imread(fname)

    # Scale images
    if SCALE:
        img = cv2.resize(img, (CALIB_FORMAT[0], CALIB_FORMAT[1]))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, BOARD_SIZE, corners2, ret)
        
        # Resize the image to display it properly
        if not SCALE:
            h, w = img.shape[:2]
            ratio = w/h
            img = cv2.resize(img, (int(ratio*1000), 1000))

        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print("No points found for " + fname)

cv2.destroyAllWindows()

# Computing calibration data
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Saving calibration data
with open(CALIB_FILE, "w") as f:
    data = {"mtx":mtx.tolist(), "dist":dist.tolist()}
    enc = json.JSONEncoder().encode(data)

    print(enc)

    f.write(enc)
    print("Data saved to " + CALIB_FILE)

img = cv2.imread(CALIB_FOLDER + "/test_undisort_s7.jpg")

h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (w, h), 1, (w, h))

img = cv2.undistort(img, mtx, dist, None, newcameramtx)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


