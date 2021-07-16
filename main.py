# coding: utf-8
"""
Script réservé à l'execution sans kivy !
"""

import cv2
import os
import matplotlib.pyplot as plt

from consts import *
from utils import *
from imgproc import *
from lines import *

if not VIDEO_MODE :
    img = cv2.imread(IMAGE_PATH)

    assert img is not None

    cv2.imshow(pipeline_v2(img))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if VIDEO_MODE:
    # Capture sur fichier vidéo
    if VIDEO_MODE == 1:
        if not os.path.exists(VIDEO_MODE):
            printd(f"Path {VIDEO_MODE} not found !")
        else:
            printd(f"Path {VIDEO_MODE} exists !")
        cap = cv2.VideoCapture(VIDEO_PATH)

    # Capture sur webcam
    else:
        cap = cv2.VideoCapture(WEBCAM_PATH)

    if cap.isOpened():
        printd("cv2 VideoCapture initialised !")
    else:
        printd("cv2 VideoCapture failed to init...")

    carState = 0

    while cap.isOpened():
        ret, frame = cap.read()

        frames, carState = pipeline_v2(frame, prevCarState=carState, onlyBirdview=False)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        if key == ord('p'):
            cv2.waitKey(-1) # Attend pour n'importe quelle touche

        # Pour afficher un max
        for k in frames.keys():
            cv2.imshow(k, frames[k])

    cap.release()
    cv2.destroyAllWindows()