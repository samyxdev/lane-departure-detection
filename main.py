# coding: utf-8
"""
Script réservé à l'execution sans kivy !
"""

import cv2
import os

from consts import *
from cv_funcs import *

if not VIDEO_MODE :
    img_names = ["nuit", "crepuscule", "journee"]

    imgs = [cv2.imread(f"data/samples/{name}.png") for name in img_names]

    for i, img in enumerate(imgs):
        print(img_names[i])

        assert img is not None

        frames = pipeline_debug(img)

        cv2.imshow(f"Otsu {img_names[i]}", frames[0])
        cv2.imshow(f"Adptative {img_names[i]}", frames[1])

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