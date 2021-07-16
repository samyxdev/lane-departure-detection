"""
Script reservé aux essais de traitement de l'image
"""

import cv2
import os
import matplotlib.pyplot as plt

from consts import *
from utils import *
from imgproc import *
from lines import *


"""
Pipeline de debug - Permet de réaliser des tests
"""
def pipeline_debug(frame):
    #frame = frame[(frame.shape[0] // 3) * 2:, :]
    f2 = custom_threshold(frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f3 = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 9)

    return f2, f3


#img_names = ["nuit", "crepuscule", "journee"]
img_names = ["nuit", "journee"]

imgs = [cv2.imread(f"data/samples/{name}.png") for name in img_names]

for i, img in enumerate(imgs):
    print(img_names[i])

    assert img is not None

    frames = pipeline_debug(img)

    cv2.imshow(f"Custom threshold {img_names[i]}", frames[0])
    cv2.imshow(f"Adaptive {img_names[i]}", frames[1])

    img = img[(img.shape[0] // 3) * 2:, :]

    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grayed " + img_names[i], grayed)

    plt.subplot(1, len(img_names), i + 1)
    plt.hist(grayed.ravel(), bins=256, range=(0, 255))
    plt.title(img_names[i])

cv2.waitKey(0)
cv2.destroyAllWindows()