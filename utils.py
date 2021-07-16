# coding: utf-8
"""
Fonctions d'affichage ou de chargement de fichiers
"""

import numpy as np
import cv2
import json

from consts import *

def printd(text):
    #Permet de distinguer nos messages de debug de ceux de kivy
    if DEBUG_PRINT:
        print("PyDebug: " + str(text))


def availableVideoSources(src=[-1, 0, 1]):
    """Permet de tester l'ouverture d'un flux vidéo
    sur opencv
    """
    print("Testing videos sources : ", src)

    workingSources = []
    for source in src:
        cap = cv2.VideoCapture(source)

        if cap is None or not cap.isOpened():
            printd(f"Warning: unable to open video source {source}")
        else:
            printd(f"Video source {source} successfully opened !")
            workingSources.append(source)

        cap.release()

    return workingSources


"""
Fonction raccourci de cv2 pour du text, NE RETOURNE RIEN
Centred: Si True, la postion sera modifiée pour déplacer le centre inférieur du texte
(à la place du coin inférieur gauche initialement)
"""
def fasttext(frame, text, pos, color=(255,)*3, centred=False, fontface=cv2.FONT_HERSHEY_SIMPLEX):
    if centred:
        pos = (pos[0] - cv2.getTextSize(text, fontface, 1, 1)[0][0]//2, pos[1])
    frame = cv2.putText(frame, text, pos, fontface, 1, color, lineType=cv2.LINE_AA)

    return None

"""
Charge les données de calibration spécifique à la caméra utilisées (dossier et
format du fichier spécifiés dans "consts.py")
"""
def load_calibration_data():
    data = {}
    with open(CALIB_FILE, "r") as f:
        data = f.read()

    dec = json.JSONDecoder().decode(data)

    # Conversion au format numpy attendu par opencv
    dec['mtx'] = np.array(dec['mtx'])
    dec['dist'] = np.array(dec['dist'])

    return dec

"""
Affiche des informations extraites à l'écran passé en argument
Rayons de courbure
Type ligne
Curseur d'écart axial voie/véhicule
Comportement véhicule
"""
def display_infos(frame, rcurv, linetype, linectrs, offset, behavior):
    # Infos sur le rayon de courbure
    fasttext(frame, "Rcourbure gauche: {:.2f} px".format(rcurv[0]), (0, 30))
    fasttext(frame, "Rcourbure droit: {:.2f} px".format(rcurv[1]), (0, 70))

    # Type de ligne
    linetype_text = ["", ""]
    for i in range(len(linetype_text)):
        if linetype[i] == 0:
            linetype_text[i] = "non detectee"
        elif linetype[i] == 1:
            linetype_text[i] = "continue"
        else: # Ligne pointillée
            linetype_text[i] = "pointillee ({})".format(linetype[i])

    fasttext(frame, "Types lignes:", (0, 140))
    fasttext(frame, " Gauche " + linetype_text[0], (0, 170))
    fasttext(frame, " Droite " + linetype_text[1], (0, 200))

    # Constantes curseurs
    cursor_w = 500
    cursor_h = 20
    cursor_y = frame.shape[0] - 50

    col = (255, 0, 0)
    if -1 in linectrs:
        col = (0, 255, 0)
    else:
        col = (255, 0, 255)

    # Axe curseur, curseur central et limites
    frame = cv2.line(frame, (CAR_CTRX - cursor_w//2, cursor_y),
                     (CAR_CTRX + cursor_w//2, cursor_y), (255,)*3, 4)
    frame = cv2.line(frame, (CAR_CTRX, cursor_y - cursor_h//2 - 5),
                     (CAR_CTRX, cursor_y + cursor_h//2 + 5), (255,)*3, 3)

    frame = cv2.line(frame, (CAR_CTRX - OFFSET_LIMIT, cursor_y - 7),
                     (CAR_CTRX - OFFSET_LIMIT, cursor_y + 5), (0, 255, 255), 2)
    frame = cv2.line(frame, (CAR_CTRX + OFFSET_LIMIT, cursor_y - 7),
                     (CAR_CTRX + OFFSET_LIMIT, cursor_y + 5), (0, 255, 255), 2)

    # Curseur position actuelle
    frame = cv2.line(frame, (CAR_CTRX + offset, cursor_y - cursor_h//2),
                     (CAR_CTRX + offset, cursor_y + cursor_h//2), col, 3)
    fasttext(frame, "Ecart vehicule/centre voie",
             (CAR_CTRX, cursor_y - 20), centred=True)

    # Affichage comportement véhicule
    behavior_texts = ["Trajectoire stable",
        "Changement de voie",
        "Sortie de route !"]

    xpos = cv2.getTextSize(behavior_texts[behavior], cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
    fasttext(frame, behavior_texts[behavior], (frame.shape[1] - xpos - 10, 30))

    return frame


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("{}, {}".format(x, y))
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            RECT_BVIEW[1][0] += 5
        else:
            RECT_BVIEW[1][0] -= 5

        print("Current: ", RECT_BVIEW[1][0])








