# coding: utf-8
"""
Fonctions de processing des images
"""

import numpy as np
import cv2

from utils import *
from lines import *
from consts import *

"""
Applique un masque sur la région d'interêt et noircir le reste de l'image

img: Image à traiter
canny: booléen a passer en True si img est un retour de la fonction cv2.Canny
region_pts: Région définie par un ensemble de points ou None pour un triangle adapté
"""
def region_mask(img, canny=False, region_pts=None):
    h, w = img.shape[:2]
    region = np.array([(0, h), (w/2, h/2), (w, h)], dtype=np.int32) if region_pts is None else region_pts

    mask = np.zeros_like(img)

    # ATTENTION : la fonction attend une liste de polygones
    if not canny:
        cv2.fillPoly(mask, [region], [255]*img.shape[2])
    else:
        cv2.fillPoly(mask, [region], 255)

    masked = cv2.bitwise_and(img, mask)

    return masked

"""
prob: Booléen pour l'approche probabilistique de Hough
"""
def houghlines_to_img(img, lines, prob=False):
    printed = [False, False] # Statut de récupération des lignes de gauche et droite

    offsets = [0, 0] # Décalage des lignes de chaque côté

    for i in range(len(lines)):
        if not prob:
            r, t = lines[i][0][0], lines[i][0][1]
            a, b = math.cos(t), math.sin(t)
            x0, y0 = a*r, b*r
            x1, y1 = int(x0 + 1500*(-b)), int(y0 + 1500*a)
            x2, y2 = int(x0 - 1500*(-b)), int(y0 - 1500*a)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        else:
            color = c_red
            line_status, offset = line_filter(lines[i][0])

            if line_status in [0, 1] and not printed[line_status]: # 1ère Ligne conservée gauche ou droite
                color = c_green
                offsets[line_status] = offset
                printed[line_status] = True

                # A sortir du if pour afficher les lignes rouges non utilisées
            cv2.line(img, (lines[i][0][0], lines[i][0][1]),
                        (lines[i][0][2], lines[i][0][3]), color, 2)

    return offsets

"""
Permet d'extraire des informations sur la ligne passée (par rapport à la route)

Retourne 0 pour la ligne de gauche, 1 pour la droite
"""
def line_filter(line):
    # Tuplet position x (inférieure, supérieure)
    x = (line[0], line[2]) if line[1] > line[3] else (line[2], line[0])

    # La ligne de gauche a son extrémité supérieure à droite de l'extrémité inférieure
    side = 0 if x[0] < x[1] else 1

    offset = (LINE_CTRL if side == 0 else LINE_CTRR) - x[0]

    return side, offset

def norme(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

"""
Transformation à vue d'oiseau
"""
def birdview_transformation(img, rect):
    maxw = int(max(norme(rect[0], rect[1]), norme(rect[2], rect[3])))
    maxh = int(max(norme(rect[0], rect[3]), norme(rect[1], rect[2])))

    dst = np.array([[0, 0],
        [maxw - 1, 0],
        [maxw - 1, maxh - 1],
        [0, maxh - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, transform_matrix, (maxw, maxh))

    return warped


"""
Deuxième version de la pipeline, avec la birdview transform
"""
def pipeline_v2(frame, calib=None, prevCarState=None, onlyBirdview=False):
    """ Deuxième version de la pipeline de production, contenant
    la birdview. Retourne une image
    Arguments :
    - A définir
    """
    t1 = cv2.getTickCount()

    if calib is not None:
        # Undisort l'image
        #frame = cv2.undistort(frame, calib["mtx"], calib["dist"], None, calib["mtx"])

        # Marque la zone des lignes (pour ésperer la retrouver droite sur la birdviw)
        #frame = cv2.polylines(frame, [RECT_LINES.astype(np.int32)], True, (255, 255, 255))
        pass

    # Transformation par threshold perso + birdview
    edged = custom_threshold(frame)
    birdview = birdview_transformation(edged, RECT_BVIEW)
    birdview, line_ctrs, rcurv, line_types = line_finder(birdview)

    # Ecart centre véhicule/centre voie
    birdview, car_offset = car_path_offset(birdview, line_ctrs)

    # Information du comportement véhicule
    birdview, behavior = car_behavior(birdview, car_offset, line_types, prevCarState)

    # Affichage informations interprétées sur la birdview
    final_birdview = display_infos(birdview, rcurv, line_types, line_ctrs, car_offset, behavior)

    # Affichage durée de traitement
    dt = (cv2.getTickCount() - t1)/cv2.getTickFrequency()
    fasttext(final_birdview, "dt: {} s".format(dt), (0, birdview.shape[0] - 5))

    # Ajoute le marquage de la birdview
    marked_frame = cv2.polylines(
        frame, [RECT_BVIEW.astype(np.int32)], True, (255, 150, 0))

    # Marquage du centre de la voiture
    marked_frame = cv2.line(marked_frame, (CAR_CTRX, CAR_CTRY - 10),
                            (CAR_CTRX, CAR_CTRY + 10), (255, 0, 255), 2)
    marked_frame = cv2.line(marked_frame, (CAR_CTRX - 10, CAR_CTRY),
                            (CAR_CTRX + 10, CAR_CTRY), (255, 0, 255), 2)

    # Pratique pour les tests kivy
    if not onlyBirdview:
        return {"original marked": marked_frame, "birdview w/ data": final_birdview}, behavior
    else:
        if not DISABLE_CAR_STATE:
            return final_birdview, behavior

        else:
            return final_birdview

"""
Custom threshold
v1
gray/light > 70
saturation < 60

v2 - Marche avec autoroute.mp4
v > 75
s < 50

v3 -
v > 135
"""

def custom_threshold(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    binary = np.zeros_like(H)
    binary[(V > 75) & (S < 50)] = 255

    return binary


"""
Pipeline de test pour le préprocessing de l'image
"""
def pipeline_preproc(frame):
    thresh = custom_threshold(frame)


    return {"original": frame, "thresh":thresh}

"""
Permet de détecter les lignes avec la méthode de la fenêtre glissante et
y applique une modélisation polynomiale pour trouver le rayon de courbure
des tracés de chaque côtés

la frame passée doit être déjà sous forme birdview

Retourne la frame et un tuple contenant les coordonées x des lignes (ou -1 quand elles sont indéterminées)
"""
def line_finder(frame, h=HIST_WINDOW_H, markings=True):
    win_nb = len(frame)//h
    half_frame_w = frame.shape[1]//2
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Pour afficher des points de couleur

    # Coordonnées des points trouvés, sur le système de coordonnées de l'écran
    leftx, lefty = [], []
    rightx, righty = [], []

    # Fenêtre glissante vers le haut
    for i in range(win_nb):
        win_sum = np.sum(frame[-(i+1)*h:-i*h, :], 0)
        lines_ctr = (np.argmax(win_sum[:half_frame_w]), np.argmax(
            win_sum[half_frame_w:]) + half_frame_w)

        # Si il y a un vrai maximum qui a été détecté (0 si aucun max)
        if lines_ctr[0] != 0: # Tracés de gauche
            if markings:
                img = cv2.circle(img, (lines_ctr[0], frame.shape[0] - i*h), 8, c_green, -1)
            leftx.append(lines_ctr[0])
            lefty.append(frame.shape[0] - h*i)

        if lines_ctr[1] != half_frame_w: # Tracés de droite
            if markings:
                img = cv2.circle(img, (lines_ctr[1], frame.shape[0] - i*h), 8, c_green, -1)
            rightx.append(lines_ctr[1])
            righty.append(frame.shape[0] - h*i)

    # Application de la modélisation polynomiale
    img, rcurvl, accl = polynomialfit_line(img, leftx, lefty)
    img, rcurvr, accr = polynomialfit_line(img, rightx, righty)

    # Détermination du type de ligne
    ltype = (line_type(lefty), line_type(righty))

    # Récupération des centres des tracés (utilisé dans car_path_offset)
    # Peut être source de bug si ne sont pas alignés sur le même y
    ctrs = [-1, -1]

    # Version plus précise (ne considère que les lignes qui ont pu être modélisées)
    if ONLY_FITTED_LINE:
        if accl:
            ctrs[0] = leftx[0]
        if accr:
            ctrs[1] = rightx[0]

    # Version qui fonctionne plus souvent
    else:
        if len(leftx):
            ctrs[0] = leftx[0]
        if len(rightx):
            ctrs[1] = rightx[0]

    return img, ctrs, (rcurvl, rcurvr), ltype