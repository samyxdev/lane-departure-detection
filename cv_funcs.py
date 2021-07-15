# coding: utf-8
"""
Fichier des fonctions partagées par le projet (scripts image et videos)
"""

import numpy as np
import cv2
import math
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
Pipeline de debug - Permet de réaliser des tests
"""
def pipeline_debug(frame):
    frame = frame[(frame.shape[0] // 3) * 2:, :]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (T, f2) = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f3 = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)

    print(T)

    return f2, f3

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
def custom_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (T, otsu_frame) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return otsu_frame

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


"""
Détermine le type de ligne
Argument:
    pty: Coordonée y des points
    text: Version texte du type de ligne
Retourne 0 pour aucune ligne, 1 pour une ligne continue, 2 pour une ligne à sections
"""
def line_type(pty):
    nb = len(pty)
    ltype = 0 # Par défaut, aucune ligne
    last_gap = 0

    if nb >= 4: # Il faut un minimum de points pour la détection
        i = 0
        endSearch = False

        while i < nb - 1 and not endSearch:
            # Ecart entre les points supérieur à la hauteur de fenêtre glissante
            gap = pty[i] - pty[i + 1]
            if gap > HIST_WINDOW_H:
                if last_gap == 0:
                    last_gap = gap

                # Seconde fois qu'on enregistre un gap, il faut donc le comparer au le précédent
                else:
                    # Le nouvel écart est similaire au précédent
                    if interv(gap, 30, 30, last_gap):
                        endSearch = True
                        last_gap = (gap + last_gap) / 2 # Moyenne des deux écarts pour être plus précis

                    else:
                        last_gap = gap

            i += 1

        # Détermination du type de ligne
        if i == nb - 1 and last_gap == 0: # Ligne continue
            ltype = 1

        elif last_gap != 0: # Ligne pointillée
            ltype = int(last_gap/HIST_WINDOW_H) # Nombres de point d'écart

    return ltype


"""
Modélisation polynomiale ligne (gauche pour l'instant)
Avec le système de coordonées de la fenêtre opencv (x, y), f(y)=x où f le polynome

Retourne l'image et le rayon de courbure lorsque c'est pertinent
"""
def polynomialfit_line(frame, ptx, pty):
    ptx, pty = np.array(ptx), np.array(pty)
    poly = np.array(())

    rcurv = 0
    accurate = False

    if len(ptx) > 1:
        poly, res, _, _, _ = np.polyfit(pty, ptx, 2, full=True)

        # Seuil de précision de la modélisation (à étalonner + TODO: Moyen générique de le faire)
        if res < 2000:
            accurate = True
            polyfunc = np.poly1d(poly)  # Fonction d'évaluation du poly

            # Calcul courbure
            rcurv = (1 + (2*poly[0]*pty[0] + poly[1])**2)**(3/2)/np.abs(2*poly[0])

            # Coordonées x de la modélisation (y est déjà défini par pty)
            pt_poly_x = [int(polyfunc(pty[i])) for i in range(len(pty))]
            for ip in range(len(pt_poly_x)):
                frame = cv2.circle(
                    frame, (pt_poly_x[ip], pty[ip]), 5, c_red, -1)

            if not VIDEO_MODE:
                """
                # Validation graphique de la modélisation polynomiale
                print("Polynome: {}, Residus: {}".format(poly, res))

                fig, ax = plt.subplots()
                ax.plot(ptx, pty, "go")
                ax.plot(pt_poly_x, pty, "r.")
                # Pour inverser l'axe y (comme sur la birdview)
                ax.set_ylim(frame.shape[0], 0)
                ax.legend(["Points détectés tracés",
                           "Modélisation polynomiale"])
                ax.set_title("Détection/Modèle pour tracés")
                """
                pass

                # Le plt.show est effectué dans le script "main_full.py"

    return frame, rcurv, accurate


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("{}, {}".format(x, y))
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            RECT_BVIEW[1][0] += 5
        else:
            RECT_BVIEW[1][0] -= 5

        print("Current: ", RECT_BVIEW[1][0])

"""
Détermine l'écart entre le centre du véhicule et le centre calculé de la voie
Peut également afficher un curseur pour visualiser cet écart
Couleur du curseur:
    rouge pour aucune ligne détectée
    vert pour une ligne
    magenta pour deux lignes

Arguments:
    img: ...
    line_ctrs: Tuple (gauche, droit) sur la coordonnée x des points à considérer des lignes
                On attend -1 sur la/les coordonnée(s) où l'on a pas trouvé de ligne
"""
def car_path_offset(frame, line_ctrs):
    off = 0
    ln_ctrs = line_ctrs[:]
    """
    Le problème de la détection basée sur qu'une seule ligne est le fait qu'on doit recourir
    à une largeur de voie fixée pour déterminer l'écart entre l'axe véhicule et le centre de la voie
    """
    if MONO_LINE:
        if ln_ctrs[0] != -1 or ln_ctrs[1] != -1:
            # Permet d'alléger le calcul de l'offset et de paramétrer la couleur en fonction de la qualité de la détection
            if -1 in ln_ctrs:
                ln_ctrs.pop(ln_ctrs.index(-1))

            off = int(sum(ln_ctrs)/len(ln_ctrs) - CAR_CTRX)

    else:
        if ln_ctrs[0] != -1 and ln_ctrs[1] != -1:
            off = int(sum(ln_ctrs)/len(ln_ctrs) - CAR_CTRX)


    return frame, off

"""
Permet d'interpréter le comportement du véhicule (changement de voie, sortie de route ou conservation de la trajectoire)
Arguments:
    frame: image sur laquelle afficher le texte
    offset: calculé par car_path_offset
    line_types: tuple du type de lignes
    prevBehav: Comportement précédent (permet de gérer les changements de voie)

Retourne:
    frame:
    behav: 0 pour comportement normal, 1 pour changement de voie,
        2 pour sortie de route (déterminé par rapport au type de ligne qu'on cross)
"""
def car_behavior(frame, offset, line_types, prevBehav):
    behav = 0
    if abs(offset) > OFFSET_LIMIT: # Changement de voie détecté
        behav = 1 # Changement de voie

        # Traversement de ligne continue ou de délimitation de route (pour l'instant calibré sur autoroute/nationale)
        if offset != 0:
            ind = 1 if offset < 0 else 0 # Côté de dépassement

            if prevBehav == 1: # Si on était déjà en changmement de voie
                # Poitilliés de côté Nationale/Autouroute ou continue
                if interv(line_types[ind], 2, 2, 8) or line_types[ind] == 1:
                    behav = 2  # Sortie de route

    return frame, behav

"""
Simple fonction de controle d'appartenance de la valeur var à l'intervalle défini par
[inf, sup] si rien n'est spécifié pour ctr
ou [ctr - inf, ctr + sup] sinon
"""
def interv(var, inf, sup, ctr=None):
    res = False
    if ctr is None:
        res = var >= inf and var <= sup
    else:
        res = var >= ctr - inf and var <= ctr + sup

    return res








