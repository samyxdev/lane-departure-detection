# coding: utf-8
"""
Fonctions de calculs de haut niveau,
liées à l'interpretation mathématiques et factuelles des tracés
"""

import numpy as np
import cv2

from consts import *

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