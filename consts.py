import numpy as np

# Triplets de couleur
c_red = (0, 0, 255)
c_green = (0, 255, 0)
c_purple = (255, 0, 255)

# Axe du véhicule (pour calculer la déviation par rapport au centre de la voie)
CAR_CTRX, CAR_CTRY = 620, 660

# Birdview zone: Top left, top right, bot right, bot left
#RECT_BVIEW = np.array([(495, 467), (692, 467), (1213, 719), (79, 719)], dtype="float32") # Légérement espacé des lignes
RECT_BVIEW = np.array([[475, 448], [704, 448], [1280, 660], [0, 660]], dtype="float32")
RECT_LINES = np.array([(507, 466), (680, 466), (1150, 717), (142, 717)], dtype="float32") # Collé au lignes

# Hauteur de la fênetre glissante de détection des tracés
HIST_WINDOW_H = 15

# Dossier des données de calibration
CALIB_FOLDER = "calib/calib_s7"
CALIB_FILE = "{}_data.json".format(CALIB_FOLDER, CALIB_FOLDER)
CALIB_FORMAT = (1280, 720) # Doit correspondre au format de la vidéo/photo d'expérimentation ?

# Paramètres détection offset
# Booléen de sélection du mode de détection : une ligne ou deux (voir note dans car_path_offset)
MONO_LINE = False
# Si activé, l'offset ne sera que calculé lorsque la modélisation a pu être faite et était assez précise
ONLY_FITTED_LINE = False
# Offset limite pour considérer un changement de voie (ou sortie de route) - A calibrer (en pixel)
OFFSET_LIMIT = 150

# Pour le script principal
# 0 pour le mode image, 1 pour les vidéos, 2 pour une cam
VIDEO_MODE = 2

IMAGE_PATH = "sortie_1_red.png"

# Chemin de la vidéo
VIDEO_PATH = "autoroute.avi"
WEBCAM_PATH = "http://192.168.0.30:8080/video"

# Pour utiliser la pipeline de preprocessing à la place de la pipeline de production
PIPLELINE_TEST = 0

# Pour utiliser les états précédents de la voiture pour anticiper la sortie de route
DISABLE_CAR_STATE = True

# Forcer les fps dans le script kivy (mettre à 0 pour désactiver le FORCE
FORCE_FPS = 45

# Pour afficher les messages de debug de nos scripts
DEBUG_PRINT = True