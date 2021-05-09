# coding: utf-8
"""
Script réservé à l'execution sous kivy -> Android

Contient la mise en page des différentes pages possibles de l'appli

Exemple d'intégration opencv à kivy:
https://github.com/liyuanrui/kivy-for-android-opencv-demo/blob/master/main.py

Pour lancer Kivy sur Windows natif : conda run ./main.py
Pour lancer Kivy sur WSL : classique
Pour générer l'APK : Sur WSL : buildozer android debug deploy run logcat
Quand aucun adb device n'est reconnu : adb kill-server puis adb start-server
Pour clean les précedents builds : buildozer appclean

Pour entrer dans le venv : source ~/.storevirtualenvs/kivy_env/bin/activate

WIP : Fix le crash sur Android
    -> Tester les permissions (qui font marcher la cam avec kivy directement sans cv2)
    -> Vu que l'ouverture de la cam marche avec kivy, on va récup la frame avec kivy
    et la passer ensuite à opencv -> Trop compliqué (et risque de ralentir le tout)
    -> On reste sur ce script et on essaye de débug cv2.VideoCapture
    2. Plus de crash mais la capture n'est pas fonctionnelle (isOpened() = False)
    -> On a supprimé les opencv_incompat_tools pour gradle mais il y a quand même
    un message étrange avec l'utilisation de kivy 1.11 (alors qu'on a 2.0)
    -> Potentiellement un recipe de Buildozer qui télécharge une mauvaise
    version de kivy
    -> Chemin des recipes : kivy_android_test\.buildozer
    \ android\platform\python-for-android\pythonforandroid\ recipes\kivy
    On a modifié pour forcer l'usage de kivy 2.0.0 et on fait buildozer appclean
    -> On peut en fait spécifier la version de kivy qu'on veut dans buildozer.spec
    avec "kivy==2.0.0" à la place de "kivy" sur la ligne requirements
    -> On va aussi passer à WSL 2 pour aller plus vite

TODO:
x Lancer le main_win.py sur WSL
x Lancer le main_kivy.py sur Windows Natif (Conda)
x Adapter le buildozer.spec (Permission, requierements)
x Opencv:
    x Installer opencv sous wsl
    x Compiler l'APK avec opencv

- Pouvoir lancer le script de test sur vidéo sur android
- Intégrer l'usage de la camera
    (après permissions)
    x Avec kivy (pour tester mais inutilisable pour la suite)
    - Avec cv2.VideoCapture (ou autre chose ?)
- Pouvoir choisir les points d'ancrage de la birdview sur le tel
    - Mettre en place un bouton permettant la mise en pause du filtrage cv2
    (qui laisse donc la video intacte, sans birdview etc.)
    - Permet avec 4 pressions de définir la zone de birdview et les enregistrer
    dans l'internal_storage
    - Pouvoir selectionner la config de points d'ancrage au début de l'exec ?
- Optimiser la conversion de frame opencv à Texture Kivy

x Changer l'icone de l'appli
x Changer le splashscreen

Notes :
Pour utiliser la caméra, il semble être nécessaire de demander la permission
à l'éxecution (en plus de le spécifier dans le buildozer.spec) avec:
from android.permissions import request_permissions, Permission
request_permissions([Permission.CAMERA])
"""

import cv2
import os

#os.environ['KIVY_GL_BACKEND'] = 'glew'

from consts import *
from cv_funcs import *
from kivy_funcs import *

import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.utils import platform

# Pour check que c'est la dernière version de kivy malgré les recipes 1.11.1 de buildozer
REQUIRED_KIVY = "2.0.0"

try:
    kivy.require(REQUIRED_KIVY)
except:
    print(f"Unexpected kivy version {kivy.__version__}, expected {REQUIRED_KIVY}")

isAndroid = platform == "android"
if isAndroid:
    from android.permissions import request_permissions, Permission
    from android.storage import app_storage_path, primary_external_storage_path

class VideoTesting(Image):
    """Page de test sur vidéos (pas sur caméra)
    Pour l'instant, les tests seront limités à la birdview
    """
    def __init__(self, capture, fps, **kwargs):
        Image.__init__(self, **kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()

        if ret:
            out_frame = pipeline_v2(frame, onlyBirdview=True)

            # Pour passer de la frame cv à la texture kivy
            buf = cv2.flip(out_frame, 0).tostring()
            image_texture = Texture.create(
                size=(out_frame.shape[1], out_frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

class MainApp(App):
    """ Classe d'initialisation de l'application
    """
    def build(self):
        printd("MainApp build ...")

        if isAndroid:
            printd("Requesting permissions...")
            request_permissions([Permission.CAMERA])

        int_video_path = os.path.join(app_storage_path(), "app", VIDEO_PATH) if isAndroid else VIDEO_PATH

        if not os.path.exists(int_video_path):
            printd(f"Path {int_video_path} not found !")
        else:
            printd(f"Path {int_video_path} exists !")

        #availableVideoSources()

        self.capture = cv2.VideoCapture(os.path.realpath(int_video_path))
        #self.capture = cv2.VideoCapture(VIDEO_PATH)
        #self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():
            printd("cv2 VideoCapture initialised !")
        else:
            printd("cv2 VideoCapture failed to init...")
            return Label(text="cv2 VideoCapture failed to init...")

        video = VideoTesting(self.capture,
            self.capture.get(cv2.CAP_PROP_FPS) if not FORCE_FPS else FORCE_FPS)

        printd("Capture set up")

        return video

    def on_stop(self):
        if VIDEO_MODE:
            self.capture.release()

if __name__ == '__main__':
    MainApp().run()

