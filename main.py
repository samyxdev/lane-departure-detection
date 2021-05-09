# coding: utf-8
"""
Script réservé à l'execution sous kivy -> Android

Contient la mise en page des différentes pages possibles de l'appli

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

