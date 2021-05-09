"""
Marche bel et bien sur Android !
"""
import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.camera import Camera

from android.permissions import request_permissions, Permission

class MainApp(App):
    def build(self):
        self.cam = Camera(play=True, index=0, resolution=(1280,720))
        return self.cam

if __name__== "__main__":
    request_permissions([Permission.CAMERA])
    MainApp().run()