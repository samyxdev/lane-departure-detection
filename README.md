# Lane Departure Detection

## Faire marcher Kivy2 sur WSL2
### Partie WSL2
* Suivre l'autre tuto pour faire fonctionner correctement WSL2
* **Ne pas ajouter la ligne** `export LIBGL_ALWAYS_INDIRECT=1`
* Ajouter le repo : `sudo add-apt-repository ppa:oibaf/graphics-drivers`

### Partie Kivy2
* Installer kivy[full] avec pip
* Installer buildozer du git clone et installer ses dependencies
* Tenter un buildozer build puis configurer correctement l'android-sdk avec:
A ajouter dans le bashrc
`export ANDROID_SDK_ROOT=~/.buildozer/lien du sdk ...`
`export PATH=$PATH:$ANDROID_SDK_ROOT/tools`
puis `source ~/.bashrc`

Erreur de "OpenCV requires Android SDK Tools revision 14 or newer.":
Si c'est pas fixé par les export d'en haut alors :
Il trouve pas les anciens tools du android sdk qu'il utilise donc

    Download cmdlines-tools from google
    Create a directory for the android sdk at buildozer android location:
    mkdir ~/.buildozer/android/platform/android-sdk
    Move the zip to this folder and unzip it
    Rename the folder
    mv tools old-tools
    Install missing dependencies & the famous tools:
    sudo ./cmd-lines/bin/sdkmanager --sdk_root=/home/<USERNAME>/.buildozer/android/platform/android-sdk/ --install "tools"
    sudo ./cmd-lines/bin/sdkmanager --sdk_root=/home/<USERNAME>/.buildozer/android/platform/android-sdk/ --install "build-tools;29.0.0-rc3"
(le reste est pas nécessaire mais dispo ici:
https://github.com/kivy/buildozer/issues/1144#issuecomment-655548082)
