# Lane Departure Detection

## Lancement
* Pour lancer Kivy sur Windows natif : conda run ./main.py
* Pour lancer Kivy sur WSL : classique
* Pour générer l'APK : Sur WSL : buildozer android debug deploy run logcat

* Quand aucun adb device n'est reconnu : adb kill-server puis adb start-server
* Pour clean les précedents builds : buildozer appclean
* Pour entrer dans le venv : source ~/.storevirtualenvs/kivy_env/bin/activate

## WIP
Fix le crash sur Android
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

## TODO:
- [x] Lancer le main_win.py sur WSL
- [x] Lancer le main_kivy.py sur Windows Natif (Conda)
- [x] Adapter le buildozer.spec (Permission, requierements)
- [x] Opencv:
    - [x] Installer opencv sous wsl
    - [x] Compiler l'APK avec opencv

- [ ] Pouvoir lancer le script de test sur vidéo sur android
- [ ] Intégrer l'usage de la camera
    (après permissions)
    - [x] Avec kivy (pour tester mais inutilisable pour la suite)
    - [ ] Avec cv2.VideoCapture (ou autre chose ?)
- [ ] Pouvoir choisir les points d'ancrage de la birdview sur le tel
    - [ ] Mettre en place un bouton permettant la mise en pause du filtrage cv2
    (qui laisse donc la video intacte, sans birdview etc.)
    - [ ] Permet avec 4 pressions de définir la zone de birdview et les enregistrer
    dans l'internal_storage
    - [ ] Pouvoir selectionner la config de points d'ancrage au début de l'exec ?
- [ ] Optimiser la conversion de frame opencv à Texture Kivy

- [x] Changer l'icone de l'appli
- [x] Changer le splashscreen

## Notes
Exemple d'intégration opencv à kivy:
https://github.com/liyuanrui/kivy-for-android-opencv-demo/blob/master/main.py

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
