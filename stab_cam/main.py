import time
import cv2
from camera import initialiser_cam, capture_frame, stop_camera
from visu import afficher_visu, dessin_aruco, texte_sur_frame, fermer_visu
from utils import recuperer_centre_marqueur

# Initialisation de la caméra
picam2 = initialiser_cam()

# Initialisation des paramètres ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters_create()

# FPS
fps = 30
delai = 1.0/fps

try:
    while True:
        temps_debut = time.time()
        frame = capture_frame(picam2)

        if frame is None or frame.size == 0:
            print("Erreur d'image")
            break

        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Détection des marqueurs ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Annotation des marqueurs détectés
        frame = dessin_aruco(frame, corners, ids)

        if ids is not None:
            print("Marqueurs détectés : ", ids.flatten())
            centre = recuperer_centre_marqueur(corners)
            corners_array = corners[0]
            pts = corners_array.reshape(-1, 4)
            print("Position des coins du marqueur :")
            print(pts) 
            print("Centre du marqueur :", centre)  

        else:
            print("Aucun marqueur détecté")
        
        cv2.imshow("Visualisation ArUco", frame)

        temps_passe = time.time() - temps_debut
        time.sleep(max(0, delai - temps_passe))  # Ajuste le délai pour respecter le FPS cible

     

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_camera(picam2)
    fermer_visu()
