import time
import cv2
import numpy as np  # pour les tableaux
from camera import initialiser_cam, capture_frame, stop_camera
from visu import afficher_visu, fermer_visu, dessin_aruco
from utils import recuperer_centre_marqueur

# Initialisation caméra
picam2 = initialiser_cam()

# Paramètres ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters_create()

# FPS 
fps = 30
delai = 1.0 / fps

# Tracker et ID précédent
tracker = None
pre_id = None

try:
    while True:
        temps_debut = time.time()
        frame = capture_frame(picam2)

        # Vérification 
        if frame is None or frame.size == 0:
            print("Erreur d'image")
            break

        # Conversion RGBA vers BGR 
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection du marqueur ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Marqueur détecté
            print("Marqueur détecté :", ids[0][0])
            pre_id = ids[0][0]  # Sauvegarde de l'ID
            tracker = None       # Réinitialisation du tracker

            # Obtention du premier marqueur détecté
            corner = corners[0]
            corner_int = corner.astype(int)           # Coins en entiers
            x, y, w, h = cv2.boundingRect(corner_int) # Rectangle
            bbox = (x, y, w, h)                       # Bounding box

            # Création et initialisation du tracker
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, bbox)

            # Centre du marqueur
            centre = recuperer_centre_marqueur([corner])
            print("Centre du marqueur :", centre)

            # Dessin du marqueur en vert
            frame = dessin_aruco(frame, [corner], ids[:1], color=(0, 255, 0))
        elif tracker is not None:
            # Utilisation du tracking
            success, bbox = tracker.update(frame)   # Mise à jour du tracker
            if success:
                x, y, w, h = bbox
                # Création des coins à partir du bounding box
                corner = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

                # Dessin du marqueur suivi en rouge
                frame = dessin_aruco(frame, [corner], np.array([[pre_id]]), color=(0, 0, 255))
                print("Tracking en cours pour le marqueur :", pre_id)
            else:
                # Suivi échoué
                tracker = None
                pre_id = None
                print("Aucun marqueur détecté ou suivi")
        else:
            # Aucun marqueur détecté ou suivi
            print("Aucun marqueur détecté ou suivi")

        # Affichage de la visualisation
        cv2.imshow("Visualisation ArUco", frame)

        # Gestion du FPS
        temps_passe = time.time() - temps_debut
        time.sleep(max(0, delai - temps_passe))

        # Sortie si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
        stop_camera(picam2)
        fermer_visu()
