import time
import cv2
import numpy as np 
from camera import initialiser_cam, capture_frame, stop_camera
from visu import afficher_visu, fermer_visu, dessin_aruco
from utils import recuperer_centre_marqueur
from moteur_scs15 import initialiser_servo, envoyer_position, correcteur_p

# Initialisation caméra
picam2 = initialiser_cam()

# Initialisation UART pour les moteurs
servo = initialiser_servo()
id_pitch = 1  # ID du moteur pour le pitch
id_roll = 2   # ID du moteur pour le roll

# Gains du correcteur P
gain_pitch = 0.1
gain_roll = 0.1

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

        # Vérification de l'image
        if frame is None or frame.size == 0:
            print("Erreur d'image")
            break

        # Conversion RGBA vers BGR 
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Marqueur détecté
            print("Marqueur détecté :", ids[0][0])
            pre_id = ids[0][0]  # Sauvegarde de l'ID
            corner = corners[0]
            centre = recuperer_centre_marqueur([corner])
            print("Centre du marqueur :", centre)

            # Dessin du marqueur en vert
            frame = dessin_aruco(frame, [corner], ids[:1], color=(0, 255, 0))
            
            if centre:
                # Calcul des erreurs
                erreur_x = centre[0] - frame.shape[1] // 2  # Erreur horizontale
                erreur_y = centre[1] - frame.shape[0] // 2  # Erreur verticale

                # Calcul des corrections
                correction_roll = correcteur_p(erreur_x, gain_roll)
                correction_pitch = correcteur_p(erreur_y, gain_pitch)

                # Envoi des nouvelles positions aux moteurs
                envoyer_position(servo, id_roll, 100 + correction_roll)  # Centré à 100
                envoyer_position(servo, id_pitch, 100 + correction_pitch)

            # Mise à jour du tracker avec le bounding box du marqueur
            corner_int = corner.astype(int)
            x, y, w, h = cv2.boundingRect(corner_int)
            bbox = (x, y, w, h)

            if tracker is None:
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)

        elif tracker is not None:
            # Suivi du marqueur
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = bbox
                corner = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
                centre = ((x + x + w) // 2, (y + y + h) // 2)

                # Calcul des erreurs
                erreur_x = centre[0] - frame.shape[1] // 2  # Erreur horizontale
                erreur_y = centre[1] - frame.shape[0] // 2  # Erreur verticale

                # Calcul des corrections
                correction_roll = correcteur_p(erreur_x, gain_roll)
                correction_pitch = correcteur_p(erreur_y, gain_pitch)

                # Envoi des nouvelles positions aux moteurs
                envoyer_position(servo, id_roll, 100 + correction_roll)
                envoyer_position(servo, id_pitch, 100 + correction_pitch)

                # Dessin du marqueur suivi en rouge
                frame = dessin_aruco(frame, [corner], np.array([[pre_id]]), color=(0, 0, 255))
                print("Tracking en cours pour le marqueur :", pre_id)
            else:
                # Suivi échoué
                tracker = None
                pre_id = None
                print("Suivi perdu. Recentrage des moteurs")
                
                # Recentrer les moteurs
                envoyer_position(servo, id_roll, 100)
                envoyer_position(servo, id_pitch, 100)
        else:
            # Aucun marqueur détecté ni suivi actif
            print("Aucun marqueur détecté ou suivi. Recentrez les moteurs.")
            
            # Recentrer les moteurs
            envoyer_position(servo, id_roll, 100)
            envoyer_position(servo, id_pitch, 100)

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
