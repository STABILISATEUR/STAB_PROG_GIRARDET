import cv2
import time
import numpy as np

# Import des fonctions depuis vos fichiers
from camera import initialiser_cam, capture_frame, stop_camera
from visu import afficher_visu, dessin_aruco, texte_sur_frame, fermer_visu
from utils import recuperer_centre_marqueur
from moteur_scs15 import initialiser_servo, envoyer_position, deg_to_pos

def main():
    """
    - Détecte un marqueur ArUco et trace ses contours en vert
    - Met en place un tracker CSRT pour suivre le marqueur (contour en rouge)
    - Pilote 2 moteurs SCS15 (daisy chain ID=1 pour pitch, ID=2 pour yaw)
      pour centrer la caméra sur le marqueur à l'aide d'un correcteur PI
      et d'une zone morte.
    """

    # --- 1) Initialisations ---
    picam2 = initialiser_cam()  # Démarrage de la caméra
    afficher_visu("AruCo Tracking")  # Fenêtre pour l'affichage
    
    # Initialisation des moteurs
    initialiser_servo()
    pitch_motor_id = 1
    yaw_motor_id   = 2

    # Valeurs initiales pour les moteurs (en degrés)
    pitch_angle = 0.0
    yaw_angle   = 0.0

    # Gains du correcteur PI (à ajuster selon votre setup)
    Kp = 0.2
    Ki = 0.01
    
    # Zone morte pour éviter de trop bouger les moteurs
    dead_zone = 10  # nombre de pixels d’erreur tolérée

    # Intégrateurs pour l’axe pitch et yaw
    integral_pitch = 0.0
    integral_yaw   = 0.0

    # Paramètres ArUco
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
    parameters   = cv2.aruco.DetectorParameters_create()

    # Initialisation du tracker
    tracker = None
    tracking = False
    bbox = None

    # Pour mesurer le temps entre deux itérations
    previous_time = time.time()

    print("Appuyez sur la touche 'q' pour quitter.")

    try:
        while True:
            # --- 2) Acquisition d’une frame ---
            frame = capture_frame(picam2)
            if frame is None:
                continue  # si la capture échoue, on recommence

            # On récupère les dimensions du frame
            frame_height, frame_width = frame.shape[:2]
            center_x = frame_width // 2
            center_y = frame_height // 2

            current_time = time.time()
            dt = current_time - previous_time
            previous_time = current_time

            # --- 3) Détection ArUco (si on n’est pas en tracking ou pour actualiser la position) ---
            if not tracking:
                # Détection des marqueurs
                corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
                
                if ids is not None and len(ids) > 0:
                    # On suppose ici qu’on ne suit qu’un seul marqueur, on prend le premier détecté
                    corners = corners[0]
                    # Dessin du contour en VERT
                    dessin_aruco(frame, [corners], color=(0, 255, 0))

                    # On récupère le bounding box (pour initialiser le tracker CSRT)
                    x_min = int(min(corners[0][:,0]))
                    y_min = int(min(corners[0][:,1]))
                    x_max = int(max(corners[0][:,0]))
                    y_max = int(max(corners[0][:,1]))
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    # Initialisation du tracker CSRT
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    tracking = True
                else:
                    # Aucun marqueur détecté : on continue
                    pass
            
            else:
                # --- 4) Tracking avec CSRT (contour en ROUGE) ---
                success, bbox = tracker.update(frame)
                if success:
                    # On récupère la bounding box trackée
                    x, y, w, h = [int(v) for v in bbox]
                    # Dessin du rectangle en rouge
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    # On peut ré-estimer (ou non) le centre du marqueur
                    # Ici on suppose que le centre du bounding box = centre du marqueur
                    marker_x = x + w//2
                    marker_y = y + h//2

                    # Correction de la position des moteurs
                    error_x = marker_x - center_x
                    error_y = marker_y - center_y

                    # Application de la zone morte
                    if abs(error_x) < dead_zone: 
                        error_x = 0
                    if abs(error_y) < dead_zone:
                        error_y = 0

                    # Correcteur PI sur le pitch (vertical) et yaw (horizontal)
                    integral_pitch += error_y * dt
                    integral_yaw   += error_x * dt

                    # Calcul de la consigne
                    cmd_pitch = Kp*error_y + Ki*integral_pitch
                    cmd_yaw   = Kp*error_x + Ki*integral_yaw

                    # Mise à jour des angles (on peut limiter la plage si nécessaire)
                    pitch_angle -= cmd_pitch  # l’axe pitch peut nécessiter un signe négatif
                    yaw_angle   += cmd_yaw

                    # Envoi des positions aux moteurs (en degrés -> position SCS15)
                    envoyer_position(pitch_motor_id, deg_to_pos(pitch_angle), time=0.2)
                    envoyer_position(yaw_motor_id, deg_to_pos(yaw_angle), time=0.2)

                else:
                    # Le tracker a perdu la cible
                    tracking = False
                    tracker = None

            # --- 5) Affichage ---
            texte = "Tracking: ON" if tracking else "Tracking: OFF"
            texte_sur_frame(frame, texte, org=(10,30), color=(255,255,255))
            
            # Affichage dans la fenêtre
            cv2.imshow("AruCo Tracking", frame)

            # Touche 'q' pour quitter
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    # --- 6) Nettoyage ---
    stop_camera(picam2)
    fermer_visu("AruCo Tracking")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
