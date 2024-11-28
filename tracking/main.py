# ---------MAIN--------------
import time
import cv2
import numpy as np
from camera import initialiser_cam, capture_frame, stop_camera
from visu import afficher_visu, fermer_visu, dessin_aruco_partiel

picam2 = initialiser_cam()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters_create()

trackers = []
ids = None

try:
    while True:
        frame = capture_frame(picam2)
        if frame is None:
            break

        # Vérifier et convertir l'image en couleur avec 3 canaux
        #if len(frame.shape) == 2:
            # Image en niveaux de gris, conversion en BGR
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            # Image avec 4 canaux (par exemple, RGBA), conversion en BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        #if frame.shape[2] == 1:
            # Image avec 1 canal, conversion en BGR
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Sinon, l'image a déjà 3 canaux (BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            trackers = []
            for corner in corners:
                # Conversion des coins en entiers
                corner_int = corner.astype(int)
                # Calcul du rectangle englobant
                x, y, w, h = cv2.boundingRect(corner_int)
                bbox = (int(x), int(y), int(w), int(h))  # Assurez-vous que bbox est composé d'entiers natifs
                # Création du tracker en utilisant cv2.legacy
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)
                trackers.append(tracker)
        else:
            corners = []
            for tracker in trackers:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = bbox
                    # Création des coins à partir du bounding box
                    corner = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
                    corners.append(corner)
            if corners:
                corners = np.array(corners)
            else:
                corners = None

        frame = dessin_aruco_partiel(frame, corners)
        afficher_visu("ArUco Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_camera(picam2)
    fermer_visu()
