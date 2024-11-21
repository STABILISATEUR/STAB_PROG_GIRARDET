import cv2
import numpy as np

# Importer le module ArUco d'OpenCV
from cv2 import aruco

# Paramètres pour le marqueur
marker_size = 200  # Taille du marqueur
marker_id = 55     # ID du marqueur ArUco

# Créer le dictionnaire ArUco
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

# Créer le marqueur
marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Convertir en image RGBA
marker_image_rgba = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGBA)

# Sauvegarder l'image RGBA
cv2.imwrite("aruco_fixed.png", marker_image_rgba)

# Afficher le marqueur
cv2.imshow("Aruco Marker", marker_image_rgba)
cv2.waitKey(0)
cv2.destroyAllWindows()
