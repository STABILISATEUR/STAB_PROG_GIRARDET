import cv2
import numpy as np

def recuperer_centre_marqueur(corners):
    """
    Calcule le centre d'un marqueur ArUco à partir de ses coins.

    Arg:
        Tableau contenant les coins du marqueur, de forme (1, 4, 2), 
        où 1 = nb marqueur 4 = coins et 2 = coordonnées (x, y).

    Ret:
        Coordonnées (x, y) du centre du marqueur.
    """

    corners_array = corners[0]  # on prend le premier élément
    pts = corners_array.reshape(-1, 2)  # Transformation en tableau de points [4, 2]
    x = pts[:, 0].mean()  # Moyenne des coordonnées x
    y = pts[:, 1].mean()  # Moyenne des coordonnées y
    return int(x), int(y)  # Retourner le centre sous forme de tuple (entiers)

