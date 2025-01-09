import cv2
import numpy as np

def afficher_visu(nom_fenetre, frame):
    cv2.imshow(nom_fenetre, frame)

def dessin_aruco(frame, corners, ids, color=(0, 0, 255), thickness=2):
    """
    Dessin des contours du marqueur
    Args:
        frame (numpy.ndarray): Image sur laquelle dessiner les marqueurs.
        corners (numpy.ndarray): Coins des marqueurs détectés,
            chaque élément contient 4 coordonnées (x, y).
        ids (numpy.ndarray | None): Identifiants des marqueurs, ou None si aucun détecté.
        color (tuple): Couleur des contours (B, G, R), par défaut rouge.
        thickness (int): Épaisseur des contours, par défaut 2.
    """
    if ids is not None:
        for corner in corners:
            pts = corner.reshape(-1, 2).astype(int)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame


def fermer_visu():
    cv2.destroyAllWindows()

def texte_sur_frame(frame, text, position, color=(255, 255, 255), font_scale=1, thickness=2):
    """
    Ajoute du texte sur une image.

    Entrée:
        frame : L'image sur laquelle écrire.
        text : Texte à afficher.
        position : Position (x, y) du texte.
        color : Couleur du texte (B, G, R).
        font_scale : Taille du texte.
        thickness : Épaisseur du texte.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def dessiner_marqueurs(frame, x, y):
    """
    Dessine un marqueur vert au point (320, 240) et un marqueur rouge au point (x, y).
    
    Args:
        frame (numpy.ndarray): L'image sur laquelle dessiner les marqueurs.
        x (int): Coordonnée X du marqueur rouge.
        y (int): Coordonnée Y du marqueur rouge.
    """
    # Dessiner un cercle vert (BGR = (0,255,0)) au point (320, 240)
    cv2.circle(frame, (320, 240), 5, (0, 255, 0), -1)
    
    # Dessiner un cercle rouge (BGR = (0,0,255)) au point (x, y)
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
