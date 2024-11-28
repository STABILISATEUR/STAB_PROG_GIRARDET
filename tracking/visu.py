import cv2
import numpy as np

def afficher_visu(nom_fenetre, frame):
    cv2.imshow(nom_fenetre, frame)

def dessin_aruco(frame, corners, ids):
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
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
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

def dessin_aruco_partiel(frame, corners, color=(0, 0, 255), thickness=2):
    """
    Dessine uniquement les parties visibles des marqueurs ArUco détectés.

    Args:
        frame (numpy.ndarray): L'image sur laquelle dessiner les marqueurs.
        corners (list ou numpy.ndarray): Les coins des marqueurs détectés.
        color (tuple): La couleur des contours.
        thickness (int): L'épaisseur des contours.

    Returns:
        numpy.ndarray: L'image avec les marqueurs dessinés partiellement.
    """
    if corners is None:
        return frame

    h, w = frame.shape[:2]
    for corner in corners:
        pts = corner.reshape(-1, 2).astype(int)
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]  # Prochain point (boucle fermée)
            # Vérifie si les deux points sont dans le champ de vision
            if (0 <= p1[0] < w and 0 <= p1[1] < h) or (0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(frame, tuple(p1), tuple(p2), color, thickness)
    return frame