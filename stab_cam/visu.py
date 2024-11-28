import cv2 

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
        text : texte a afficher 
        position : Position (x, y) du texte.
        color : Couleur du texte (B, G, R).
        font_scale : Taille du texte.
        thickness : Épaisseur du texte.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


