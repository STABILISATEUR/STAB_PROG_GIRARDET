import time
import cv2
import numpy as np

from camera import initialiser_cam, capture_frame, stop_camera
from visu import fermer_visu, dessin_aruco, dessiner_marqueurs, texte_sur_frame  # Ajout de texte_sur_frame
from utils import recuperer_centre_marqueur
from moteur_scs15 import initialiser_servo, envoyer_position, deg_to_pos

#------------------------------------------------------------------------
# Paramètres généraux
#------------------------------------------------------------------------

# IDs des servos
ID_SERVO_PITCH = 1
ID_SERVO_YAW = 2

# Taille du carré ArUco en cm
ARUCO = 5

# Paramètres du contrôleur PI
KP_PITCH = 1.0   # Gain proportionnel pour le Pitch (ajusté)
KI_PITCH = 0.2   # Gain intégral pour le Pitch (ajusté)
KP_YAW   = 15.0  # Gain proportionnel pour le Yaw (ajusté)
KI_YAW   = 0.05  # Gain intégral pour le Yaw (ajusté)

# Limite de l'intégrale (Anti-Windup)
INTEGRAL_LIMIT = 20.0

# Angle neutre et limites du servo
ANGLE_NEUTRE = 100.0
PITCH_MIN = 10.0   # Limite minimale ajustée
PITCH_MAX = 190.0  # Limite maximale ajustée
YAW_MIN = 20.0      # Limite minimale ajustée 
YAW_MAX = 180.0     # Limite maximale ajustée

# Zone morte (en pixels) pour ignorer de petites erreurs
ACCEPTANCE_X = 3
ACCEPTANCE_Y = 3

# Paramètres de tracking
MAX_PERTE = 15  # Nombre d'images maximum sans détection avant recadrage
REF_PIXELS = 20
REF_DISTANCE = 0.5
MIN_GAIN = 0.1
MAX_GAIN = 100

# Timeout pour annuler le tracking (en secondes)
TRACKING_TIMEOUT = 2.0

# Configuration ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters_create()

#------------------------------------------------------------------------
# Classe du contrôleur PI
#------------------------------------------------------------------------
class PIController:
    def __init__(self, kp, ki, integral_limit):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.last_time = time.time()
    
    def reset(self):
        """Réinitialise l'intégrale et le temps."""
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error):
        """
        Calcule la sortie du contrôleur PI
        Args:
            error: erreur actuelle (consigne - mesure)
        Returns:
            float: commande calculée
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt < 0.01:  # Protection contre un dt trop petit
            dt = 0.01
            
        # Mise à jour de l'intégrale (anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Calcul de la commande
        output = (self.kp * error) + (self.ki * self.integral)
        
        self.last_time = current_time
        return output

#------------------------------------------------------------------------
# Programme principal
#------------------------------------------------------------------------
def main():
    # Initialisation de la caméra et des servos
    picam2 = initialiser_cam()
    servo = initialiser_servo()

    # Création des contrôleurs PI
    pi_pitch = PIController(KP_PITCH, KI_PITCH, INTEGRAL_LIMIT)
    pi_yaw   = PIController(KP_YAW,   KI_YAW,   INTEGRAL_LIMIT)

    # Mise en position neutre initiale
    pos_init = deg_to_pos(ANGLE_NEUTRE)
    envoyer_position(servo, ID_SERVO_PITCH, pos_init)
    envoyer_position(servo, ID_SERVO_YAW,   pos_init)
    time.sleep(1)

    # Initialisation des angles actuels
    current_pitch = ANGLE_NEUTRE
    current_yaw = ANGLE_NEUTRE

    # Première capture pour connaître la taille de l'image
    frame = capture_frame(picam2)
    if frame is None or frame.size == 0:
        print("[ERREUR] Impossible de capturer une image. Arrêt.")
        stop_camera(picam2)
        return

    # La caméra est à l'envers, on retourne l'image de 180°
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Centre de l'image
    x_image = frame.shape[1] // 2
    y_image = frame.shape[0] // 2

    # Initialisation de l'enregistrement vidéo
    # ----------------------------------------------------
    # Définir le codec et créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Vous pouvez choisir un autre codec si nécessaire
    out = cv2.VideoWriter('visualisation.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    # ----------------------------------------------------

    # Tracker initialement nul
    tracker = None
    dernier_id_marqueur = None
    marker_perdu_compteur = 0

    # Contrôle de la fréquence des mises à jour
    dernier_update = time.time()

    # Initialisation du timestamp de la dernière détection
    last_detection_time = time.time()

    try:
        while True:
            frame = capture_frame(picam2)
            if frame is None or frame.size == 0:
                print("[ERREUR] Impossible de capturer une image. Arrêt.")
                break

            # Retourner l'image pour compenser la caméra à l’envers
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Conversion RGBA -> BGR si nécessaire
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Détection ArUco
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coins, ids, _ = cv2.aruco.detectMarkers(gris, aruco_dict, parameters=parameters)

            correction_yaw = 0
            correction_pitch = 0

            current_time = time.time()

            if ids is not None and len(ids) > 0:
                # On a détecté au moins un marqueur
                id_marqueur = ids[0][0]
                coin_marqueur = coins[0]
                centre = recuperer_centre_marqueur([coin_marqueur])
                
                # Dessin du marqueur détecté en vert
                frame = dessin_aruco(frame, [coin_marqueur], ids[:1], color=(0, 255, 0))

                if centre is not None:
                    # Calcul des deltas par rapport au centre de l'image
                    delta_x = centre[0] - x_image
                    delta_y = centre[1] - y_image

                    # Zone morte
                    if abs(delta_x) < ACCEPTANCE_X:
                        delta_x = 0
                    if abs(delta_y) < ACCEPTANCE_Y:
                        delta_y = 0

                    # Normalisation pour rester ~[-1, +1]
                    erreur_x = float(delta_x) / float(x_image)
                    erreur_y = float(delta_y) / float(y_image)

                    # Vérifier si 0.2 secondes se sont écoulées depuis la dernière mise à jour
                    if (current_time - dernier_update) >= 0.2:
                        # Mise à jour des contrôleurs PI
                        correction_yaw   = pi_yaw.update(erreur_x)
                        correction_pitch = pi_pitch.update(erreur_y)

                        # Debug des corrections
                        print(f"[DEBUG] Correction Yaw: {correction_yaw:.3f}, Correction Pitch: {correction_pitch:.3f}")

                        # Mise à jour des angles actuels
                        # ** Ajustement des signes pour une direction correcte **
                        # MODIFICATION : Utiliser += au lieu de -= pour cohérence
                        current_pitch -= correction_pitch  # Augmente pour regarder vers le haut
                        current_yaw   += correction_yaw    # Augmente pour tourner vers la droite

                        # Limiter les angles pour éviter le dépassement du servo
                        current_pitch = np.clip(current_pitch, PITCH_MIN, PITCH_MAX)
                        current_yaw   = np.clip(current_yaw, YAW_MIN, YAW_MAX)  # Limiter Yaw si nécessaire

                        # Debug des angles
                        print(f"[INFO] ArUco={id_marqueur} | ErX={erreur_x:.3f} ErY={erreur_y:.3f} | "
                              f"Pitch={current_pitch:.2f} Yaw={current_yaw:.2f}")

                        # Conversion en positions internes et envoi
                        pos_pitch = deg_to_pos(current_pitch)
                        pos_yaw   = deg_to_pos(current_yaw)
                        envoyer_position(servo, ID_SERVO_PITCH, pos_pitch)
                        envoyer_position(servo, ID_SERVO_YAW,   pos_yaw)

                        # --------------------------
                        # Ajout de l'affichage des commandes moteurs
                        texte_sur_frame(
                            frame, 
                            f"Pitch: {current_pitch:.2f}", 
                            (10, 30),  # Position en haut à gauche
                            color=(255, 0, 0),  # Bleu
                            font_scale=0.7, 
                            thickness=2
                        )
                        texte_sur_frame(
                            frame, 
                            f"Yaw: {current_yaw:.2f}", 
                            (10, 60),  # Position en dessous du Pitch
                            color=(255, 0, 0),  # Bleu
                            font_scale=0.7, 
                            thickness=2
                        )
                        # --------------------------

                        dernier_update = current_time  # Mettre à jour le timestamp

                    # Mise à jour du timestamp de la dernière détection
                    last_detection_time = current_time

                    # Affichage visuel : centre de l'image et centre du marqueur
                    dessiner_marqueurs(frame, x_image, y_image)
                    dessiner_marqueurs(frame, centre[0], centre[1])

                    # --------------------------
                    # Ajout de l'affichage de l'ID
                    if centre is not None:
                        texte_sur_frame(
                            frame, 
                            f"ID: {id_marqueur}", 
                            (centre[0] + 10, centre[1] - 10),  # Position ajustée légèrement par rapport au centre
                            color=(0, 255, 0),  # Vert
                            font_scale=0.5, 
                            thickness=2
                        )
                    # --------------------------

                # Mise à jour d'une zone_suivi pour le tracker
                x_tr, y_tr, w_tr, h_tr = cv2.boundingRect(coin_marqueur.astype(int))
                zone_suivi = (x_tr, y_tr, w_tr, h_tr)

                if tracker is None:
                    # Initialisation du tracker KCF
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, zone_suivi)
                else:
                    # Ré-initialiser le tracker avec la nouvelle zone_suivi
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, zone_suivi)

                dernier_id_marqueur = id_marqueur
                marker_perdu_compteur = 0

            else:
                # Vérifier si le tracking doit être annulé
                if tracker is not None and (current_time - last_detection_time) > TRACKING_TIMEOUT:
                    print("[INFO] Timeout atteint. Annulation du tracking.")
                    tracker = None
                    dernier_id_marqueur = None
                    # Réinitialisation des contrôleurs PI et servos
                    pi_pitch.reset()
                    pi_yaw.reset()
                    current_pitch = ANGLE_NEUTRE
                    current_yaw = ANGLE_NEUTRE
                    pos_neutre = deg_to_pos(ANGLE_NEUTRE)
                    envoyer_position(servo, ID_SERVO_PITCH, pos_neutre)
                    envoyer_position(servo, ID_SERVO_YAW,   pos_neutre)
                    continue  # Passer à l'itération suivante

                # Marqueur non détecté => tentative de fallback via tracker
                if tracker is not None:
                    success, zone_suivi = tracker.update(frame)
                    if success:
                        x_tr, y_tr, w_tr, h_tr = zone_suivi
                        # Centre estimé de la zone trackée
                        centre_x = int(x_tr + w_tr / 2)
                        centre_y = int(y_tr + h_tr / 2)

                        delta_x = centre_x - x_image
                        delta_y = centre_y - y_image

                        if abs(delta_x) < ACCEPTANCE_X:
                            delta_x = 0
                        if abs(delta_y) < ACCEPTANCE_Y:
                            delta_y = 0

                        erreur_x = float(delta_x) / float(x_image)
                        erreur_y = float(delta_y) / float(y_image)

                        # Vérifier si 0.2 secondes se sont écoulées depuis la dernière mise à jour
                        if (current_time - dernier_update) >= 0.2:
                            # Mise à jour des contrôleurs PI
                            correction_yaw   = pi_yaw.update(erreur_x)
                            correction_pitch = pi_pitch.update(erreur_y)

                            # Debug des corrections
                            print(f"[DEBUG] Tracking Correction Yaw: {correction_yaw:.3f}, Correction Pitch: {correction_pitch:.3f}")

                            # Mise à jour des angles actuels
                            # MODIFICATION : Utiliser += au lieu de -= pour cohérence
                            current_pitch -= correction_pitch  # Augmente pour regarder vers le haut
                            current_yaw   += correction_yaw    # Augmente pour tourner vers la droite

                            # Limiter les angles pour éviter le dépassement du servo
                            current_pitch = np.clip(current_pitch, PITCH_MIN, PITCH_MAX)
                            current_yaw   = np.clip(current_yaw, YAW_MIN, YAW_MAX)  # Limiter Yaw si nécessaire

                            # Debug des angles
                            print(f"[INFO] Tracking | ErX={erreur_x:.3f} ErY={erreur_y:.3f} | "
                                  f"Pitch={current_pitch:.2f} Yaw={current_yaw:.2f}")

                            # Conversion en positions internes et envoi
                            pos_pitch = deg_to_pos(current_pitch)
                            pos_yaw   = deg_to_pos(current_yaw)
                            envoyer_position(servo, ID_SERVO_PITCH, pos_pitch)
                            envoyer_position(servo, ID_SERVO_YAW,   pos_yaw)

                            # --------------------------
                            # Ajout de l'affichage des commandes moteurs
                            texte_sur_frame(
                                frame, 
                                f"Pitch: {current_pitch:.2f}", 
                                (10, 30),  # Position en haut à gauche
                                color=(255, 0, 0),  # Bleu
                                font_scale=0.7, 
                                thickness=2
                            )
                            texte_sur_frame(
                                frame, 
                                f"Yaw: {current_yaw:.2f}", 
                                (10, 60),  # Position en dessous du Pitch
                                color=(255, 0, 0),  # Bleu
                                font_scale=0.7, 
                                thickness=2
                            )
                            # --------------------------

                            dernier_update = current_time  # Mettre à jour le timestamp

                        # Dessin d'un rectangle rouge suivi
                        coin_suivi = np.array([
                            [x_tr,        y_tr],
                            [x_tr + w_tr, y_tr],
                            [x_tr + w_tr, y_tr + h_tr],
                            [x_tr,        y_tr + h_tr]
                        ], dtype=np.float32)
                        frame = dessin_aruco(frame, [coin_suivi],
                                             np.array([[dernier_id_marqueur if dernier_id_marqueur else -1]]),
                                             color=(0, 0, 255))

                        # --------------------------
                        # Optionnel : Afficher l'ID même en tracking
                        if dernier_id_marqueur is not None:
                            texte_sur_frame(
                                frame, 
                                f"ID: {dernier_id_marqueur}", 
                                (centre_x + 10, centre_y - 10),  # Position ajustée
                                color=(0, 255, 0),  # Vert
                                font_scale=0.5, 
                                thickness=2
                            )
                        # --------------------------

                        # Mise à jour du timestamp de la dernière détection
                        marker_perdu_compteur += 1

                    else:
                        # Tracker a échoué
                        marker_perdu_compteur += 1
                        if marker_perdu_compteur > MAX_PERTE:
                            print("[INFO] Tracker a échoué. Remise à zéro.")
                            # Remise à zéro
                            pi_pitch.reset()
                            pi_yaw.reset()
                            current_pitch = ANGLE_NEUTRE
                            current_yaw = ANGLE_NEUTRE
                            pos_neutre = deg_to_pos(ANGLE_NEUTRE)
                            envoyer_position(servo, ID_SERVO_PITCH, pos_neutre)
                            envoyer_position(servo, ID_SERVO_YAW,   pos_neutre)
                            tracker = None
                            dernier_id_marqueur = None
                else:
                    # Pas de tracker => on incrémente simplement
                    marker_perdu_compteur += 1
                    if marker_perdu_compteur > MAX_PERTE:
                        # Recadrage
                        pi_pitch.reset()
                        pi_yaw.reset()
                        current_pitch = ANGLE_NEUTRE
                        current_yaw = ANGLE_NEUTRE
                        pos_neutre = deg_to_pos(ANGLE_NEUTRE)
                        envoyer_position(servo, ID_SERVO_PITCH, pos_neutre)
                        envoyer_position(servo, ID_SERVO_YAW,   pos_neutre)
                        tracker = None
                        dernier_id_marqueur = None

            # --------------------------
            # Ajout de l'enregistrement du cadre
            out.write(frame)
            # --------------------------

            # Affichage
            cv2.imshow("Visualisation", frame)

            # Sortie si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Libération des ressources
        stop_camera(picam2)
        fermer_visu()
        # --------------------------
        # Libération de l'enregistrement vidéo
        out.release()
        # --------------------------

if __name__ == "__main__":
    main()
