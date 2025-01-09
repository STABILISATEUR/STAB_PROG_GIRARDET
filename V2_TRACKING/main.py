import time
import cv2
import numpy as np

from camera import initialiser_cam, capture_frame, stop_camera
from visu import fermer_visu, dessin_aruco, dessiner_marqueurs
from utils import recuperer_centre_marqueur
from moteur_scs15 import initialiser_servo, envoyer_position, deg_to_pos

#------------------------------------------------------------------------
# Paramètres généraux
#------------------------------------------------------------------------

# IDs des servos
ID_SERVO_PITCH = 2
ID_SERVO_YAW = 1

# Taille du carré ArUco en cm
ARUCO = 5

# Paramètres du contrôleur PI
# (Ajustez ces gains si la caméra est très proche, ex. ~20 cm)
KP_PITCH = 5   # Gain proportionnel pour le Pitch
KI_PITCH = 0.1   # Gain intégral pour le Pitch
KP_YAW   = 10   # Gain proportionnel pour le Yaw
KI_YAW   = 0.2   # Gain intégral pour le Yaw

# Limite de l'intégrale (Anti-Windup)
INTEGRAL_LIMIT = 20.0

# Angle neutre et limites du servo
ANGLE_NEUTRE = 100.0
PITCH_MIN = 55.0
PITCH_MAX = 145.0

# Zone morte (en pixels) pour ignorer de petites erreurs
ACCEPTANCE_X = 3
ACCEPTANCE_Y = 3

# Paramètres de tracking
MAX_PERTE = 15  # Nombre d'images maximum sans détection avant recadrage
REF_PIXELS = 20
REF_DISTANCE = 0.5
MIN_GAIN = 0.1
MAX_GAIN = 100

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

    # Tracker initialement nul
    tracker = None
    dernier_id_marqueur = None
    marker_perdu_compteur = 0

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

                    # Mise à jour des contrôleurs PI
                    correction_yaw   = pi_yaw.update(erreur_x)
                    correction_pitch = pi_pitch.update(erreur_y)

                    # Angles finaux
                    #
                    angle_pitch = ANGLE_NEUTRE - correction_pitch
                    angle_yaw   = ANGLE_NEUTRE + correction_yaw

                    # Limiter le pitch (éviter dépassement du servo)
                    angle_pitch = np.clip(angle_pitch, PITCH_MIN, PITCH_MAX)

                    # Debug
                    print(f"[INFO] ArUco={id_marqueur} | ErX={erreur_x:.3f} ErY={erreur_y:.3f} | "
                          f"Pitch={angle_pitch:.2f} Yaw={angle_yaw:.2f}")

                    # Conversion en positions internes et envoi
                    pos_pitch = deg_to_pos(angle_pitch)
                    pos_yaw   = deg_to_pos(angle_yaw)
                    envoyer_position(servo, ID_SERVO_PITCH, pos_pitch)
                    time.sleep(0.5)
                    envoyer_position(servo, ID_SERVO_YAW,   pos_yaw)

                    # Affichage visuel : centre de l'image et centre du marqueur
                    dessiner_marqueurs(frame, x_image, y_image)
                    dessiner_marqueurs(frame, centre[0], centre[1])

                # Mise à jour d'une zone_suivi pour le tracker
                x_tr, y_tr, w_tr, h_tr = cv2.boundingRect(coin_marqueur.astype(int))
                zone_suivi = (x_tr, y_tr, w_tr, h_tr)

                if tracker is None:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, zone_suivi)
                else:
                    # Si le tracker existe déjà, on peut le ré-initialiser pour "coller" au nouveau repère
                    tracker.clear()  # Libère l'ancien tracker
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, zone_suivi)

                dernier_id_marqueur = id_marqueur
                marker_perdu_compteur = 0

            else:
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

                        correction_yaw   = pi_yaw.update(erreur_x)
                        correction_pitch = pi_pitch.update(erreur_y)

                        angle_pitch = ANGLE_NEUTRE - correction_pitch
                        angle_yaw   = ANGLE_NEUTRE + correction_yaw

                        angle_pitch = np.clip(angle_pitch, PITCH_MIN, PITCH_MAX)

                        pos_pitch = deg_to_pos(angle_pitch)
                        pos_yaw   = deg_to_pos(angle_yaw)
                        envoyer_position(servo, ID_SERVO_PITCH, pos_pitch)
                        time.sleep(0.5)
                        envoyer_position(servo, ID_SERVO_YAW,   pos_yaw)

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

                        # On n'a pas détecté le marqueur, mais le tracker voit "quelque chose"
                        # On n'incrémente donc pas marker_perdu_compteur trop vite
                        marker_perdu_compteur += 1
                    else:
                        # Tracker a échoué
                        marker_perdu_compteur += 1
                        if marker_perdu_compteur > MAX_PERTE:
                            # Remise à zéro
                            pi_pitch.reset()
                            pi_yaw.reset()
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
                        pos_neutre = deg_to_pos(ANGLE_NEUTRE)
                        envoyer_position(servo, ID_SERVO_PITCH, pos_neutre)
                        envoyer_position(servo, ID_SERVO_YAW,   pos_neutre)
                        tracker = None
                        dernier_id_marqueur = None

            # Affichage
            cv2.imshow("Visualisation ArUco (Pitch/Yaw) + Fallback Tracking", frame)

            # Sortie si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_camera(picam2)
        fermer_visu()

if __name__ == "__main__":
    main()
