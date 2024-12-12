import serial

# Initialisation UART pour les servos
def initialiser_servo(port="/dev/serial0", baudrate=1000000):
    return serial.Serial(port, baudrate, timeout=1)

# Envoi d'une position à un servo
def envoyer_position(servo, servo_id, position):
    # Limiter la position entre 0° et 200°
    position = max(0, min(200, position))
    valeur = int(position / 200.0 * 0x03FF)
    commande = [0xFF, 0xFF, servo_id, 0x07, 0x03, 0x2A, valeur & 0xFF, (valeur >> 8) & 0xFF]
    checksum = (~sum(commande[2:]) & 0xFF)
    commande.append(checksum)
    servo.write(bytearray(commande))

# Calcul de l'erreur et correction avec un P
def correcteur_p(erreur, gain):
    return gain * erreur
