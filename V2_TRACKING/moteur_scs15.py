import serial

def initialiser_servo(port="/dev/serial0", baudrate=1000000):
    """
    Initialise la connexion série vers le servo SCS15.
    Renvoie un objet 'serial.Serial'.
    """
    return serial.Serial(port, baudrate, timeout=1)

def deg_to_pos(deg):
    """
    Convertit un angle (en degrés) dans la plage [0°, 200°] en une position interne [0, 1023].
    0° correspond à ~0 ticks, 200° correspond à ~1023 ticks.
    """
    pos = int((deg / 200.0) * 1023)
    # On s'assure que la position reste dans [0,1023]
    if pos < 0:
        pos = 0
    elif pos > 1023:
        pos = 1023
    return pos

def envoyer_position(servo, servo_id, position, time_val=505, speed_val=505):
    """
    Envoie une commande de position au servo SCS15 via le port série 'servo'.
    
    servo     : objet serial.Serial
    servo_id  : ID unique du servo (entre 1 et 253)
    position  : valeur entre 0 et 1023 (0 = 0°, 1023 ~ 200°)
    time_val  : temps (0-1023) pour l'action
    speed_val : vitesse (0-1023, 0 = vitesse max)
    """
    # Limitation des valeurs selon la datasheet
    position = max(0, min(1023, int(position)))
    time_val = max(0, min(1023, int(time_val)))
    speed_val = max(0, min(1023, int(speed_val)))

    # Conversion en octets (High puis Low)
    pos_h, pos_l = (position >> 8) & 0xFF, position & 0xFF
    t_h, t_l = (time_val >> 8) & 0xFF, time_val & 0xFF
    s_h, s_l = (speed_val >> 8) & 0xFF, speed_val & 0xFF

    start_addr = 0x2A  # Adresse de la position cible
    instruction = 0x03 # Instruction WRITE

    # Le length se calcule comme suit :
    # length = Nombre d'octets suivant l'ID (instruction, addr, params) + 1 pour le checksum
    # ici : Instruction(1) + Addr(1) + PosH(1) + PosL(1) + TH(1) + TL(1) + SH(1) + SL(1)
    # = 8 octets de data + 1 (pour l'instruction length) = 9
    length = 9

    # Construction du paquet
    # Format : [0xFF, 0xFF, ID, LENGTH, INSTRUCTION, ADDR, DATA..., CHECKSUM]
    packet = [
        0xFF, 0xFF,
        servo_id,
        length,
        instruction,
        start_addr,
        pos_h, pos_l,
        t_h, t_l,
        s_h, s_l
    ]

    # Calcul du checksum
    chk = (~sum(packet[2:]) & 0xFF)
    packet.append(chk)

    # Envoi du paquet
    servo.write(bytearray(packet))

def correcteur_p(erreur, gain):
    """
    Correcteur proportionnel simple : sortie = gain * erreur.
    """
    return gain * erreur
