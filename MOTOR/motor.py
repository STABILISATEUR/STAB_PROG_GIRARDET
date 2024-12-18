import serial
import time

# Ouvrir le port série (vérifiez que vous avez les permissions, 
# éventuellement : sudo usermod -a -G dialout pi)
ser = serial.Serial('/dev/serial0', baudrate=1000000, timeout=0.1)

def write_position(servo_id, position, time_val=0, speed_val=0):
    """
    Envoie une commande de position au servo SCS15.
    position : valeur entre 0 et 1023 (0 = 0°, 1023 ~ 200°)
    time_val : temps (en unité de 11.2 ms selon la doc, ou autre, selon doc)
    speed_val : vitesse (0 = max)
    
    Le protocole SCS :
    Packet : 0xFF,0xFF, ID, LENGTH, INSTRUCTION, START_ADDR, PARAMS..., CHECKSUM

    Pour définir la position, le temps, la vitesse :
    START_ADDR = 0x29
    PARAMS = [PosH, PosL, TimeH, TimeL, SpeedH, SpeedL]
    """

    pos_h = (position >> 8) & 0xFF
    pos_l = position & 0xFF
    t_h = (time_val >> 8) & 0xFF
    t_l = time_val & 0xFF
    s_h = (speed_val >> 8) & 0xFF
    s_l = speed_val & 0xFF

    start_addr = 0x29
    # Instruction write = 0x03
    # Nombre de paramètres = 6
    # LENGTH = 1 (pour l'adresse) + 6 (param) + 2 (INSTR + CHK) = 9
    length = 9

    packet = [0xFF, 0xFF, servo_id, length, 0x03, start_addr,
              pos_h, pos_l, t_h, t_l, s_h, s_l]

    # Calcul du checksum
    chk = 0
    for b in packet[2:]:
        chk += b
    chk = (~chk) & 0xFF
    packet.append(chk)

    ser.write(bytearray(packet))

try:
    servo_id = 1
    while True:
        print("Déplacement à 0°")
        angle = 0
        pos = int(angle / 200.0 * 1023)
        write_position(servo_id, pos, time_val=0, speed_val=1023) # 500 et 0 à adapter
        time.sleep(2)

        print("Déplacement à 200°")
        angle = 200
        pos = int(angle / 200.0 * 1023)
        write_position(servo_id, pos, time_val=0, speed_val=1023)
        time.sleep(2)

except KeyboardInterrupt:
    print("Arrêt du programme")
finally:
    ser.close()
