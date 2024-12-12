import serial
import time

# Initialisation de la connexion UART
ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

try:
    while True:
        # Envoi d'une commande test
        message = "Commande Test 123"
        ser.write(message.encode())
        print(f"Envoyé : {message}")
        time.sleep(1)


except KeyboardInterrupt:
    print("Arrêt du programme.")
finally:
    ser.close()
