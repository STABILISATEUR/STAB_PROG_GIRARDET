from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
preview_config = picam2.configure(picam2.create_video_configuration())
picam2.configure(preview_config)
picam2.start()


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters_create()

try:
    while True:

        frame = picam2.capture_array()

        if frame is None or frame.size == 0:
            print("Erreur : image vide ou non valide.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None:
            for corner in corners:
                pts = corner.reshape(-1, 2).astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            print("Marqueurs détectés : ", ids.flatten())
        else:
            print("Aucun marqueur détecté")
        
        cv2.imshow("Frame with ArUco markers", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

finally:
    picam2.stop()
    cv2.destroyAllWindows()


