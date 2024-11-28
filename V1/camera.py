from picamera2 import Picamera2

def initialiser_cam():
	#Fonction initialisant la caméra et retournant Picamera2 configuré
	picam2 = Picamera2() 
	preview_config = picam2.configure(picam2.create_video_configuration())
	picam2.configure(preview_config)
	picam2.start()
	return picam2
	
def capture_frame(picam2):
	#Capture d'image
	return picam2.capture_array()
	
def stop_camera(picam2):
	picam2.stop
