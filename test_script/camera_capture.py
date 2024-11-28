from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from time import sleep

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)

#POUR VIDEO: ------------------------------------------------------------

picam2.configure(picam2.create_video_configuration())
encoder = H264Encoder(10000000)

picam2.start_recording(encoder, '/home/pi/Pictures/test.h264')
sleep(5)

#POUR PHOTO :------------------------------------------------------------

#picam2.configure(picam2.create_preview_configuration())

#picam2.start()
#sleep(10)

#picam2.capture_file('/home/valou/Pictures/photo.jpg')

#-----------------------------------------------------------------------

#picam2.stop_recording()
picam2.stop()

print("Photo/vidéo enregistrée dans le repértoire Pictures")
