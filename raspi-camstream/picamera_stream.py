import time
import cv2
import numpy as np
import picamera
import picamera.array

class VideoStream:
    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 24
        self.rawCapture = picamera.array.PiRGBArray(self.camera, size=self.camera.resolution)

        time.sleep(2)  # Allow the camera to warm up

    def start(self):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            cv2.imshow("Frame", image)
            self.rawCapture.truncate(0)  # Clear the stream for the next frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.camera.close()

# Start the video stream
vs = VideoStream()
vs.start()
