import cv2
import time
from threading import Thread

class VideoStream:
    def __init__(self, src='dev/video0'):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # Set the resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Initialize the video stream and start capturing frames in a separate thread
vs = VideoStream().start()
frame_count = 0
start_time = time.time()

while True:
    frame = vs.read()
    cv2.imshow('Camera Feed', frame)

    # Increase frame count and calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time

    # Calculate FPS every second
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
