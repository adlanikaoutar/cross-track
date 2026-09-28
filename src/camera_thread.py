import cv2
import threading
import time


class CameraThread:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        self.ret = False
        self.frame = None
        self.running = False
        self.read_lock = threading.Lock()

        if not self.cap.isOpened():
            print(f"[ERREUR] Impossible d'ouvrir {name}")
        else:
            print(f"[OK] {name} connectée.")
            self.running = True
            self.thread = threading.Thread(target=self.update,
                                           args=(), daemon=True)
            self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.read_lock:
                if ret:
                    self.ret = ret
                    self.frame = frame
                else:
                    self.ret = False
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()