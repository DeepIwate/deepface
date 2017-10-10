import os
import sys
from glob import glob
from datetime import datetime
import random
import time
import traceback

import cv2

from face_detection import FaceExtractor
from gan_wrapper import GANWrapper

CAPTURE_FOLDER = "../data/capture"
DATA_FOLDER = "../data/lfw/"
USE_GAN = True

# Fake web camera: provides random images peridiocally from a folder
# Useful in cases when there is no web camera
class FolderWebCamera:
    def __init__(self, folderPath, delay=0.5):
        self.folderPath = folderPath
        self.files = files = glob(os.path.join(folderPath, "*/*.jpg"))
        self.delay = delay
        self.lastTime = 0

    def read(self):
        now = time.time()
        if now - self.lastTime > self.delay:
            imgFile = random.choice(self.files)
            self.img = cv2.imread(imgFile)
            self.lastTime = now
        return True, self.img.copy()

    def release(self):
        pass

if __name__ == "__main__":
    fe = FaceExtractor()
    cap = cv2.VideoCapture(0)

    ganWrapper = None
    if USE_GAN:
        try:
            ganWrapper = GANWrapper()
        except:
            traceback.print_exc()
            print("Failed to load GAN. Will not display autoencoded preview.")

    if cap == None or not cap.isOpened():
        print("No camera found. Using folder {} instead".format(DATA_FOLDER))
        cap = FolderWebCamera(DATA_FOLDER)

    while(True):
        ret, frame = cap.read()

        if ret == False:
            print("Capture Failed")
            break

        #fe.extractFace(frame)
        rect = fe.detectFace(frame)

        if rect is not None:
            color = (255, 255, 255)
            cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

        # Display the resulting frame
        cv2.imshow('Web Camera', frame)
        img = fe.extractFace(frame)
        if img is not None:
            cv2.imshow('Extracted', img)
            if ganWrapper:
                autoencoded = ganWrapper.autoencode(img)
                # for some reason we cannot display autoencoded properly
                # (wrong format?)
                # so first we save it to disk and then read it again
                cv2.imwrite("_temp.png", autoencoded)
                autoencoded = cv2.imread("_temp.png")
                cv2.imshow('Autoencoded', autoencoded)

        # key function : q(quiet), s(save)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and rect is not None:
            img = fe.extractFace(frame) # fe.saveFace(frame)
            timeStr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if not os.path.exists(CAPTURE_FOLDER):
                os.makedirs(CAPTURE_FOLDER)
            cv2.imwrite(os.path.join(CAPTURE_FOLDER, timeStr + '.jpg'), img)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
