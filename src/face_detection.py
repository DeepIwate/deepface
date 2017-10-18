import os
import sys
from glob import glob
import shutil

import cv2
from tqdm import tqdm

class FaceExtractor:
    def __init__(self):
        XML_PATH = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
        if not os.path.exists(XML_PATH):
            print("Missing haarcascade_frontalface_default.xml")
            sys.exit(-1)
        self.cascade = cv2.CascadeClassifier(XML_PATH)

    def detectFace(self, img):
        # face recognition (detect face)
        facerect = self.cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #facerect = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5,minSize=(10, 10))
        dst = None
        if len(facerect) > 0:
            return facerect[0]

    def extractFace(self, img):
        rect = self.detectFace(img)
        return self.extractFaceFromRect(img, rect)

    def extractFaceFromRect(self, img, rect, resize=True):
        if rect is None:
            return

        dst = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        if dst is not None:
            if resize:
                # resize all images to 128x128 size
                return cv2.resize(dst, (128, 128))
            else:
                return dst

    def extractAllFaces(self):
        # Data folders
        DATA_DIR = "../data/"
        # Input data
        LFW_DATA_DIR = os.path.join(DATA_DIR, "lfw")
        # Output data, extracted faces
        FACES_DIR = os.path.join(DATA_DIR, "faces")

        # Cleanup output folder
        if os.path.exists(FACES_DIR):
            print("Deleting exisitng {} folder.".format(FACES_DIR))
            shutil.rmtree(FACES_DIR)
        if not os.path.exists(FACES_DIR):
            print("Creating {} folder.".format(FACES_DIR))
            os.makedirs(FACES_DIR)

        files = glob(os.path.join(LFW_DATA_DIR, "*/*.jpg"))
        index = 1
        for fpath in tqdm(files):
            img = cv2.imread(fpath)
            dst = self.extractFace(img)
            # write to file
            if dst is not None:
                outputFile = os.path.join(FACES_DIR, "{}.jpg".format(index))
                cv2.imwrite(outputFile, dst)
                index = index + 1

if __name__ == "__main__":
    fe = FaceExtractor()
    fe.extractAllFaces()
