import os
import sys
from glob import glob
from datetime import datetime
import random
import time
import traceback

import cv2
import numpy as np

from face_detection import FaceExtractor
from gan_wrapper import GANWrapper
from paint import CVMouseEvent

CAPTURE_FOLDER = "../data/capture"
DATA_FOLDER = "../data/lfw/"
USE_GAN = True
# Fake web camera: provides random images peridiocally from a folder
# Useful in cases when there is no web camera
class FolderWebCamera:
    def __init__(self, folderPath, delay=0.5):
        self.folderPath = folderPath
        self.files = files = glob(os.path.join(folderPath, "./*.jpg"))
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

class WebCamera:
    def __init__(self):
        self.fe = FaceExtractor()
        self.cap = cv2.VideoCapture(0)

        self.ganWrapper = None
        if USE_GAN:
            try:
                self.ganWrapper = GANWrapper()
            except:
                traceback.print_exc()
                print("Failed to load GAN. Will not display autoencoded preview.")

        if self.cap == None or not self.cap.isOpened():
            print("No camera found. Using folder {} instead".format(DATA_FOLDER))
            self.cap = FolderWebCamera(DATA_FOLDER)

        self.inputWindowName = "Web Camera"
        cv2.namedWindow(self.inputWindowName)

        # Original input
        self.captureFrame = None
        # Annotated input (input with annotated face capture rectangle)
        self.annotatedFrame = None
        # Edited input (entire image)
        self.editFrame = None
        # Rectangle of detected face (original input)
        self.extractRect = None

        self.paintColors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),(0,0,0),(255,255,255)]
        self.paintColor = self.paintColors[-2]
        def brushPaint(x, y):
            if self.editMode:
                cv2.circle(self.editFrame, (x, y), 3, self.paintColor, -1)
                #You should imshow when you edit something
                cv2.imshow(self.inputWindowName, self.editFrame)
                
        # CVMouseEventクラスによるドラッグ描画関数の登録
        mouse_event = CVMouseEvent(drag_func=brushPaint)
        mouse_event.setCallBack(self.inputWindowName)

        self.editMode = False
        self.exit = False

    def processNotEdit(self):
        ret, self.captureFrame = self.cap.read()
        self.annotatedFrame = self.captureFrame.copy()
        if ret == False:
            raise "Capture Failed"

        self.extractRect = self.fe.detectFace(self.captureFrame)

        if self.extractRect is not None:
            rectColor = (255, 255, 255)
            cv2.rectangle(self.annotatedFrame,
                tuple(self.extractRect[0:2]),
                tuple(self.extractRect[0:2] + self.extractRect[2:4]),
                rectColor,
                thickness=2)

    def run(self):
        while True:
            if not self.editMode:
                self.processNotEdit()

            # Display the resulting frame
            if self.editMode:
                cv2.imshow(self.inputWindowName, self.editFrame)
            else:
                cv2.imshow(self.inputWindowName, self.annotatedFrame)

            if self.extractRect is not None:
                if self.editMode:
                    extracted = self.fe.extractFaceFromRect(self.editFrame,
                                                            self.extractRect)
                else:
                    extracted = self.fe.extractFaceFromRect(self.captureFrame,
                                                            self.extractRect)
                    self.extractedFrame = self.fe.extractFaceFromRect(self.captureFrame,
                                                                      self.extractRect,
                                                                      resize=False)

                #cv2.imshow('Extracted', extracted)
                if not os.path.exists("tempdir"):
                    os.makedirs("tempdir")
                cv2.imwrite(os.path.join("tempdir", 'tmp.png'), extracted)
                if self.ganWrapper:
                    #autoencoded = ganWrapper.autoencode(extracted, "tempdir")
                    output = self.pipeline(extracted)
                    cv2.imshow('Output', output)

            self.processKeys()
            if self.exit:
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def pipeline(self, img):
        self.modifiedCaptureFrame = self.captureFrame.copy()
        r = self.extractRect
        origShape = self.modifiedCaptureFrame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]].shape
        result = self.ganWrapper.autoencode(img)
        result = cv2.resize(result, (origShape[0], origShape[1]), interpolation = cv2.INTER_CUBIC)
        result = np.asarray(result).astype('uint8')

        if not self.editMode:
            self.originalOutput = result
            # difference between autoencoder output and original face
            self.diffOrig = self.extractedFrame - self.originalOutput
            #cv2.imshow("DiffOrig", self.diffOrig)
            #cv2.imshow("DiffOrig+Old", self.diffOrig + self.originalOutput)

        newSubFrame = result
        # TODO: improve diff function to produce fewer artefacts
        # START

        newSubFrame = newSubFrame + self.diffOrig
        newSubFrame = np.clip(newSubFrame, 0, 255)

        # with detail (includes detail from input):
        self.modifiedCaptureFrame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = newSubFrame
        # without detail (autoencoder output):
        # self.modifiedCaptureFrame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = result
        # END

        #print(result)
        #print(res2)

        return self.modifiedCaptureFrame

    def processKeys(self):
        # key function : q(quiet), s(save)
        
        if self.editMode:
            #when user is editing, waitkey should be large
            #But output is slow when waitkey is too large
            #Maybe 500 or 750 is better. I don't know.
            key = cv2.waitKey(1000) & 0xFF
        else:
            #when user is not editing, waitkey should be small
            key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.exit = True
        elif key == ord('p'):
            self.editMode = not self.editMode
            if self.editMode:
                self.editFrame = self.captureFrame.copy()
                #When user is editing,release the caputure.
                self.cap.release()
            else:
                #When user finished editing ,restart the capture.
                self.cap = cv2.VideoCapture(0)
        elif key == ord('s') and self.extractRect is not None:
            img = self.fe.extractFace(self.captureFrame) # self.fe.saveFace(frame)
            timeStr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if not os.path.exists(CAPTURE_FOLDER):
                os.makedirs(CAPTURE_FOLDER)
            cv2.imwrite(os.path.join(CAPTURE_FOLDER, timeStr + '.jpg'), img)


if __name__ == "__main__":
    wc = WebCamera()
    wc.run()
