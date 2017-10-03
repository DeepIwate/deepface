import os
import sys
import cv2
from datetime import datetime
out_path = "/tmp/"


class FaceExtractor:
    def __init__(self):
        XML_PATH = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")

        if not os.path.exists(XML_PATH):
            print("Missing haarcascade_frontalface_default.xml")
            sys.exit(-1)
        self.cascade = cv2.CascadeClassifier(XML_PATH)

    def extractFace(self, frame):
        color = (255, 255, 255)
        # face recognition (detect face)
        facerect = self.cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

        if len(facerect) > 0:
            for rect in facerect:
                # draw rectangle
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
        # Display the resulting frame
        cv2.imshow('frame', frame)

    def saveFace(self, frame):
        facerect = self.cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

        if len(facerect) > 0:
            dst = None
            for rect in facerect:
                dst = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

            if dst is not None:
                # resize all images to 128x128 size
                return cv2.resize(dst, (128, 128))

if __name__ == "__main__":
    fe = FaceExtractor()
    cap = cv2.VideoCapture(0)

    if cap == None:
        print("Camera not found")
        sys.exit(-1)

    while(True):
        ret, frame = cap.read()

        if ret == False:
            print("Capture Failed")
            break

        fe.extractFace(frame)

        # key function : q(quiet), s(save)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            img = fe.saveFace(frame)
            time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            cv2.imwrite(out_path + time + '.jpg', img)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
