from face_detection import FaceExtractor

# setup webcam
fe = FaceExtractor()

# loop
    img = getImage()
    fe.extractFace(img)
    # display

# close webcam
