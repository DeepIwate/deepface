
# coding: utf-8

# In[24]:

import numpy as np
import cv2
from datetime import datetime


# In[25]:

out_jpg = "/home/yura/Desktop/output/"
cascade_path = "/home/yura/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"


# In[26]:

cap = cv2.VideoCapture(0)
color = (255, 255, 255)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #CascadeClassifier
    cascade = cv2.CascadeClassifier(cascade_path)

    #face recognition (detect face)
    facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, 
                                        minNeighbors=2, minSize=(10, 10))
    
    for rect in facerect:
        #draw rectangle
        cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), color, thickness=2)
        #face crop
        dst = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = out_jpg + time + '.jpg'
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    #key function : q(quiet), s(save)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(save_path, dst)
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:



