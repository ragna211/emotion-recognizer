# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:56:44 2018

@author: HP
"""

import numpy as np
import cv2
import os, os.path

os.makedirs("C:/Users/HP/Desktop/downloads/output/sad")
                        
face_cascade = cv2.CascadeClassifier('C:/Users/HP/Downloads/Module_1_Face_Recognition/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
#haar cascade features for face imported

eye_cascade = cv2.CascadeClassifier('C:/Users/HP/Downloads/Module_1_Face_Recognition/Module_1_Face_Recognition/haarcascade_eye.xml')
#haar cascade features for eye imported
DIR = 'C:/Users/HP/Desktop/downloads/downloads/person sad/'  #diretory where files are located
numPics = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])  #checks the number of images which are to be cropped
p=os.listdir(DIR)
for pic in p:
    try:
        img = cv2.imread(DIR+str(pic))
        height = img.shape[0]
        width = img.shape[1]
        size = height * width
        
        if size > (500^2):
            r = 500.0 / img.shape[1]
            dim = (500, int(img.shape[0] * r))
            img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img = img2         #image resized
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            eyesn = 0
            
            for (x,y,w,h) in faces:
                imgCrop = img[y:y+h,x:x+w]                     #image cropped, basically selecting a sub matrix in numpy matrix
                #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)----------->basically forms a rectangle around face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    eyesn = eyesn +1
                    if eyesn >= 2:              #save image iff there are two eyes, else its not face
                        cv2.imwrite("C:/Users/HP/Desktop/downloads/output/sad/"+str(pic)+".jpg", imgCrop)
                                                                                                
                        #cv2.imshow('img',imgCrop)
                        print("Image"+str(pic)+" has been processed and cropped")
                        k = cv2.waitKey(30) & 0xff
                        if k == 27:
                            break
    except:
        print("Image couldnot be processed")
#cap.release()
print("All images have been processed!!!")
cv2.destroyAllWindows()
cv2.destroyAllWindows()