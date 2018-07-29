# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 22:12:29 2018

@author: HP
"""
# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers

face_cascade = cv2.CascadeClassifier('C:/Users/HP/Downloads/Module_1_Face_Recognition/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')




# Initialising the CNN
classifier = Sequential()    #object initialized

#convolutional layers arnot added in ANN.In it works start from adding dense layer/fully connected layers

# Step 1 - Convolution
classifier.add(Conv2D(128, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

#32 feature detectors of 3x3 size 
#we need to define input_shapein first layer only, further it knows that output of prev layer is its input

#we use relu in starting layers and sigmoid(binary)/sofmax(multi class) in the last layers

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#they are generally stacked with an increasing number of filters in each layer. Each successive layer can have 
#two to four times the number of filters in the previous layer. This helps the network learn hierarchical
#features.

# Step 3 - Flattening
classifier.add(Flatten())
#flattening --->converting matrices to linear arrays


# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#full connected layer1

classifier.add(Dense(units = 3, activation = 'softmax'))
#fully connected layer 2

# Compiling the CNN
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)

classifier.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
#data input and scaling and data increase
#we are basically increasing the input data by ImageDataGenerator function. By flipping,applying shear to the same image

train_datagen = ImageDataGenerator(rescale = 1./255,    #converting 0->255 to 0->1 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#extra trainig data generated

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/HP/Desktop/downloads/output/train',
                                                 target_size = (128, 128),#size of input image, we above mentioned 64x64
                                                 batch_size = 1,#numbr of images after which weights are updated
                                                 class_mode = 'categorical'#multiclass classification
                                                 )
#images are compressed to the size of 64x64
#training the model. Training basically takes place with the help of object of Sequential class

test_set = test_datagen.flow_from_directory('C:/Users/HP/Desktop/downloads/output/test',
                                            target_size = (128, 128),
                                            batch_size = 1,
                                            class_mode = 'categorical')
#testing the data

classifier.fit_generator(training_set,
                         samples_per_epoch = 90, #number of training example photos
                         epochs =40,  #your choice
                         validation_data = test_set, #number of testing photos
                         validation_steps = 25)
#greater is the difference bw training and test set accuracy,greater is overfitting
#making new predictions

import numpy as np
from keras.preprocessing import image

import os
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly if not cap.isOpened():     raise IOError("Cannot open webcam")
#roi_color is actually fed to prediction model but actually full frame with ectangle around faces is shown in webcam

while True:
    ret, frame = cap.read()
    #input from webcam stored in variable frame
    #ret is just a boolean,telling whether it is able to capture frame or not 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #converting frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#face detection
    i=0
    for (x,y,w,h) in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #rectangle around my face drawn
        roi_gray = gray[y:y+h, x:x+w]#image cropped 
        roi_color = frame[y:y+h, x:x+w]#colored image cropped
        #ret is boolean, to check if frame is coming or not 
        roi_color = cv2.resize(roi_color, (128,128),
                       interpolation=cv2.INTER_AREA)#image resized becz only 128x128 image is to be fed into model

        test_image=image.img_to_array(roi_color)#third dimension added becz input is supposed to be 128x128x3 and not 128x128
        test_image=np.expand_dims(test_image,axis=0)#1 more dimension added because predict function needs 4 dimenions.
        #This dimension takes the batch size. i.e. on how many images are to be fed in model at one time.I kept batch size=1
        result=classifier.predict(test_image)
     
        if (result[0][0] == 1):#result.all to handle multiple faces
            print("angry")
            cv2.putText(frame,"angry",(x,h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            #this is how to put text on image
        if (result[0][1] == 1):
            print("sad")
            cv2.putText(frame,"sad",(x,h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        if (result[0][2] == 1):
            print("smiling")
            cv2.putText(frame,"smiling",(x,h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        #video.write_videofile('test.avi', fps=30, codec='mpeg4')
        if (c == 27):
            #27 is ASCII value of esc
            i=1
            break
    if (i==1):
        break

#training_set.class_indices
cap.release()
cv2.destroyAllWindows()