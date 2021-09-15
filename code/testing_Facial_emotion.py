import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import face_recognition
import keras

model = load_model("model_testV3.h5")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_image  = cv2.imread('1.jpg')        
plt.imshow(face_image)

face_image.shape

plt.imshow(cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB))

gray = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
    
faces = faceCascade.detectMultiScale(gray,1.1,4)

for(x, y, w, h) in faces:
    cv2.rectangle(face_image,(x, y), (x+w, y+h), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB))

face_image.shape

face_locations = face_recognition.face_locations(face_image)
top, right, bottom, left = face_locations[0]
face_image = face_image[top:bottom, left:right]
#plt.imshow(face_image)
plt.imshow(cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB))

face_image.shape.

# resizing the image
face_image = cv2.resize(face_image, (48,48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])


face_image.shape

predicted_class = np.argmax(model.predict(face_image))

emotion_labels= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

label_map = dict((v,k) for k,v in emotion_labels.items()) 
predicted_label = label_map[predicted_class]

print("The fecial emotion is",'\033[1m' + predicted_label)

