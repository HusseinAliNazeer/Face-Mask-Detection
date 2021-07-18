# %%
from keras.models import load_model
import cv2
import numpy as np 
import os
os.chdir("/home/huseein/My_Learning/Jupyter NoteBooks/Face Mask Detection")
# %%
model = load_model('cnn_model.hdf5')
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_label = {0: 'MASK', 1: 'NO MASK'}
# %%
def detect_mask(gray):
    faces = face_model.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # returns a list of (x,y,w,h) tuples
    new_img = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)  # colored output image
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = new_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 4)
        cv2.rectangle(new_img, (x, y), (x + w, y + h) ,(0, 255, 0),4)
    return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
# %%
# this for video 
video_capture = cv2.VideoCapture(0)  # We turn the webcam on.
while True:  # We repeat infinitely (until break):
    _, frame = video_capture.read()  # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)  # We do some colour transformations.
    canvas = detect_mask(gray)  # We get the output of our detect function.
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    cv2.imshow('Video', canvas)  # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # If we type on the keyboard:
        break  # We stop the loop.

video_capture.release()  # We turn the webcam off.
cv2.destroyAllWindows()  # We destroy all the windows inside which the images were displayed.
