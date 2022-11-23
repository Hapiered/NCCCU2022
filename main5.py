import dlib
import cv2
import numpy as np
import face_recognition
import pandas as pd

image = face_recognition.load_image_file(r"D:\1mylearningdata\1mphil\Competition\NCCCU2022\1my\train\data\3\b.jpg")
image = cv2.medianBlur(image, 5)
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
face_locations = face_recognition.face_locations(image)
img = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]

cv2.imshow("image", image)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
