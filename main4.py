import dlib
import cv2
import numpy as np
import face_recognition
import pandas as pd


def getFaceEncoding(src):
    image = face_recognition.load_image_file(src)
    image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)
    face_locations = face_recognition.face_locations(image)
    if face_locations == []:
        return np.array([])
    img = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    return face_encoding


def getEuDist(img_encoding1, img_encoding2):
    img_encoding1 = np.array(img_encoding1)
    img_encoding2 = np.array(img_encoding2)
    diff = np.subtract(img_encoding1, img_encoding2)
    dist = np.sqrt(np.sum(np.square(diff)))
    return dist


def getSimDist(img_encoding1, img_encoding2):
    img_encoding1 = np.array(img_encoding1)
    img_encoding2 = np.array(img_encoding2)
    dist = np.linalg.norm(img_encoding1 - img_encoding2)
    sim = 1.0 / (1.0 + dist)
    return sim

# train_data=pd.read_csv("train\annos.csv")
img_encoding1 = getFaceEncoding("a.jpg")
img_encoding2 = getFaceEncoding("b.jpg")
# img_encoding1 = getFaceEncoding("telangpu.png")
# img_encoding2 = getFaceEncoding("telangpu2.png")
if (img_encoding1.size == 0) or (img_encoding2.size == 0):
    print("no")
else:
    eu_dist = getEuDist(img_encoding1, img_encoding2)
    sim_dist = getSimDist(img_encoding1, img_encoding2)

    print(eu_dist)
    print(sim_dist)
