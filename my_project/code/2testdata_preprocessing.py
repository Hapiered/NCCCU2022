import cv2
import numpy as np
import face_recognition
import pandas as pd
import os


def getFaceEncoding(src):
    image = face_recognition.load_image_file(src)
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    image = cv2.medianBlur(image, 7)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
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


id_imgGroup = []


test_path = "../init_data/toUser/test/data"
file = os.walk(test_path)
subdirs = os.listdir(test_path)
test_data = pd.DataFrame(columns=['EuDist', 'SimDist'], index=range(len(subdirs)))

for sub_path, sub_dir_list, sub_file_list in file:

    for sub_dir_name in sub_dir_list:
        sub_file = os.walk(os.path.join(sub_path, sub_dir_name))

        for sub2_path, sub2_dir_list, sub2_file_list in sub_file:
            img_encoding1 = getFaceEncoding(os.path.join(sub2_path, sub2_file_list[0]))
            img_encoding2 = getFaceEncoding(os.path.join(sub2_path, sub2_file_list[1]))
            if (img_encoding1.size == 0) or (img_encoding2.size == 0):
                continue
            else:
                print(sub_dir_name)
                eu_dist = getEuDist(img_encoding1, img_encoding2)
                sim_dist = getSimDist(img_encoding1, img_encoding2)
                test_data.at[int(sub_dir_name), "EuDist"] = eu_dist
                test_data.at[int(sub_dir_name), "SimDist"] = sim_dist

test_data.to_csv("../init_data/temp_data/test_data.csv", index_label="id")
