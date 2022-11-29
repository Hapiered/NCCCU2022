import numpy as np
import pandas as pd

# 自动生成训练集和测试集模块
from sklearn.model_selection import train_test_split
# 计算auc模块
from sklearn.metrics import roc_auc_score
# K近邻分类器、决策树分类器、高斯朴素贝叶斯函数
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# 打乱数据模块
from sklearn.utils import shuffle
# 输出模型模块
import joblib

import cv2
import face_recognition
import os
import sys

""" 
function：提取人脸特征并编码
Parameters:
    src - 图片地址
Returns:
    face_encoding - 提取的人脸特征编码
"""


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


""" 
function：计算两向量间的欧氏距离
Parameters:
    img_encoding1 - 图片1的人脸128维编码
    img_encoding2 - 图片2的人脸128维编码
Returns:
    dist - 欧氏距离
"""


def getEuDist(img_encoding1, img_encoding2):
    img_encoding1 = np.array(img_encoding1)
    img_encoding2 = np.array(img_encoding2)
    diff = np.subtract(img_encoding1, img_encoding2)
    dist = np.sqrt(np.sum(np.square(diff)))
    return dist


""" 
function：计算两向量间的余弦相似距离
Parameters:
    img_encoding1 - 图片1的人脸128维编码
    img_encoding2 - 图片2的人脸128维编码
Returns:
    sim - 余弦相似距离
"""


def getSimDist(img_encoding1, img_encoding2):
    img_encoding1 = np.array(img_encoding1)
    img_encoding2 = np.array(img_encoding2)
    dist = np.linalg.norm(img_encoding1 - img_encoding2)
    sim = 1.0 / (1.0 + dist)
    return sim


DT = joblib.load("my_project\model\model.h5")
def predict(file1, file2):
    """
        以下是同学进行判断的代码
        此处省略直接返回0.2
    """
    img_encoding1 = getFaceEncoding(file1)
    img_encoding2 = getFaceEncoding(file2)
    if (img_encoding1.size == 0) or (img_encoding2.size == 0):
        eu_dist=10
    else:
        eu_dist = getEuDist(img_encoding1, img_encoding2)
        
    
    dt_predict_proba = DT.predict_proba(np.array(eu_dist).reshape(1, -1))
    print(dt_predict_proba[0][1])
    return dt_predict_proba[0][1]


def main(to_pred_dir, result_save_path):
    subdirs = os.listdir(to_pred_dir)  # name
    labels = []
    for subdir in subdirs:
        result = predict(os.path.join(to_pred_dir, subdir, "a.jpg"), os.path.join(to_pred_dir, subdir, "b.jpg"))
        labels.append(result)
    fw = open(result_save_path, "w")
    fw.write("id,label\n")
    for subdir, label in zip(subdirs, labels):
        fw.write("{},{}\n".format(subdir, label))
    fw.close()


if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)
