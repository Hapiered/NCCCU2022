import os
import sys
import cv2
import numpy as np
import face_recognition
import pandas as pd
from keras.models import load_model



def getFaceEncoding(src):
    # 加载图片
    image = face_recognition.load_image_file(src)
    # 放大图片
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    # 高斯均值滤波
    image = cv2.medianBlur(image, 7)
    # 非本地滤波
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # 提取图片中人脸区域
    face_locations = face_recognition.face_locations(image)
    if face_locations == []:
        return np.array([])
    img = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
    # 对人脸区域进行128维编码
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    return face_encoding


"""
function：计算两向量之差
Parameters:
    img_encoding1 - 图片1的人脸128维编码
    img_encoding2 - 图片2的人脸128维编码
Returns:
    diff - 两向量之差
"""


def getDiff(img_encoding1, img_encoding2):
    img_encoding1 = np.array(img_encoding1)
    img_encoding2 = np.array(img_encoding2)
    diff = np.subtract(img_encoding1, img_encoding2)
    return diff






global X_all_test
global x_temp
X_all_test = np.empty(shape=[0, 128])
x_temp = np.ones(128)
# 加载模型
MODEL1 = load_model("model.h5")
def predict(file1,file2):
    """
        以下是同学进行判断的代码
        此处省略直接返回0.2
    """
    # 设置空数组存储特征向量

    # 遍历图片数据集，降噪，提取特征，计算特征差的向量128维，并保存向量

    img_encoding1 = getFaceEncoding(file1)
    img_encoding2 = getFaceEncoding(file2)
    # 遇到噪声较大的图片，设置其向量差为非常大的值，让模型判断两张图片不一样
    if (img_encoding1.size == 0) or (img_encoding2.size == 0):
        x_test = [x_temp]
    else:
        x_test = [getDiff(img_encoding1, img_encoding2)]

    # 预测
    predict = MODEL1.predict(np.array(x_test).reshape(1, -1))
    # 输出结果
    print(predict[0][0])
    return predict[0][0]

def main(to_pred_dir,result_save_path):
    subdirs = os.listdir(to_pred_dir) # name
    labels = []
    for subdir in subdirs:
        result = predict(os.path.join(to_pred_dir,subdir,"a.jpg"),os.path.join(to_pred_dir,subdir,"b.jpg"))
        labels.append(result)
    fw = open(result_save_path,"w")
    fw.write("id,label\n")
    for subdir,label in zip(subdirs,labels):
        fw.write("{},{}\n".format(subdir,label))
    fw.close()

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)