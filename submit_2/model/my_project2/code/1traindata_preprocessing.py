import cv2
import numpy as np
import face_recognition
import pandas as pd
import os

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


# 读取训练集中csv文件，用于获取标签
train_data = pd.read_csv("../init_data/toUser/train/annos.csv")
# 设置空数组存储特征向量和标签
X_all = np.empty(shape=[0, 128])
Y_all = np.empty(shape=[0, 1])
# 遍历图片数据集，降噪，提取特征，计算特征差的向量128维，并保存向量和对应的标签
train_path = "../init_data/toUser/train/data"
file = os.walk(train_path)
for sub_path, sub_dir_list, sub_file_list in file:
    for sub_dir_name in sub_dir_list:
        sub_file = os.walk(os.path.join(sub_path, sub_dir_name))

        for sub2_path, sub2_dir_list, sub2_file_list in sub_file:
            img_encoding1 = getFaceEncoding(os.path.join(sub2_path, sub2_file_list[0]))
            img_encoding2 = getFaceEncoding(os.path.join(sub2_path, sub2_file_list[1]))
            # 跳过因噪声较大而没有提取到特征的照片
            if (img_encoding1.size == 0) or (img_encoding2.size == 0):
                continue
            else:
                print(sub_dir_name)
                Y_all = np.append(Y_all, [train_data.at[int(sub_dir_name), "label"]])
                X_all = np.append(X_all, [getDiff(img_encoding1, img_encoding2)], axis=0)

np.save("../init_data/temp_data/train_X_all.npy",X_all)
np.save("../init_data/temp_data/train_Y_all.npy",Y_all)