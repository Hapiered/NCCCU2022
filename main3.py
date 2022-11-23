import dlib
import cv2
import numpy as np
import sys

maxdist = 0
detector = dlib.get_frontal_face_detector()


def face_locator(img):
    '''
    脸部定位,如果有多张人脸则返回最大的人脸
    '''
    dets = detector(img,1)
    if not dets:
        return None
    return max(dets,key=lambda d:d.area())#TODO


predictor = dlib.shape_predictor(r'D:\1mylearningdata\1mphil\Competition\NCCCU2022\1-eye-dingzhen-identify-main\res\shape_predictor_68_face_landmarks.dat')


def extract_features(img, face_loc):
    '''
    利用dlib的68点模型,提取特征
    '''
    landmark = predictor(img, face_loc)
    key_points = []
    for i in range(68):
        pos = landmark.part(i)
        # 转换成np数组方便计算
        key_points.append(np.array([pos.x, pos.y], dtype=np.int32))
    return key_points


def cal(std_keypoints, self_keypoints):
    new_std = []
    new_self = []
    for i in range(68):
        '''
        将绝对坐标转换为相对坐标
        '''
        new_std.append(std_keypoints[i] - std_keypoints[0])
        new_self.append(self_keypoints[i] - self_keypoints[0])
    sum = 0
    for i in range(68):
        sum += np.linalg.norm(new_std[i] - new_self[i])
    rate = 1-np.tanh(sum/10000)
    return rate


def main():
    std_img = cv2.imread(r'a.jpg')
    std_face_loc = face_locator(std_img)
    std_keypoints = extract_features(std_img, std_face_loc)
    self_img = cv2.imread(r'b.jpg')
    self_face_loc = face_locator(self_img)
    self_keypoints = extract_features(self_img, self_face_loc)
    rate = cal(std_keypoints, self_keypoints)
    print(rate)


if __name__ == '__main__':
    main()
