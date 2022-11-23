import face_recognition
import numpy as np
import cv2

'''
对比两张图片是否属于同一人
'''


def face_re(face1, face2):
    face_encoding1 = []
    is_face1 = face_recognition.face_encodings(face_recognition.load_image_file(face1))

    if (len(is_face1) <= 0):
        print('未识别人脸1')
        return '未识别人脸1'
    else:
        face_encoding1.append(is_face1[0])
    face_encoding2 = []
    is_face2 = face_recognition.face_encodings(face_recognition.load_image_file(face2))
    if (len(is_face2) <= 0):
        print('未识别人脸2')
        return '未识别人脸2'
    else:
        face_encoding2.append(is_face2[0])
    match = face_recognition.compare_faces(np.array(is_face1), np.array(is_face2), tolerance=0.5)

    print(match)
    if match[0]:
        print('图片识别为同一人')
        return True
    else:
        print('图片识别不是一个人')
        return False

'''
判断是否是人脸
'''
def is_face(face_img):
    is_face1 = face_recognition.face_encodings(face_recognition.load_image_file(face_img))

    if (len(is_face1) <= 0):
        print('未识别人脸1')
        return False
    else:
        print('识别人脸')
    return True

src1, src2 = r"D:\1mylearningdata\1mphil\研究生竞赛\全国高校计算机能力挑战赛2022\telangpu.png", \
    r"D:\1mylearningdata\1mphil\研究生竞赛\全国高校计算机能力挑战赛2022\telangpu2.png"
face_re(src1, src2)