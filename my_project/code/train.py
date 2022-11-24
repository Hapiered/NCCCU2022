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


def load_dataset(data_csv_path, eu_dist=1):  # 读取特征文件列表和标签文件列表的内容，归并后返回
    all_data = pd.read_csv(data_csv_path)
    all_data = all_data.fillna(all_data.mean()["EuDist":"SimDist"])
    all_data = shuffle(all_data)
    EuDist1 = all_data["EuDist"].values.tolist()
    EuDist = []
    for i in EuDist1:
        EuDist.append([i])

    SimDist1 = all_data["SimDist"].values.tolist()
    SimDist = []
    for i in SimDist1:
        SimDist.append([i])

    label = all_data["label"].values.tolist()

    if eu_dist == 1:
        return EuDist, label
    else:
        return SimDist, label


def split_dataset(X, y, ratio=0.8):
    X_train, y_train = X[:int(ratio * len(X))], y[:int(ratio * len(y))]  # 前四个数据作为训练集
    x_test, y_test = X[int(ratio * len(X)):], y[int(ratio * len(y)):]   # 最后一个数据作为测试集
    X_train, x_, y_train, y_ = train_test_split(X_train, y_train, test_size=0.1)  # 使用全量数据作为训练集，split函数打乱训练集
    return X_train, y_train, x_test, y_test


def train_model():
    dt_AUC_max = -1
    knn_AUC_max = -1
    gnb_AUC_max = -1
    for i in range(10000):
        X, y = load_dataset("../init_data/temp_data/train_data.csv")
        X_train, y_train, x_test, y_test = split_dataset(X, y, ratio=0.8)

        # # 创建knn分类器，并在测试集上进行预测
        # knn = KNeighborsClassifier().fit(X_train, y_train)
        # knn_predict = knn.predict(x_test)  # 得到分类结果
        # knn_predict_proba = knn.predict_proba(x_test)
        # if knn_AUC_max < roc_auc_score(y_test, knn_predict_proba.max(axis=1)):
        #     knn_AUC_max = roc_auc_score(y_test, knn_predict_proba.max(axis=1))
        #     joblib.dump(knn, "model/knn_model.h5")
        #     print("best_knn_AUC is", knn_AUC_max)

        # 创建决策树分类器，并在测试集上进行预测
        dt = DecisionTreeClassifier().fit(X_train, y_train)
        dt_predict = dt.predict(x_test)  # 得到分类结果
        dt_predict_proba = dt.predict_proba(x_test)
        if dt_AUC_max < roc_auc_score(y_test, dt_predict_proba.max(axis=1)):
            dt_AUC_max = roc_auc_score(y_test, dt_predict_proba.max(axis=1))
            joblib.dump(dt, "model/model.h5")
            print("best_dt_AUC is", dt_AUC_max)

        # # 创建贝叶斯分类器，并在测试集上进行预测
        # gnb = GaussianNB().fit(X_train, y_train)
        # gnb_predict = gnb.predict(x_test)  # 得到分类结果
        # gnb_predict_proba = gnb.predict_proba(x_test)
        # if gnb_AUC_max < roc_auc_score(y_test, gnb_predict_proba.max(axis=1)):
        #     gnb_AUC_max = roc_auc_score(y_test, gnb_predict_proba.max(axis=1))
        #     joblib.dump(gnb, "model/gnb_model.h5")
        #     print("best_gnb_AUC is", gnb_AUC_max)


train_model()
