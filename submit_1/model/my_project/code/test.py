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

 # 读取csv文件，提取特征列表和标签列表的内容，并返回
def load_testdata(testdata_csv_path, eu_dist=1): 
    all_data = pd.read_csv(testdata_csv_path)
    all_data = all_data.fillna(all_data.mean()["EuDist":"SimDist"])

    EuDist1 = all_data["EuDist"].values.tolist()
    EuDist = []
    for i in EuDist1:
        EuDist.append([i])

    SimDist1 = all_data["SimDist"].values.tolist()
    SimDist = []
    for i in SimDist1:
        SimDist.append([i])

    if eu_dist == 1:
        return EuDist
    else:
        return SimDist
    
# 对数据做预测，包括三个模型：knn，决策树dt和贝叶斯gnb，选用AUC指标更好的决策树dt，其余部分注释
def predict():
    x_test = load_testdata("../init_data/temp_data/test_data.csv")

    result_table = pd.DataFrame(columns=["label"], index=range(len(x_test)))

    # # 加载knn分类器，并在测试集上进行预测
    # knn = joblib.load("model/knn_model.h5")
    # knn_predict = knn.predict(x_test)  # 得到分类结果
    # knn_predict_proba = knn.predict_proba(x_test)
    # result_table["label"] = pd.DataFrame(knn_predict_proba)[1]
    # result_table.to_csv("../result/result.csv", index_label="id")

    # 加载决策树分类器，并在测试集上进行预测
    dt = joblib.load("../model/model.h5")
    dt_predict = dt.predict(x_test)  # 得到分类结果
    dt_predict_proba = dt.predict_proba(x_test)
    result_table["label"] = pd.DataFrame(dt_predict_proba)[1]
    result_table.to_csv("../result/result.csv", index_label="id")

    # # 加载贝叶斯分类器，并在测试集上进行预测
    # gnb = joblib.load("model/gnb_model.h5")
    # gnb_predict = gnb.predict(x_test)  # 得到分类结果
    # gnb_predict_proba = gnb.predict_proba(x_test)
    # result_table["label"] = pd.DataFrame(gnb_predict_proba)[1]
    # result_table.to_csv("../result/result.csv", index_label="id")