import numpy as np
import pandas as pd


from keras.models import load_model
# 加载特征向量输入数据
x_test = np.load("../init_data/temp_data/test_X_all.npy")
# 初始化输出表
result_table = pd.DataFrame(columns=["label"], index=range(len(x_test)))
# 加载模型
model = load_model("model/model.h5")
# 预测
predict = model.predict(x_test)
# 输出结果
result_table["label"] = pd.DataFrame(predict)[0]
result_table.to_csv("../result/result.csv", index_label="id")
