# 解决思路

1. 放大图片，对图片降噪处理。观察图片数据集有噪声。分为如下两个滤波器
   - 高斯均值滤波 ``cv2.medianBlur()``
   - 非本地滤波 ``cv2.fastNlMeansDenoisingColored()``
2. 提取人脸区域，对特征进行编码。``face_recognition``库
3. 计算欧式距离和余弦相似距离，最终选用指标为欧氏距离
4. 利用距离和标签来训练模型，这是简单的二分类模型。可以选用knn，决策树和贝叶斯。
5. 经过测试效果最好的是决策树，所以选用决策树作为最终方案，进行训练和预测。

# 环境！！！

(按顺序安装，如果某行安装失败，可能与网络有关，请多次重复安装，或换源)

主要在安装dlib库出现问题，如果已经安装dlib库，可直接安装requirements.txt中的库

```bash
conda create -n my1 python=3.9
my_project\doc> conda activate my1
my_project\doc> pip install CMake
my_project\doc> conda install -c conda-forge dlib
my_project\doc> pip install -r requirements.txt
```

# 运行顺序

```bash
# 训练图片处理。时间较长，600组训练集大概8分钟左右
my_project\code> python 1traindata_preprocessing.py
# 预测图片处理。时间较长
my_project\code> pyhton 2testdata_preprocessing.py
# 训练模型
my_project\code> python train.py ../init_data/toUser/train ./model/model.h5
# 预测
my_project\code> python test.py ../init_data/toUser/test ../result/result.csv
```

# 官方要求目录的新增文件

```bash
# 本项目训练了3个分类模型，knn，决策树dt和贝叶斯gnb，生成的模型如下
# 发现决策树针对AUC指标效果更好，所以选用决策树dt作为最终模型model.h5
my_project\code\model\dt_model.h5
my_project\code\model\gnb_model.h5
my_project\code\model\knn_model.h5

# 处理图片数据的中间数据，train是训练集的图片，test是从训练集抽取的一部分图片
# 存储的是两张图片人脸特征的距离，两列EuDist和SimDist，分别为欧式距离和余弦相似距离
# 在训练模型时调用train_data.csv，做预测时调用test_data.csv
my_project\init_data\temp_data\test_data.csv
my_project\init_data\temp_data\train_data.csv
```
