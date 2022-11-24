# 解决思路

模型架构，输入层relu，中间层relu，输出层sigmoid

1. 放大图片，对图片降噪处理。观察图片数据集有噪声。分为如下两个滤波器
   - 高斯均值滤波 ``cv2.medianBlur()``
   - 非本地滤波 ``cv2.fastNlMeansDenoisingColored()``
2. 提取人脸区域，对特征进行编码128维。``face_recognition``库
3. 计算特征向量之差，为128维，作为输入
4. 利用特征向量之差和标签来训练模型，这是二分类模型。

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
# 训练图片处理。时间较长，600组训练集大概20分钟左右
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
# 处理图片数据的中间数据
# 训练数据集：train_X_all是所有图片特征向量之差，train_Y_all是所有对应的标签
# 预测数据集：test_X_all是所有图片特征向量之差
# 在训练模型时调用train_X_all.npy，train_Y_all.npy，做预测时调用test_X_all.npy
my_project\init_data\temp_data\train_X_all.npy
my_project\init_data\temp_data\train_Y_all.npy
my_project\init_data\temp_data\test_X_all.npy
```
