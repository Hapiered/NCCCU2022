import numpy as np

# 构建网络
from keras import models
from keras import layers


x_train = np.load("../init_data/temp_data/train_X_all.npy")
y_train = np.load("../init_data/temp_data/train_Y_all.npy")


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(128,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, epochs=50, batch_size=5)
# 保存模型
model.save("model/model.h5")
