# 运行顺序

python 1traindata_preprocessing.py
pyhton 2testdata_preprocessing.py
python train.py ../init_data/toUser/train ./model/model.h5
python test.py ../init_data/toUser/test ../result/result.csv
