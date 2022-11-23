import os, sys
import cv2

def predict(file1,file2):
    img1 = cv2.imread(file1) 
    img2 = cv2.imread(file2)
    """
        以下是同学进行判断的代码
        此处省略直接返回0.2
    """
    return 0.2

def main(to_pred_dir,result_save_path):
    subdirs = os.listdir(to_pred_dir) # name
    labels = []
    for subdir in subdirs:
        result = predict(os.path.join(to_pred_dir,subdir,"a.jpg"),os.path.join(to_pred_dir,subdir,"b.jpg"))
        labels.append(int(result))
    fw = open(result_save_path,"w")
    fw.write("id,label\n")
    for subdir,label in zip(subdirs,labels):
        fw.write("{},{}\n".format(subdir,label))
    fw.close()

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)