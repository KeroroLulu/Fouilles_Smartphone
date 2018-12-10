from sklearn import preprocessing
from sklearn.svm import SVC
import numpy as np


def read_data(fileX, fileY):
    X = []
    Y = []
    count = 0
    for line in open(fileX, 'r').readlines():
        tmp = line.split(' ')
        x_tmp = []
        for i in range(len(tmp)):
            x_tmp.append(float(tmp[i]))
        X.append(x_tmp)
        count += 1
    count = 0
    for line in open(fileY, 'r').readlines():
        Y.append(int(line))
        count += 1
    return (X, Y)


def eval_svc(predicts, y_test):
    mistake = 0
    nb_lines = 0
    for i in range(len(predicts)):
        if predicts[i] != y_test[i]:
            mistake += 1
        nb_lines += 1
    return mistake/nb_lines


(train_X, train_Y) = read_data("HAPT_Data_Set/Train/X_train.txt",
                               "HAPT_Data_Set/Train/y_train.txt")
(test_X, test_Y) = read_data("HAPT_Data_Set/Test/X_test.txt",
                             "HAPT_Data_Set/Test/y_test.txt")
X_train = np.array(train_X)
y_train = np.array(train_Y)
X_test = np.array(test_X)
y_test = np.array(test_Y)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
model = SVC(C=1.0, kernel='rbf', gamma="scale")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(eval_svc(predictions, y_test))
