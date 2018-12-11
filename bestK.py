from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math
import random


def read_data(fileXtr, fileYtr, fileXte, fileYte):
    X = []
    Y = []
    for line in open(fileXtr, 'r').readlines():
        tmp = line.split(' ')
        x_tmp = []
        for i in range(len(tmp)):
            x_tmp.append(float(tmp[i]))
        X.append(x_tmp)
    for line in open(fileXte, 'r').readlines():
        tmp = line.split(' ')
        x_tmp = []
        for i in range(len(tmp)):
            x_tmp.append(float(tmp[i]))
        X.append(x_tmp)
    for line in open(fileYtr, 'r').readlines():
        Y.append(int(line))
    for line in open(fileYte, 'r').readlines():
        Y.append(int(line))
    return (X, Y)


def eval_kneigh(predicts, y_test):
    mistake = 0
    nb_lines = 0
    for i in range(len(predicts)):
        if predicts[i] != y_test[i]:
            mistake += 1
        nb_lines += 1
    return mistake/nb_lines


def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out


def find_best_k(X, y):
    skf = StratifiedKFold(n_splits=10)
    best_k_t = {}
    for train_index, test_index in skf.split(X, y):
        compare = 1.1
        best_k = -1
        X_tr, X_te = [X[i] for i in train_index], [X[i] for i in test_index]
        y_tr, y_te = [y[i] for i in train_index], [y[i] for i in test_index]
        for i in sampled_range(1, 9830, 100):
            neigh = KNeighborsClassifier(n_neighbors=i, algorithm="auto", p=1)
            neigh.fit(X_tr, y_tr)
            predictions = neigh.predict(X_te)
            tmp = eval_kneigh(predictions, y_te)
            if tmp < compare:
                compare = tmp
                best_k = i
        if best_k not in best_k_t:
            best_k_t[best_k] = 1
        else:
            best_k_t[best_k] += 1
    max = 0
    k = 0
    for best_k, count in best_k_t.items():
        if count > max:
            max = count
            k = best_k
    return k


(X1, y1) = read_data("Train/X_train.txt",
                     "Train/y_train.txt",
                     "Test/X_test.txt",
                     "Test/y_test.txt")
X = np.array(X1)
y = np.array(y1)
print(find_best_k(X, y))
