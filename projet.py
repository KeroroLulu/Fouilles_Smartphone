# coding=utf-8

import math
import random
from sklearn.model_selection import StratifiedKFold


def read_data(fileX, fileY):
    X = []
    Y = []
    for line in open(fileX, 'r').readlines():
        tmp = line.split(' ')
        x_tmp = []
        for i in range(len(tmp)):
            x_tmp.append(float(tmp[i]))
        X.append(x_tmp)
    count = 0
    for line in open(fileY, 'r').readlines():
        Y.append(int(line))
    return (X, Y)


def simple_distance(data1, data2):
    res = 0
    for i in range(len(data1)):
        res += (data1[i] - data2[i]) ** 2
    return math.sqrt(res)


def k_nearest_neighbors(x, points, dist_function, k):
    if k > len(points):
        return None
    tmp = []
    for i in range(len(points)):
        tmp.append([dist_function(x, points[i]), i])
    tmp.sort(key=lambda x: x[0])
    res = []
    for i in range(k):
        res.append(tmp[i][1])
    return res


def predict_smart(x, train_x, train_y, dist_function, k):
    tmp = k_nearest_neighbors(x, train_x, dist_function, k)
    if tmp is None:
        return None
    res = dict()
    for i in tmp:
        if train_y[i] not in res:
            res[train_y[i]] = 1
        else:
            res[train_y[i]] += 1
    max = 0
    state = -1
    for s, count in res.items():
        if count > max:
            max = count
            state = s
    return state


def eval_smart_classifier(train_x, train_y, test_x, test_y, classifier, dist_function, k):
    mistake = 0
    nb_lines = 0
    for i in range(len(test_x)):
        if classifier(test_x[i], train_x, train_y, dist_function, k) != test_y[i]:
            mistake += 1
        nb_lines += 1
    return mistake/nb_lines


train_X, train_Y = read_data("HAPT_Data_Set/Train/X_train.txt", "HAPT_Data_Set/Train/y_train.txt")
test_X, test_Y = read_data("./HAPT_Data_Set/Test/X_test.txt", "./HAPT_Data_Set/Test/y_test.txt")
print(eval_smart_classifier(train_X, train_Y, test_X, test_Y, predict_smart, simple_distance, 5))


def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out


def find_best_k(train_x, train_y, dist_function):
    skf = StratifiedKFold(n_splits=10)
    best_k_t = {}
    for train_index, test_index in skf.split(train_x, train_y):
        compare = 1.1
        best_k = -1
        X_train, X_test = [train_x[i] for i in train_index], [train_x[i] for i in test_index]
        y_train, y_test = [train_y[i] for i in train_index], [train_y[i] for i in test_index]
        for i in sampled_range(1, 1000, 10):
            tmp = eval_cancer_classifier(X_train, y_train, X_test, y_test, is_cancerous_knn, dist_function, i)
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
