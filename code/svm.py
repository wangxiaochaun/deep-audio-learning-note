# coding=UTF-8
# 使用svm做分类
# Author: Fing
# Date: 2017-12-03
# Modification: 2020-04-27

import numpy as np 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import sys
from extract_feature import parse_predict_file


def train(X_train, X_valid, y_train, y_valid):
    # 读取训练集数据.npy
    print('fitting...')
    # 设置svm的参数，参数可调，这里默认是Fing的
    clf = SVC(C=20.0, gamma=0.00001)
    # 在训练集上拟合
    clf.fit(X_train, y_train)
    # 预测在验证集上的结果
    acc = clf.score(X_valid, y_valid)
    print("acc=%0.4f" % acc)
    return clf, acc


def split_train_test(epoch):
    """
    我理解的n折交叉验证，准确率奇高
    """
    X_train, y_train = np.empty((0, 193)), np.empty(0)
    # print(X_train.shape) (0, 193)
    X_valid, y_valid = np.empty((0, 193)), np.empty(0)
    for i in range(1, 6):
        X = np.load('fold'+str(epoch)+'_feature.npy')
        # print(X.shape) (400, 193)
        y = np.load('fold'+str(epoch)+'_labels.npy')
        if i == epoch:
            X_valid = np.concatenate([X_valid, X], axis=0)
            y_valid = np.concatenate([y_valid, y], axis=0)
        else:
            X_train = np.concatenate([X_train, X], axis=0)
            y_train = np.concatenate([y_train, y], axis=0) 
    
    return X_train, X_valid, y_train, y_valid


def inner_split_train_test(epoch):
    """
    按照esc-50数据集baseline，对每一折内部切分trainset和testset
    """
    X = np.load('fold'+str(epoch)+'_feature.npy')
    y = np.load('fold'+str(epoch)+'_labels.npy')

    # 把这一折的数据拆分成train和test（感觉其实会有问题，同一场景的不同
    # 音频有可能被分到两边
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def test():
    """测试在esc-50上的表现
    """
    accs = []
    for i in range(1, 6):
        # simple SVM
        # X_train, X_valid, y_train, y_valid = inner_split_train_test(i)
        X_train, X_valid, y_train, y_valid = split_train_test(i)
        clf, acc = train(X_train, X_valid, y_train, y_valid)
        accs.append(acc)
        # 保存svm模型
        with open('save\\clf'+str(i)+'.pickle', 'wb') as f:
            pickle.dump(clf, f)
    print(np.mean(accs))


def predict(filename):
    # 读取待测音频
    features = parse_predict_file(filename)
    # 读取保存的svm模型
    for i in range(1, 6):
        with open('save\\clf'+str(i)+'.pickle', 'rb') as f:
            clf = pickle.load(f)
            print(clf.predict(features))


if __name__=="__main__":
    # test()
    predict("..\\media\\4-99644-B-4.wav")
        