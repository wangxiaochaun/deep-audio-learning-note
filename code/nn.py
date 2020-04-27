# coding=UTF-8
# Author: Fing
# Date: 2017-12-03
# Modification: 2020-04-27

import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# 全局变量
num_classes = 50

def net():
    """搭建多层感知机
    """
    # 定义网络结构
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=193))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # 打印模型概述信息
    model.summary()

    return model


def train(X_train, X_test, y_train, y_test):
    """训练
    """
    # 要把labels变成one-hot向量
    y_train = keras.utils.to_categorical(y_train-1, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test-1, num_classes=num_classes)

    # 读模型
    model = net()
    # 编译模型
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1000, batch_size=64)
    score, acc = model.evaluate(X_test, y_test, batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    return model, acc


def n_fold_cross(fold):
    """做n折交叉验证
    """
    # 读取数据
    X_train, y_train = np.empty((0, 193)), np.empty(0)
    # print(X_train.shape) (0, 193)
    X_valid, y_valid = np.empty((0, 193)), np.empty(0)
    for i in range(1, 6):
        X = np.load('model\\fold'+str(i)+'_feature.npy')
        # print(X.shape) (400, 193)
        y = np.load('model\\fold'+str(i)+'_labels.npy')
        if i == fold:
            X_valid = np.concatenate([X_valid, X], axis=0)
            y_valid = np.concatenate([y_valid, y], axis=0)
        else:
            X_train = np.concatenate([X_train, X], axis=0)
            y_train = np.concatenate([y_train, y], axis=0) 
    
    return X_train, X_valid, y_train, y_valid



def inner_fold_cross(fold):
    """Fing做的n折验证，将每一折内部分为train和test
    """
    # 读取数据
    X = np.load("model\\fold"+str(fold)+"_feature.npy")
    y = np.load("model\\fold"+str(fold)+"_labels.npy").ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    return X_train, X_test, y_train, y_test


def test():
    """测试在ESC-50上的性能
    """
    accs = []
    for i in range(1, 6):
        print("第"+str(i)+"折交叉验证...")
        # 每折内切分
        # X_train, X_test, y_train, y_test = inner_fold_cross(i)
        # 用我的想法，全部数据集切分
        X_train, X_test, y_train, y_test = n_fold_cross(i)
        model, acc = train(X_train, X_test, y_train, y_test)
        # 保存模型
        model.save('save\\keras_nn_'+str(i)+'.h5')
        accs.append(acc)
    print("平均accuracy：%0.4f" % np.mean(accs))


if __name__=="__main__":
    test()