# coding=UTF-8
# Author: Fing
# Date: 2017-12-03
# Modification: 2020-04-28

import time
import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import numpy as np 
from sklearn.model_selection import train_test_split
from extract_feature import parse_predict_file


class_count = 50 # 分类类别数


def net():
    # 构建网络结构
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    # 加深网络
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(256, 3, activation='relu'))
    # model.add(Conv1D(256, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))

    return model


def train(X_train, X_test, y_train, y_test):
    # 将标签变成符合keras的one-hot向量
    y_train = keras.utils.to_categorical(y_train, num_classes=class_count)
    y_test = keras.utils.to_categorical(y_test, num_classes=class_count)
    # 这里Fing做了一个trick，把特征从一维变成了三维，这个思路不敢苟同？
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # 编译网络
    model = net()
    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
    start = time.time()
    model.fit(X_train, y_train, batch_size=64, epochs=500)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)

    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Training took: %d seconds' % int(time.time() - start))
    
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
        X_train, X_test, y_train, y_test = inner_fold_cross(i)
        # 用我的想法，全部数据集切分
        # X_train, X_test, y_train, y_test = n_fold_cross(i)
        # X_train, X_test, y_train, y_test = inner_fold_cross(i)
        model, acc = train(X_train, X_test, y_train, y_test)
        # 保存模型
        model.save('save\\keras_cnn_'+str(i)+'.h5')
        accs.append(acc)
    for i in range(len(accs)):
        print(accs[i])
    print("平均accuracy：%0.4f" % np.mean(accs))


def predict(filename):
    """预测某一段音频的类别
    """
    # 提取音频特征
    features = parse_predict_file(filename)
    features = np.expand_dims(features, axis=2)
    for i in range(1, 6):
        model = keras.models.load_model('save\\keras_cnn_'+str(i)+'.h5')
        # 直接预测结果是onehot向量，需要通过argmax输出类别
        predict = np.argmax(model.predict(features), axis=1)
        # predict = model.predict_classes(features)
        print(predict)


if __name__=='__main__':
    test()
    # predict("..\\media\\3-65748-A-12.wav")

