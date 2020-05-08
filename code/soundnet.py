# !coding = UTF-8
# Author: wangx
# Date: 2020-05-08
# 测试soundnet(pre-trained)在ESC-50上的表现

from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer, Dense
from keras.models import Sequential
import numpy as np 
import librosa
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import backend as K 
from keras.optimizers import RMSprop, Adam


def preprocess(audio):
    '''
    视频预处理
    '''
    audio *= 256.0 # SoundNet requires that the range [-256, 256]
    # Reshaping the audio data so it fits into the graph
    # [batch_size, num_samples, num_filter_channels]
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    '''
    用librosa读音频文件
    '''
    sample_rate = 22050 # SoundNet works on mono audio files with a sample rate of 22050
    audio, _ = librosa.load(audio_file, dtype='float32', sr=sample_rate, mono=True)
    audio = preprocess(audio)
    return audio


def build_model():
    '''
    网络模型：加载预训练模型
    '''
    model_weights = np.load('model/sound8.npy', allow_pickle=True, encoding='latin1').item()
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, None, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},
                         ]

    for x in filter_parameters:
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']


            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(Activation('relu'))
        if 'pool_size' in x:
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model


def load_dataset(folds=None):
    '''
    试图在这里实现对ESC-50数据集的处理
    '''
    test = pd.read_csv('D:/Project/ESC-50-master/meta/esc50.csv', sep=',')
    # 我个人认为这种读取音频的方式不好，太占用内存了
    data = []           # 音频
    list_target = []    # 类别
    list_fold = []      # 折

    for file_name, target, fold in zip(list(test['filename']),
                                                list(test['target']),
                                                list(test['fold'])):
        # 如果是指定某一折
        if folds != None:
            if fold == folds:
                audio = load_audio('D:/Project/ESC-50-master/audio/' + file_name)
                data.append(audio)
                list_target.append(target)
                list_fold.append(fold)
        else:
            audio = load_audio('D:/Project/ESC-50-master/audio/' + file_name)
            data.append(audio)
            list_target.append(target)
            list_fold.append(fold)
    
    assert len(data) == len(list_target) == len(list_fold)
    # print(len(data))
    
    return data, list_target, list_fold


def getActivations(data, model, number_layer):
    '''
    获取网络某一层的输出
    '''
    intermediate_tensor = []
    get_layer_output = K.function([model.layers[0].input],
                                [model.layers[number_layer].output])
    
    for audio in data:
        # 截取某一层的输出
        layer_output = get_layer_output([audio])[0]
        tensor = layer_output.reshape(1, -1)
        intermediate_tensor.append(tensor[0])

    return intermediate_tensor


def train():
    '''
    在ESC-50上训练
    '''
    model = build_model()
    model.summary()

    num_classes = 50

    # 截取最后一层输出，然后用dense convolutional layer训练
    classifier = Sequential()
    classifier.add(Dense(num_classes, activation='softmax', input_shape=(3328,)))
    classifier.summary()

    classifier.compile(loss='categorical_crossentropy',
                        optimizer=Adam(),
                        metrics=['accuracy'])

    batch_size = 64
    epochs = 20

    tag = 'whole'

    # 两种n折交叉验证方式
    avg_acc = []
    if tag == 'inner':
        # 首先是每折内部划分，fit，然后取平均
        for i in range(1, 6):
            print("Proceeding..." + str(i) + ' folds')
            data, labels, _ = load_dataset(i)
            x = np.asarray(getActivations(data, model, 22))
            y = np.asarray(labels)
            # print(x.shape)
            # print(y.shape)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)

            classifier.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

            score = classifier.evaluate(x_test, y_test, verbose=0)
            print('Test loss: ', score[0])
            print('Test accuracy: ', score[1])
            avg_acc.append(score[1])
    elif tag == 'whole':
        # 整体划分，然后求平均       
        for fold in range(1, 6):
            print('Training...' + str(fold) + ' folds')
            x_train, y_train = np.empty((0, 3328)), np.empty(0)
            x_test, y_test = np.empty((0, 3328)), np.empty(0)
            for i in range(1, 6):
                data, labels, _ = load_dataset(i)
                data = np.asarray(getActivations(data, model, 22))
                labels = np.asarray(labels)
                if fold == i:
                    x_test = np.concatenate([x_test, data], axis=0)
                    y_test = np.concatenate([y_test, labels], axis=0)
                else:
                    x_train = np.concatenate([x_train, data], axis=0)
                    y_train = np.concatenate([y_train, labels], axis=0)
            
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)

            history = classifier.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

            score = classifier.evaluate(x_test, y_test, verbose=0)
            print('Test loss: ', score[0])
            print('Test accuracy: ', score[1])
            avg_acc.append(score[1])
            
    print('Average loss: ', np.mean(avg_acc))


if __name__=='__main__':
    train()
