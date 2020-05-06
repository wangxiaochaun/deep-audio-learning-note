# !conding=UTF-8
# Author: 
# Date:
# Modification: 2020-05-06

from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential
import numpy as np 
import librosa


def preprocess(audio):
    '''音频预处理
    '''
    audio *= 256.0 # SoundNet needs the range to be between -256 and 256
    # Reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    # print("audio shape: ", np.shape(audio))
    return audio


def load_audio(audio_file):
    '''读取音频文件
    '''
    sample_rate = 22050 # SoundNet works on mono audio files with a sample rate of 22050
    audio, sr = librosa.load(audio_file, dtype='float32', sr=sample_rate, mono=True)
    audio = preprocess(audio)
    return audio


def build_model():
    '''网络
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is kept as sound8.npy)
    注意：这里的网络已经是预训练好的，省去了训练过程
    '''
    model_weights = np.load('model/sound8.npy', allow_pickle=True, encoding="latin1").item()
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


def predict_scene_from_audio_file(audio_file):
    '''预测
    '''
    model = build_model()
    audio = load_audio(audio_file)
    print(audio.shape)
    return model.predict(audio)


def predictions_to_scenes(prediction):
    scenes = []
    with open('categories/categories_places2.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            scenes.append(categories[np.argmax(prediction[0, p, :])])
    
    return scenes


prediction = predict_scene_from_audio_file('../media/railroad_audio.wav')
print(prediction.shape) # [1, 4, #num_categories]

print(predictions_to_scenes(prediction))
print(np.shape(predictions_to_scenes(prediction)))

prediction2 = predict_scene_from_audio_file('../media/demo.mp3')
print(prediction2.shape)
print(predictions_to_scenes(prediction2))

prediction3 = predict_scene_from_audio_file('../media/airplane.wav')
print(predictions_to_scenes(prediction3))

import matplotlib.pyplot as plt 


def figura(prediction):
    print(prediction.shape)
    fig, ax = plt.subplots()
    ax.plot(prediction[0, 0, :])
    ax.plot(prediction[0, 1, :], c='r')
    plt.show()


figura(prediction)

model = build_model()
model.summary()

from keras import backend as K 

# Function to obtain the output of Max pooling 1d24 (poo;5)
get_22th_layer_output = K.function([model.layers[0].input],
                                    [model.layers[22].output])



import pandas as pd 
test = pd.read_csv('E:/dataset/ESC-50-master/meta/esc50.csv', sep=',')
data = []
list_target = []
list_category = []

for file_name, target, category, esc10 in zip(list(test['filename']),
                                            list(test['target']),
                                            list(test['category']),
                                            list(test['esc10'])):
    if esc10 == True:
        audio = load_audio('E:/dataset/ESC-50-master/audio/' + file_name)
        data.append(audio)
        list_target.append(target)
        list_category.append(category)


get = get_22th_layer_output([data[155]])[0]
# print('get_22th_layer_output([data[155]])[0]', get)
# get = model.predict([data[155]])
# print('model.predict([data[155]])', get)
datos = np.asarray([data[155], data[155]]).reshape(1, -1, 1)
print(datos.shape)

p = model.predict(datos)
print(p.shape)
print(predictions_to_scenes(p))
tensor = get.reshape(1, -1)
print(tensor.shape)

datos = np.asarray([data[5],data[5],data[5]]).reshape(1,-1,1)
print(datos.shape)
p = model.predict(datos)
print(p.shape)
print(predictions_to_scenes(p))
tensor = get.reshape(1,-1)
print(tensor.shape)


def getActivations(data, number_layer):
    '''
    '''
    intermediate_tensor = []
    get_layer_output = K.function([model.layers[0].input],
                        [model.layers[number_layer].output])
    
    for audio in data:
        # get hidden representation
        layer_output = get_layer_output([audio])[0]
        tensor = layer_output.reshape(1, -1)
        intermediate_tensor.append(tensor[0])
    return intermediate_tensor


from sklearn.manifold import TSNE
from time import time


def toTSNE(intermediate_tensor, target, number_layer, ax):
    '''
    '''
    t0 = time()
    # define dimension of the graph
    tsne = TSNE(n_components=2, random_state=0)
    # assign activation from data and obtain tsne representation
    intermediates_tsne = tsne.fit_transform(intermediate_tensor)

    ax.scatter(x=intermediates_tsne[:, 0], y=intermediates_tsne[:, 1], c=target, alpha = 0.7, cmap=plt.cm.Spectral)
    ax.title("TSNE layer %i (time %.2fs)" % (number_layer, time() - t0))
    ax.colorbar()


activations = getActivations(data, 22)

fig = plt.figure(figsize=(5, 5))
toTSNE(activations, list_target, 22, plt)

