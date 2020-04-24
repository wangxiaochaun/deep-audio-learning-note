# 测试各种音频特征的辅助代码
# author: wangx
# date: 2020-04-24

import numpy as np 
import librosa
import soundfile as sf 


def extract_feature(file_name=None):
    if file_name:
        print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')

    # print('sample_rate: {}'.format(sample_rate)) # 44100

    if X.ndim > 1: X = X[:, 0] # 如果声道数大于1，只取第1个声道
    X = X.T # 转置

    # print(X.shape) # [7026095,]

    # short term fourier transform 
    stft = np.abs(librosa.stft(X))

    # print(stft.shape) # [1+stft/2, # segments]

    # chroma
    # 沿着分帧的维度取均值，最终向量是12维
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    
    # print(chroma.shape)

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # print(mfccs.shape)

    # mel-spectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    
    # print(mel.shape)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)


if __name__=='__main__':
    extract_feature('..\\media\\example.wav')