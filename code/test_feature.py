# 测试各种音频特征的辅助代码
# author: wangx
# date: 2020-04-24

import numpy as np 
import librosa
import librosa.display
import soundfile as sf

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt


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
    # 沿着分帧的维度取均值，最终向量是40维
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # print(mfccs.shape)

    # mel-spectrogram
    # 沿着分帧的维度取均值，最终向量是128维（等于mel滤波器的数量）
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    
    # print(mel.shape)

    # spectral contrast
    # 沿着分帧的维度取均值，最终向量是7维（n_bands+1)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    
    # print(contrast.shape)

    # tonnetz centroid feature
    # 沿着分帧的维度取均值，最终向量是6维
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    # print(tonnetz.shape)

    return mfccs, chroma, mel, contrast, tonnetz



def test_spectrogram():
    """声谱图可视化的测试程式
    """
    # 读取测试信号
    y, sr = librosa.load("..\\media\\example.wav")
    f, t, Sxx = signal.spectrogram(y, sr)
    plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [Sec]')
    plt.show()


def test_melspectrogram():
    """梅尔频谱可视化的测试程式
    """
    # 读取测试信号
    y, sr = librosa.load("..\\media\\example.wav")
    D = librosa.stft(y)
    D = np.abs(D) ** 2 # 功率谱
    S = librosa.feature.melspectrogram(S=D, sr=sr)

    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S, x_axis='time',
                             y_axis='mel', sr=sr,
                            )
    plt.colorbar()
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             x_axis='time',
                             y_axis='mel', sr=sr,
                            )
    plt.colorbar()
    plt.title('Mel-frequency spectrogram (dB)')
    plt.tight_layout()
    plt.show()


def test_mfcc():
    y, sr = librosa.load("..\\media\\example.wav")
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    # extract_feature('..\\media\\example.wav')
    # test_spectrogram()
    test_mfcc()