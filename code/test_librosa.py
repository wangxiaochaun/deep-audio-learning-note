# 这个文档主要用来辅助学习LibROSA
# author: wangx
# data: 2020-04-20

import librosa
import numpy as np
import librosa.display
from matplotlib import pyplot as plt

def sample():
    filename = librosa.util.example_audio_file()

    # y - waveform; sr - sampling rate
    y, sr = librosa.load(filename)

    # default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(beat_times)


def advanced_sample():
    y, sr = librosa.load(librosa.util.example_audio_file())
    # 默认采样率是22050，如果要保留原始采样率，显式使用`sr=None`
    # print(np.shape(y))

    # 写音频
    librosa.output.write_wav('example.wav', y=y, sr=22050)

    # 计算时间序列的持续时间(秒)
    duration = librosa.get_duration(y=y)
    print('duration:{:.2f} seconds'.format(duration))

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussive into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    # print('tempo:{:.2f}'.format(tempo)) # tempo: 103.36
    # print(np.shape(beat_frames)) # [98.]
    # 心跳帧的序号，不规律



    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13) #[13, #分帧数]
    # print(np.shape(mfcc))

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc) #[13, #分帧数]
    # print(np.shape(mfcc_delta))

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames) #[13+13, #beat_frame+1]
    # print(np.shape(beat_mfcc_delta))
    # beat_frames一共是98帧，按照[0,T]范围，将音频分为99个区间，每个区间计算特征值
    # 的平均值(mean)


    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)


    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])


if __name__=='__main__':

    # sample()
    advanced_sample()
