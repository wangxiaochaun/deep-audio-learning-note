# sounddevice包的辅助代码
# author: wangx
# date: 2020-04-23

import sounddevice as sd
import librosa
import numpy as np
import soundfile as sf

def generate_audio(filename=None):
    if filename == None:
        # 如果没有指定音频文件，就自己生成一个440Hz的正弦波
        sr = 44100 # Hz, sample rate
        f = 440 # Hz
        length = 5 # second
        myarray = np.arange(sr * length) # total signals
        myarray = np.sin(2 * np.pi * f / sr * myarray)

    else:
        # 如果指定了文件，就读文件
        myarray, sr = librosa.core.load(filename, sr=None)

    return myarray, sr


def play():
    # 播放声音
    # Note：这里的路径是windows format的
    y, sr = generate_audio('..\\media\\finalfantasy.wav')
    sd.play(y, sr)
    sd.wait()


def device_info():
    # 显示系统所有的声音设备
    sd.query_devices()


def device_rec():
    # 录制声音
    # duration: 时长
    # samplerate：采样率
    # channels：声道
    # dtype：精度，默认float32
    # 返回：np.array
    duration = 15 # second
    sr = 44100
    channels = 2
    myrecording = sd.rec(int(duration * sr), channels=channels)
    sd.wait()
    # librosa.output.write_wav('myvoice.wav', myrecording, sr=44100)
    sf.write('myvoice.wav', myrecording, 44100)


if __name__=='__main__':
    play()
    # device_rec()

