# coding=UTF-8
# Author: Karol J. Piczak
# Date: 2015
# Modification: 2020-04-30

import numpy as np 
import librosa
import librosa.display
import pydub
import os
import glob
import random
from matplotlib import pyplot as plt
import pandas as pd


class Clip:
    """A single 5-sec long recording."""
    RATE = 44100 # All recording is ESC are 44.1 kHZ
    FRAME = 412  # Frame size in samples
    CATEGORIES = 50 # categories

    class Audio:
        """The actual audio data of the clip

            Uses a context manager to load/unload the raw audio data.
            This way clips can be processed sequentially with reasonable memory usage.
        """

        def __init__(self, path):
            # path: 音频文件的绝对路径
            self.path = path
        
        def __enter__(self):
            # Actual recordings are sometimes not frame accurate,
            # so we trim/overlay to exactly 5 seconds
            self.data = pydub.AudioSegment.silent(duration=5000)
            self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path)[0:5000])
            self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5) # convert to float
            return self

        def __exit__(self, exception_type, exception_value, trackback):
            if exception_type is not None:
                print(exception_type, exception_value, trackback)
            del self.data
            del self.raw


    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        # print('filename: {0}'.format(self.filename))
        self.path = os.path.abspath(filename)
        # print('path: {0}'.format(self.path))
        self.category = self.filename.split('.')[0].split('-')[-1]

        self.audio = Clip.Audio(self.path)

        with self.audio as audio:
            self._compute_mfcc(audio)
            self.__computer_zcr(audio)


    # 计算音频的短时特征（可扩展）
    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length,
        # 128 bands)
        self.melspectrogram = librosa.feature.melspectrogram(audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME)
        self.logamplitude = librosa.power_to_db(self.melspectrogram)
        # self.mfcc = librosa.feature.mfcc(audio.raw, sr=Clip.RATE, n_mfcc=13).T
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=13).T
    
    def __computer_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = librosa.feature.zero_crossing_rate(audio.raw)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME) : (index + 1) * Clip.FRAME]
    

def display_audio():
    all_recordings = glob.glob('..\\audio\\*.wav')
    clip = Clip(all_recordings[random.randint(0, len(all_recordings) - 1)])

    with clip.audio as audio:
        plt.subplot(2, 1, 1)
        plt.title('{0}: {1}'.format(clip.category, clip.filename))
        plt.plot(np.arange(0, len(audio.raw)) / 44100.0, audio.raw)

        plt.subplot(2, 1, 2)
        librosa.display.specshow(clip.logamplitude, sr=44100, x_axis='frames',
                                y_axis='linear', cmap='RdBu_r')
        plt.show()


"""更多关于数据集预览的图片
"""
def add_subplot_axes(ax, position):
    # 辅助函数
    box = ax.get_position()

    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]

    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], facecolor='w')

def plot_clip_overview(clip, ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_wavefrom = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])

    with clip.audio as audio:
        ax_wavefrom.plot(np.arange(0, len(audio.raw)) / float(Clip.RATE), audio.raw)
        ax_wavefrom.get_xaxis().set_visible(False)
        ax_wavefrom.get_yaxis().set_visible(False)
        ax_wavefrom.set_title('{0} \n {1}'.format(clip.category, clip.filename),
                            {'fontsize': 8}, y=1.03)
        librosa.display.specshow(clip.logamplitude, sr=Clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
        ax_spectrogram.get_xaxis().set_visible(False)
        ax_spectrogram.get_yaxis().set_visible(False)

def plot_clip_overviews():
    # 最外层函数
    categories = 10
    clips_shown = 7
    f, axes = plt.subplots(categories, clips_shown, figsize=(clips_shown * 2, categories * 2),
                        sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.35)

    clip_10 = load_dataset('..\\audio\\')

    for c in range(0, categories):
        for i in range(0, clips_shown):
            plot_clip_overview(clip_10[c][i], axes[c, i])
    
    plt.savefig('..\\audio\\plot_clip_overviews.png')


'''怎么处理新的数据集格式，最终还是按照类别搞了一个二维数组
'''
def load_dataset(name):
    # name:数据所在目录
    clips = []
    
    filenames = []
    for _, _, names in os.walk(name):
        for item in names:
            filenames.append(item)

    # print(len(filenames))

    # 默认应该是Clip.CATEGORIES，但是太吃内存，改用10
    for i in range(0, 10):
        categories = []
        for clip in filenames:
            if clip.split('.')[0].split('-')[-1] == str(i):
                categories.append(Clip(os.path.join(name, clip)))
        # print(len(categories))
        clips.append(categories)

    # sprint(np.shape(clips))
    
    print('All {0} recordings loaded.'.format(name))
    return clips


if __name__=="__main__":
    # display_audio()
    plot_clip_overviews()
    
    
