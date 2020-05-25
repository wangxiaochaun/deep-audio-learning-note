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
import seaborn as sb


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
            self.raw = (np.frombuffer(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
            return(self)

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
        # print(np.shape(self.melspectrogram))
        self.logamplitude = librosa.core.amplitude_to_db(self.melspectrogram)
        # print(np.shape(self.logamplitude))
        # self.mfcc = librosa.feature.mfcc(audio.raw, sr=Clip.RATE, n_mfcc=13, hop_length=Clip.FRAME)
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=13).transpose()
        # print(np.shape(self.mfcc))
    
    def __computer_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))

        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(frame)))))

        self.zcr = np.asarray(self.zcr)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME) : (index + 1) * Clip.FRAME]

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)
    

def display_audio():
    all_recordings = glob.glob('D:\\Project\\ESC-50-master\\audio\\*.wav')
    clip = Clip(all_recordings[random.randint(0, len(all_recordings) - 1)]) # 随机挑一个clip

    with clip.audio as audio:
        plt.subplot(2, 1, 1)
        plt.title('{0}: {1}'.format(clip.category, clip.filename))
        plt.plot(np.arange(0, len(audio.raw)) / 44100.0, audio.raw)

        plt.subplot(2, 1, 2)
        librosa.display.specshow(clip.logamplitude, sr=44100, x_axis='frames',
                                y_axis='linear', cmap='RdBu_r')
        plt.show()


"""
更多关于数据集预览的图片
"""
def add_subplot_axes(ax, position):
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
    f, axes = plt.subplots(categories, clips_shown, figsize=(clips_shown * 2, categories * 2), sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.35)

    clip_50 = load_dataset('D:\\Project\\ESC-50-master\\audio\\')

    for c in range(0, categories):
        for i in range(0, clips_shown):
            plot_clip_overview(clip_50[c][i], axes[c, i])
    
    plt.savefig('D:\\Project\\deep-audio-learning-note\\media\\plot_clip_overviews.png')


'''
怎么处理新的数据集格式，最终还是按照类别搞了一个二维数组
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
    for i in range(0, Clip.CATEGORIES):
        categories = []
        for clip in filenames:
            if clip.split('.')[0].split('-')[-1] == str(i):
                print('loading {0}'.format(clip))
                categories.append(Clip(os.path.join(name, clip)))
        # print(len(categories))
        clips.append(categories)

    # print(np.shape(clips))
    
    print('All {0} recordings loaded.'.format(name))
    return clips


'''
单个clip上特征分布的可视化
'''
def plot_single_clip(clip):
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clip.mfcc)[1]))
    # print(col_names)
    MFCC = pd.DataFrame(clip.mfcc[:, :], columns=col_names)
    # print(MFCC)
    # print(np.shape(MFCC))

    f = plt.figure(figsize=(10, 6))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.75])
    ax_mfcc.set_xlim(-400, 400)
    # ax_mfcc.set_yticks(np.linspace(0, 12, num=13), MFCC.columns)
    ax_zcr = add_subplot_axes(ax, [0.0, 0.85, 1.0, 0.05])
    ax_zcr.set_xlim(0.0, 1.0)

    plt.title('Feature distribution across frames of a single clip ({0} : {1})'.format(clip.category, clip.filename), y=1.5)
    # print(MFCC.columns)
    # MFCC.boxplot(column=list(MFCC.columns), ax=ax_mfcc)
    sb.boxplot(data=MFCC, order=list(reversed(MFCC.columns)), orient='h', ax=ax_mfcc)
    sb.boxplot(pd.DataFrame(clip.zcr, columns=['ZCR']), orient='h', ax=ax_zcr)

    plt.savefig('D:\\Project\\deep-audio-learning-note\\media\\plot_single_clip.png', dpi=300, bbox_inches='tight')


# 单个clip的直方图可视化
def plot_single_feature_one_clip(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.10)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[2], ax=ax)


# 所有clips的箱图可视化
def plot_single_feature_all_clips(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.boxplot(data=feature, orient="h", order=list(reversed(feature.columns)), ax=ax)


# 所有clips的直方图可视化
def plot_single_feature_aggregate(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[1], ax=ax)


def generate_feature_summary(category, clip, coefficient):
    title = "{0} : {1}".format(clips_50[category][clip].category, clips_50[category][clip].filename)
    MFCC = pd.DataFrame()
    aggregate = []
    for i in range(0, len(clips_50[category])):
        MFCC[i] = clips_50[category][i].mfcc[:, coefficient]
        aggregate = np.concatenate([aggregate, clips_50[category][i].mfcc[:, coefficient]])
    
    f = plt.figure(figsize=(14, 12))
    f.subplots_adjust(hspace=0.6, wspace=0.3)

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

    ax1.set_xlim(0.0, 0.5)
    ax2.set_xlim(-100, 250)
    ax4.set_xlim(-100, 250)

    plot_single_feature_one_clip(clips_50[category][clip].zcr, 'ZCR distribution across frames\n{0}'.format(title), ax1)
    plot_single_feature_one_clip(clips_50[category][clip].mfcc[:, coefficient], "MFCC_{0} distribution across frames\n{1}".format(coefficient, title), ax2)
    plot_single_feature_all_clips(MFCC, 'Differences in MFCC_{0} distribution\nbetween clips of {1}'.format(coefficient, clips_50[category][clip].category), ax3)
    plot_single_feature_aggregate(aggregate, 'Aggregate MFCC_{0} distribution\n(bag-of-frames across all clips\nof {1}'.format(coefficient, clips_50[category][clip].category), ax4)

    plt.savefig('D:\\Project\\deep-audio-learning-note\\media\\plot_feature_summary_new.png', dpi=300, bbox_inches='tight')


# 特征可视化大杂烩
def plot_all_features_aggregate(clips, ax):
    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 0.85, 1.0])
    ax_zcr = add_subplot_axes(ax, [0.9, 0.0, 0.1, 1.0])
    
    sb.set_style('ticks')
    
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clips[0].mfcc)[1]))
    aggregated_mfcc = pd.DataFrame(clips[0].mfcc[:, :], columns=col_names)

    for i in range(1, len(clips)):
        aggregated_mfcc = aggregated_mfcc.append(pd.DataFrame(clips[i].mfcc[:, :], columns=col_names))
        
    aggregated_zcr = pd.DataFrame(clips[0].zcr, columns=['ZCR']) 
    for i in range(1, len(clips)):
        aggregated_zcr = aggregated_zcr.append(pd.DataFrame(clips[i].zcr, columns=['ZCR']))
    
    sb.despine(ax=ax_mfcc, right=True, left=False)
    ax.set_title('Aggregate distribution: {0}'.format(clips[0].category), y=1.10, fontsize=10)
    sb.boxplot(data=aggregated_mfcc, order=aggregated_mfcc.columns, ax=ax_mfcc)
    ax_mfcc.set_xticklabels(range(0, 13), rotation=90, fontsize=8)
    ax_mfcc.set_xlabel('MFCC', fontsize=8)
    ax_mfcc.set_ylim(-150, 200)
    ax_mfcc.set_yticks((-150, -100, -50, 0, 50, 100, 150, 200))
    ax_mfcc.set_yticklabels(('-150', '', '', '0', '', '', '', '200'))
    
    sb.despine(ax=ax_zcr, right=False, left=True)
    sb.boxplot(data=aggregated_zcr, order=aggregated_zcr.columns, ax=ax_zcr)
    ax_zcr.set_ylim(0.0, 0.5)
    ax_zcr.set_yticks((0.0, 0.25, 0.5))
    ax_zcr.set_yticklabels(('0.0', '', '0.5'))


if __name__=="__main__":
    # display_audio()
    # plot_clip_overviews()
    clips_50 = load_dataset('D:\\Project\\ESC-50-master\\audio\\')
    # plot_single_clip(clips_50[20][0])
    # generate_feature_summary(10, 0, 1)
    #----------------------------------
    categories = 50
    f, axes = plt.subplots(int(np.ceil(categories / 3.0)), 3, figsize=(14, categories * 1))
    f.subplots_adjust(hspace=0.8, wspace=0.4)

    for c in range(0, categories):
        ax = axes.flat[c]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plot_all_features_aggregate(clips_50[c], ax=ax)
    
    plt.savefig('D:\\Project\\deep-audio-learning-note\\media\\plot_all_features_aggregate.png', transparant=True, dpi=300, bbox_inches='tight')


    
    
