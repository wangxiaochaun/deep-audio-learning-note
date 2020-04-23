# 深度数字语音处理

> #### [引言](#引言)|[有用的工具](#有用的工具)|[知识体系](#知识体系)|[数学](#数学)|[传统特征](#传统特征)
>
><a href="GitHub last commit"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/wangxiaochaun/deep-audio-learning-note?style=flat-square" /></a>&nbsp;
 <a href="GitHub repo size"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/wangxiaochaun/deep-audio-learning-note?style=flat-square"></a>&nbsp;
 <a href="GitHub release"><img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/wangxiaochaun/deep-audio-learning-note?include_prereleases&style=flat-square"></a>&nbsp;

 ## 引言

 ## 有用的工具

 >#### [语音数据集](#语音数据集)|[奇怪的网站](#奇怪的网站)|[开发环境](#开发环境)|[第三方库](#第三方库)

 ### 语音数据集

| 数据集 | 描述 |
| :--- | :--- |
|[ESC-50 dataset](https://github.com/karolpiczak/ESC-50) | 该数据集由2000条语音组成，每条语音5秒，共分为50个语义类别（每个语义类别40条)|
||

### 奇怪的网站

[shields.io](https://shields.io/) : 一个用来生成各种markdown小图标的网站

### 开发环境

| Software    | Description    |
| :- | :- |
| Anaconda | pythong环境管理 |
| PyCharm  | python最好用的IDE之一，community即可|
| VS Code  | 万金油IDE+Text Editor，PyCharm coomunity无法使用版本控制后的github最佳选择之一|
| Xshell/Xftp | 远程服务器管理工具，可以支持本地tensorboard|

### 第三方库

>[soundfile](#soundfile)|[LibROSA](#LibROSA)|[Sounddevice](#Sounddevice)

#### soundfile

>跨平台的音频读写包
>
>`pip/conda install soundfile`, Linux可能需要 `apt-get install libsndfile1`
>
>Read/Write: `soundfile.read(filename)`, `soundfile.write(filename, data, samplerate)` (WAV, FLAC, OGG, MAT)
>
>Blocking: `soundfile.blocks(filename, blocksize, overlap)`
>
>SoundFile: `soundfile.SoundFile(filaname, I/O mode)` / `soundfile.close()`
>RAW files: 需要指定读入声音文件的类型
```python
import soundfile as sf

data, samplerate = sf.read('myfile.raw', channels=1, samperate=44100, subtype='FLOAT')
```
>x86机器默认`endian='LITTLE'`,老机器上可能需要指定`endian='BIG'`
>
>Virtual IO
```python
import io
import soundfile as sf
from urllib.request import urlopen

url = "http://tinyurl.com/shepard-risset"
data, samplerate = sf.read(io.BytesIO(urlopen(url).read()))
```
> 有可能出现写OGG文件为空的情况

#### LibROSA

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/mel_filter_bank.png" width="50%" height="50%" alt="Mel滤波器组示意图" title="Mel滤波器组示意图" align="right" />

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/log_mel_spectrogram.png" width="50%" height="50%" alt="Log-Mel功率谱示意图" title="Log-Mel功率谱示意图" align="right" />

>音乐和音频分析python包
>
>**Install**:`pip conda install librosa` or `conda install -c conda-forge librosa`
>
>*Windows*需要另外安装*ffmpeg*来支持更多的音频格式
>
>Note：LibROSA有很大一部分module和function是为music processing and analysis服务的

>**CoreIO and DSP**：包括音频处理(`load`,`resample`,`zero_crossings`)，谱表示(`stft`,`istft`,`cqt`,`icqt`),幅度变换(`amplitude_to_db`,`db_to_power`),时域与频域转换(`frames_to_samples`,`frames_to_time`,`samples_to_frames`),音高与调音
>
>**Display**：可视化功率谱、波形图等(`specshow`,`waveplot`)
>
>**Feature extraction**：梅尔功率谱图(`melspectrogram`)、mfcc(`mfcc`)、过零率(`zero_crossing_rate`)、特征转换(`inverse.mel_to_stft`,`inverse.mfcc_to_mel`)等
>
><font color=Green>*TODO待补完*</font>


[Example for LibROSA](https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/code/test_librosa.py)


#### Sounddevice

>sounddevice是一个与Numpy兼容的录音以及播放声音的包
>
>主要功能是播放和录音，以及一些交互式控制音频设备的方法

[Example for Sounddevice](https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/code/test_sounddevice.py)

## 知识体系
>#### [音频格式](#音频格式)|

### 音频格式

**采样频率**：1秒钟采样次数，大部分采样频率是44.1KHz
**采样位数（位深，精度，比特）**：类似图像的位数，CD音频是16bit
**比特率（音频位速，码率）**：单位时间内传送的比特数bps

```matlab
CD音频比特率 = 44.1KHz * 16bit * 2 channels = 1411.2Kbps
```
>16bit/44.1KHz是CD音频的采样
>
>24bit/48KHz是DVD音频的采样
>
>24bit/192KHz是蓝光中音频的采样

## 数学
>#### [STFT](#STFT)|[CQT](#CQT)



## 传统特征
>#### [色度特征](#色度特征)|

### 色度特征

色度特征是色度向量（Chroma vector）和色度谱（Chromagram）的统称。色度向量是一个含有12个元素的向量，这些元素分别代表一段时间（如1帧）内12个音级中的能量，不同八度（音高，pitch）的同一音级能量累加，色度图谱则是色度向量的序列（时间扩展）。




