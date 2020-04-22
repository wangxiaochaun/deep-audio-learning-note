# 深度数字语音处理

> #### [引言](#引言)|[有用的工具](#有用的工具)|[知识体系](#知识体系)|[数学](#数学)|[传统特征](#传统特征)
>
><a href="GitHub last commit"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/wangxiaochaun/deep-audio-learning-note?style=flat-square" /></a>&nbsp;
 <a href="GitHub repo size"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/wangxiaochaun/deep-audio-learning-note?style=flat-square"></a>&nbsp;
 <a href="GitHub release"><img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/wangxiaochaun/deep-audio-learning-note?include_prereleases&style=flat-square"></a>&nbsp;

 ## 引言

 ## 有用的工具

 >#### [语音数据集](#语音数据集)|[奇怪的网站](#奇怪的网站)|[开发环境](#开发环境)|[3rd Party Lib](#3rd Party Lib)

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

### 3rd Party Lib

**soundfile**
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


## 知识体系



## 数学


## 传统特征




