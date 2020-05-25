# ESC-50数据集的分析

>###### [单个音频波形图和对应声谱图的可视化](#单个音频波形图和对应声谱图的可视化)|[数据集样例的随机可视化](#数据集样例的随机可视化)

原始资料来自原作者的jupyter notebook<sup>[[jupyter]](#https://nbviewer.jupyter.org/github/karoldvl/paper-2015-esc-dataset/blob/master/Notebook/ESC-Dataset-for-Environmental-Sound-Classification.ipynb#Introduction)</sup>

这里的一些可视化工具，**可能**可以用来新的数据集分析，以及算法性能比较。

本人誊写的代码在这里<sup>[[source]]()</sup>

### 单个音频波形图和对应声谱图的可视化

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/display_audio.png" width="60%" height="60%" alt="单个音频波形图和对应声谱图" title="单个音频波形图和对应声谱图" align="center" />

### 数据集样例的随机可视化

以10类为例

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/plot_clip_overviews.png" width="100%" height="100%" alt="数据集样例" title="数据集样例" align="center" />

### 特征可视化

这里是用来分析所使用的音频特征的区分度（有效性）。可视化的方法有很多，但是背后其实是降维。以常用的音频特征MFCC为例，首先看一下在单个音频clip上的分布表现。这里用的seaborn的boxplot实现了一个特征分布的箱图。箱图是一个看起来高大上的统计指标，主要用来表征数据的分布情况。和简单的使用均值和方差相比，箱图可以反映更多的信息。seaborn是matplotlib的高级封装版，仅此而已。

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/plot_single_clip.png" width="100%" height="100%" alt="单clip特征箱图" title="单clip特征箱图" align="center" />

注意$MFCC_0$的意思不大，因为它的分布实在是有点平凡。但是注意到$MFCC_1$和$MFCC_2$的分布差异很明显。（在特征表示里，特征的各个维度区分度diversity越明显越好）

我们可以进一步分析$MFCC_1$在不同clips之间的表现。同样带上过零率。过零率的区分度和MFCCs还是很大的。

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/plot_feature_summary.png" width="100%" height="100%" alt="跨clips的特征可视化" title="跨clips的特征可视化" align="center" />

上面是第20类（Crying baby）的特征可视化，再看一看Rain这个类的特征可视化：

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/plot_feature_summary_new.png" width="100%" height="100%" alt="跨clips的特征可视化2" title="跨clips的特征可视化2" align="center" />

两张图一比较，就能发现这两类声音的特征$MFF_{1}$分布差异还是很明显的。

最后来一张50类的全家福：

<img src="https://github.com/wangxiaochaun/deep-audio-learning-note/blob/master/media/plot_all_features_aggregate.png" width="100%" height="100%" alt="所有类别的特征可视化" title="所有类别的特征可视化" align="center" />



