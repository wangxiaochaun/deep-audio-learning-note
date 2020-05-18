# Deep Learning for Environmentally Robust Speech Recognition: An Overview of Recent Developments

## Deep learning for automatic speech recognition survey[<sup>[1]</sup>](#ref_1)

这篇综述质量尚可（废话，TIST的文章能不好）。TIST是什么？PIA打飞，百度去。通讯作者是Schuller，被引次数超过3万的大佬。听大佬的，少走弯路（Doge脸）

那这篇文章的关注点其实很聚焦：鲁棒的语音识别，也就是对有噪声语音的识别。所关注的技术是当红炸子鸡deep learning。

有噪语音的识别，可以类比失真图像的识别。因为图像也好，语音也罢，本质上都是信号。那么有噪语音的问题其实也可以类比为图像失真——一种非线性的降质过程。这篇文章主要关注的噪声是non-stational的，就是非稳定的、突发的噪声，类似图像中的油渍、空洞等。而传统失真，比如高斯白噪声，是稳定的，可以用线性方程描述的（时域的卷积=频域的乘积）。

ASR大致是两步走：第一步是从语音信号提取特征；第二步是构建声音模型，然后做一个分类。前者被作者称为front-end technique，后者被称为back-end technique。

### Front-end techniques

所谓front-end，有点像是feature representation。但是又不太一样。front-end technique 重点是从ASR的处理过程出发，更广泛地说，是speech application的一环。更像是pre-processing。其主要目的是从带噪声的声音中估计原始无失真信号。这就和图像去噪有点像了，或许和图像本征分解、去模糊等也类同的。本质上，还是一个非线性的、病态的问题。那么常用手段其实也就来了。

把深度学习用于front-end，无非就是如何训练的问题。目前（按这篇文章的2017年）主要还是有监督的。然后更细分一点，训练数据是用从原始无失真语音信号（clean speech）提取的特征，还是从原始无失真与噪声信号取mask后的特征，分为mapping-based和masking-based methods。

Mapping-based method的实质是解一个优化方程：

$$
\mathcal{J}(\theta)=\frac{1}{N}\sum_{n=1}^{N}||F(\vec{y}_{n})-\vec{x}_{n}||^{2}
$$

其中$\vec{y}$是输入特征，$\vec{x}$是目标特征，$\theta$就是网络要学习的参数了。

这里按照历史变革，很自然有三种可用的网络技术：

1. 自编码器或者玻尔兹曼机。这两种其实应算做是无监督方法（聚类）。但是在编码器之后加一个有监督的解码器，也就能做去噪了；
2. LSTM-RNN。这是语音界老大哥，好处是保留了上下文信息。坏处是太没特色。硬要说的话，LSTM过分依赖标注信息吧；([[2]](#ref_2))
3. CNN。学界前辈们发现语谱图其实也是一张图以后，就开始把CNN用在语谱图上了，各种语谱图，什么Mel谱图，log-Mel谱图都行。这类图一般横轴是时间，纵轴是频率信息。这样一来，用CNN卷积的话，从某种程度上算是保留了时空与的上下文结构。但是有一点——与幅值几乎同等重要的相位信息没了。([[3]](#ref_3))。2016年，WaveNet横空出世，据说可以保留所有音频信息。([[4]](#ref_4))

最后自然就是生成式网络啦——
4. GAN。据说可以秒杀传统方法，比如维纳滤波器（看来比LSTM/CNN还是差一丢丢）。([[5]](#ref_5)[[6]](#ref_6))

来看大佬总结的deep learning杀进语音识别带来的变化：

1. 超强的计算能力使得直接从原始数据获取特征表示成为可能；
2. 新型网络架构，如dilated CNN，可以显著降低计算负载；
3. ~~云计算使得计算上述任务称为可能~~(俺们小作坊不关心)

但是大佬指出一个问题：就是大家在看待语谱图的时候，是不是太自然地将他与普通图像等量齐观了呢？普通图像中，相邻像素往往是相似的；但是语谱图里，沿着时间方向很相像，但是沿着频率方向相关性很小(必须的……因为是经过STFT了嘛)。这块需要留意。


<font size=2><div id="ref_1"></div>
[1] Jiang, Dan-Ning, Lie Lu, Hong-Jiang Zhang, Jian-Hua Tao, and Lian-Hong Cai. “Music type classification by spectral contrast feature.” In Multimedia and Expo, 2002. ICME‘02. Proceedings. 2002 IEEE International Conference on, vol. 1, pp. 113-116. IEEE, 2002.</font>

<font size=2><div id="ref_2"></div>
[2]Wollmer, Martin, et al. "Feature enhancement by bidirectional LSTM networks for conversational speech recognition in highly non-stationary noise." 2013 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2013.</font>

<font size=2><div id="ref_3"></div>
[3]Park S R, Lee J. A fully convolutional neural network for speech enhancement[J]. arXiv preprint arXiv:1609.07132, 2016.
</font>

<font size=2><div id="ref_4"></div>
[4]Oord A, Dieleman S, Zen H, et al. Wavenet: A generative model for raw audio[J]. arXiv preprint arXiv:1609.03499, 2016.
</font>

<font size=2><div id="ref_5"></div>
Pascual S, Bonafonte A, Serra J. SEGAN: Speech enhancement generative adversarial network[J]. arXiv preprint arXiv:1703.09452, 2017.
</font>

<font size=2><div id="ref_6"></div>
Michelsanti D, Tan Z H. Conditional generative adversarial networks for speech enhancement and noise-robust speaker verification[J]. arXiv preprint arXiv:1709.01703, 2017.
</font>
