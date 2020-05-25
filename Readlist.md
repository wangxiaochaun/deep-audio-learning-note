# Deep Learning for Environmentally Robust Speech Recognition: An Overview of Recent Developments

## Deep learning for automatic speech recognition survey[<sup>[1]</sup>](#ref_1)

这篇综述质量尚可（废话，TIST的文章能不好）。TIST是什么？PIA打飞，百度去。通讯作者是Schuller，被引次数超过3万的大佬。听大佬的，少走弯路（Doge脸）

那这篇文章的关注点其实很聚焦：鲁棒的语音识别，也就是对有噪声语音的识别。所关注的技术是当红炸子鸡deep learning。

有噪语音的识别，可以类比失真图像的识别。因为图像也好，语音也罢，本质上都是信号。那么有噪语音的问题其实也可以类比为图像失真——一种非线性的降质过程。这篇文章主要关注的噪声是non-stational的，就是非稳定的、突发的噪声，类似图像中的油渍、空洞等。而传统失真，比如高斯白噪声，是稳定的，可以用线性方程描述的（时域的卷积=频域的乘积）。

ASR大致是两步走：第一步是从语音信号提取特征；第二步是构建声音模型，然后做一个分类。前者被作者称为front-end technique，后者被称为back-end technique。

### Front-end techniques

所谓front-end，有点像是feature representation。但是又不太一样。front-end technique 重点是从ASR的处理过程出发，更广泛地说，是speech application的一环。更像是pre-processing。其主要目的是从带噪声的声音中估计原始无失真信号。这就和图像去噪有点像了，或许和图像本征分解、去模糊等也类同的。本质上，还是一个非线性的、病态的问题。那么常用手段其实也就来了。

把深度学习用于front-end，无非就是如何训练的问题。目前（按这篇文章的2017年）主要还是有监督的。然后更细分一点，训练数据是用从原始无失真语音信号（clean speech）提取的特征，还是从原始无失真与噪声信号取mask后的特征，分为mapping-based和masking-based methods。

**Mapping-based method**的实质是解一个优化方程：

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

**Masking-based**试图学习从带噪声的语谱图$Y(n,f)$到时-频mask$M(n,f)$的回归函数：

$$
Y(n,f)\rightarrow M(n,f)
$$

Masking-based方法首先要圈定mask的形态。用什么样的mask？一种是binary-based mask。这种二值的T-F mask有点像Indicator function，给一个SNR的阈值，大于阈值的认为是干净语音主导的，小于阈值的认为是噪声主导的。这么搞出来的mask是一个二值矩阵（Ideal binary mask，IBM）；另一种是ratio-based mask，T-F mask的每个单元用的是干净语音和噪声语音的一种soft ratio（Ideal ratio mask，IRM）。相较而言，后者比前者保有的信息更多([[7]](#ref_7))。通过估计mask，就可以做到语音分解。T-F mask也可以用Mel-frequency语谱图，log-Mel-frequency语谱图替代。后者效果更好（一切荣耀归于Mel！）。另外，估计mask的训练方式也从DNN转向LSTM-RNN。

当然，IBM/IRM方法也存在上述致命缺陷，就是没有把纤维信息考虑进去。事实上，相位信息对语音增强是很有用的。因此，又有一种mask叫Phase-sensitive mask（PSM），把干净语音和噪声语音的相位角差引入mask表示；还有进一步保留相位信息的complex IRM。

确定的mask之后，就是定义目标函数了。Input $\vec{y}$是从噪声信号$Y(n,f)$得到的；目标$\vec{x}$是根据干净语音和噪声语音计算的T-F mask，$\theta$是网络参数。目标函数就是：

$$
\mathcal{J}(\theta)=\frac{1}{N}\sum_{n=1}^{N}||F(\vec{y}_{n}-M(n,f)||^{2}
$$

估计到mask$\hat{M}(n,f)=F(\vec{y}_{n})$后，可以把它与带噪信号的频谱卷积，然后再变换回时域，就能把干净信号分离出来（这里的mask实质上就是信号处理里的滤波器）。这类目标函数叫MA（Mask approximation）。

还有一类目标函数叫Signal approximation（SA），是比较卷积后信号频谱与干净信号频谱的MSE：

$$
\mathcal{J}(\theta)=\frac{1}{N}\sum_{n=1}^{N}||\vec{y}_{n}\otimes\hat{M}(n,f)-\vec{x}_{n}||^{2}
$$

使用SA比MA好一点，原因不明（本文认为是source separation）。但这里有一个思路可以借鉴，就是ResNet里为何使用Residual而不是sum？应该还是数值计算的问题。最后还有考虑相位信息的目标函数Phase-sensitive SA。

多任务网络也已经发现了这片热土——同时训练noise-speech和mask([[8]](#ref_8))。

### Back-end techniques

现在看看后端技术。这块对应语音处理的第二步，就是通过比较输入语音（语音特征）与预设的语音模型，来完成某项任务。在综述里，后端技术是指，输入的就是未经处理的带噪信号，直接通过神经网络来完成语音任务。和使用前端技术的方法相比，网络结构，甚至包括语音模型（acoustic model）都可能要改变。

我们知道传统语音任务这块是构建GMM-HMM模型，学习输入语音特征和预设语音模型的关系。这是个线性化的过程。当然，核方法也能做到非线性。可是用DNN的话，分类能力可以大大提高。最早的工作就是DNN-HMM。通过DNN学习到具有辨识能力的特征（discriminative feature），然后交给HMM预测。Multk-stream HMM是另一种结构创新，把DNN和传统GMM-HMM模型结合起来。做法是搞一个双流特征，一支走GMM的路，一支走RNN的路，然后再combine一下交给HMM。也可以用LSTM层替换全卷积层。DNN-HMM的局限性是它还是把命运交给了HMM。随着DNN高歌猛进，HMM已经沦为了basiline这样的角色([[9]](#ref_9))。

一种解决训练数据带噪声的方法就是扩大训练样本，使得样本带有各种噪声，这样提高学习模型的鲁棒性。这种方法太莽，学士们觉得还是修改预训练的语音模型（AM）比较优雅（model adaptation）。但是，修改AM的权重可能导致过拟合。所以，只能修改一部分网络参数。注意这里是指AM模型([[10]](#ref_10))。

除了修改语音模型之外，还可以在训练AM的时候就让它对噪声敏感（Noise-aware Training，NAT）。这是源于深度学习本身的技术，在输入的时候，除了输入信号（带噪或者不带噪），再附加上估计的噪声。这样，DNN就能学习带噪语音和噪声的关系，从而有益于后面的语音任务，比如音调识别。这个有点像图像处理里面的balance。那么怎么估计这个噪声？传统方法MMSE可行，i-vectors也行。注意i-vector的元素可以是MFCCs，也可以是DNN学习的特征等等。

注意这类噪声估计方法默认噪声在一个utterance（发声）内是稳态的，所以估计的噪声可以应用于整个utterance。实际上这不一定成立。学士们于是搞出了dynamic NAT，使得估计的噪声是时变的。

除了这些方法，多任务学习也可以用在AM学习上。网络可以是DNN，也可以是LSTM（手动摊手）。

### Joint Front- and Back-end训练技术

*中庸之为德也，其至矣乎。*既然有前端，有后端，就必然要有联合。

>Most research efforts on flighting with the environmental
noise in the past few year were separately made on the system
front end or back end.

总觉得原文这里的flighting用fighting更贴切。speech/feature enhancement和speech recognition单打独斗好多年。而且，前端用的metric（MSE，SNR等等）往往和后端任务要求的（预测精度等）大不一样。（当然有研究验证了前者指标和后者是正相关的，不然这前面的工作就白搞了。）

何不联合起来？最直接的方法就是用前端得到的增强后的信号来re-train那个pre-trained AM。这不需要改变pipeline，只不过是一个re-training。更优雅（意味着更复杂）的方式是joint DNN。比如，把两个预训练的DNN concate起来，第一个DNN做去噪，第二个做声调识别。这两个连起来做一个fine-tune([[11]](#ref_11))。

有串联就有并联。并联的人认为串联训练受限于单向通信（uni-directional communication）。于是就有平行网络，然后把每个子网络隐层激活给联结起来，再送到下一个隐层。([[12]](#ref_12))

最后就是end-to-end，一个网络做成所有事情。这里有一个经典工作，“very deep” CNN（不过和视觉的very deep比可差多了）。([[13]](#ref_13))

### Multi-channel techniques

文章这一节其实是个插曲，描述了一个新型的ASR场景：麦克风阵列（Microphone arrays）。这个应用场景很容易理解，有点像多视点相机阵列。不同阵列的信号，可以通过acoustical beamforming的方式给变成单通道的。beamforming是波束成形的意思。beamforming之后，再通过一个post-filtering增强一哈。最后和单通道方法一样，使用back-end技术完成任务。Deep learning也可以用在这里，或用来改进传统的beamforming和post-filtering，或用来做联合训练。单独back-end技术和前面的一样。

最后是总结：（1）从手工设计特征到保留整个信息（例如相位）的自动特征；（2）从front-和back-end分别改进到end-to-end。随着手持设备发展，语音数据越来越多。总之呢，刷分是大公司的事儿，我们呢得夹缝里找创新咯。

这篇综述到此为止。

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
[5]Pascual S, Bonafonte A, Serra J. SEGAN: Speech enhancement generative adversarial network[J]. arXiv preprint arXiv:1703.09452, 2017.
</font>

<font size=2><div id="ref_6"></div>
[6]Michelsanti D, Tan Z H. Conditional generative adversarial networks for speech enhancement and noise-robust speaker verification[J]. arXiv preprint arXiv:1709.01703, 2017.
</font>

<font size=2><div id="ref_7"></div>
[7]Wang Y, Narayanan A, Wang D L. On training targets for supervised speech separation[J]. IEEE/ACM transactions on audio, speech, and language processing, 2014, 22(12): 1849-1858.
</font>

<font size=2><div id="ref_8"></div>
[8]Huang P S, Kim M, Hasegawa-Johnson M, et al. Deep learning for monaural speech separation[C]//2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014: 1562-1566.
</font>

<font size=2><div id="ref_9"></div>
[9]Amodei, Dario, et al. "Deep speech 2: End-to-end speech recognition in english and mandarin." International conference on machine learning. 2016.
</font>

<font size=2><div id="ref_10"></div>
[10]Mirsamadi, Seyedmahdad, and John HL Hansen. "A study on deep neural network acoustic model adaptation for robust far-field speech recognition." Sixteenth Annual Conference of the International Speech Communication Association. 2015.
</font>

<font size=2><div id="ref_11"></div>
[11]Lee, Kang Hyun, et al. "Two-stage noise aware training using asymmetric deep denoising autoencoder." 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.
</font>

<font size=2><div id="ref_12"></div>
[12]Ravanelli, Mirco, et al. "A network of deep neural networks for distant speech recognition." 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
</font>

<font size=2><div id="ref_13"></div>
[13]Qian, Yanmin, et al. "Very deep convolutional neural networks for noise robust speech recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 24.12 (2016): 2263-2276.
</font>