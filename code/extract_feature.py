# 测试从音频文件提取传统特征
# 所设计的特征来自test_feature，包括mfccs,chroma,mel,contrast和tonnetz

from test_feature import extract_feature
import numpy as np 
import os


def parse_audio_files(parent_dir, file_ext='*.ogg'):
    