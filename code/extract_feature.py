# 测试从音频文件提取传统特征
# 所设计的特征来自test_feature，包括mfccs,chroma,mel,contrast和tonnetz

from test_feature import extract_feature
import numpy as np 
import os


def parse_audio_files(sub_dir):
    features, labels = np.empty((0, 193)), np.empty(0)

    for root, dirs, files in os.walk(sub_dir):
        for name in files:
            # 获得音频文件的绝对路径
            temp = os.path.join(sub_dir, name)
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(temp)
            except Exception as e:
                print("[Error] extract feature error in %s. %s" % (name, e))
                continue
            # horizon 在水平方向上平铺
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz]) # [193,]
            # vertical 在垂直方向上平铺，通过复用feature，实现叠加
            # 每一行是一个音频文件的feature
            features = np.vstack([features, ext_features])
            # 获得类别标签 0-50
            label = name.split('.')[0].split('-')[3]
            # 叠加
            labels = np.append(labels, label)
        print("Extract %s features done." % (sub_dir))
    
    return np.array(features), np.array(labels, dtype=np.int)
            

def parse_predict_file(filename):
    """这里需要输入绝对路径
    """
    features = np.empty((0, 193))
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(filename)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])
    print(features.shape)
    print("Extract %s features done." % (filename))
    return np.array(features)


def main():
    """提取数据集esc-50的特征，并保存成npy
    """
    # 数据集根目录
    data_path = "E:\\dataset\\audio\\"
    # 5-fold 结构
    # keep fold5 as testing data
    # 对传统方法，用其余4个folds训练
    # 对深度学习方法，使用4折交叉训练
    sub_dirs = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    for sub_dir in sub_dirs:        
        features, labels = parse_audio_files(data_path+sub_dir)
        np.save(sub_dir+'_feature.npy', features)
        np.save(sub_dir+'_labels.npy', labels)


if __name__=="__main__":
    main()