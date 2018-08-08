# coding: UTF-8
import numpy as np
import glob  # ファイルを取得
from sklearn.tree import DecisionTreeClassifier # 決定木
from sklearn.model_selection import train_test_split #訓練データとテストデータを分ける

from sklearn.model_selection import cross_val_score #交差検証
import librosa  # MFCC


########ハイパーパラメータ########
depth = 3 #木の深さ(何回質問しますか？)
leaf = 4 #各ノードにおける最低サンプル数(枝に生えてる葉っぱの枚数)
###############################


def getMfcc(filename):
    y, sr = librosa.load(filename)
    return librosa.feature.mfcc(y=y, sr=sr)


# 正解ラベル
cls_dic = {'double_tap': 0, 'double_swipe': 1, 'single_swipe': 2, 'circle': 3, 'flip': 4}


#データとラベルのセットを作る
file_list = glob.glob('./dataset/*/*.wav')
dataset = []

for filename in file_list:
    label = filename.split('/')[-2] #'/'で区切られた文字列を後ろから2番目の塊を取ってくる
    dataset.append(np.array([filename, cls_dic[label]]))
dataset = np.array(dataset)
print('number of sample: ' + str(dataset.shape[0]))


#wavファイルの特徴量を抽出する
data = []
for i in range(dataset.shape[0]):
    feature = getMfcc(dataset[i,0])
    data.append(feature.reshape(-1))
data = np.array(data) #未知の入力
target = dataset[:,1] #教師ラベル

#決定木モデルを生成
tree = DecisionTreeClassifier(min_samples_leaf=leaf, max_depth=depth)

#交差検証
scores = cross_val_score(tree, data, target) #cv:分割数
# 各分割におけるスコア
print('Cross-Validation scores: {}'.format(scores))
# スコアの平均値
print('Average score: {:.3f}'.format(np.mean(scores)))
print('depth: ' + str(depth) + '\nmin_sample_leaf: ' + str(leaf))
