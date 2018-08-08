# coding: UTF-8
import numpy as np
import glob  # ファイルを取得
from sklearn.tree import DecisionTreeClassifier # 決定木
from sklearn.tree import export_graphviz  # 決定木の図示化
import graphviz #dotファイルを表示
from sklearn.model_selection import train_test_split #訓練データとテストデータを分ける
from sklearn.model_selection import cross_val_score #交差検証
import librosa  # MFCC
from sklearn.externals import joblib #モデルを保存


########ハイパーパラメータ########
depth = 3 #木の深さ(何回質問しますか？)
leaf = 4 #各ノーどにおける最低サンプル数
###############################

########モデルを保存############
flag = 0 #保存するなら'1'，しないなら'0'
######保存したモデルを使う#######
switch = 1 #保存したモデルを使うなら'1'．使わないなら'0'
##############################

###########正解ラベル###########
cls_dic = {'double_tap': 0, 'double_swipe': 1, 'single_swipe': 2, 'circle': 3, 'flip': 4}
##############################

##MFCCにかけて特徴量を抽出
def getMfcc(filename):
    y, sr = librosa.load(filename)
    return librosa.feature.mfcc(y=y, sr=sr)


#データとラベルのセットを作る
file_list = glob.glob('./dataset/*/*.wav')
dataset = []

for filename in file_list:
    label = filename.split('/')[-2] #'/'で区切られた文字列を後ろから2番目の塊を取ってくる
    dataset.append(np.array([filename, cls_dic[label]])) #[学習データ,教師ラベル]のセットを作成
dataset = np.array(dataset) #ndarrayに変更
print('number of sample: ' + str(dataset.shape[0]))



#訓練データとテストデータに分ける
train_set, test_set = train_test_split(dataset, test_size=0.2)
#print('trainng_set:',train_set.shape)
#print('test_set:',test_set.shape)


#データをMFCCにかけて特徴量抽出
x_train = []
for i in range(train_set.shape[0]):
    feature = getMfcc(train_set[i,0]) #MFCCにより特徴量を抽出
    #print(feature.shape)
    x_train.append(feature.reshape(-1)) #特徴量の次元を規格化
x_train = np.array(x_train) #2次元ndarrayに変換
y_train = train_set[:,1]



#テストデータをMFCCにかけて特徴量抽出
x_test = []
for i in range(test_set.shape[0]):
    feature = getMfcc(test_set[i,0])
    x_test.append(feature.reshape(-1))
x_test = np.array(x_test)
y_test = test_set[:,1]


#データセットの表示
#print('training_data:', x_train.shape, y_train.shape)
#print('test_data:', x_test.shape, y_test.shape)

# 決定木のモデル生成
if switch == 0:
    tree = DecisionTreeClassifier(min_samples_leaf=leaf, max_depth=depth)
    tree.fit(x_train, y_train)
    if flag == 1:
        joblib.dump(tree, 'decisiontree_mfcc.pkl')
    print('Accuracy on training: {:.3f}'.format(tree.score(x_train,y_train)))
    print('Accuracy on test: {:.3f}'.format(tree.score(x_test,y_test)))

else:
    tree = joblib.load('decisiontree_mfcc.pkl')
    print('Accuracy on test: {:.3f}'.format(tree.score(x_test,y_test)))


#決定木の可視化
export_graphviz(tree, out_file='ScratchInput_MFCC.dot', class_names=list(cls_dic.keys()), filled=False, rounded=True)
with open('ScratchInput_MFCC.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
