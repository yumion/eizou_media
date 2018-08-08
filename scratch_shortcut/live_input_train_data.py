# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt #入力波形を表示させる用
import os
from sklearn.externals import joblib #モデルを保存・呼び出し

##音声入力関連##
import pyaudio  # 音声入力
import wave #波形保存
from datetime import datetime #時刻を取得
from time import sleep #delaytime


###### 採 取 す る サ ン プ ル 数 #####

number_of_sample = 10

####################################


##音声入力に必要なパラメータ
FORMAT = pyaudio.paInt16
CHANNELS = 1  # モノラル
RATE = 44100  # サンプルレート
CHUNK = 2**10  # データ点数
RECORD_SECONDS = 2  # 録音する時間の長さ
THRESHHOLD = 0.01 #閾値を超えたら録音開始 max=1 0〜1の割合に規格化してある

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)


#出力先フォルダの確保
output_directory = 'live_dataset_' + datetime.today().strftime("%m%d%H%M")
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)


print('Recording...')
cnt = 0

while True:
    data = stream.read(CHUNK)
    #sleep(0.2) #マイクを開いたときに入るノイズを回避->これを入れるとオーバーフローしてしまう
    x = np.frombuffer(data, dtype="int16") / 32768.0

    if x.max() > THRESHHOLD:
        print("Start")

        filename = output_directory + '/' + datetime.today().strftime("%Y%m%d%H%M%S") + ".wav"
        print(cnt, filename)

        all = []
        all.append(data)
        for i in range(0, int(RATE / CHUNK * int(RECORD_SECONDS))):
            data = stream.read(CHUNK)
            all.append(data)
        data = b''.join(all)

        out = wave.open(filename,'w')
        out.setnchannels(CHANNELS)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(data)
        out.close()

        print("Saved")
        cnt += 1

        #sound = np.frombuffer(data, dtype="int16") / 32768.0
        #plt.figure(figsize=(15,3))
        #plt.plot(sound)
        #plt.show()->これを入れるとオーバーフローしてしまう

    if cnt > number_of_sample:
        break


stream.close()
audio.terminate()
print('-*-finished-*-')
