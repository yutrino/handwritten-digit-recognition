# csv/train/*のデータで訓練、csv/testのデータでテスト(evaluate)するプログラム
from tensorflow.keras import layers, models
import glob
import numpy
import matplotlib.pyplot as plt


# 訓練用データの作成
SIZE = 28
QUANTITY = 3000
NUM = QUANTITY * 10
data1 = numpy.zeros((NUM, SIZE, SIZE))
i = 0
for dir_number in range(10):
    files = glob.glob("csv/train/" + str(dir_number) + "/*.csv") 
    
    for num in range(QUANTITY):
        csv_name = "csv/train/" + str(dir_number) + "/img_gray_resize" + str(num) + ".csv"
        data1[i] = numpy.genfromtxt(csv_name, delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
        i = i + 1
data1 = data1.reshape((NUM, 28, 28, 1))
train_images = data1

data2 = numpy.zeros(NUM)
label = 0
for i in range(NUM):
    if i == QUANTITY * (label + 1):
        label = label + 1
    data2[i] = label
train_labels = data2


# テスト用データの作成
files = glob.glob("csv/test/*.csv")
NUM = len(files)
SIZE = 28

data1 = numpy.zeros((NUM, SIZE, SIZE)) 
for num in range(NUM):
    csv_name = "csv/test/img_gray_resize" + str(num) + ".csv"
    data1[num] = numpy.genfromtxt(csv_name, delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
data1 = data1.reshape((NUM, 28, 28, 1))
test_images = data1

data2 = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_labels = data2


# ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0


# モデルを作成
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.fit(train_images, train_labels, epochs=1)
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)



# グラフを作成
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

metrics = ['loss', 'accuracy']  # 使用する評価関数を指定

plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意

for i in range(len(metrics)):

    metric = metrics[i]

    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示
    
plt.show()  # グラフの表示