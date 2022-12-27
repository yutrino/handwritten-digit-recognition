# MNISTのデータで訓練、csv/testのデータでテスト(evaluate)する
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import glob
import numpy


# 訓練用データの作成
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))


# テスト用データを作成（csvのデータを配列data1に入れる）
files = glob.glob('csv/test/*.csv') 
NUM = len(files)
SIZE = 28
data1 = numpy.zeros((NUM, SIZE, SIZE))

# ダメなやり方（テストデータの配置がぐちゃぐちゃになる）
# i = 0
# for csv_name in files:
#     data1[i] = numpy.genfromtxt(csv_name, delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
#     i = i + 1

# OKなやり方
i = 0
for num in range(NUM):
    csv_name = 'csv/test/img_gray_resize' + str(i) + '.csv'
    data1[i] = numpy.genfromtxt(
        csv_name,
        delimiter=",",
        skip_header=0,
        skip_footer=0,
        usecols=(range(0, SIZE)))
    i = i + 1
    
# OKな方の意味
# data1[0] = numpy.genfromtxt('csv/test/img_gray_resize1.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[1] = numpy.genfromtxt('csv/test/img_gray_resize2.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[2] = numpy.genfromtxt('csv/test/img_gray_resize3.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[3] = numpy.genfromtxt('csv/test/img_gray_resize4.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[4] = numpy.genfromtxt('csv/test/img_gray_resize5.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[5] = numpy.genfromtxt('csv/test/img_gray_resize6.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[6] = numpy.genfromtxt('csv/test/img_gray_resize7.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[7] = numpy.genfromtxt('csv/test/img_gray_resize8.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[8] = numpy.genfromtxt('csv/test/img_gray_resize9.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
# data1[9] = numpy.genfromtxt('csv/test/img_gray_resize10.csv', delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))

data1 = data1.reshape((NUM, 28, 28, 1))
test_images = data1

# 正解データ
data2 = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_labels = data2

# ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
