# csv2のデータで訓練、csv1のデータでテスト(evaluate)するプログラム
from tensorflow.keras import layers, models
import glob
import numpy


# 訓練用データの作成
files = glob.glob('csv2/*.csv') 
NUM = len(files)
SIZE = 28

data1 = numpy.zeros((NUM, SIZE, SIZE))
i = 0
for num in range(NUM):
    csv_name = 'csv2/img_gray_resize' + str(i + 1) + '.csv'
    data1[i] = numpy.genfromtxt(
        csv_name,
        delimiter=",",
        skip_header=0,
        skip_footer=0,
        usecols=(range(0, SIZE)))
    i = i + 1
data1 = data1.reshape((NUM, 28, 28, 1))
train_images = data1

data2 = numpy.zeros(NUM)
label = 0
for i in range(NUM):
    data2[i] = label
    if (i + 1) / (NUM / 10) == label + 1:
        label = label + 1
train_labels = data2


# テスト用データの作成
files = glob.glob('csv1/*.csv')
NUM = len(files)
SIZE = 28

data1 = numpy.zeros((NUM, SIZE, SIZE)) 
i = 0
for num in range(NUM):
    csv_name = 'csv1/img_gray_resize' + str(i + 1) + '.csv'
    data1[i] = numpy.genfromtxt(
        csv_name,
        delimiter=",",
        skip_header=0,
        skip_footer=0,
        usecols=(range(0, SIZE)))
    i = i + 1
data1 = data1.reshape((NUM, 28, 28, 1))
test_images = data1

data2 = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_labels = data2


# ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0


# モデルを作成
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# テスト用データでモデルを評価
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)