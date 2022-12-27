# MNISTデータをPNGで保存
import os
from torchvision import datasets

# 保存先フォルダ設定
rootdir = "MNIST"
traindir = rootdir + "/train"
testdir = rootdir + "/test"

# MNIST データセット読み込み
train_dataset = datasets.MNIST(root=rootdir, train=True, download=True)
test_dataset = datasets.MNIST(root=rootdir, train=False, download=True)

# 画像保存 train
number = 0
for img, label in train_dataset:
    savedir = traindir + "/" + str(label)
    os.makedirs(savedir, exist_ok=True)
    savepath = savedir + "/" + str(number).zfill(5) + ".png"
    img.save(savepath)
    number = number + 1
    print(savepath)

# 画像保存 test
number = 0
for img, label in test_dataset:
    savedir = testdir + "/" + str(label)
    os.makedirs(savedir, exist_ok=True)
    savepath = savedir + "/" + str(number).zfill(5) + ".png"
    img.save(savepath)
    number = number + 1
    print(savepath)