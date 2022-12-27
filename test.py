import numpy
import glob

# i = 1
# for num in range(10):
#     file = 'csv1/img_gray_resize' + str(i) + '.csv'
#     print(file)
#     i = i + 1

# NUM = 10
# i = 1
# for num in range(NUM):
#     csv_name = 'csv1/img_gray_resize' + str(i) + '.csv'
#     print(csv_name)
#     i = i + 1

files = glob.glob('csv/train/*.csv') 
NUM = len(files)
SIZE = 28

data2 = numpy.zeros(NUM)
label = 0
for i in range(NUM):
    data2[i] = label
    if (i + 1) / (NUM / 10) == label + 1:
        label = label + 1
train_labels = data2

print(NUM)
print(train_labels)