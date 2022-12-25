import glob
import numpy

files = glob.glob('csv1/*.csv') 
NUM = len(files)
SIZE = 28
data1 = numpy.zeros((NUM, SIZE, SIZE))


i = 0
for csv_name in files:
    data1[i] = numpy.genfromtxt(csv_name, delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
    i= i + 1

for j in range(0, NUM):
    for x in range(0, SIZE):
        for y in range(0, SIZE):
            print(int(data1[j, x, y]), end='')
        print('')
    print('')