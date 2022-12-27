# train用画像をcsvに変換してcsv2に格納
import glob
from PIL import Image
import numpy

for dir_number in range(10):
    files = glob.glob('png/train/' + str(dir_number) + '/*.png')

    SIZE = 28
    num = 0
    for img in files:
        image = Image.open(img).convert('L')
        image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        img_gray_resize = numpy.array(image)
        
        out_file = "csv/train/" + str(dir_number) + "/img_gray_resize" + str(num) + ".csv"
        numpy.savetxt(out_file, img_gray_resize, delimiter=',', fmt="%.5f")
        num = num + 1