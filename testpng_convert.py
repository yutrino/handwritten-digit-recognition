# png_test3.py
# test用画像をcsvに変換してcsv1に格納
import glob
from PIL import Image
import numpy


files = glob.glob('png/test/*.png')

SIZE = 28
num = 0
for img in files:
    image = Image.open(img).convert('L')
    image = image.resize((SIZE, SIZE), Image.LANCZOS)
    img_gray_resize = numpy.array(image)
    
    out_file = "csv/test/img_gray_resize" + str(num) + ".csv"
    numpy.savetxt(out_file, img_gray_resize, delimiter=',', fmt="%.5f")
    num = num + 1